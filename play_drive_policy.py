"""Play a trained driving policy.

Loads a saved policy produced by `train_drive_policy.py` and runs it in the
`TwoLaneOvertaking-v0` environment. Supports interactive manual `turn` input or
randomized `user_turn` commands for testing.
"""

from __future__ import annotations

import argparse
import random
import threading
import queue
import sys
from pathlib import Path
from collections import deque
import time

import gymnasium as gym
import numpy as np

import pygame

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:
    raise SystemExit("PyTorch is required. Install with 'pip install torch'.") from exc

from overtaking_environment import TwoLaneOvertakingEnv, register_two_lane_overtaking_env
from highway_env.road.lane import StraightLane


def _flatten_observation(obs) -> np.ndarray:
    # The environment returns structured observations; the policy expects a flat vector.
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _lane_id(vehicle) -> int:
    # Map the vehicle's lateral position to a lane index using the lane width.
    return int(round(vehicle.position[1] / StraightLane.DEFAULT_WIDTH))


def _get_keyboard_input() -> tuple[int, int, bool]:
    # Poll the current keyboard state every frame so held keys stay active.
    """Read keyboard input and return (user_turn, user_speed_delta, quit).
    
    Key mappings:
      't' -> lane change (user_turn=1)
      'w' -> speed up (user_speed_delta=1)
      's' -> speed down (user_speed_delta=-1)
      'q' -> quit
    """
    keys = pygame.key.get_pressed()
    
    user_turn = 1 if keys[pygame.K_t] else 0
    user_speed_delta = 0
    if keys[pygame.K_w]:
        user_speed_delta = 1
    elif keys[pygame.K_s]:
        user_speed_delta = -1
    
    quit_requested = keys[pygame.K_q]
    
    return user_turn, user_speed_delta, quit_requested


class ActorCritic(nn.Module):
    # Shared trunk followed by separate actor and critic heads.
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        # The shared trunk extracts a compact latent representation from the stacked input.
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        # The actor head predicts the two raw control values for continuous driving.
        self.actor = nn.Linear(hidden, act_dim)
        # The critic head estimates state value for logging and potential training use.
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def play(policy_path: Path, episodes: int, max_steps: int, stack_size: int, render: bool, user_turn_prob: float, manual_input: bool):
    # Register the custom highway-env environment before creating it.
    register_two_lane_overtaking_env()

    # Use the same environment settings as training so the checkpoint stays compatible.
    config = TwoLaneOvertakingEnv.default_config()
    config.update(
        {
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "dynamical": True,
            },
            "manual_control": True,
            "traffic_pattern": "MIXED",
        }
    )

    # Render in a window only when requested; otherwise run headless.
    render_mode = "human" if render else None
    # env = gym.make("TwoLaneOvertaking-v0", render_mode=render_mode, config=config)
    env = TwoLaneOvertakingEnv(config=config, render_mode=render_mode)


    try:
        # Load the saved policy checkpoint on the active device.
        data = torch.load(policy_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    except Exception as exc:
        # Newer PyTorch versions may default to weights_only=True which
        # rejects some saved objects. Retry with weights_only=False for
        # backwards compatibility when loading local trusted checkpoints.
        try:
            data = torch.load(policy_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
        except Exception:
            raise
    obs_dim = int(data.get("obs_dim"))
    act_dim = int(data.get("act_dim"))
    # Load stack_size from checkpoint; if not present, use the provided value
    checkpoint_stack_size = data.get("stack_size")
    if checkpoint_stack_size is not None:
        # Match the observation stacking width used during training.
        stack_size = int(checkpoint_stack_size)
        print(f"Loaded stack_size={stack_size} from checkpoint")

    # Rebuild the policy architecture and restore its learned weights.
    policy = ActorCritic(obs_dim=obs_dim, act_dim=act_dim)
    policy.load_state_dict(data["state_dict"])
    policy.eval()

    # Load log_std for continuous action distribution
    checkpoint_log_std = data.get("log_std")
    if checkpoint_log_std is None:
        # Fallback for old checkpoints without log_std
        log_std = torch.ones(act_dim) * 0.0
    else:
        log_std = checkpoint_log_std

    if manual_input:
        # Explain the held-key controls before entering the simulation loop.
        print("Starting interactive simulation.")
        print("  Hold 't' = lane change, hold 'w' = speed up, hold 's' = slow down, press 'q' = quit")

    # This outer loop keeps the script ready for multiple episodes if the flow is expanded later.
    for _ in range(1, 2):
        ep = 1
        # Reset the environment and initialize the stacked observation history.
        obs, _ = env.reset()
        flat = _flatten_observation(obs)
        obs_stack = deque(maxlen=stack_size)
        for _ in range(stack_size):
            obs_stack.append(flat.copy())

        total_reward = 0.0
        step = 0
        user_turn = 0
        user_speed_delta = 0
        while True:
            # Advance one control cycle at a time.
            step += 1
            if manual_input:
                # Use the current pressed-key state so held keys are not missed between frames.
                user_turn, user_speed_delta, quit_requested = _get_keyboard_input()
                if quit_requested:
                    print("\nQuitting.")
                    env.close()
                    return
            else:
                # During non-interactive runs, synthesize user commands probabilistically.
                user_turn = 1 if random.random() < user_turn_prob else 0
                user_speed_delta = random.choice([-1, 0, 1]) if random.random() < 0.1 else 0

            # Append the user inputs to the stacked observation so the policy can condition on them.
            obs_vec = np.concatenate(list(obs_stack), axis=0)
            obs_vec = np.concatenate([obs_vec, np.array([float(user_turn), float(user_speed_delta)], dtype=np.float32)], axis=0)
            obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0)

            with torch.no_grad():
                # The actor predicts the raw mean action, then we squash and sample a bounded continuous command.
                logits, value = policy(obs_tensor)
                # Apply tanh to squash to [-1, 1]
                mean = torch.tanh(logits)
                std = torch.exp(log_std)
                action = torch.clamp(mean + torch.randn_like(mean) * std, -1.0, 1.0).squeeze(0).cpu().numpy()
            # Send the continuous two-dimensional action directly to the environment.
            next_obs, reward, terminated, truncated, info = env.step(action)
            ego = env.unwrapped.controlled_vehicles[0]
            lane = _lane_id(ego)
            gap_ahead = float("inf")
            # compute gap ahead roughly
            ego_x = float(ego.position[0])
            best = float("inf")
            # Scan vehicles in the same lane to estimate how much room is ahead of the ego vehicle.
            for v in env.unwrapped.road.vehicles:
                if v is ego:
                    continue
                if _lane_id(v) != lane:
                    continue
                dx = float(v.position[0] - ego_x)
                if dx > 0 and dx < best:
                    best = dx
            if best < 1e6:
                gap_ahead = best

            # Print a compact step-by-step trace so it is easy to inspect policy behavior.
            print(f"Ep {ep} Step {step} | reward={reward:.3f} | total={total_reward+reward:.3f} | action={action} | user_turn={user_turn} | user_speed={user_speed_delta:+d} | lane={lane} | gap_ahead={gap_ahead:.2f} | crashed={info.get('crashed', False)}")

            total_reward += float(reward)

            # Shift the observation stack forward with the newest frame.
            flat_next = _flatten_observation(next_obs)
            obs_stack.append(flat_next)

            if render:
                # some renderers require calling env.render(); gymnasium with render_mode='human' may render on step
                try:
                    # Keep the render call guarded so headless backends do not crash the loop.
                    env.render()
                except Exception:
                    pass

            if terminated or truncated:
                # End the current episode as soon as the environment reports termination.
                print(f"Episode {ep} finished after {step} steps | total_reward={total_reward:.3f}")
                break
            # Pace the loop to the environment's policy frequency so keyboard input and rendering stay in sync.
            time.sleep(1.0 / env.unwrapped.config["policy_frequency"])
        else:
            print(f"Episode {ep} reached max steps | total_reward={total_reward:.3f}")

    # Always close the environment on exit so the window and simulator resources are released cleanly.
    env.close()


def parse_args():
    # Define the command-line interface for selecting the checkpoint and interactive mode.
    p = argparse.ArgumentParser(description="Play a trained driving policy.")
    p.add_argument("--policy", type=Path, default=Path("artifacts/drive_policy.pt"))
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--stack-size", type=int, default=3)
    p.add_argument("--render", action="store_true")
    p.add_argument("--user-turn-prob", type=float, default=0.0)
    p.add_argument("--manual_input", action="store_true", help="Interactive mode: type 't' for lane change, 's' for speed up, 'd' for slow down, 'q' to quit.")
    return p.parse_args()


def main():
    # Parse CLI options and dispatch into the interactive playback loop.
    args = parse_args()
    play(
        policy_path=args.policy,
        episodes=args.episodes,
        max_steps=args.max_steps,
        stack_size=args.stack_size,
        render=args.render,
        user_turn_prob=args.user_turn_prob,
        manual_input=args.manual_input,
    )


if __name__ == "__main__":
    main()
