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

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:
    raise SystemExit("PyTorch is required. Install with 'pip install torch'.") from exc

from overtaking_environment import TwoLaneOvertakingEnv, register_two_lane_overtaking_env
from highway_env.road.lane import StraightLane

try:
    from pynput import keyboard
except ModuleNotFoundError:
    keyboard = None


def _flatten_observation(obs) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _lane_id(vehicle) -> int:
    return int(round(vehicle.position[1] / StraightLane.DEFAULT_WIDTH))


class KeyboardState:
    """Track held keyboard input so commands stay active while a key is pressed."""

    def __init__(self) -> None:
        self.turn = 0
        self.speed_delta = 0
        self.quit = False
        self._listener = None

    def _on_press(self, key):
        try:
            char = key.char.lower()
        except AttributeError:
            return

        if char == "t":
            self.turn = 1
        elif char == "w":
            self.speed_delta = 1
        elif char == "s":
            self.speed_delta = -1
        elif char == "q":
            self.quit = True

    def _on_release(self, key):
        try:
            char = key.char.lower()
        except AttributeError:
            return

        if char == "t":
            self.turn = 0
        elif char in {"w", "s"}:
            self.speed_delta = 0

    def start(self) -> None:
        if keyboard is None:
            raise SystemExit(
                "pynput is required for held-key input. Install it with 'pip install pynput' or run without --manual_input."
            )
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def play(policy_path: Path, episodes: int, max_steps: int, stack_size: int, render: bool, user_turn_prob: float, manual_input: bool):
    register_two_lane_overtaking_env()

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

    render_mode = "human" if render else None
    # env = gym.make("TwoLaneOvertaking-v0", render_mode=render_mode, config=config)
    env = TwoLaneOvertakingEnv(config=config, render_mode=render_mode)


    try:
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
        stack_size = int(checkpoint_stack_size)
        print(f"Loaded stack_size={stack_size} from checkpoint")

    policy = ActorCritic(obs_dim=obs_dim, act_dim=act_dim)
    policy.load_state_dict(data["state_dict"])
    policy.eval()

    keyboard_state = None
    if manual_input:
        print("Starting continuous simulation.")
        print("  Hold 't' = lane change, hold 'w' = speed up, hold 's' = slow down, press 'q' = quit")
        keyboard_state = KeyboardState()
        keyboard_state.start()

    # for ep in range(1, episodes + 1):
    for _ in range(1,2):
        ep = 1
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
            step += 1
            # Check for held key state (non-blocking)
            if manual_input:
                if keyboard_state is None:
                    raise RuntimeError("Keyboard state not initialized.")
                if keyboard_state.quit:
                    print("\nQuitting.")
                    keyboard_state.stop()
                    env.close()
                    return
                user_turn = keyboard_state.turn
                user_speed_delta = keyboard_state.speed_delta
            # else:
            #     user_turn = 1 if random.random() < user_turn_prob else 0
            #     user_speed_delta = random.choice([-1, 0, 1]) if random.random() < 0.1 else 0

            obs_vec = np.concatenate(list(obs_stack), axis=0)
            obs_vec = np.concatenate([obs_vec, np.array([float(user_turn), float(user_speed_delta)], dtype=np.float32)], axis=0)
            obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0)

            with torch.no_grad():
                logits, value = policy(obs_tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze(0)
                action = int(np.argmax(probs))
            next_obs, reward, terminated, truncated, info = env.step(action)
            ego = env.unwrapped.controlled_vehicles[0]
            lane = _lane_id(ego)
            gap_ahead = float("inf")
            # compute gap ahead roughly
            ego_x = float(ego.position[0])
            best = float("inf")
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

            print(f"Ep {ep} Step {step} | reward={reward:.3f} | total={total_reward+reward:.3f} | action={action} | user_turn={user_turn} | user_speed={user_speed_delta:+d} | lane={lane} | gap_ahead={gap_ahead:.2f} | crashed={info.get('crashed', False)}")

            total_reward += float(reward)

            flat_next = _flatten_observation(next_obs)
            obs_stack.append(flat_next)

            if render:
                # some renderers require calling env.render(); gymnasium with render_mode='human' may render on step
                try:
                    env.render()
                except Exception:
                    pass

            if terminated or truncated:
                print(f"Episode {ep} finished after {step} steps | total_reward={total_reward:.3f}")
                break
            time.sleep(1.0 / env.unwrapped.config["policy_frequency"])
        else:
            print(f"Episode {ep} reached max steps | total_reward={total_reward:.3f}")

    if keyboard_state is not None:
        keyboard_state.stop()
    env.close()


def parse_args():
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
