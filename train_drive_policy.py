"""Train a driving policy that: stays in target lane, avoids off-road, slows for close obstacles,
responds to user "turn" input by changing lanes, and responds to user speed commands.

This implements a compact PPO trainer in PyTorch that conditions the policy on:
- binary `user_turn` input for lane changes
- ternary `user_speed_delta` input for speed control (-1=slow, 0=maintain, +1=speed up)

Safety constraints:
- Minimum safe distance from obstacles in same lane (enforced via reward shaping)
- Prevents speed up if too close to vehicle ahead
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import math
import random
import time

import gymnasium as gym
import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:
    raise SystemExit("PyTorch is required. Install with 'pip install torch'.") from exc

from overtaking_environment import register_two_lane_overtaking_env, TwoLaneOvertakingEnv
from overtaking_constants import ACTION_SLOWER, ACTION_FASTER, ACTION_IDLE, ACTION_LANE_LEFT, ACTION_LANE_RIGHT
from highway_env.road.lane import StraightLane


def _flatten_observation(obs) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _lane_id(vehicle) -> int:
    return int(round(vehicle.position[1] / StraightLane.DEFAULT_WIDTH))


def _nearest_same_lane_ahead(env, ego) -> float:
    ego_x = float(ego.position[0])
    best = float("inf")
    lid = _lane_id(ego)
    for v in env.road.vehicles:
        if v is ego:
            continue
        if _lane_id(v) != lid:
            continue
        dx = float(v.position[0] - ego_x)
        if dx > 0 and dx < best:
            best = dx
    return best if best < 1e6 else float("inf")


def _nearest_same_lane_ahead_speed(env, ego) -> float:
    ego_x = float(ego.position[0])
    best_dx = float("inf")
    best_speed = float("inf")
    lid = _lane_id(ego)
    for v in env.road.vehicles:
        if v is ego:
            continue
        if _lane_id(v) != lid:
            continue
        dx = float(v.position[0] - ego_x)
        if dx > 0 and dx < best_dx:
            best_dx = dx
            best_speed = float(getattr(v, "speed", 0.0))
    return best_speed


def _desired_lane_change_action(lane_id: int) -> int:
    # In this two-lane setup, lane 0 is the right lane and lane 1 is the left lane.
    # If the ego is in the right lane, a requested lane change should be a left turn.
    # If the ego is in the left lane, a requested lane change should be a right turn.
    return ACTION_LANE_LEFT if lane_id == 0 else ACTION_LANE_RIGHT


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)


@dataclass
class RolloutBatch:
    obs: list
    actions: list
    logps: list
    rewards: list
    dones: list
    values: list

    def clear(self):
        self.obs.clear(); self.actions.clear(); self.logps.clear(); self.rewards.clear(); self.dones.clear(); self.values.clear()


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    # `values` is expected to be length `len(rewards) + 1` where the last
    # element is the bootstrap value for the state following the final reward.
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        nextvalue = values[t + 1]
        delta = rewards[t] + gamma * nextvalue * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values[:-1]
    return advantages, returns


def train(
    epochs: int,
    steps_per_epoch: int,
    max_ep_len: int,
    lr: float,
    device: torch.device,
    stack_size: int,
    user_turn_prob: float,
    output: Path,
    verbose: bool = False,
    min_safe_dist_base: float = 15.0,
    resume: Path | None = None,
):
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

    env = TwoLaneOvertakingEnv(config=config, render_mode=None)

    print(f"Action Space: {env.action_space.shape[0]} | Observation Space: {env.observation_space.shape}")

    # Build observation shape by resetting once
    init_obs, _ = env.reset()
    flat = _flatten_observation(init_obs)
    obs_dim = flat.size * stack_size + 2  # +2 for user_turn and user_speed_delta
    act_dim = env.action_space.shape[0]

    policy = ActorCritic(obs_dim, act_dim).to(device)
    log_std = torch.nn.Parameter(torch.zeros(act_dim, device=device))
    optimizer = torch.optim.Adam(list(policy.parameters()) + [log_std], lr=lr)
    start_epoch = 1

    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        if "state_dict" not in checkpoint:
            raise ValueError(f"Checkpoint {resume} does not contain 'state_dict'.")

        checkpoint_obs_dim = checkpoint.get("obs_dim")
        checkpoint_act_dim = checkpoint.get("act_dim")
        if checkpoint_obs_dim is not None and int(checkpoint_obs_dim) != obs_dim:
            raise ValueError(
                f"Checkpoint obs_dim={checkpoint_obs_dim} does not match current obs_dim={obs_dim}. "
                "Use the same stack_size/user-input format used during checkpoint training."
            )
        if checkpoint_act_dim is not None and int(checkpoint_act_dim) != act_dim:
            raise ValueError(
                f"Checkpoint act_dim={checkpoint_act_dim} does not match current act_dim={act_dim}."
            )

        policy.load_state_dict(checkpoint["state_dict"])

        # Load log_std if available in checkpoint
        checkpoint_log_std = checkpoint.get("log_std")
        if checkpoint_log_std is not None:
            log_std.data = checkpoint_log_std.to(device)

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        checkpoint_epoch = checkpoint.get("epoch")
        if checkpoint_epoch is not None:
            start_epoch = int(checkpoint_epoch) + 1
        print(f"Resumed from {resume} at epoch {start_epoch}")

    batch = RolloutBatch([], [], [], [], [], [])

    def get_action_and_value(obs_tensor):
        logits, value = policy(obs_tensor)
        # Apply tanh to squash to [-1, 1]
        mean = torch.tanh(logits)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        # Clamp action to [-1, 1] to ensure it's in bounds
        action_clamped = torch.clamp(action, -1.0, 1.0)
        logp = dist.log_prob(action_clamped).sum(dim=-1)
        return action_clamped.squeeze(0).cpu().numpy(), logp.item(), value.item()

    def pack_obs(stack, user_turn_flag, user_speed_delta):
        arr = np.concatenate(list(stack), axis=0)
        return np.concatenate([
            arr,
            np.array([float(user_turn_flag), float(user_speed_delta)], dtype=np.float32)
        ], axis=0)

    for epoch in range(1, epochs + 1):
        obs_stack = deque(maxlen=stack_size)
        obs0, _ = env.reset()
        flat0 = _flatten_observation(obs0)
        for _ in range(stack_size):
            obs_stack.append(flat0.copy())

        ep_ret = 0.0
        ep_len = 0
        start_time = time.time()

        for step in range(steps_per_epoch):
            # sample a user turn command randomly with configured probability
            user_turn = 1 if random.random() < user_turn_prob else 0
            # sample a user speed command: -1 (slow down), 0 (maintain), +1 (speed up)
            # with equal probability for each command
            user_speed_delta = random.choice([-1, 0, 1])

            obs_vec = pack_obs(obs_stack, user_turn, user_speed_delta)
            obs_tensor = torch.from_numpy(obs_vec).float().to(device).unsqueeze(0)
            action, logp, value = get_action_and_value(obs_tensor)

            prev_lane = _lane_id(env.unwrapped.controlled_vehicles[0])

            next_obs, reward, terminated, truncated, info = env.step(action)

            # Reward shaping: strong penalty for off-road (y outside lane band)
            ego = env.unwrapped.controlled_vehicles[0]
            y = float(ego.position[1])
            lane_count = env.unwrapped.config.get("lanes_count", 2)
            lane_width = StraightLane.DEFAULT_WIDTH
            min_y = -lane_width * 0.5
            max_y = lane_width * (lane_count - 0.5)
            if y < min_y or y > max_y:
                reward -= 5.0
            
            # Potential-based reward: encourage staying centered or moving to target lane based on user_turn
            # Compute distance to nearest lane edge to discourage drifting toward boundaries
            
            # Determine target lateral position based on user_turn input
            if user_turn == 1:
                # User is requesting a lane change: target the adjacent lane center
                # In a two-lane setup, lanes are centered at 0 and lane_width
                if prev_lane == 0:
                    target_y = lane_width  # Move to left lane (higher y)
                else:
                    target_y = 0.0  # Move to right lane (lower y)
            else:
                # No lane change requested: maintain current lane center
                target_y = prev_lane * lane_width
            
            # Compute distance from current position to target lane center
            distance_to_target = abs(y - target_y)
            
            # Compute distance to nearest lane edge to discourage drifting toward boundaries
            distance_to_lower_edge = y - min_y
            distance_to_upper_edge = max_y - y
            distance_to_nearest_edge = min(distance_to_lower_edge, distance_to_upper_edge)
            
            # Apply a smooth penalty as vehicle approaches lane edges (within 0.3m)
            # The potential function increases penalty as we get closer to edges
            edge_buffer_threshold = 0.3  # meters
            if distance_to_nearest_edge < edge_buffer_threshold and distance_to_nearest_edge > 0:
                # Quadratic penalty: getting closer to edge increases penalty non-linearly
                lateral_potential_penalty = -2.0 * ((edge_buffer_threshold - distance_to_nearest_edge) / edge_buffer_threshold) ** 2
                reward += lateral_potential_penalty
            elif distance_to_nearest_edge <= 0:
                # Already off-road (covered by hard penalty above, but for safety)
                pass
            else:
                # Safe distance from edges: reward movement toward target lane
                # The reward increases as the vehicle gets closer to the target lateral position
                lane_width_half = lane_width / 2.0
                centering_bonus = 0.1 * max(0.0, 1.0 - distance_to_target / lane_width_half)
                reward += centering_bonus

            # Penalize collisions heavily (env already has collision reward but be explicit)
            if info.get("crashed", False) or terminated and reward < 0:
                reward -= 10.0

            # Safe following distance enforcement
            ego_speed = float(getattr(ego, "speed", 0.0))
            gap_ahead = _nearest_same_lane_ahead(env.unwrapped, ego)
            current_lane = _lane_id(ego)

            ###################################################
            # Minimum Distance Reward shaping
            # Minimum safe distance = 15 + 0.5 * speed (rough rule of thumb)
            min_safe_dist = min_safe_dist_base + 0.5 * ego_speed
            if gap_ahead != float("inf"):
                # Distance Penalty
                # Penalize being too close to vehicle ahead.
                if gap_ahead < min_safe_dist:
                    # Strong penalty if closing distance.
                    reward -= 6.0 * (min_safe_dist - gap_ahead) / min_safe_dist
                    # Velocity Penalty
                    # Additional penalty for high speed when gap is small (encourage safe deceleration).
                    reward -= 5.0 * max(0.0, ego_speed - 8.0) * (min_safe_dist - gap_ahead) / min_safe_dist
                    # Reward the slow-down action itself when the ego is inside the safe following distance.
                    # This is independent of user speed input so the policy learns to brake whenever needed.
                    if action[0] < 0.0:
                        reward += 1.0     
            # Reward compliance with user speed commands (if safe)
            elif (gap_ahead == float("inf") or gap_ahead >= min_safe_dist) and ((user_speed_delta == 1 and action[0] > 0.0) or (user_speed_delta == -1 and action[0] < 0.0)):
                # User wants to speed up and it's safe -> small reward for attempting acceleration
                reward += 1.0

            # Discourage lane change unless the user explicitly requested it.
            if current_lane != prev_lane and user_turn == 0:
                reward -= 3.0
            elif current_lane != prev_lane and user_turn == 1:
                desired_lane_action = _desired_lane_change_action(prev_lane)
                if action == desired_lane_action:
                    reward += 1.0



            # Reward for staying alive.
            reward += 0.05

            batch.obs.append(obs_vec)
            batch.actions.append(action)
            batch.logps.append(logp)
            batch.rewards.append(reward)
            batch.dones.append(bool(terminated or truncated))
            batch.values.append(value)

            if verbose:
                print(f"Step {step+1}/{steps_per_epoch} | reward={reward:.3f} | action={action} | user_turn={user_turn} | user_speed={user_speed_delta:+d}")

            ep_ret += reward
            ep_len += 1

            # push next observation
            flat_next = _flatten_observation(next_obs)
            obs_stack.append(flat_next)

            if terminated or truncated or ep_len >= max_ep_len:
                # reset episode
                obs0, _ = env.reset()
                flat0 = _flatten_observation(obs0)
                obs_stack.clear()
                for _ in range(stack_size):
                    obs_stack.append(flat0.copy())
                ep_ret = 0.0
                ep_len = 0

        # At epoch end, compute advantages
        # Append last bootstrap value 0.0 for simplicity
        values = np.array(batch.values + [0.0], dtype=np.float32)
        rewards = np.array(batch.rewards, dtype=np.float32)
        dones = np.array(batch.dones, dtype=np.float32)
        advantages, returns = compute_gae(rewards, values, dones)

        # Compute epoch reward statistics BEFORE clearing batch
        avg_reward = float(np.mean(rewards)) if len(rewards) > 0 else float("nan")
        std_reward = float(np.std(rewards)) if len(rewards) > 0 else float("nan")

        # Convert to tensors
        obs_tensor = torch.from_numpy(np.stack(batch.obs)).float().to(device)
        actions_tensor = torch.from_numpy(np.stack(batch.actions)).float().to(device)
        old_logps_tensor = torch.from_numpy(np.array(batch.logps)).float().to(device)
        adv_tensor = torch.from_numpy(advantages).float().to(device)
        ret_tensor = torch.from_numpy(returns).float().to(device)

        # PPO updates
        for _ in range(8):
            logits, vals = policy(obs_tensor)
            # Apply tanh to squash to [-1, 1]
            mean = torch.tanh(logits)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            # Clamp actions to [-1, 1]
            actions_clamped = torch.clamp(actions_tensor, -1.0, 1.0)
            logps = dist.log_prob(actions_clamped).sum(dim=-1)
            ratio = torch.exp(logps - old_logps_tensor)
            clip = 0.2
            pg_loss = -torch.mean(torch.min(ratio * adv_tensor, torch.clamp(ratio, 1 - clip, 1 + clip) * adv_tensor))
            v_loss = torch.mean((ret_tensor - vals) ** 2)
            entropy = torch.mean(dist.entropy().sum(dim=-1))
            loss = pg_loss + 0.5 * v_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        # Clear batch
        batch.clear()

        # Print epoch summary
        print(
            f"Epoch {epoch}/{epochs} | Time {time.time()-start_time:.1f}s | Steps {steps_per_epoch} | avg_reward={avg_reward:.3f} | std_reward={std_reward:.3f} | loss={loss.item():.3f}"
        )

    # Save model
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": policy.state_dict(),
        "log_std": log_std.data,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "stack_size": stack_size,
        "epoch": epoch
    }, output)
    env.close()
    print(f"Saved trained drive policy to {output}")


def parse_args():
    p = argparse.ArgumentParser(description="Train a driving policy conditioned on user turn input.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps-per-epoch", type=int, default=2000)
    p.add_argument("--max-ep-len", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--stack-size", type=int, default=3)
    p.add_argument("--user-turn-prob", type=float, default=0.05, help="Probability a user 'turn' input will be present at each step during training")
    p.add_argument("--output", type=Path, default=Path("artifacts/drive_policy.pt"))
    p.add_argument("--resume", type=Path, default=None, help="Path to a saved checkpoint to continue training from.")
    p.add_argument("--verbose", action="store_true", help="Print per-step rewards and actions to the console.")
    p.add_argument("--min-safe-dist-base", type=float, default=15.0, help="Base minimum safe distance in meters (total = base + 0.5*speed).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        max_ep_len=args.max_ep_len,
        lr=args.lr,
        device=device,
        stack_size=args.stack_size,
        user_turn_prob=args.user_turn_prob,
        output=args.output,
        resume=args.resume,
        verbose=args.verbose,
        min_safe_dist_base=args.min_safe_dist_base,
    )


if __name__ == "__main__":
    main()
