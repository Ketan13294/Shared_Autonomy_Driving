"""Collect overtaking trajectories and train an image-stack MLP policy.

This script uses the same overtaking environment as `main.py`, records stacked
Grayscale observations, drives the environment with the rule-based controller,
and trains a small MLP to imitate the controller's continuous actions:
`[acceleration, steering]`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required to train the overtaking policy. Install it with 'pip install torch'."
    ) from exc

from highway_env.road.lane import StraightLane
from overtaking_constants import ACTION_FASTER, ACTION_IDLE, ACTION_LANE_LEFT, ACTION_LANE_RIGHT, ACTION_SLOWER, FOLLOW_DISTANCE_MIN, FOLLOW_DISTANCE_OK, GAP_REQUIRED_AHEAD, GAP_REQUIRED_BEHIND, SAFE_RETURN_GAP

from overtaking_constants import (
    FOLLOW_DISTANCE_MIN,
    FOLLOW_DISTANCE_OK,
    GAP_REQUIRED_AHEAD,
    GAP_REQUIRED_BEHIND,
    PATTERN_CLUSTER,
    PATTERN_MIXED,
    PATTERN_OVERTAKING,
    PATTERN_SPARSE,
    PATTERNS,
    SAFE_RETURN_GAP,
)
from overtaking_controller import OvertakingController
from overtaking_environment import TwoLaneOvertakingEnv, register_two_lane_overtaking_env


OBSERVATION_SHAPE = (128, 64)


def _lane_id(vehicle) -> int:
    return int(round(vehicle.position[1] / StraightLane.DEFAULT_WIDTH))


def _signed_dist_nearest_in_lane(env, ego, lane_id: int) -> float:
    ego_x = float(ego.position[0])
    best_dist = 999.0
    found_any = False
    for vehicle in env.road.vehicles:
        if vehicle is ego:
            continue
        if _lane_id(vehicle) != lane_id:
            continue
        dx = float(vehicle.position[0] - ego_x)
        if not found_any or abs(dx) < abs(best_dist):
            best_dist = dx
            found_any = True
    return best_dist


def _scan_lane(env, ego, lane_id: int) -> tuple[float, float]:
    ego_x = float(ego.position[0])
    ahead_min = 500.0
    behind_min = 500.0
    for vehicle in env.road.vehicles:
        if vehicle is ego:
            continue
        if _lane_id(vehicle) != lane_id:
            continue
        dx = float(vehicle.position[0] - ego_x)
        if dx > 0:
            ahead_min = min(ahead_min, dx)
        else:
            behind_min = min(behind_min, -dx)
    return ahead_min, behind_min


def _flatten_observation(obs) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _teacher_user_input(env) -> tuple[int, int]:
    """Generate a user-input pair that produces overtaking behavior.

    Returns (user_speed_delta, user_turn):
    - user_speed_delta: -1 slow down, 0 maintain, +1 speed up
    - user_turn: -1 request right lane change, 0 no turn, +1 request left lane change
    """
    ego = env.controlled_vehicles[0]
    ego_lane = _lane_id(ego)
    other_lane = ego_lane ^ 1

    dist_to_blocker = _signed_dist_nearest_in_lane(env, ego, lane_id=ego_lane)
    gap_ahead_other, gap_behind_other = _scan_lane(env, ego, lane_id=other_lane)

    user_speed_delta = ACTION_IDLE
    if 0.0 < dist_to_blocker < FOLLOW_DISTANCE_MIN:
        user_speed_delta = ACTION_SLOWER
    elif dist_to_blocker > FOLLOW_DISTANCE_OK:
        user_speed_delta = ACTION_FASTER

    user_turn = ACTION_IDLE
    lane_change_safe = (
        gap_ahead_other > GAP_REQUIRED_AHEAD
        and gap_behind_other > GAP_REQUIRED_BEHIND
    )

    if ego_lane == 0:
        if 0.0 < dist_to_blocker < FOLLOW_DISTANCE_OK and lane_change_safe:
            user_turn = ACTION_LANE_RIGHT
    else:
        # Merge back after the overtake once the original lane is safely clear behind us.
        if gap_behind_other > SAFE_RETURN_GAP and lane_change_safe:
            user_turn = ACTION_LANE_LEFT

    return user_speed_delta, user_turn

class BehaviorMLP(nn.Module):
    """MLP that maps stacked grayscale observations to continuous actions."""

    def __init__(self, input_dim: int, action_dim: int = 2, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def collect_trajectories(
    episodes: int,
    max_steps: int,
    seed: int,
    stack_size: int,
    traffic_pattern: str,
) -> dict[str, np.ndarray]:
    register_two_lane_overtaking_env()
    env = gym.make(
        "TwoLaneOvertaking-v0",
        render_mode=None,
        config=_build_env_config(stack_size=stack_size, traffic_pattern=traffic_pattern),
    )
    controller = OvertakingController()

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    episode_ids: list[int] = []
    step_ids: list[int] = []
    patterns: list[str] = []

    try:
        for episode in range(episodes):
            pattern = traffic_pattern
            env.unwrapped.config["traffic_pattern"] = pattern
            obs, _ = env.reset(seed=seed + episode)
            controller.reset()
            env.unwrapped.controller = controller

            for step in range(max_steps):
                teacher_user_input = _teacher_user_input(env.unwrapped)
                action = env.unwrapped.controller.act(obs, env.unwrapped, teacher_user_input)

                observations.append(_flatten_observation(obs))
                actions.append(np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0))
                episode_ids.append(episode)
                step_ids.append(step)
                patterns.append(pattern)

                obs, reward, terminated, truncated, _ = env.step(action)
                rewards.append(float(reward))
                dones.append(bool(terminated or truncated))
                if terminated or truncated:
                    break
    finally:
        env.close()

    if not observations:
        raise RuntimeError("No trajectory samples were collected.")

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "episode_ids": np.asarray(episode_ids, dtype=np.int32),
        "step_ids": np.asarray(step_ids, dtype=np.int32),
        "patterns": np.asarray(patterns),
    }


def train_policy(
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[BehaviorMLP, dict[str, np.ndarray | int]]:
    if len(observations) < 8:
        raise ValueError("Need at least 8 samples to train the policy.")

    permutation = np.random.permutation(len(observations))
    observations = observations[permutation]
    actions = np.clip(actions[permutation], -1.0, 1.0)

    split_index = max(1, int(len(observations) * 0.8))
    train_obs = observations[:split_index]
    train_actions = actions[:split_index]
    val_obs = observations[split_index:]
    val_actions = actions[split_index:]

    obs_mean = train_obs.mean(axis=0)
    obs_std = train_obs.std(axis=0)
    obs_std[obs_std < 1e-6] = 1.0

    train_inputs = torch.from_numpy((train_obs - obs_mean) / obs_std).float()
    train_targets = torch.from_numpy(train_actions).float()
    val_inputs = torch.from_numpy((val_obs - obs_mean) / obs_std).float()
    val_targets = torch.from_numpy(val_actions).float()

    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = BehaviorMLP(input_dim=train_inputs.shape[1], action_dim=train_targets.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0

        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            batch_n = batch_inputs.shape[0]
            running_loss += float(loss.item()) * batch_n
            running_count += batch_n

        model.eval()
        with torch.no_grad():
            train_predictions = model(train_inputs.to(device)).cpu()
            val_predictions = model(val_inputs.to(device)).cpu() if len(val_inputs) else torch.empty(0)

        train_mae = float(torch.mean(torch.abs(train_predictions - train_targets)).item())
        train_loss = running_loss / max(1, running_count)

        if len(val_inputs):
            val_loss = float(criterion(val_predictions, val_targets).item())
            val_mae = float(torch.mean(torch.abs(val_predictions - val_targets)).item())
        else:
            val_loss = float("nan")
            val_mae = float("nan")

        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_mae={train_mae:.4f} | "
            f"val_loss={val_loss:.4f} | val_mae={val_mae:.4f}"
        )

    bundle = {
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "input_dim": int(train_inputs.shape[1]),
        "action_dim": int(train_targets.shape[1]),
    }
    return model, bundle


def save_dataset(dataset: dict[str, np.ndarray], output_path: Path, stack_size: int, traffic_pattern: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=dataset["observations"],
        actions=dataset["actions"],
        rewards=dataset["rewards"],
        dones=dataset["dones"],
        episode_ids=dataset["episode_ids"],
        step_ids=dataset["step_ids"],
        patterns=dataset["patterns"],
        stack_size=np.asarray(stack_size, dtype=np.int32),
        traffic_pattern=np.asarray(traffic_pattern),
        observation_shape=np.asarray(OBSERVATION_SHAPE, dtype=np.int32),
    )


def save_policy(model: BehaviorMLP, bundle: dict[str, np.ndarray | int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_mean": bundle["obs_mean"],
            "obs_std": bundle["obs_std"],
            "input_dim": bundle["input_dim"],
            "action_dim": bundle["action_dim"],
        },
        output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect overtaking trajectories and train an image-stack MLP policy."
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of trajectory episodes to collect.")
    parser.add_argument("--max-steps", type=int, default=600, help="Maximum steps per episode.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--stack-size", type=int, default=4, help="Number of grayscale frames stacked by the environment.")
    parser.add_argument(
        "--traffic-pattern",
        type=str,
        default=PATTERN_OVERTAKING,
        choices=[PATTERN_SPARSE, PATTERN_CLUSTER, PATTERN_MIXED, PATTERN_OVERTAKING],
        help="Traffic preset used while collecting trajectories.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=Path("artifacts/overtaking_trajectories.npz"),
        help="Where to save the collected trajectory dataset.",
    )
    parser.add_argument(
        "--policy-output",
        type=Path,
        default=Path("artifacts/overtaking_behavior_mlp.pt"),
        help="Where to save the trained policy checkpoint.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only generate the dataset and skip training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = collect_trajectories(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        stack_size=args.stack_size,
        traffic_pattern=args.traffic_pattern,
    )

    save_dataset(dataset, args.dataset_output, args.stack_size, args.traffic_pattern)
    print(
        f"Collected {len(dataset['observations'])} samples from {args.episodes} episodes | "
        f"saved dataset to {args.dataset_output}"
    )

    if args.collect_only:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, bundle = train_policy(
        observations=dataset["observations"],
        actions=dataset["actions"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
    )

    model.cpu()
    save_policy(model, bundle, args.policy_output)
    print(f"Saved trained policy to {args.policy_output}")


if __name__ == "__main__":
    main()
