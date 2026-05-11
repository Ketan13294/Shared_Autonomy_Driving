"""Train a Torch policy that predicts whether overtaking is safe.

This module collects rollouts from the custom TwoLaneOvertaking environment,
labels each timestep using simple rule-based gap checks (the existing
overtaking safety thresholds), and trains a small MLP to predict the
binary safe/unsafe label from a short stack of consecutive environment
observations. The saved bundle contains the model weights and the
normalization statistics required to run the policy later.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import deque

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

from overtaking_constants import (
    FOLLOW_DISTANCE_OK,
    GAP_REQUIRED_AHEAD,
    GAP_REQUIRED_BEHIND,
    PATTERNS,
)
from overtaking_controller import OvertakingController
from overtaking_environment import register_two_lane_overtaking_env


class SafetyPolicy(nn.Module):
    """Small MLP returning two logits for two safety predictions.

    Outputs logits in the order: [safe_to_keep_speed, safe_to_overtake].
    Use `torch.sigmoid` to convert each logit to a probability.
    """
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _lane_id(vehicle) -> int:
    # Convert the vehicle lateral position into an integer lane id
    return int(round(vehicle.position[1] / StraightLane.DEFAULT_WIDTH))


def _signed_dist_nearest_in_lane(env, ego, lane_id: int) -> float:
    # Signed distance (dx) to the nearest vehicle in `lane_id`.
    # Positive means the vehicle is ahead of ego, negative behind.
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
    # Return (gap_ahead, gap_behind) measured from ego's x position
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
    # Convert the environment observation (which may be an array or
    # nested structure) into a flat float32 vector for the network.
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _stack_observations(stack: deque[np.ndarray], stack_size: int) -> np.ndarray:
    # Build a temporal input by concatenating the most recent
    # `stack_size` flattened observations. If the stack is shorter than
    # `stack_size` (beginning of an episode) the earliest frame is
    # duplicated to pad the stack so the input size is consistent.
    frames = list(stack)
    if not frames:
        raise ValueError("Observation stack is empty.")

    while len(frames) < stack_size:
        frames.insert(0, frames[0].copy())

    return np.concatenate(frames, axis=0)

def _safe_to_change_lane(env) -> int:
    """Return 1 if it is safe to change into the adjacent lane, 0 otherwise.

    Rules (approximate):
    - If there are no vehicles in the adjacent lane, it's safe.
    - If there are vehicles in both lanes, check the nearest vehicle ahead and behind in the target lane, and the nearest vehicle ahead in the current lane:
      - It's safe to change lanes if:
        - The nearest vehicle ahead in the current lane is at a comfortable distance (> FOLLOW_DISTANCE_OK), AND
        - The gap ahead in the target lane is greater than GAP_REQUIRED_AHEAD, AND
        - The gap behind in the target lane is greater than GAP_REQUIRED_BEHIND.
    """
    ego = env.controlled_vehicles[0]
    ego_lane_id = _lane_id(ego)

    # Find any other lane id present among vehicles; prefer the first one that's not ego's lane.
    other_lane_id = None
    for v in env.road.vehicles:
        if v is ego:
            continue
        lid = _lane_id(v)
        if lid != ego_lane_id:
            other_lane_id = lid
            break

    # If no other lanes/vehicles are present, consider it safe to change lanes.
    if other_lane_id is None:
        return 1

    # Check presence of vehicles in both lanes (excluding ego).
    presence_in_ego_lane = any(_lane_id(v) == ego_lane_id and v is not ego for v in env.road.vehicles)
    presence_in_other_lane = any(_lane_id(v) == other_lane_id and v is not ego for v in env.road.vehicles)

    # If vehicles are not present in both lanes (i.e., either lane is empty), it's safe to change lanes.
    if not (presence_in_ego_lane and presence_in_other_lane):
        return 1

    # Nearest vehicle ahead in ego lane (blocker metric) and gaps in the target lane.
    dist_to_blocker = _signed_dist_nearest_in_lane(env, ego, lane_id=ego_lane_id)
    gap_ahead_target, gap_behind_target = _scan_lane(env, ego, lane_id=other_lane_id)

    return int(
        dist_to_blocker > FOLLOW_DISTANCE_OK
        and gap_ahead_target > GAP_REQUIRED_AHEAD
        and gap_behind_target > GAP_REQUIRED_BEHIND
    )

def _safe_to_overtake(env) -> int:
    # Determine whether it is safe to overtake into the adjacent lane.
    ego = env.controlled_vehicles[0]
    ego_lane_id = _lane_id(ego)

    # Find any other lane id present among vehicles; prefer the first one that's not ego's lane.
    other_lane_id = None
    for v in env.road.vehicles:
        if v is ego:
            continue
        lid = _lane_id(v)
        if lid != ego_lane_id:
            other_lane_id = lid
            break

    # If no other lanes/vehicles are present, consider it safe to overtake.
    if other_lane_id is None:
        return 1


    # Check presence of vehicles in both lanes (excluding ego).
    presence_in_ego_lane = any(_lane_id(v) == ego_lane_id and v is not ego for v in env.road.vehicles)
    presence_in_other_lane = any(_lane_id(v) == other_lane_id and v is not ego for v in env.road.vehicles)

    # If vehicles are not present in both lanes (i.e., either lane is empty), it's safe to overtake.
    if not (presence_in_ego_lane and presence_in_other_lane):
        return 1

    # Nearest vehicle ahead/behind in ego lane (blocker metric) and gaps in the target lane.
    dist_to_blocker = _signed_dist_nearest_in_lane(env, ego, lane_id=ego_lane_id)
    gap_ahead_target, gap_behind_target = _scan_lane(env, ego, lane_id=other_lane_id)

    return int(
        0.0 < dist_to_blocker <= FOLLOW_DISTANCE_OK
        and gap_ahead_target > GAP_REQUIRED_AHEAD
        and gap_behind_target > GAP_REQUIRED_BEHIND
    )


def _safe_to_keep_speed(env) -> int:
    """Return 1 if it is safe to keep the current speed in the current lane.

    Rules (approximate):
    - If the ego is far from all vehicles (beyond FOLLOW_DISTANCE_OK), it's safe.
    - Otherwise, if the nearest vehicle in the ego lane is ahead and at a
      comfortable distance (> FOLLOW_DISTANCE_OK) it's safe.
    - If the ego is closing quickly on a slower vehicle (time-to-collision < 4s), it's unsafe.
    """
    ego = env.controlled_vehicles[0]
    ego_x = float(ego.position[0])
    ego_speed = float(getattr(ego, "speed", 0.0))

    # Minimal absolute longitudinal distance to any other vehicle
    min_abs_dx = 1e6
    for v in env.road.vehicles:
        if v is ego:
            continue
        dx = float(v.position[0] - ego_x)
        min_abs_dx = min(min_abs_dx, abs(dx))

    if min_abs_dx > FOLLOW_DISTANCE_OK:
        return 1

    # Check nearest vehicle in ego's lane
    lane_id = _lane_id(ego)
    dist_same_lane = _signed_dist_nearest_in_lane(env, ego, lane_id=lane_id)
    if dist_same_lane > FOLLOW_DISTANCE_OK:
        return 1

    # If there is a vehicle ahead in the same lane, check time-to-collision
    if dist_same_lane > 0 and dist_same_lane < 999.0:
        # find that vehicle's speed
        ahead_speed = None
        ego_x = float(ego.position[0])
        best_dx = 1e6
        for v in env.road.vehicles:
            if v is ego:
                continue
            if _lane_id(v) != lane_id:
                continue
            dx = float(v.position[0] - ego_x)
            if dx > 0 and dx < best_dx:
                best_dx = dx
                ahead_speed = float(getattr(v, "speed", 0.0))

        if ahead_speed is not None:
            rel_speed = ego_speed - ahead_speed
            # if ego is faster and closing fast, compute time to collision
            if rel_speed > 0:
                ttc = dist_same_lane / rel_speed if rel_speed > 1e-6 else float("inf")
                if ttc < 4.0:
                    return 0
    return 1


def collect_dataset(
    episodes: int,
    max_steps: int,
    seed: int,
    exploration_rate: float,
    stack_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    register_two_lane_overtaking_env()
    env = gym.make("TwoLaneOvertaking-v0", render_mode=None)
    controller = OvertakingController()
    rng = np.random.default_rng(seed)

    features: list[np.ndarray] = []
    labels: list[list[int]] = []

    # Rollout collection: run `episodes` episodes and record
    # (stacked_observation, label) pairs at each timestep.
    try:
        for episode in range(episodes):
            pattern = PATTERNS[episode % len(PATTERNS)]
            env.unwrapped.config["traffic_pattern"] = pattern
            obs, info = env.reset(seed=seed + episode)
            controller.reset()

            # Initialize an observation stack for temporal context.
            obs_stack: deque[np.ndarray] = deque(maxlen=stack_size)
            initial_frame = _flatten_observation(obs)
            obs_stack.append(initial_frame)

            for _ in range(max_steps):
                # Append the current stacked input and its rule-based
                # safety label. The stack contains the most recent
                # observations (flattened) so the model can infer
                # velocities and motion.
                features.append(_stack_observations(obs_stack, stack_size))
                # Two supervision signals per timestep:
                #  - safe to keep current speed in current lane
                #  - safe to perform an overtaking maneuver now
                keep_label = _safe_to_keep_speed(env.unwrapped)
                overtake_label = _safe_to_overtake(env.unwrapped)
                change_lane_label = _safe_to_change_lane(env.unwrapped)
                labels.append([keep_label, change_lane_label])

                if rng.random() < exploration_rate:
                    action = env.action_space.sample()
                else:
                    action = controller.act(obs, env.unwrapped)

                obs, reward, terminated, truncated, info = env.step(action)
                # Push the latest raw observation onto the stack for the
                # next training sample. We flatten here to keep the
                # stored frames compact and consistent.
                obs_stack.append(_flatten_observation(obs))
                if terminated or truncated:
                    break
    finally:
        env.close()

    if not features:
        raise RuntimeError("No training samples were collected.")

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def train_policy(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[SafetyPolicy, dict[str, np.ndarray | float]]:
    if len(features) < 8:
        raise ValueError("Need at least 8 samples to train the policy.")

    permutation = np.random.permutation(len(features))
    features = features[permutation]
    labels = labels[permutation]

    split_index = max(1, int(len(features) * 0.8))
    train_features = features[:split_index]
    train_labels = labels[:split_index]
    val_features = features[split_index:]
    val_labels = labels[split_index:]

    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0

    train_inputs = torch.from_numpy((train_features - feature_mean) / feature_std).float()
    # labels are two-dimensional: (N, 2)
    train_targets = torch.from_numpy(train_labels).float()
    val_inputs = torch.from_numpy((val_features - feature_mean) / feature_std).float()
    val_targets = torch.from_numpy(val_labels).float()

    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=batch_size,
        shuffle=True,
    )

    model = SafetyPolicy(input_dim=train_inputs.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0

        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            # logits: (B,2), targets: (B,2)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_inputs.shape[0]
            running_loss += float(loss.item()) * batch_size_actual
            running_count += batch_size_actual

        model.eval()
        with torch.no_grad():
            train_logits = model(train_inputs.to(device)).cpu()
            val_logits = model(val_inputs.to(device)).cpu() if len(val_inputs) else torch.empty(0)

        # Compute per-task accuracies
        train_pred = (torch.sigmoid(train_logits) >= 0.5).float()
        train_accuracy_keep = float((train_pred[:, 0] == train_targets[:, 0]).float().mean().item())
        train_accuracy_overtake = float((train_pred[:, 1] == train_targets[:, 1]).float().mean().item())
        train_accuracy = 0.5 * (train_accuracy_keep + train_accuracy_overtake)

        if len(val_inputs):
            val_pred = (torch.sigmoid(val_logits) >= 0.5).float()
            val_accuracy_keep = float((val_pred[:, 0] == val_targets[:, 0]).float().mean().item())
            val_accuracy_overtake = float((val_pred[:, 1] == val_targets[:, 1]).float().mean().item())
            val_accuracy = 0.5 * (val_accuracy_keep + val_accuracy_overtake)
            val_loss = float(criterion(val_logits, val_targets).item())
        else:
            val_accuracy = float("nan")
            val_loss = float("nan") 

        avg_loss = running_loss / max(1, running_count)
        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"train_loss={avg_loss:.4f} | "
            f"train_acc_keep={train_accuracy_keep:.3f} "
            f"keep_overtake_avg={train_accuracy:.3f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.3f}"
        )

    bundle = {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "input_dim": int(train_inputs.shape[1]),
    }
    return model, bundle


def save_policy(model: SafetyPolicy, bundle: dict[str, np.ndarray | float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": bundle["feature_mean"],
            "feature_std": bundle["feature_std"],
            "input_dim": bundle["input_dim"],
        },
        output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Torch policy that predicts whether overtaking is safe."
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of rollout episodes to collect.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per rollout episode.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--exploration-rate", type=float, default=0.25, help="Probability of sampling a random action during data collection.")
    parser.add_argument("--stack-size", type=int, default=4, help="Number of consecutive environment snapshots to stack per training example.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overtaking_safety_policy.pt"),
        help="Where to save the trained policy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    features, labels = collect_dataset(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        exploration_rate=args.exploration_rate,
        stack_size=args.stack_size,
    )

    positive_rate = float(labels.mean()) if len(labels) else 0.0
    print(f"Collected {len(features)} samples | positive_rate={positive_rate:.3f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, bundle = train_policy(
        features=features,
        labels=labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
    )

    model.cpu()
    save_policy(model, bundle, args.output)
    print(f"Saved trained policy to {args.output}")


if __name__ == "__main__":
    main()