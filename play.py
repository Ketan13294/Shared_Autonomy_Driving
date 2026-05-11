"""Playable keyboard control with optional safety-policy indicator.

When run with --policy, loads a trained overtaking-safety classifier
and displays a real-time indicator showing if overtaking is predicted
as safe. Human keyboard controls remain active.
"""

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pygame

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None

from overtaking_environment import TwoLaneOvertakingEnv


class PolicyNet(nn.Module):
    """Network architecture matching train_overtaking_policy.SafetyPolicy.

    This model returns two logits: [safe_to_keep_speed, safe_to_overtake].
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

    def forward(self, x):
        return self.network(x)


def _get_keyboard_action() -> np.ndarray:
    """Read keyboard input and return continuous action [throttle, steering].
    
    Arrow keys:
      Up    -> throttle +1 (accelerate)
      Down  -> throttle -1 (brake)
      Left  -> steer -1 (left)
      Right -> steer +1 (right)
    """
    keys = pygame.key.get_pressed()
    
    throttle = 0.0
    if keys[pygame.K_UP]:
        throttle = 1.0
    elif keys[pygame.K_DOWN]:
        throttle = -1.0
    
    steering = 0.0
    if keys[pygame.K_LEFT]:
        steering = -1.0
    elif keys[pygame.K_RIGHT]:
        steering = 1.0
    
    return np.array([throttle, steering], dtype=np.float32)


def _flatten_observation(obs) -> np.ndarray:
    """Convert environment observation to flat float32 vector."""
    return np.asarray(obs, dtype=np.float32).reshape(-1)

 
def _stack_observations(stack: deque, stack_size: int) -> np.ndarray:
    """Concatenate stacked observations for temporal context."""
    frames = list(stack)
    while len(frames) < stack_size:
        frames.insert(0, frames[0].copy())
    return np.concatenate(frames, axis=0)


def load_policy(policy_path: Path, stack_size: int):
    """Load saved policy bundle and return model, normalization stats."""
    if torch is None:
        raise SystemExit("PyTorch is required to load a policy.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(policy_path, map_location=device, weights_only=False)
    input_dim = int(bundle["input_dim"])
    feature_mean = np.asarray(bundle["feature_mean"])
    feature_std = np.asarray(bundle["feature_std"])

    model = PolicyNet(input_dim=input_dim)
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device)
    model.eval()

    return model, feature_mean, feature_std, device


def predict_safety(model, feature_mean, feature_std, obs_stack, stack_size, device):
    """Run policy on stacked observations and return two safety probabilities.

    Returns (keep_prob, overtake_prob).
    """
    stacked = _stack_observations(obs_stack, stack_size)
    normed = (stacked - feature_mean) / feature_std
    x = torch.from_numpy(normed).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).cpu().numpy().ravel()
        probs = 1.0 / (1.0 + np.exp(-logits))
    # logits order: [keep, overtake]
    return float(probs[0]), float(probs[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Playable overtaking scenario with optional safety indicator.")
    parser.add_argument("--policy", type=Path, default=None, help="Path to saved policy bundle (optional).")
    parser.add_argument("--stack-size", type=int, default=4, help="Number of stacked observations for policy.")
    args = parser.parse_args()

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

    env = TwoLaneOvertakingEnv(config=config, render_mode="human")
    obs, info = env.reset()

    # Initialize policy if provided
    model = None
    feature_mean = None
    feature_std = None
    device = None

    if args.policy is not None:
        model, feature_mean, feature_std, device = load_policy(args.policy, args.stack_size)
        print(f"Loaded policy from {args.policy}")

        # Initialize observation stack
        obs_stack = deque(maxlen=args.stack_size)
        obs_stack.append(_flatten_observation(obs))
    else:
        obs_stack = None

    print(
        "Keyboard controls:\n"
        "  Up / Down    : accelerate / brake\n"
        "  Left / Right : steer left / right\n"
        "  Close the window or press Ctrl+C to quit\n"
    )

    if model is not None:
        print("Safety indicator: [SAFE] = prob >= 0.5, [UNSAFE] = prob < 0.5\n")

    try:
        step = 0
        while True:
            # Read keyboard input every frame
            action = _get_keyboard_action()
            
            obs, reward, terminated, truncated, info = env.step(action)

            # Update observation stack and check safety
            if model is not None:
                obs_stack.append(_flatten_observation(obs))
                keep_prob, overtake_prob = predict_safety(model, feature_mean, feature_std, obs_stack, args.stack_size, device)
                status_ot = "SAFE" if overtake_prob >= 0.5 else "UNSAFE"
                status_keep = "SAFE" if keep_prob >= 0.5 else "UNSAFE"
                print(f"[step {step}] Keep: {status_keep} (p={keep_prob:.3f}) | Overtake: {status_ot} (p={overtake_prob:.3f})")
            else:
                print(f"[step {step}] No policy loaded.")

            if terminated or truncated:
                obs, info = env.reset()
                if model is not None:
                    obs_stack.clear()
                    obs_stack.append(_flatten_observation(obs))
                print("Episode reset.")
                step = 0
            else:
                step += 1

            time.sleep(1.0 / env.unwrapped.config["policy_frequency"])
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()