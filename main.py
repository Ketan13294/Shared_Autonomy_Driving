"""Command-line entry point for the overtaking simulation."""

import argparse
import time

import gymnasium as gym

from highway_env.road.lane import StraightLane

from overtaking_constants import PATTERNS, STATE_CLEAR
from overtaking_controller import OvertakingController
from overtaking_environment import register_two_lane_overtaking_env


def run_simulation(
    n_episodes: int = 3,
    render: bool = True,
    max_steps: int = 600,
) -> None:
    """Run the overtaking scenario for *n_episodes* episodes."""
    register_two_lane_overtaking_env()
    render_mode = "human" if render else None
    env = gym.make("TwoLaneOvertaking-v0", render_mode=render_mode)
    controller = OvertakingController()

    for episode in range(n_episodes):
        pattern = PATTERNS[episode % len(PATTERNS)]
        env.unwrapped.config["traffic_pattern"] = pattern

        obs, info = env.reset()
        controller.reset()

        print(
            f"\n{'═' * 62}\n"
            f"  Episode {episode + 1}/{n_episodes}  |  Pattern: {pattern}\n"
            f"{'═' * 62}"
        )

        total_reward = 0.0
        step = 0
        done = False
        last_state = None

        while not done and step < max_steps:
            action = controller.act(obs, env.unwrapped)

            if controller.state != last_state:
                ego = env.unwrapped.controlled_vehicles[0]
                ego_lane = int(round(ego.position[1] / StraightLane.DEFAULT_WIDTH))
                print(
                    f"  [step {step:>4}]  State → {controller.state:<12}  "
                    f"ego_x={ego.position[0]:>7.1f} m  "
                    f"speed={ego.speed:>5.1f} m/s  "
                    f"lane={ego_lane}"
                )
                last_state = controller.state

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if render:
                time.sleep(1.0 / env.unwrapped.config["policy_frequency"])

            step += 1

        ego = env.unwrapped.controlled_vehicles[0]
        collided = getattr(ego, "crashed", False)
        success = controller.state == STATE_CLEAR
        print(
            f"\n  Episode summary\n"
            f"    Steps       : {step}\n"
            f"    Total reward: {total_reward:.2f}\n"
            f"    Final state : {controller.state}\n"
            f"    Overtake    : {'✓ success' if success else '✗ incomplete'}\n"
            f"    Collision   : {'YES ⚠' if collided else 'no'}\n"
        )

    env.close()
    print("Simulation complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-lane overtaking simulation (gymnasium + highway-env)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run (default: 3, one per pattern)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable graphical rendering (useful for headless environments)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Maximum simulation steps per episode (default: 600)",
    )
    args = parser.parse_args()

    run_simulation(
        n_episodes=args.episodes,
        render=not args.no_render,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()