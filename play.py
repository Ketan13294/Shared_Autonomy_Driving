"""Playable keyboard control for the overtaking scenario."""

import time

from overtaking_environment import TwoLaneOvertakingEnv


def main() -> None:
    config = TwoLaneOvertakingEnv.default_config()
    config.update(
        {
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "dynamical": False,
            },
            "manual_control": True,
            "traffic_pattern": "SPARSE",
        }
    )

    env = TwoLaneOvertakingEnv(config=config, render_mode="human")
    obs, info = env.reset()

    print(
        "Keyboard controls:\n"
        "  Up / Down    : accelerate / brake\n"
        "  Left / Right : steer left / right\n"
        "  Close the window or press Ctrl+C to quit\n"
    )

    try:
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, info = env.reset()
                print("Episode reset.")

            time.sleep(1.0 / env.unwrapped.config["policy_frequency"])
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()