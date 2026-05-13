"""Command-line entry point for the overtaking simulation."""

import argparse
from logging import config
import time

import gymnasium as gym
import pygame
import numpy as np
import threading

from highway_env.road.lane import StraightLane

from overtaking_constants import PATTERNS, STATE_CLEAR, ACTION_FASTER, ACTION_SLOWER, ACTION_LANE_LEFT, ACTION_LANE_RIGHT, ACTION_IDLE, FOLLOW_DISTANCE_MIN, FOLLOW_DISTANCE_OK, GAP_REQUIRED_AHEAD, GAP_REQUIRED_BEHIND, SAFE_RETURN_GAP
from overtaking_controller import OvertakingController
from overtaking_environment import register_two_lane_overtaking_env, TwoLaneOvertakingEnv

class KeyboardPoller:
    """Background thread that polls pygame events and updates key state.

    This keeps keyboard polling independent of the main simulation loop so
    that held keys and quick presses are captured reliably.
    """

    def __init__(self, poll_hz: int = 60) -> None:
        pygame.init()
        self._poll_hz = max(1, int(poll_hz))
        self.user_turn: int = 0
        self.user_speed_delta: int = 0
        self.quit_requested: bool = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        clock = pygame.time.Clock()
        while not self._stop.is_set():
            try:
                pygame.event.pump()
                keys = pygame.key.get_pressed()

                if keys[pygame.K_UP]:
                    self.user_turn = 1
                elif keys[pygame.K_DOWN]:
                    self.user_turn = -1
                else:
                    self.user_turn = 0

                if keys[pygame.K_RIGHT]:
                    self.user_speed_delta = 1
                elif keys[pygame.K_LEFT]:
                    self.user_speed_delta = -1
                else:
                    self.user_speed_delta = 0
                self.quit_requested = bool(keys[pygame.K_q])
            except Exception:
                # Defensive: ignore pygame errors during shutdown
                pass
            clock.tick(self._poll_hz)

    def stop(self, timeout: float = 0.5) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)


def run_simulation(
    n_episodes: int = 3,
    render: bool = True,
    max_steps: int = 600,
) -> None:
    """Run the overtaking scenario for *n_episodes* episodes."""
    register_two_lane_overtaking_env()

    config = TwoLaneOvertakingEnv.default_config()
    config.update({"manual_control": False})

    render_mode = "human" if render else None
    env = gym.make("TwoLaneOvertaking-v0", render_mode=render_mode,config=config)
    
    controller = OvertakingController()
    # Start a background keyboard poller so key reads are independent of loop
    poller = KeyboardPoller(poll_hz=100)
    # Initialize a clock for pacing the main loop
    try:
        clock = pygame.time.Clock()
    except Exception:
        clock = None

    for episode in range(n_episodes):
        print(f"\n=== Starting episode {episode + 1}/{n_episodes} ===")
        pattern = PATTERNS[episode % len(PATTERNS)]
        env.unwrapped.config["traffic_pattern"] = "OVERTAKING"

        obs, info = env.reset()
        controller.reset()
        env.unwrapped.controller = controller

        total_reward = 0.0
        step = 0
        done = False
        last_state = None

        while not done and step < max_steps:
            # Read the most-recent keyboard state from the background poller
            user_turn = poller.user_turn
            user_speed_delta = poller.user_speed_delta
            # user_speed_delta, user_turn = _teacher_user_input(env.unwrapped)
            quit_requested = poller.quit_requested

            action = controller.act(obs, env.unwrapped, [user_speed_delta, user_turn],False)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            policy_freq = int(env.unwrapped.config.get("policy_frequency", 5))
            if clock is not None:
                clock.tick(max(1, policy_freq))
            else:
                time.sleep(1.0 / max(1, policy_freq))

            step += 1

    env.close()
    # Ensure the poller thread is stopped cleanly
    try:
        poller.stop()
    except Exception:
        pass
    print("Simulation complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-lane overtaking simulation (gymnasium + highway-env)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
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