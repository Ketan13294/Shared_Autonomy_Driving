"""
Two-lane overtaking simulation using gymnasium and highway-env.

Scenario
--------
* Two-lane straight highway.
* Ego vehicle starts in the RIGHT lane (lane 0, y = 0 m) behind a slow
  blocker car.
* Ego vehicle executes an overtaking manoeuvre:
    1. Follow the blocker at a safe distance.
    2. Wait for a safe gap in the LEFT overtaking lane (lane 1, y = 4 m).
    3. Merge LEFT into lane 1 (LANE_RIGHT action increases lane id 0 → 1).
    4. Accelerate past the blocker.
    5. Return RIGHT to lane 0 (LANE_LEFT action decreases lane id 1 → 0).
* Lane 1 contains other vehicles arranged in three distinct traffic patterns
  that cycle across episodes:
    - SPARSE   : two fast cars widely spaced  (easy gap to exploit)
    - CLUSTER  : three cars in a tight bunch   (requires patience)
    - MIXED    : alternating fast/slow vehicles (realistic mixed traffic)

Highway-env lane-index convention (two-lane road)
--------------------------------------------------
  Lane 0  y =  0 m  RIGHT / slow lane   – ego starts here, blocker here
  Lane 1  y =  4 m  LEFT  / fast lane   – overtaking lane

  LANE_LEFT  action (id 0) : lane_id − 1  (move toward lane 0 → rightward)
  LANE_RIGHT action (id 2) : lane_id + 1  (move toward lane 1 → leftward)

Usage
-----
    python overtaking_simulation.py            # render with pygame
    python overtaking_simulation.py --no-render  # headless (CI / testing)
    python overtaking_simulation.py --episodes 5
"""

import argparse
import time
from typing import Optional

import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401 – registers highway environments

from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.envs import HighwayEnv

# ─── Traffic pattern names ────────────────────────────────────────────────────
PATTERN_SPARSE = "SPARSE"
PATTERN_CLUSTER = "CLUSTER"
PATTERN_MIXED = "MIXED"
PATTERNS = [PATTERN_SPARSE, PATTERN_CLUSTER, PATTERN_MIXED]

# ─── Overtaking state machine ─────────────────────────────────────────────────
STATE_FOLLOW = "FOLLOW"          # trailing the blocker, managing following gap
STATE_CHECKING = "CHECKING"      # waiting for a safe gap in lane 1
STATE_MERGE_OUT = "MERGE_OUT"    # executing the merge into lane 1
STATE_PASSING = "PASSING"        # in lane 1, accelerating past the blocker
STATE_MERGE_BACK = "MERGE_BACK"  # returning to lane 0
STATE_CLEAR = "CLEAR"            # overtake complete, cruising

# DiscreteMetaAction indices (DiscreteMetaAction.ACTIONS_ALL)
ACTION_LANE_LEFT = 0   # lane_id − 1  (right-ward; toward lane 0)
ACTION_IDLE = 1        # maintain current target speed and lane
ACTION_LANE_RIGHT = 2  # lane_id + 1  (left-ward; toward lane 1)
ACTION_FASTER = 3      # increase target speed one level
ACTION_SLOWER = 4      # decrease target speed one level

# Safety thresholds (metres)
GAP_REQUIRED_BEHIND = 25.0   # minimum clear distance behind ego in lane 1
GAP_REQUIRED_AHEAD = 50.0    # minimum clear distance ahead of ego in lane 1
SAFE_RETURN_GAP = 35.0       # how far ego must be ahead of blocker before merging back
FOLLOW_DISTANCE_MIN = 20.0   # emergency brake threshold (slow down hard)
FOLLOW_DISTANCE_OK = 40.0    # comfortable following distance


# ─── Custom environment ───────────────────────────────────────────────────────

class TwoLaneOvertakingEnv(HighwayEnv):
    """
    A two-lane highway environment purpose-built for overtaking.

    Differences from the default HighwayEnv
    ----------------------------------------
    * Exactly two lanes.
    * Automatic vehicle population is disabled; vehicles are placed
      manually in ``_create_vehicles`` so we control their pattern.
    * The ego vehicle always starts in lane 0 at a fixed position.
    * A slow blocker vehicle is placed a short distance ahead.
    * The current traffic pattern (SPARSE / CLUSTER / MIXED) is set via
      ``self.config["traffic_pattern"]`` before each reset.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                # Road geometry
                "lanes_count": 2,
                "road_length": 1500,          # metres
                # Ego vehicle – starts in lane 0 (right / slow lane)
                "initial_lane_id": 0,
                "ego_speed": 22.0,            # m/s  (~79 km/h); one step above minimum
                # Blocker vehicle in lane 0
                "blocker_speed": 15.0,        # m/s  (~54 km/h) – deliberately slow
                "blocker_distance": 120.0,    # metres ahead of ego at episode start
                # Action / observation
                "action": {
                    "type": "DiscreteMetaAction",
                    "target_speeds": [15, 20, 25, 30],  # 4 levels; 15 m/s minimum
                },
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 8,
                    "features": ["presence", "x", "y", "vx", "vy"],
                    "normalize": False,
                    "absolute": True,
                },
                # Reward shaping
                "collision_reward": -5.0,
                "high_speed_reward": 0.3,
                "right_lane_reward": 0.1,
                "lane_change_reward": -0.05,
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                # Episode length
                "duration": 90,              # seconds
                # Simulation
                "simulation_frequency": 15,
                "policy_frequency": 5,
                # Traffic pattern – overridden per episode
                "traffic_pattern": PATTERN_SPARSE,
                # Disable automatic random vehicle spawning
                "vehicles_count": 0,
            }
        )
        return config

    # ------------------------------------------------------------------
    def _create_road(self) -> None:
        """Two perfectly straight lanes."""
        net = RoadNetwork()
        width = StraightLane.DEFAULT_WIDTH  # 4 m
        length = self.config["road_length"]

        for lane_id in range(2):
            origin = np.array([0.0, lane_id * width])
            end = np.array([float(length), lane_id * width])

            if lane_id == 0:
                line_types = [LineType.CONTINUOUS_LINE, LineType.STRIPED]
            else:
                line_types = [LineType.STRIPED, LineType.CONTINUOUS_LINE]

            net.add_lane(
                "start", "end",
                StraightLane(origin, end, line_types=line_types, speed_limit=30.0),
            )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    # ------------------------------------------------------------------
    def _create_vehicles(self) -> None:
        """Place all vehicles deterministically according to the traffic pattern."""
        self.controlled_vehicles = []

        # ── Ego vehicle (lane 0 / right lane, position x = 50 m) ──────
        ego_lane = self.road.network.get_lane(("start", "end", 0))
        ego_x = 50.0
        ego = MDPVehicle(
            road=self.road,
            position=ego_lane.position(ego_x, 0),
            heading=ego_lane.heading_at(ego_x),
            speed=self.config["ego_speed"],
            target_speeds=np.array([15.0, 20.0, 25.0, 30.0]),
        )
        self.controlled_vehicles.append(ego)
        self.road.vehicles.append(ego)

        # ── Slow blocker (lane 0, ahead of ego) ───────────────────────
        blocker_x = ego_x + self.config["blocker_distance"]
        blocker = IDMVehicle(
            road=self.road,
            position=ego_lane.position(blocker_x, 0),
            heading=ego_lane.heading_at(blocker_x),
            speed=self.config["blocker_speed"],
            enable_lane_change=False,   # stays in lane 0 throughout
        )
        self.road.vehicles.append(blocker)

        # ── Traffic in lane 1 (left / overtaking lane, pattern-dep.) ──
        lane1 = self.road.network.get_lane(("start", "end", 1))
        pattern = self.config["traffic_pattern"]

        if pattern == PATTERN_SPARSE:
            self._add_sparse_traffic(lane1)
        elif pattern == PATTERN_CLUSTER:
            self._add_cluster_traffic(lane1)
        elif pattern == PATTERN_MIXED:
            self._add_mixed_traffic(lane1)

    # ------------------------------------------------------------------
    # Traffic-pattern helpers
    # ------------------------------------------------------------------

    def _make_idm(
        self,
        lane: StraightLane,
        x: float,
        speed: float,
        enable_lane_change: bool = False,
    ) -> IDMVehicle:
        """Convenience factory: create an IDMVehicle at longitudinal position *x*."""
        v = IDMVehicle(
            road=self.road,
            position=lane.position(x, 0),
            heading=lane.heading_at(x),
            speed=speed,
            enable_lane_change=enable_lane_change,
        )
        self.road.vehicles.append(v)
        return v

    def _add_sparse_traffic(self, lane: StraightLane) -> None:
        """
        SPARSE pattern – two fast cars with a large gap between them.

        Timeline / layout (x-axis, metres):
            ego ~50 m     blocker ~170 m
            car A at 200 m  (fast – 28 m/s)      ← passes quickly
            < gap ~200 m >
            car B at 400 m  (fast – 26 m/s)
        The ego can exploit the gap after car A passes.
        """
        self._make_idm(lane, x=200.0, speed=28.0)
        self._make_idm(lane, x=400.0, speed=26.0)

    def _add_cluster_traffic(self, lane: StraightLane) -> None:
        """
        CLUSTER pattern – three cars bunched closely together, then a long gap.

        Layout:
            car A at 130 m  (20 m/s)
            car B at 155 m  (20 m/s)
            car C at 180 m  (20 m/s)
            < long gap ~250 m >
            car D at 430 m  (22 m/s)
        The ego must wait for the entire cluster to clear lane 1 before merging.
        """
        self._make_idm(lane, x=130.0, speed=20.0)
        self._make_idm(lane, x=155.0, speed=20.0)
        self._make_idm(lane, x=180.0, speed=20.0)
        self._make_idm(lane, x=430.0, speed=22.0)

    def _add_mixed_traffic(self, lane: StraightLane) -> None:
        """
        MIXED pattern – alternating fast/slow vehicles at medium spacing.

        Layout:
            car A at 160 m (fast – 27 m/s)
            car B at 240 m (slow – 18 m/s)
            car C at 330 m (fast – 29 m/s)
            car D at 440 m (slow – 17 m/s)
        Requires careful timing: gaps open and close dynamically.
        """
        self._make_idm(lane, x=160.0, speed=27.0)
        self._make_idm(lane, x=240.0, speed=18.0)
        self._make_idm(lane, x=330.0, speed=29.0)
        self._make_idm(lane, x=440.0, speed=17.0)


# Register the custom environment with gymnasium
gym.register(
    id="TwoLaneOvertaking-v0",
    entry_point=TwoLaneOvertakingEnv,
)


# ─── Overtaking controller ────────────────────────────────────────────────────

class OvertakingController:
    """
    A rule-based state-machine controller for the overtaking manoeuvre.

    Lane-change direction reminder
    --------------------------------
    Ego starts in lane 0 (right / slow lane, y = 0 m).
    To overtake, it moves to lane 1 (left / fast lane, y = 4 m).

      LANE_RIGHT action (id 2): lane_id + 1  →  lane 0 → lane 1  (merge left)
      LANE_LEFT  action (id 0): lane_id − 1  →  lane 1 → lane 0  (return right)

    States
    ------
    FOLLOW     → trailing the blocker at a safe following distance
    CHECKING   → gap might be opening; verifying it is truly safe
    MERGE_OUT  → issuing LANE_RIGHT until ego is in lane 1
    PASSING    → in lane 1, accelerating past the blocker
    MERGE_BACK → issuing LANE_LEFT until ego is back in lane 0
    CLEAR      → overtake complete; cruising at full speed
    """

    def __init__(self) -> None:
        self.state = STATE_FOLLOW

    def reset(self) -> None:
        self.state = STATE_FOLLOW

    def act(self, obs: np.ndarray, env: "TwoLaneOvertakingEnv") -> int:
        """
        Choose a DiscreteMetaAction index given the current environment state.

        Parameters
        ----------
        obs : kinematics observation array  (vehicles_count × features)
        env : the live environment instance
        """
        ego = env.controlled_vehicles[0]
        ego_x = ego.position[0]
        ego_lane = int(round(ego.position[1] / StraightLane.DEFAULT_WIDTH))

        # Signed longitudinal distance to the nearest vehicle in lane 0.
        # Positive = that vehicle is ahead of ego; negative = it is behind.
        dist_to_blocker = self._signed_dist_nearest_in_lane(env, ego, lane_id=0)

        # Gaps in lane 1 relative to the ego's x position
        gap_ahead_l1, gap_behind_l1 = self._scan_lane(env, ego, lane_id=1)

        # ── State machine ─────────────────────────────────────────────
        if self.state == STATE_CLEAR:
            # Overtake complete: cruise at maximum speed
            return ACTION_FASTER

        if self.state == STATE_FOLLOW:
            # Slow down hard if dangerously close
            if dist_to_blocker < FOLLOW_DISTANCE_MIN:
                return ACTION_SLOWER
            # Comfortable following: look for an opportunity to overtake
            if dist_to_blocker < FOLLOW_DISTANCE_OK:
                self.state = STATE_CHECKING
            return ACTION_IDLE

        if self.state == STATE_CHECKING:
            gap_ok = (
                gap_ahead_l1 > GAP_REQUIRED_AHEAD
                and gap_behind_l1 > GAP_REQUIRED_BEHIND
            )
            if gap_ok:
                self.state = STATE_MERGE_OUT
            # Speed management while waiting
            if dist_to_blocker < FOLLOW_DISTANCE_MIN:
                return ACTION_SLOWER
            if dist_to_blocker > FOLLOW_DISTANCE_OK * 2:
                # Blocker drifted far; stop checking, resume following
                self.state = STATE_FOLLOW
            return ACTION_IDLE

        if self.state == STATE_MERGE_OUT:
            if ego_lane == 1:
                # Successfully in the overtaking lane
                self.state = STATE_PASSING
                return ACTION_FASTER
            return ACTION_LANE_RIGHT   # id + 1: lane 0 → lane 1

        if self.state == STATE_PASSING:
            # Accelerate until comfortably ahead of the blocker
            if dist_to_blocker < -SAFE_RETURN_GAP:
                # We are ahead; check lane 0 is clear before merging back
                gap_behind_l0 = self._gap_behind_in_lane(env, ego, lane_id=0)
                if gap_behind_l0 > 20.0:
                    self.state = STATE_MERGE_BACK
            return ACTION_FASTER

        if self.state == STATE_MERGE_BACK:
            if ego_lane == 0:
                self.state = STATE_CLEAR
                return ACTION_IDLE
            return ACTION_LANE_LEFT   # id − 1: lane 1 → lane 0

        return ACTION_IDLE  # fallback

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _signed_dist_nearest_in_lane(
        env: "TwoLaneOvertakingEnv",
        ego: MDPVehicle,
        lane_id: int,
    ) -> float:
        """
        Return the signed longitudinal distance to the nearest vehicle in
        *lane_id*.  Positive means the vehicle is ahead of the ego; negative
        means it is behind.  Returns +999 when no vehicle is in the lane.
        """
        ego_x = ego.position[0]
        best_dist = 999.0          # start with large positive (no vehicle found)
        found_any = False
        for v in env.road.vehicles:
            if v is ego:
                continue
            v_lane = int(round(v.position[1] / StraightLane.DEFAULT_WIDTH))
            if v_lane != lane_id:
                continue
            dx = v.position[0] - ego_x
            if not found_any or abs(dx) < abs(best_dist):
                best_dist = dx
                found_any = True
        return best_dist

    @staticmethod
    def _scan_lane(
        env: "TwoLaneOvertakingEnv",
        ego: MDPVehicle,
        lane_id: int,
    ) -> tuple[float, float]:
        """
        Return *(gap_ahead, gap_behind)* in *lane_id* relative to the ego's
        longitudinal position.  Values are clipped to 500 m when no vehicle
        is found in that direction.
        """
        ego_x = ego.position[0]
        ahead_min = 500.0
        behind_min = 500.0
        for v in env.road.vehicles:
            if v is ego:
                continue
            v_lane = int(round(v.position[1] / StraightLane.DEFAULT_WIDTH))
            if v_lane != lane_id:
                continue
            dx = v.position[0] - ego_x
            if dx > 0:
                ahead_min = min(ahead_min, dx)
            else:
                behind_min = min(behind_min, -dx)
        return ahead_min, behind_min

    @staticmethod
    def _gap_behind_in_lane(
        env: "TwoLaneOvertakingEnv",
        ego: MDPVehicle,
        lane_id: int,
    ) -> float:
        """Distance to the nearest vehicle BEHIND the ego in *lane_id*."""
        ego_x = ego.position[0]
        behind_min = 500.0
        for v in env.road.vehicles:
            if v is ego:
                continue
            v_lane = int(round(v.position[1] / StraightLane.DEFAULT_WIDTH))
            if v_lane != lane_id:
                continue
            dx = ego_x - v.position[0]
            if dx > 0:
                behind_min = min(behind_min, dx)
        return behind_min


# ─── Simulation runner ────────────────────────────────────────────────────────

def run_simulation(
    n_episodes: int = 3,
    render: bool = True,
    max_steps: int = 600,
) -> None:
    """
    Run the overtaking scenario for *n_episodes* episodes.

    Parameters
    ----------
    n_episodes : int
        Number of episodes to run.  Each episode cycles through a traffic
        pattern (SPARSE → CLUSTER → MIXED → SPARSE → …).
    render : bool
        Whether to open a pygame window.  Set to False for headless runs.
    max_steps : int
        Maximum simulation steps per episode (default 600, ≈ 120 s at 5 Hz).
    """
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

            # Log controller-state transitions
            if controller.state != last_state:
                ego = env.unwrapped.controlled_vehicles[0]
                ego_lane = int(
                    round(ego.position[1] / StraightLane.DEFAULT_WIDTH)
                )
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


# ─── Entry point ─────────────────────────────────────────────────────────────

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
