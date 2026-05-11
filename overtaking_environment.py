"""Custom two-lane overtaking environment."""

import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401 - registers highway environments

from highway_env.envs import HighwayEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle, ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

from overtaking_constants import PATTERN_CLUSTER, PATTERN_MIXED, PATTERN_SPARSE


class TwoLaneOvertakingEnv(HighwayEnv):
    """A two-lane highway environment purpose-built for overtaking."""

    @classmethod
    def default_config(cls) -> dict:
        # Keep the environment short and structured so overtaking is the main challenge.
        config = super().default_config()
        config.update(
            {
                # Two lanes are enough to represent the passing decision without adding unnecessary complexity.
                "lanes_count": 2,
                # Shorter roads keep episodes focused on a few overtaking opportunities.
                "road_length": 1500,
                # The ego starts in the right lane, which is the natural place to begin a pass from.
                "initial_lane_id": 0,
                # The ego and blocker speeds create a clear incentive to change lanes.
                "ego_speed": 22.0,
                "blocker_speed": 15.0,
                "blocker_distance": 30.0,
                # Discrete actions make the task easier to learn and evaluate than continuous control.
                # "action": {
                #     "type": "DiscreteMetaAction",
                #     "target_speeds": [15, 17.5, 20, 22.5, 25, 27.5, 30],
                # },
                # Continuous actions let the policy choose any acceleration and steering, which is more realistic but harder to learn from.
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": True,
                    "lateral": True,
                    "dynamical": True,
                },
                # Kinematics observations expose nearby vehicles directly to the policy.
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 8,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h", "ang_off"],
                    "normalize": True,
                    "absolute": True,
                },
                # Image observations expose nearby vehicles to the policy through rendering, which is more realistic but harder to learn from.
                # "observations":{
                #     "type" : "GrayscaleObservation",
                #     "observation_shape": (128, 84),
                #     "stack_size": 4,
                #     "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                #     "scaling": 1.75,
                # },
                # Reward values encourage safe progress while penalizing collisions and unnecessary weaving.
                "collision_reward": -5.0,
                "high_speed_reward": 0.3,
                "right_lane_reward": 0.0,
                "lane_change_reward": 0.00,
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                # Episodes are long enough for one or two overtakes, but not long enough to become a cruising task.
                "duration": 90,
                "simulation_frequency": 15,
                "policy_frequency": 2,
                # Traffic presets let training cover different densities without changing the environment definition.
                "traffic_pattern": PATTERN_CLUSTER,
                "vehicles_count": 0,
                "manual_control": False,
            }
        )
        return config

    def _create_road(self) -> None:
        net = RoadNetwork()
        width = StraightLane.DEFAULT_WIDTH
        length = self.config["road_length"]

        # Lane 0 is the ego lane; lane 1 is the adjacent passing lane.
        # Distinct line markings help the rendered scene stay readable.
        for lane_id in range(2):
            origin = np.array([0.0, lane_id * width])
            end = np.array([float(length), lane_id * width])

            if lane_id == 0:
                line_types = [LineType.CONTINUOUS_LINE, LineType.STRIPED]
            else:
                line_types = [LineType.STRIPED, LineType.CONTINUOUS_LINE]

            net.add_lane(
                "start",
                "end",
                StraightLane(origin, end, line_types=line_types, speed_limit=30.0),
            )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        self.controlled_vehicles = []

        # Randomly assign ego and blocker to lanes and positions.
        ego_lane_id = int(self.np_random.choice([0, 1]))
        blocker_lane_id = int(self.np_random.choice([0, 1]))
        
        # Sample longitudinal positions from a range to avoid spawning too close to the road edge.
        ego_x = float(self.np_random.uniform(40.0, 150.0))
        blocker_x = float(self.np_random.uniform(40.0, 300.0))
        
        # Ensure ego and blocker are not too close (at least 30 units apart if in the same lane).
        if ego_lane_id == blocker_lane_id and abs(ego_x - blocker_x) < 30.0:
            blocker_x = ego_x + 50.0
        
        ego_lane = self.road.network.get_lane(("start", "end", ego_lane_id))
        ego_vehicle_class = self._ego_vehicle_class()
        ego_kwargs = {
            "road": self.road,
            "position": ego_lane.position(ego_x, 0),
            "heading": ego_lane.heading_at(ego_x),
            "speed": self.config["ego_speed"],
        }
        if ego_vehicle_class is MDPVehicle:
            ego_kwargs["target_speeds"] = np.array([15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0])
        ego = ego_vehicle_class(**ego_kwargs)
        self.controlled_vehicles.append(ego)
        self.road.vehicles.append(ego)

        blocker_lane = self.road.network.get_lane(("start", "end", blocker_lane_id))
        blocker = IDMVehicle(
            road=self.road,
            position=blocker_lane.position(blocker_x, 0),
            heading=blocker_lane.heading_at(blocker_x),
            speed=self.config["blocker_speed"],
            enable_lane_change=False,
        )
        self.road.vehicles.append(blocker)

        pattern = self.config["traffic_pattern"]
        
        # Add traffic to any lane not occupied by ego (to give the overtaking task variety).
        for lane_id in [0, 1]:
            if lane_id != ego_lane_id:
                lane = self.road.network.get_lane(("start", "end", lane_id))
                if pattern == PATTERN_SPARSE:
                    self._add_sparse_traffic(lane)
                elif pattern == PATTERN_CLUSTER:
                    self._add_cluster_traffic(lane)
                elif pattern == PATTERN_MIXED:
                    self._add_mixed_traffic(lane)

    def _ego_vehicle_class(self):
        # Continuous control uses the base Vehicle class; discrete control uses highway-env's MDPVehicle wrapper.
        action_type = self.config.get("action", {}).get("type", "DiscreteMetaAction")
        if action_type == "ContinuousAction":
            return ControlledVehicle
        return MDPVehicle

    def _make_idm(
        self,
        lane: StraightLane,
        x: float,
        speed: float,
        enable_lane_change: bool = False,
    ) -> IDMVehicle:
        # Central helper for all background traffic so every preset is built from the same placement logic.
        vehicle = IDMVehicle(
            road=self.road,
            position=lane.position(x, 0),
            heading=lane.heading_at(x),
            speed=speed,
            enable_lane_change=enable_lane_change,
        )
        self.road.vehicles.append(vehicle)
        return vehicle

    def _make_random_idm(
        self,
        lane: StraightLane,
        x_min: float,
        x_max: float,
        speed: float,
        enable_lane_change: bool = False,
    ) -> IDMVehicle:
        # Sample a longitudinal position so each reset can produce a different traffic layout.
        x = float(self.np_random.uniform(x_min, x_max))
        return self._make_idm(lane, x=x, speed=speed, enable_lane_change=enable_lane_change)

    def _add_sparse_traffic(self, lane: StraightLane) -> None:
        # Sparse traffic gives the ego more open space, which makes the pass easier and the rewards steadier.
        self._make_random_idm(lane, x_min=180.0, x_max=320.0, speed=28.0)
        self._make_random_idm(lane, x_min=380.0, x_max=560.0, speed=26.0)

    def _add_cluster_traffic(self, lane: StraightLane) -> None:
        # Clustered traffic places several slow vehicles close together so the ego has to react to a queue.
        self._make_random_idm(lane, x_min=120.0, x_max=160.0, speed=20.0)
        self._make_random_idm(lane, x_min=145.0, x_max=185.0, speed=20.0)
        self._make_random_idm(lane, x_min=170.0, x_max=215.0, speed=20.0)
        self._make_random_idm(lane, x_min=400.0, x_max=520.0, speed=22.0)

    def _add_mixed_traffic(self, lane: StraightLane) -> None:
        # Mixed traffic alternates between faster and slower cars to create a less predictable passing lane.
        self._make_random_idm(lane, x_min=140.0, x_max=220.0, speed=27.0)
        self._make_random_idm(lane, x_min=220.0, x_max=300.0, speed=18.0)
        self._make_random_idm(lane, x_min=300.0, x_max=380.0, speed=29.0)
        self._make_random_idm(lane, x_min=420.0, x_max=520.0, speed=17.0)

    def _draw_ego_speed_overlay(self):
        """Draw ego speed in the rendered GUI (best-effort; only when pygame is used)."""
        try:
            import pygame
            import math

            screen = None
            try:
                screen = pygame.display.get_surface()
            except Exception:
                screen = None
            if screen is None:
                return

            if not getattr(self, "controlled_vehicles", None):
                return

            ego = self.controlled_vehicles[0]
            # Try several ways to obtain ego speed so overlay works across vehicle implementations
            speed = getattr(ego, "speed", None)
            if speed is None:
                # Some vehicles expose velocity as a vector or scalar
                vel = getattr(ego, "velocity", None) or getattr(ego, "vel", None)
                try:
                    if vel is None:
                        # Try a method
                        vel = ego.get_velocity()
                except Exception:
                    vel = None
                if vel is not None:
                    try:
                        # If vel is a sequence, take its norm; if scalar, use it directly
                        # Convert possible types (numpy arrays, lists, vectors) to numbers
                        if hasattr(vel, "__iter__"):
                            vals = list(vel)
                            # Take first two components if available
                            x = float(vals[0]) if len(vals) >= 1 else 0.0
                            y = float(vals[1]) if len(vals) >= 2 else 0.0
                            import math

                            speed = math.hypot(x, y)
                        else:
                            # Some vehicles return custom vector objects with x/y attrs
                            x = getattr(vel, "x", None)
                            y = getattr(vel, "y", None)
                            if x is not None and y is not None:
                                import math

                                speed = math.hypot(float(x), float(y))
                            else:
                                speed = float(vel)
                    except Exception:
                        speed = None
            if speed is None:
                return

            closest_ahead = None
            ego_lane = None
            ego_x = None
            try:
                ego_lane = ego.lane_index[2]
                ego_x = float(ego.position[0])
            except Exception:
                ego_lane = None
                ego_x = None

            if ego_lane is not None and ego_x is not None and getattr(self, "road", None) is not None:
                for vehicle in getattr(self.road, "vehicles", []):
                    if vehicle is ego:
                        continue
                    try:
                        if vehicle.lane_index[2] != ego_lane:
                            continue
                        delta_x = float(vehicle.position[0]) - ego_x
                        if delta_x >= 0.0 and (closest_ahead is None or delta_x < closest_ahead):
                            closest_ahead = delta_x
                    except Exception:
                        continue

            distance_text = "N/A" if closest_ahead is None else f"{closest_ahead:.2f}"
            text = f"Speed: {speed:.2f} | Closest car ahead: {distance_text}"
            pygame.font.init()
            font = pygame.font.SysFont(None, 28)
            surf = font.render(text, True, (255, 255, 255))

            sw, sh = screen.get_size()
            tw, th = surf.get_size()
            # Upper-right with small margin
            screen.blit(surf, (sw - tw - 10, 10))
            pygame.display.flip()
        except Exception:
            # Never raise from rendering overlay
            return

    def render(self, *args, **kwargs):
        """Override render to include ego speed overlay."""
        result = super().render(*args, **kwargs)
        self._draw_ego_speed_overlay()
        return result


def register_two_lane_overtaking_env() -> None:
    """Register the custom gymnasium environment once."""
    try:
        # Gymnasium keeps a global registry, so duplicate registration is harmless but ignored.
        gym.register(
            id="TwoLaneOvertaking-v0",
            entry_point=TwoLaneOvertakingEnv,
        )
    except Exception as exc:
        if "TwoLaneOvertaking-v0" not in str(exc):
            raise