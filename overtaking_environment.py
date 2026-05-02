"""Custom two-lane overtaking environment."""

import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401 - registers highway environments

from highway_env.envs import HighwayEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle

from overtaking_constants import PATTERN_CLUSTER, PATTERN_MIXED, PATTERN_SPARSE


class TwoLaneOvertakingEnv(HighwayEnv):
    """A two-lane highway environment purpose-built for overtaking."""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "lanes_count": 2,
                "road_length": 1500,
                "initial_lane_id": 0,
                "ego_speed": 22.0,
                "blocker_speed": 15.0,
                "blocker_distance": 60.0,
                "action": {
                    "type": "DiscreteMetaAction",
                    "target_speeds": [15, 20, 25, 30],
                },
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 8,
                    "features": ["presence", "x", "y", "vx", "vy"],
                    "normalize": False,
                    "absolute": True,
                },
                "collision_reward": -5.0,
                "high_speed_reward": 0.3,
                "right_lane_reward": 0.1,
                "lane_change_reward": -0.05,
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "duration": 90,
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "traffic_pattern": PATTERN_SPARSE,
                "vehicles_count": 0,
            }
        )
        return config

    def _create_road(self) -> None:
        net = RoadNetwork()
        width = StraightLane.DEFAULT_WIDTH
        length = self.config["road_length"]

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

        ego_lane = self.road.network.get_lane(("start", "end", 0))
        ego_x = 50.0
        ego_vehicle_class = self._ego_vehicle_class()
        ego_kwargs = {
            "road": self.road,
            "position": ego_lane.position(ego_x, 0),
            "heading": ego_lane.heading_at(ego_x),
            "speed": self.config["ego_speed"],
        }
        if ego_vehicle_class is MDPVehicle:
            ego_kwargs["target_speeds"] = np.array([15.0, 20.0, 25.0, 30.0])
        ego = ego_vehicle_class(**ego_kwargs)
        self.controlled_vehicles.append(ego)
        self.road.vehicles.append(ego)

        blocker_x = ego_x + self.config["blocker_distance"]
        blocker = IDMVehicle(
            road=self.road,
            position=ego_lane.position(blocker_x, 0),
            heading=ego_lane.heading_at(blocker_x),
            speed=self.config["blocker_speed"],
            enable_lane_change=False,
        )
        self.road.vehicles.append(blocker)

        lane1 = self.road.network.get_lane(("start", "end", 1))
        pattern = self.config["traffic_pattern"]

        if pattern == PATTERN_SPARSE:
            self._add_sparse_traffic(lane1)
        elif pattern == PATTERN_CLUSTER:
            self._add_cluster_traffic(lane1)
        elif pattern == PATTERN_MIXED:
            self._add_mixed_traffic(lane1)

    def _ego_vehicle_class(self):
        action_type = self.config.get("action", {}).get("type", "DiscreteMetaAction")
        if action_type == "ContinuousAction":
            return Vehicle
        return MDPVehicle

    def _make_idm(
        self,
        lane: StraightLane,
        x: float,
        speed: float,
        enable_lane_change: bool = False,
    ) -> IDMVehicle:
        vehicle = IDMVehicle(
            road=self.road,
            position=lane.position(x, 0),
            heading=lane.heading_at(x),
            speed=speed,
            enable_lane_change=enable_lane_change,
        )
        self.road.vehicles.append(vehicle)
        return vehicle

    def _add_sparse_traffic(self, lane: StraightLane) -> None:
        self._make_idm(lane, x=200.0, speed=28.0)
        self._make_idm(lane, x=400.0, speed=26.0)

    def _add_cluster_traffic(self, lane: StraightLane) -> None:
        self._make_idm(lane, x=130.0, speed=20.0)
        self._make_idm(lane, x=155.0, speed=20.0)
        self._make_idm(lane, x=180.0, speed=20.0)
        self._make_idm(lane, x=430.0, speed=22.0)

    def _add_mixed_traffic(self, lane: StraightLane) -> None:
        self._make_idm(lane, x=160.0, speed=27.0)
        self._make_idm(lane, x=240.0, speed=18.0)
        self._make_idm(lane, x=330.0, speed=29.0)
        self._make_idm(lane, x=440.0, speed=17.0)


def register_two_lane_overtaking_env() -> None:
    """Register the custom gymnasium environment once."""
    try:
        gym.register(
            id="TwoLaneOvertaking-v0",
            entry_point=TwoLaneOvertakingEnv,
        )
    except Exception as exc:
        if "TwoLaneOvertaking-v0" not in str(exc):
            raise