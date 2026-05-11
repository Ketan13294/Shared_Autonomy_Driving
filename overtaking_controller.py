"""Rule-based state-machine controller for the overtaking manoeuvre.

Returns continuous 2D actions [acceleration, steering] in [-1, 1] mapped from discrete states.
"""

import numpy as np

from highway_env.road.lane import StraightLane
from highway_env.vehicle.controller import MDPVehicle, ControlledVehicle

from overtaking_constants import (
    ACTION_FASTER,
    ACTION_IDLE,
    ACTION_LANE_LEFT,
    ACTION_LANE_RIGHT,
    ACTION_SLOWER,
    FOLLOW_DISTANCE_MIN,
    FOLLOW_DISTANCE_OK,
    GAP_REQUIRED_AHEAD,
    GAP_REQUIRED_BEHIND,
    SAFE_RETURN_GAP,
    STATE_CHECKING,
    STATE_CLEAR,
    STATE_FOLLOW,
    STATE_MERGE_BACK,
    STATE_MERGE_OUT,
    STATE_PASSING,
)
from overtaking_environment import TwoLaneOvertakingEnv


class OvertakingController:
    """A rule-based state-machine controller for the overtaking manoeuvre."""

    def __init__(self) -> None:
        self.state = STATE_CLEAR

    def reset(self) -> None:
        self.state = STATE_CLEAR

    def act_upper(self, obs: np.ndarray, env: "TwoLaneOvertakingEnv", user_input: tuple) -> int:
        ego = env.controlled_vehicles[0]
        ego_lane = int(round(ego.position[1] / StraightLane.DEFAULT_WIDTH))

        user_speed, user_steering = user_input

        dist_to_blocker = self._signed_dist_nearest_in_lane(env, ego, lane_id=0)
        gap_ahead_l1, gap_behind_l1 = self._scan_lane(env, ego, lane_id=ego_lane ^ 1)

        action = ACTION_IDLE
        gap_ok = (
            gap_ahead_l1 > GAP_REQUIRED_AHEAD
            and gap_behind_l1 > GAP_REQUIRED_BEHIND
        )
        if dist_to_blocker < FOLLOW_DISTANCE_MIN:
            return ACTION_SLOWER
        
        if user_speed:
            if user_speed > 0:
                return ACTION_FASTER
            elif user_speed < 0:
                return ACTION_SLOWER

        if user_steering:
            if ego_lane == 1:
                return ACTION_LANE_LEFT
            if ego_lane == 0:
                return ACTION_LANE_RIGHT

        # if self.state == STATE_MERGE_BACK:
        #     if ego_lane == 0:
        #         self.state = STATE_CLEAR
        #         return ACTION_IDLE
        #     return ACTION_LANE_LEFT

        return ACTION_IDLE

    def act(self, obs: np.ndarray, env: "TwoLaneOvertakingEnv", user_input: tuple) -> np.ndarray:
        """Execute the state machine and return a 2D continuous action [acceleration, steering].
        
        Maps discrete actions to continuous outputs:
        - ACTION_FASTER → [+1.0, 0.0]  (accelerate, neutral steering)
        - ACTION_SLOWER → [-1.0, 0.0]  (brake, neutral steering)
        - ACTION_LANE_LEFT → [0.0, -1.0]  (neutral speed, steer left)
        - ACTION_LANE_RIGHT → [0.0, +1.0]  (neutral speed, steer right)
        - ACTION_IDLE → [0.0, 0.0]  (no action)
        """
        # Get the discrete action from the state machine
        action = self.act_upper(obs, env, user_input)
        
        ego = env.controlled_vehicles[0]
        ego_lane = int(round(ego.position[1] / StraightLane.DEFAULT_WIDTH))
        ego_velocity_x = ego.velocity[0]
        ego_velocity_y = ego.velocity[1]
        ego_velocity = np.linalg.norm([ego_velocity_x, ego_velocity_y])
        
        command = [0.0,0.0]
        # Map discrete action to continuous 2D output [acceleration, steering]
        if action == ACTION_FASTER:
            command[0] = ego.speed_control(target_speed=(ego_velocity + 1.0))
        elif action == ACTION_SLOWER:
            command[0] = ego.speed_control(target_speed=(ego_velocity - 1.0))
        elif action == ACTION_LANE_LEFT:
            command[1] = ego.steering_control(target_lane_index=LaneIndex(1))
        elif action == ACTION_LANE_RIGHT:
            command[1] = ego.steering_control(target_lane_index=LaneIndex(0))
        else:  # ACTION_IDLE or default
            command = np.array([0.0, 0.0], dtype=np.float32)
        
        print(command)
        return np.array(command, dtype=np.float32)


    @staticmethod
    def _signed_dist_nearest_in_lane(
        env: "TwoLaneOvertakingEnv",
        ego: MDPVehicle,
        lane_id: int,
    ) -> float:
        ego_x = ego.position[0]
        best_dist = 999.0
        found_any = False
        for vehicle in env.road.vehicles:
            if vehicle is ego:
                continue
            vehicle_lane = int(round(vehicle.position[1] / StraightLane.DEFAULT_WIDTH))
            if vehicle_lane != lane_id:
                continue
            dx = vehicle.position[0] - ego_x
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
        ego_x = ego.position[0]
        ahead_min = 500.0
        behind_min = 500.0
        for vehicle in env.road.vehicles:
            if vehicle is ego:
                continue
            vehicle_lane = int(round(vehicle.position[1] / StraightLane.DEFAULT_WIDTH))
            if vehicle_lane != lane_id:
                continue
            dx = vehicle.position[0] - ego_x
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
        ego_x = ego.position[0]
        behind_min = 500.0
        for vehicle in env.road.vehicles:
            if vehicle is ego:
                continue
            vehicle_lane = int(round(vehicle.position[1] / StraightLane.DEFAULT_WIDTH))
            if vehicle_lane != lane_id:
                continue
            dx = ego_x - vehicle.position[0]
            if dx > 0:
                behind_min = min(behind_min, dx)
        return behind_min