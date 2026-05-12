"""Rule-based state-machine controller for the overtaking manoeuvre.

Returns continuous 2D actions [acceleration, steering] in [-1, 1] mapped from discrete states.
"""

import numpy as np

from highway_env.road.lane import StraightLane
from highway_env.road.road import LaneIndex
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
        # A live reference to the ego vehicle so controller and env share the same object
        self.v = None
        # Track the target lane for persistent steering during merges
        self.merge_out_target_lane = None

    def reset(self) -> None:
        self.state = STATE_CLEAR
        # Clear the cached ego reference; will be re-attached on next action
        self.v = None
        # Clear the merge target when resetting
        self.merge_out_target_lane = None

    def act_upper(self, obs: np.ndarray, env: "TwoLaneOvertakingEnv", user_input: tuple) -> tuple[int, int]:
        """
        Return separate speed and heading actions.
        
        Steering logic: if a merge is in progress (merge_out_target_lane is set),
        persist the steering action until:
          1. Lane change completes (ego reaches target lane), OR
          2. User provides a different steering input
        
        Returns:
            (speed_action, heading_action) where each is -1 (slow/left), 0 (idle), or +1 (fast/right)
        """
        # Ensure we have a live reference to the ego vehicle (attach on first use)
        ego = env.controlled_vehicles[0]
        ego_lane = ego.lane_index[2]
        user_speed, user_steering = user_input

        dist_to_blocker = self._signed_dist_nearest_in_lane(env, ego, lane_id=ego_lane)
        gap_ahead_l1, gap_behind_l1 = self._scan_lane(env, ego, lane_id=ego_lane ^ 1)

        action_speed = ACTION_IDLE
        if dist_to_blocker > 0 and dist_to_blocker < FOLLOW_DISTANCE_MIN:
            action_speed = ACTION_SLOWER
        elif dist_to_blocker < 0 and -dist_to_blocker < FOLLOW_DISTANCE_MIN:
            action_speed = ACTION_FASTER
        elif user_speed:
            if user_speed > 0:
                action_speed = ACTION_FASTER
            elif user_speed < 0:
                action_speed = ACTION_SLOWER

        # Heading action logic with persistent steering
        gap_ok = (
            gap_ahead_l1 > GAP_REQUIRED_AHEAD
            and gap_behind_l1 > GAP_REQUIRED_BEHIND
        )

        # Check if lane change has completed
        if self.merge_out_target_lane is not None and ego_lane == self.merge_out_target_lane:
            # Lane change successful, clear the target
            self.merge_out_target_lane = None

        # If user provides new steering input, update target; otherwise persist
        action_steering = ACTION_IDLE
        if user_steering != 0:
            # User is providing steering input
            if gap_ok:
                if user_steering > 0:
                    action_steering = ACTION_LANE_LEFT  # Lane left
                    self.merge_out_target_lane = 0
                elif user_steering < 0:
                    action_steering = ACTION_LANE_RIGHT   # Lane right
                    self.merge_out_target_lane = 1
        elif self.merge_out_target_lane is not None:
            # User is not steering, but we're in the middle of a merge; persist
            if self.merge_out_target_lane == 1:
                action_steering = ACTION_LANE_RIGHT  # Continue RIGHT
            elif self.merge_out_target_lane == 0:
                action_steering = ACTION_LANE_LEFT   # Continue LEFT

        return (action_speed, action_steering)

    def act(self, obs: np.ndarray, env: "TwoLaneOvertakingEnv", user_input: tuple) -> np.ndarray:
        """Execute the state machine and return a 2D continuous action [acceleration, steering].
        
        Combines separate speed and heading actions (from act_upper) into continuous outputs:
        - speed_action ∈ {-1, 0, +1} → acceleration via speed_control
        - heading_action ∈ {-1, 0, +1} → steering via steering_control, with persistence during merges
        """
        # Attach ego vehicle reference if not already attached
        ego = env.controlled_vehicles[0]

        self.v = ControlledVehicle(ego.road, ego.position, ego.heading, ego.speed, ego.lane_index)        # Get speed and heading actions separately
        action_speed, action_steering = self.act_upper(obs, env, user_input)

        ego_lane = int(round(ego.position[1] / StraightLane.DEFAULT_WIDTH))
        ego_velocity_x = ego.velocity[0]
        ego_velocity_y = ego.velocity[1]
        ego_velocity = np.linalg.norm([ego_velocity_x, ego_velocity_y])

        command = np.array([0.0, 0.0])  # [acceleration, steering]
        # Map discrete action to continuous 2D output [acceleration, steering]
        if action_speed == ACTION_FASTER:
            command[0] = self.v.speed_control(target_speed=(ego_velocity + 0.5))
        elif action_speed == ACTION_SLOWER:
            command[0] = self.v.speed_control(target_speed=(ego_velocity - 1.0))
        elif action_speed == ACTION_IDLE:
            command[0] = self.v.speed_control(target_speed=ego_velocity)

        _from, _to, _id = ego.lane_index
        lane_1 = (_from, _to, 1)
        lane_0 = (_from, _to, 0)

        # Apply heading action (with persistence during merge)
        if action_steering == ACTION_LANE_LEFT:  # Lane left
            command[1] = self.v.steering_control(target_lane_index=lane_0)
        elif action_steering == ACTION_LANE_RIGHT:  # Lane right
            command[1] = self.v.steering_control(target_lane_index=lane_1)
        else:
            if ego_lane == 0:
                command[1] = self.v.steering_control(target_lane_index=lane_0)
            else:
                command[1] = self.v.steering_control(target_lane_index=lane_1)
        return command.flatten().astype(np.float32)


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
            vehicle_lane = vehicle.lane_index[2]#int(round(vehicle.position[1] / StraightLane.DEFAULT_WIDTH))
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
            vehicle_lane = vehicle.lane_index[2]#int(round(vehicle.position[1] / StraightLane.DEFAULT_WIDTH))
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
            vehicle_lane = vehicle.lane_index[2]#int(round(vehicle.position[1] / StraightLane.DEFAULT_WIDTH))
            if vehicle_lane != lane_id:
                continue
            dx = ego_x - vehicle.position[0]
            if dx > 0:
                behind_min = min(behind_min, dx)
        return behind_min