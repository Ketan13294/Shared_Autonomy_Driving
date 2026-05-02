"""Rule-based controller for the overtaking manoeuvre."""

import numpy as np

from highway_env.road.lane import StraightLane
from highway_env.vehicle.controller import MDPVehicle

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


class OvertakingController:
    """A rule-based state-machine controller for the overtaking manoeuvre."""

    def __init__(self) -> None:
        self.state = STATE_FOLLOW

    def reset(self) -> None:
        self.state = STATE_FOLLOW

    def act(self, obs: np.ndarray, env: "TwoLaneOvertakingEnv") -> int:
        ego = env.controlled_vehicles[0]
        ego_lane = int(round(ego.position[1] / StraightLane.DEFAULT_WIDTH))

        dist_to_blocker = self._signed_dist_nearest_in_lane(env, ego, lane_id=0)
        gap_ahead_l1, gap_behind_l1 = self._scan_lane(env, ego, lane_id=1)

        if self.state == STATE_CLEAR:
            return ACTION_FASTER

        if self.state == STATE_FOLLOW:
            if dist_to_blocker < FOLLOW_DISTANCE_MIN:
                return ACTION_SLOWER
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
            if dist_to_blocker < FOLLOW_DISTANCE_MIN:
                return ACTION_SLOWER
            if dist_to_blocker > FOLLOW_DISTANCE_OK * 2:
                self.state = STATE_FOLLOW
            return ACTION_IDLE

        if self.state == STATE_MERGE_OUT:
            if ego_lane == 1:
                self.state = STATE_PASSING
                return ACTION_FASTER
            return ACTION_LANE_RIGHT

        if self.state == STATE_PASSING:
            if dist_to_blocker < -SAFE_RETURN_GAP:
                gap_behind_l0 = self._gap_behind_in_lane(env, ego, lane_id=0)
                if gap_behind_l0 > 20.0:
                    self.state = STATE_MERGE_BACK
            return ACTION_FASTER

        if self.state == STATE_MERGE_BACK:
            if ego_lane == 0:
                self.state = STATE_CLEAR
                return ACTION_IDLE
            return ACTION_LANE_LEFT

        return ACTION_IDLE

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