from enum import IntEnum


class PedMode(IntEnum):
    WAITING = 0
    WALKING_SIDEWALK = 1
    CROSSING_ROAD = 2
    ROAD_TO_SIDEWALK = 3
    CHECKING_TRAFFIC = 4


class PedModeManager:
    """
    This class represents a simple finite state machine that controls the transitions between the different pedestrian
    modes (e.g., waiting, walking on sidewalk, crossing road, ...).
    """

    def __init__(self, ped_name, target_speed, initial_mode, crossing_speed_factor, crossing_safety_margin):
        self.sim_time = 0
        self.ped_name = ped_name
        self.initial_target_speed = target_speed
        self.crossing_speed = crossing_speed_factor * target_speed
        self.crossing_safety_margin = crossing_safety_margin

        self.target_speed = target_speed
        self.current_mode = initial_mode
        self.next_mode_time = -1
        self.waiting_time = 5

    def tick(self, sim_time):
        """Called during each simulation step."""
        self.sim_time = sim_time
        if self.current_mode == PedMode.WAITING:
            if self.next_mode_time <= sim_time:
                self._activate_mode(PedMode.WALKING_SIDEWALK)

    def set_mode(self, new_mode):
        """
        Set new desired mode for pedestrian. Depending on the current mode an intermediate mode may be activated first.
        """

        if self.current_mode == PedMode.WALKING_SIDEWALK and new_mode == PedMode.CROSSING_ROAD:
            self._activate_mode(PedMode.CHECKING_TRAFFIC)
        elif self.current_mode == PedMode.CROSSING_ROAD and new_mode == PedMode.WALKING_SIDEWALK:
            self._activate_mode(PedMode.ROAD_TO_SIDEWALK)
        else:
            self._activate_mode(new_mode)

    def _activate_mode(self, mode):
        """Actually internally activate new mode for pedestrian."""

        if mode == PedMode.WAITING:
            self.target_speed = 0
            self.next_mode_time = self.sim_time + self.waiting_time
            self.current_mode = mode

        elif mode == PedMode.WALKING_SIDEWALK:
            self.target_speed = self.initial_target_speed
            self.current_mode = mode

        elif mode == PedMode.CROSSING_ROAD:
            self.target_speed = self.crossing_speed
            self.current_mode = mode

        elif mode == PedMode.ROAD_TO_SIDEWALK:
            self.current_mode = mode

        elif mode == PedMode.CHECKING_TRAFFIC:
            self.target_speed = 0
            self.current_mode = mode
