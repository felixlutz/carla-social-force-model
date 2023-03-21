import numpy as np

import stateutils
from ped_mode_manager import PedMode


class PedState:
    """
    Tracks the state of pedestrians using a structured numpy array that contains information on
    name, carla id, location vector, current velocity vector, next waypoint, radius and target speed
    of every pedestrian modeled by the pedestrian simulation.
    """

    def __init__(self, sfm_config):
        self.max_speed_factor = sfm_config.get('max_speed_factor', 1.3)

        self.ped_state_dtype = [('name', 'U8'), ('id', 'i4'), ('loc', 'f8', (3,)), ('vel', 'f8', (3,)),
                                ('next_waypoint', 'f8', (3,)), ('mode', 'O'), ('radius', 'f8'),
                                ('target_speed', 'f8')]

        self.state = None

        # list of all states to record simulation
        self.all_states = {}

    def add_pedestrian(self, initial_ped_state):
        """
        Adds a new pedestrian to the pedestrian state array.
        :param initial_ped_state:
        """
        new_ped = np.expand_dims(np.array(initial_ped_state, dtype=self.ped_state_dtype), axis=0)

        if self.state is None:
            self.state = new_ped
        else:
            self.state = np.append(self.state, new_ped, axis=0)

    def remove_pedestrian(self, ped_name):
        """
        Remove a pedestrian from the pedestrian state array.
        :param ped_name:
        """
        self.state = np.delete(self.state, np.where(self.state['name'] == ped_name), axis=0)

    def size(self):
        return self.state.shape[0]

    def name(self):
        return self.state['name']

    def walker_id(self):
        return self.state['id']

    def loc(self):
        return self.state['loc']

    def vel(self):
        return self.state['vel']

    def next_waypoint(self):
        return self.state['next_waypoint']

    def mode(self):
        return self.state['mode']

    def radius(self):
        return self.state['radius']

    def target_speed(self):
        return self.state['target_speed']

    def max_speed(self):
        return self.target_speed() * self.max_speed_factor

    def speeds(self):
        """Return the speeds corresponding to a given state."""
        return stateutils.speeds(self.state)

    def update_state(self, walker_id, location, velocity):
        self.state['loc'][self.state['id'] == walker_id] = location
        self.state['vel'][self.state['id'] == walker_id] = velocity

    def update_next_waypoint(self, ped_name, next_waypoint_tuple):
        next_waypoint, crossing_road = next_waypoint_tuple
        self.state['next_waypoint'][self.state['name'] == ped_name] = next_waypoint

        if crossing_road:
            ped_mode = PedMode.CROSSING_ROAD
        else:
            ped_mode = PedMode.WALKING_SIDEWALK

        self.state['mode'][self.state['name'] == ped_name][0].set_mode(ped_mode)

    def apply_current_mode(self):
        self.state['target_speed'] = [mode.target_speed for mode in self.state['mode']]

    def desired_directions(self):
        return stateutils.desired_directions(self.state)

    def record_current_state(self, sim_time):
        last_ped_state = self.state.copy()
        for ped in last_ped_state:
            ped['mode'] = ped['mode'].current_mode
        self.all_states[sim_time] = last_ped_state

    def get_all_states(self):
        return self.all_states

