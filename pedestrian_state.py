import numpy as np

import stateutils
from ped_mode_state_machine import PedMode


class PedState:
    """
    Tracks the state of pedestrians using a structured numpy array that contains information on
    name, carla id, location vector, current velocity vector, next waypoint, radius and target speed
    of every pedestrian modeled by the pedestrian simulation.
    """

    def __init__(self, step_length, sfm_config):
        self.step_length = step_length
        self.max_speed_factor = sfm_config.get('max_speed_factor', 1.3)

        self.ped_state_dtype = [('name', 'U8'), ('id', 'i4'), ('loc', 'f8', (3,)), ('vel', 'f8', (3,)),
                                ('next_waypoint', 'f8', (3,)), ('mode', 'O'), ('radius', 'f8'),
                                ('target_speed', 'f8')]

        self.state = None
        self.new_velocities = None

        # list of all states to record simulation
        self.all_states = []

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

    def size(self) -> int:
        return self.state.shape[0]

    def name(self) -> np.ndarray:
        return self.state['name']

    def walker_id(self) -> np.ndarray:
        return self.state['id']

    def loc(self) -> np.ndarray:
        return self.state['loc']

    def vel(self) -> np.ndarray:
        return self.state['vel']

    def next_waypoint(self) -> np.ndarray:
        return self.state['next_waypoint']

    def mode(self) -> np.ndarray:
        return self.state['mode']

    def radius(self) -> np.ndarray:
        return self.state['radius']

    def target_speed(self) -> np.ndarray:
        return self.state['target_speed']

    def max_speed(self) -> np.ndarray:
        return self.target_speed() * self.max_speed_factor

    def speeds(self) -> np.ndarray:
        """Return the speeds corresponding to a given state."""
        return stateutils.speeds(self.state)

    def calculate_new_velocities(self, force):
        """Calculate new desired velocities according to forces"""
        # desired velocity
        desired_velocity = self.vel() + self.step_length * force
        desired_velocity = self.capped_velocity(desired_velocity, self.max_speed())

        self.new_velocities = self.state[['id', 'vel']]
        self.new_velocities['vel'] = desired_velocity

    def get_new_velocities(self) -> np.ndarray:
        return self.new_velocities

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

    def desired_directions(self) -> np.ndarray:
        return stateutils.desired_directions(self.state)

    def record_current_state(self):
        self.all_states.append(self.state.copy())

    def get_all_states(self) -> np.ndarray:
        return np.stack(self.all_states)

    @staticmethod
    def capped_velocity(desired_velocity, max_velocity) -> np.ndarray:
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        desired_speeds[desired_speeds == 0.0] = 1.0
        factor = np.minimum(1.0, max_velocity / desired_speeds)
        return desired_velocity * np.expand_dims(factor, -1)
