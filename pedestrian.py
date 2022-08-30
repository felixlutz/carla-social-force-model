import numpy as np

import stateutils


class PedState:
    """Tracks the state of pedestrians"""

    def __init__(self, initial_state, step_length, sfm_config):
        self.sfm_config = sfm_config
        self.step_length = step_length
        self.agent_radius = self.sfm_config.get('agent_radius', 0.35)
        self.max_speed_multiplier = sfm_config.get('max_speed_multiplier', 1.3)

        self.initial_speeds = stateutils.speeds(initial_state)
        self.max_speeds = self.max_speed_multiplier * self.initial_speeds

        self.state = initial_state
        self.new_velocities = initial_state[['id', 'vel']]

        # list of all states to record simulation
        self.all_states = [self.state.copy()]


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

    def dest(self) -> np.ndarray:
        return self.state['dest']

    def radius(self) -> np.ndarray:
        return self.state['radius']

    def tau(self) -> np.ndarray:
        return self.state['tau']

    def speeds(self) -> np.ndarray:
        """Return the speeds corresponding to a given state."""
        return stateutils.speeds(self.state)

    def calculate_new_velocities(self, force):
        """Calculate new desired velocities according to forces"""
        # desired velocity
        desired_velocity = self.vel() + self.step_length * force
        desired_velocity = self.capped_velocity(desired_velocity, self.max_speeds)

        self.new_velocities['vel'] = desired_velocity

    def get_new_velocities(self) -> np.ndarray:
        return self.new_velocities

    def update_state(self, walker_id, location, velocity):
        self.state['loc'][self.state['id'] == walker_id] = location
        self.state['vel'][self.state['id'] == walker_id] = velocity

    def desired_directions(self) -> np.ndarray:
        return stateutils.desired_directions(self.state)

    def record_current_state(self):
        self.all_states.append(self.state.copy())

    def get_all_states(self):
        return np.stack(self.all_states)

    @staticmethod
    def capped_velocity(desired_velocity, max_velocity) -> np.ndarray:
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        desired_speeds[desired_speeds == 0.0] = 1.0
        factor = np.minimum(1.0, max_velocity / desired_speeds)
        return desired_velocity * np.expand_dims(factor, -1)
