import numpy as np

import stateutils
from fieldofview import FieldOfView
from potentials import PedPedPotential, PedSpacePotential

MAX_SPEED_MULTIPLIER = 1.3  # with respect to initial speed


class PedestrianSimulation:
    def __init__(self, initial_ped_state, obstacles, sfm_config, walker_dict, step_length):
        self.sfm_config = sfm_config
        self.walker_dict = walker_dict

        self.obstacles = obstacles

        # update pedestrian states with correct CARLA actor ids
        self.initial_ped_state = add_walker_ids_to_ped_state(initial_ped_state, walker_dict)
        print(*self.initial_ped_state, sep="\n")

        self.state = self.initial_ped_state

        self.new_velocities = self.initial_ped_state[['id', 'vel']]

        self.initial_speeds = stateutils.speeds(initial_ped_state)
        self.max_speeds = MAX_SPEED_MULTIPLIER * self.initial_speeds

        self.delta_t = step_length
        self.V = PedPedPotential(self.delta_t)

        if self.obstacles is not None:
            self.U = PedSpacePotential(self.obstacles)
        else:
            self.U = None

        self.w = FieldOfView()

    def f_ab(self):
        """Compute f_ab."""
        return -1.0 * self.V.grad_r_ab(self.state)

    def f_aB(self):
        """Compute f_aB."""
        if self.U is None:
            return None
        return -1.0 * self.U.grad_r_aB(self.state)

    def capped_velocity(self, desired_velocity):
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, self.max_speeds / desired_speeds)
        factor[desired_speeds == 0] = 0.0
        return desired_velocity * np.expand_dims(factor, 1)

    def tick(self):
        """Do one step in the simulation and update the state in place."""

        # skip social force calculations if pedestrian state matrix is empty
        if self.state.size == 0:
            return

        # accelerate to desired velocity
        e = stateutils.desired_directions(self.state)
        vel = self.state['vel']
        tau = np.expand_dims(self.state['tau'], 1)
        F0 = 1.0 / tau * (np.expand_dims(self.initial_speeds, 1) * e - vel)

        # repulsive terms between pedestrians
        f_ab = self.f_ab()
        w = np.expand_dims(self.w(e, -f_ab), -1)
        F_ab = w * f_ab

        # repulsive terms between pedestrians and boundaries
        F_aB = self.f_aB()
        z_values = np.zeros(F_aB.shape[0:2])
        F_aB = np.dstack((F_aB, z_values))

        # social force
        F = F0 + np.sum(F_ab, axis=1) + np.sum(F_aB, axis=1)
        # desired velocity
        w = vel + self.delta_t * F
        # velocity
        self.new_velocities['vel'] = self.capped_velocity(w)

    def close(self):
        pass

    def update_ped_info(self, walker_id, location, velocity):
        self.state['loc'][self.state['id'] == walker_id] = location
        self.state['vel'][self.state['id'] == walker_id] = velocity

    def get_new_velocities(self):
        return self.new_velocities


def add_walker_ids_to_ped_state(ped_state, walker_dict):
    for pedestrian in ped_state:
        role_name = pedestrian[0]
        pedestrian[1] = walker_dict.get(role_name, -1)

    updated_ped_state = np.delete(ped_state, np.where(ped_state['id'] == -1)[0], axis=0)

    return updated_ped_state
