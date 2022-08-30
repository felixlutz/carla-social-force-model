import numpy as np

import forces
from pedestrian import PedState


class PedestrianSimulation:
    def __init__(self, initial_ped_state, obstacles, sfm_config, walker_dict, step_length):
        self.sfm_config = sfm_config
        self.walker_dict = walker_dict

        self.obstacles = obstacles

        # update pedestrian states with correct CARLA actor ids
        self.initial_ped_state = clean_up_ped_state(initial_ped_state)
        print(*self.initial_ped_state, sep="\n")

        self.peds = PedState(self.initial_ped_state, step_length, sfm_config)

        self.delta_t = step_length
        self.forces = self.init_forces()

    def init_forces(self):
        activated_forces = self.sfm_config['forces']
        force_list = []

        # initialize social forces according to config
        if activated_forces.get('goal_attractive_force', False):
            force_list.append(forces.GoalAttractiveForce(self.delta_t, self.sfm_config))
        if activated_forces.get('pedestrian_force', False):
            force_list.append(forces.PedestrianForce(self.delta_t, self.sfm_config))
        if activated_forces.get('obstacle_force', False):
            force_list.append(forces.ObstacleForce(self.delta_t, self.sfm_config, self.obstacles))
        if activated_forces.get('ped_repulsive_force', False):
            force_list.append(forces.PedRepulsiveForce(self.delta_t, self.sfm_config))
        if activated_forces.get('space_repulsive_force', False):
            force_list.append(forces.SpaceRepulsiveForce(self.delta_t, self.sfm_config, self.obstacles))

        return force_list

    def tick(self):
        """Do one step in the simulation"""

        # record current state for plotting
        self.peds.record_current_state()

        # skip social force calculations if pedestrian state matrix is empty
        if self.peds.size() == 0:
            return

        F = sum(map(lambda x: x.get_force(self.peds), self.forces))

        self.peds.calculate_new_velocities(F)

    def close(self):
        pass

    def update_ped_info(self, walker_id, location, velocity):
        self.peds.update_state(walker_id, location, velocity)

    def get_new_velocities(self):
        return self.peds.get_new_velocities()

    def get_obstacles(self):
        return self.obstacles

    def get_states(self):
        return self.peds.get_all_states()


def clean_up_ped_state(ped_state):

    updated_ped_state = np.delete(ped_state, np.where(ped_state['id'] == -1)[0], axis=0)

    return updated_ped_state
