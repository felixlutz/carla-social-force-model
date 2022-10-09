import numpy as np

import forces
from pedestrian_state import PedState


class PedestrianSimulation:
    def __init__(self, borders, border_section_info, obstacles, sfm_config, step_length):
        self.sfm_config = sfm_config

        self.borders = borders
        self.section_info = border_section_info
        self.obstacles = obstacles

        self.peds = PedState(step_length, sfm_config)

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
        if activated_forces.get('border_force', False):
            force_list.append(forces.BorderForce(self.delta_t, self.sfm_config, self.borders, self.section_info))
        if activated_forces.get('obstacle_evasion_force', False):
            force_list.append(forces.ObstacleEvasionForce(self.delta_t, self.sfm_config, self.obstacles))
        if activated_forces.get('ped_repulsive_force', False):
            force_list.append(forces.PedRepulsiveForce(self.delta_t, self.sfm_config))
        if activated_forces.get('space_repulsive_force', False):
            force_list.append(forces.SpaceRepulsiveForce(self.delta_t, self.sfm_config, self.borders))

        return force_list

    def tick(self):
        """Do one step in the simulation"""

        # skip social force calculations if pedestrian state matrix is empty or None
        if self.peds.state is None or self.peds.size() == 0:
            return

        # record current state for plotting
        self.peds.record_current_state()

        F = sum(map(lambda x: x.get_force(self.peds), self.forces))

        self.peds.calculate_new_velocities(F)

    def close(self):
        pass

    def get_arrived_peds(self, distance_threshold):
        if self.peds.state is None:
            return []

        diff = self.peds.next_waypoint()[:, :2] - self.peds.loc()[:, :2]
        distance = np.linalg.norm(diff, axis=-1)

        arrived_peds = self.peds.name()[distance < distance_threshold]

        return arrived_peds

    def spawn_pedestrian(self, initial_ped_state):
        self.peds.add_pedestrian(initial_ped_state)

    def destroy_pedestrian(self, ped_name):
        self.peds.remove_pedestrian(ped_name)

    def update_ped_info(self, walker_id, location, velocity):
        self.peds.update_state(walker_id, location, velocity)

    def get_new_velocities(self):
        return self.peds.get_new_velocities()

    def get_borders(self):
        return self.borders

    def get_states(self):
        return self.peds.get_all_states()
