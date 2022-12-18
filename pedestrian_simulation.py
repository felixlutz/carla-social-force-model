import numpy as np

import forces
from check_traffic import check_traffic
from ped_mode_state_machine import PedMode
from pedestrian_state import PedState


class PedestrianSimulation:
    def __init__(self, borders, border_section_info, obstacles, sfm_config, step_length):
        self.sfm_config = sfm_config

        self.borders = borders
        self.section_info = border_section_info

        self.static_obstacles = obstacles
        self.dyn_obs_ids = []
        self.dyn_obstacles = []
        self.dyn_obs_heading = []
        self.dyn_obs_vel = []
        self.dyn_obs_extent = []
        self.all_dyn_obs_states = {}

        self.peds = PedState(step_length, sfm_config)

        self.delta_t = step_length
        self.forces = self.init_forces()

    def init_forces(self):
        activated_forces = self.sfm_config['forces']
        force_dict = {}

        # initialize social forces according to config
        if activated_forces.get('goal_attractive_force', False):
            force_dict['goal_attractive_force'] = forces.GoalAttractiveForce(self.delta_t, self.sfm_config)
        if activated_forces.get('pedestrian_force', False):
            force_dict['pedestrian_force'] = forces.PedestrianForce(self.delta_t, self.sfm_config)
        if activated_forces.get('border_force', False):
            force_dict['border_force'] = forces.BorderForce(self.delta_t, self.sfm_config, self.borders,
                                                            self.section_info)
        if activated_forces.get('static_obstacle_force', False):
            force_dict['static_obstacle_force'] = forces.ObstacleEvasionForce(self.delta_t, self.sfm_config)
            force_dict['static_obstacle_force'].update_obstacles(self.static_obstacles)
        if activated_forces.get('dynamic_obstacle_force', False):
            force_dict['dynamic_obstacle_force'] = forces.ObstacleEvasionForce(self.delta_t, self.sfm_config, True)
        if activated_forces.get('ped_repulsive_force', False):
            force_dict['ped_repulsive_force'] = forces.PedRepulsiveForce(self.delta_t, self.sfm_config)
        if activated_forces.get('space_repulsive_force', False):
            force_dict['space_repulsive_force'] = forces.SpaceRepulsiveForce(self.delta_t, self.sfm_config,
                                                                             self.borders)

        return force_dict

    def tick(self, sim_time):
        """Do one step in the simulation"""
        # skip social force calculations if pedestrian state matrix is empty or None
        if self.peds.state is None or self.peds.size() == 0:
            return

        self.peds.apply_current_mode()
        for mode in self.peds.state['mode']:
            mode.tick(sim_time)

        for ped in self.peds.state[[m.current_mode == PedMode.CHECKING_TRAFFIC for m in self.peds.mode()]]:
            ready_to_cross = True
            if self.dyn_obstacles:
                ready_to_cross = check_traffic(ped, self.dyn_obstacles, self.dyn_obs_vel, self.dyn_obs_extent)

            if ready_to_cross:
                ped['mode'].set_mode(PedMode.CROSSING_ROAD)

        # record current state for plotting
        self.peds.record_current_state(sim_time)

        if self.dyn_obstacles:
            self.record_dyn_obstacle_states(sim_time)

        F = sum(map(lambda x: x.get_force(self.peds), self.forces.values()))

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

    def update_dynamic_obstacles(self, dynamic_obstacles):
        self.dyn_obs_ids, obstacle_pos, self.dyn_obs_heading, self.dyn_obs_vel, self.dyn_obs_extent, borders \
            = dynamic_obstacles
        self.dyn_obstacles = list(zip(obstacle_pos, borders))

        if 'dynamic_obstacle_force' in self.forces and self.dyn_obstacles:
            self.forces['dynamic_obstacle_force'].update_obstacles(self.dyn_obstacles)
            self.forces['dynamic_obstacle_force'].update_obstacle_velocities(self.dyn_obs_vel)

    def get_new_velocities(self):
        return self.peds.get_new_velocities()

    def get_borders(self):
        return self.borders

    def get_static_obstacles(self):
        return self.static_obstacles

    def get_dynamic_obstacles(self):
        return self.dyn_obstacles

    def record_dyn_obstacle_states(self, sim_time):
        obstacle_loc, _ = zip(*self.dyn_obstacles)
        veh_state_dtype = [('id', 'i4'), ('loc', 'f8', (2,)), ('heading', 'f8'), ('vel', 'f8', (2,)),
                           ('extent', 'f8', (2,))]
        veh_state = np.empty(len(self.dyn_obs_ids), dtype=veh_state_dtype)
        veh_state['id'] = self.dyn_obs_ids
        veh_state['loc'] = obstacle_loc
        veh_state['heading'] = self.dyn_obs_heading
        veh_state['vel'] = self.dyn_obs_vel
        veh_state['extent'] = self.dyn_obs_extent

        self.all_dyn_obs_states[sim_time] = veh_state

    def get_states(self):
        return self.peds.get_all_states()
