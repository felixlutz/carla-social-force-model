import logging
from abc import ABC, abstractmethod
from itertools import compress

import numpy as np

import stateutils
from ped_mode_state_machine import PedMode
from potentials import PedPedPotential, PedSpacePotential
from fieldofview import FieldOfView


class Force(ABC):
    """Force base class"""

    def __init__(self, step_length, sfm_config):
        super().__init__()
        self.step_length = step_length
        self.sfm_config = sfm_config
        self.use_ped_radius = sfm_config.get('use_ped_radius', False)

    @abstractmethod
    def _get_force(self, ped_state) -> np.ndarray:
        """
        Abstract class to get social forces
        return: an array of force vectors for each pedestrian
        """
        raise NotImplementedError

    def get_force(self, ped_state, debug=False):
        force = self._get_force(ped_state)
        if debug:
            logging.debug(f"{type(self).__name__}:\n {repr(force)}")
        return force


class GoalAttractiveForce(Force):
    """
    Goal attractive force based on the original paper "Social force model for pedestrian dynamics"
    from Helbing and Molnár (1995)
    """

    def __init__(self, step_length, sfm_config):
        super().__init__(step_length, sfm_config)

        self.tau = self.sfm_config.get('goal_force', {}).get('tau', 0.5)

    def _get_force(self, peds):
        tau = np.full([peds.size(), 1], self.tau)
        target_speed = np.expand_dims(peds.target_speed(), -1)
        desired_direction = peds.desired_directions()
        velocity = peds.vel()
        F_0 = 1.0 / tau * (target_speed * desired_direction - velocity)

        return F_0


class PedRepulsiveForce(Force):
    """
    Ped to ped repulsive force based on the original paper "Social force model for pedestrian dynamics"
    from Helbing and Molnár (1995)
    """

    def __init__(self, step_length, sfm_config):
        super().__init__(step_length, sfm_config)

        self.ped_ped_config = self.sfm_config['ped_ped_potential']
        self.V = PedPedPotential(self.step_length, self.ped_ped_config['v0'], self.ped_ped_config['sigma'])

        self.fov_config = self.sfm_config['field_of_view']
        self.fov = FieldOfView(self.fov_config['two_phi'], self.fov_config['out_of_view_factor'])

    def _get_force(self, peds):
        f_ab = -1.0 * self.V.grad_r_ab(peds.state)

        w = np.expand_dims(self.fov(peds.desired_directions(), -f_ab), -1)
        F_ab = w * f_ab
        return np.sum(F_ab, axis=1)


class SpaceRepulsiveForce(Force):
    """
    Obstacles to ped repulsive force based on the original paper "Social force model for pedestrian dynamics"
    from Helbing and Molnár (1995)
    """

    def __init__(self, step_length, sfm_config, obstacles):
        super().__init__(step_length, sfm_config)
        self.obstacles = obstacles

        if self.obstacles is not None:
            self.ped_space_config = self.sfm_config['ped_space_potential']
            self.U = PedSpacePotential(self.obstacles, self.ped_space_config['u0'], self.ped_space_config['r'])
        else:
            self.U = None

    def _get_force(self, peds):
        if self.U is None:
            F_aB = np.zeros((peds.size(), 0, 3))
        else:
            F_aB = -1.0 * self.U.grad_r_aB(peds.state)
            # append z=0 to force vectors to make them 3D
            z_values = np.zeros(F_aB.shape[0:2])
            F_aB = np.dstack((F_aB, z_values))

            # deactivate obstacle force for pedestrians that are crossing the road
            crossing_road = [m.current_mode in [PedMode.CROSSING_ROAD, PedMode.ROAD_TO_SIDEWALK] for m in peds.mode()]
            F_aB[crossing_road] *= 0

        return np.sum(F_aB, axis=1)


class PedestrianForce(Force):
    """
    Calculates the social force between pedestrians based on the paper "Experimental study of the behavioural
    mechanisms underlying self-organization in human crowds" form Moussaïd et. al (2009)
    """

    def __init__(self, step_length, sfm_config):
        super().__init__(step_length, sfm_config)

        # set model parameters
        self.ped_force_config = self.sfm_config['pedestrian_force']
        self.lambda_weight = self.ped_force_config.get('lambda', 2.0)
        self.A = self.ped_force_config.get('A', 4.5)
        self.gamma = self.ped_force_config.get('gamma', 0.35)
        self.n = self.ped_force_config.get('n', 2.0)
        self.n_prime = self.ped_force_config.get('n_prime', 3.0)
        self.epsilon = self.ped_force_config.get('epsilon', 0.005)

    def _get_force(self, peds):
        loc_diff = stateutils.all_diffs(peds.loc())
        diff_direction, diff_length = stateutils.normalize(loc_diff)
        vel_diff = -1.0 * stateutils.all_diffs(peds.vel())

        # subtract the radii of the pedestrians from the distances between each other
        if self.use_ped_radius:
            sum_radii = stateutils.all_sums(peds.radius())
            diff_length -= sum_radii

        # compute interaction direction t_ij
        interaction_vec = self.lambda_weight * vel_diff + diff_direction
        interaction_direction, interaction_length = stateutils.normalize(interaction_vec)

        # compute n_ij (normal vector of t_ij orientated to the left)
        left_normal_direction = np.zeros(np.shape(interaction_direction))
        left_normal_direction[..., 0] = interaction_direction[..., 1] * -1
        left_normal_direction[..., 1] = interaction_direction[..., 0]

        # compute angle theta (between interaction and position difference direction)
        theta = stateutils.angle_diff_2d(diff_direction, interaction_direction)

        # compute model parameter B = gamma * ||D||
        B = self.gamma * interaction_length

        # apply bias to right-hand side for evasions
        # (negative epsilon due to left-handed Unreal Engine coordinate system?)
        theta += B * (-self.epsilon)

        # deceleration force along interaction direction t_ij
        f_v_value = (-1.0 * self.A
                     * np.exp(-1.0 * diff_length / B - np.square(self.n_prime * B * theta)))

        # force describing the directional change along n_ij (normal vector of t_ij orientated to the left)
        f_theta_value = (-1.0 * self.A * np.sign(theta)
                         * np.exp(-1.0 * diff_length / B - np.square(self.n * B * theta)))

        # build force vectors from force value and direction
        f_v = np.expand_dims(f_v_value, -1) * interaction_direction
        f_theta = np.expand_dims(f_theta_value, -1) * left_normal_direction

        force = f_v + f_theta

        return np.sum(force, axis=1)


class BorderForce(Force):
    """
    Calculates the force between pedestrians and the nearest border (e.g. sidewalk border) based on the paper
    "Experimental study of the behavioural mechanisms underlying self-organization in human crowds" form
    Moussaïd et. al (2009)
    """

    def __init__(self, step_length, sfm_config, borders, section_info):
        super().__init__(step_length, sfm_config)
        self.borders = borders
        section_info = np.array(section_info)
        self.section_center = np.vstack(section_info[:, 0])
        self.section_length = section_info[:, 1]

        self.border_force_config = self.sfm_config['border_force']
        self.a = self.border_force_config.get('a', 3.0)
        self.b = self.border_force_config.get('b', 0.1)

    def _get_force(self, peds):

        if not self.borders:
            return np.zeros((peds.size(), 3))

        forces = []

        for ped in peds.state:
            loc = ped['loc'][:2]

            # filter out borders that are too far away to be relevant
            distances = np.linalg.norm(loc - self.section_center, axis=-1)
            distance_filter = distances < self.section_length
            close_borders = list(compress(self.borders, distance_filter))

            # get the closest point of each border within relevant range
            closest_i = [np.argmin(np.linalg.norm(loc - border, axis=-1)) for border in close_borders]
            closest_points = [border[i] for border, i in zip(close_borders, closest_i)]

            if closest_points:
                direction, distance = stateutils.normalize(loc - closest_points)

                if self.use_ped_radius:
                    distance -= ped['radius']

                f = direction * self.a * np.exp(-1.0 * np.expand_dims(distance, -1) / self.b)

                forces.append(np.sum(f, axis=0))
            else:
                forces.append(np.zeros(2))

        force = np.array(forces)

        # append z=0 to force vectors to make them 3D
        z_values = np.zeros(len(force))
        force = np.column_stack((force, z_values))

        # deactivate border force for pedestrians that are crossing the road
        crossing_road = [m.current_mode in [PedMode.CROSSING_ROAD, PedMode.ROAD_TO_SIDEWALK] for m in peds.mode()]
        force[crossing_road] *= 0

        return force


class ObstacleEvasionForce(Force):
    """
    Calculates the social force between pedestrians and obstacles based on the pedestrian interaction force of the paper
    "Experimental study of the behavioural mechanisms underlying self-organization in human crowds" form Moussaïd
    et. al (2009)
    """

    def __init__(self, step_length, sfm_config, dynamic=False):
        super().__init__(step_length, sfm_config)
        self.obstacle_locs = None
        self.obstacle_borders = None
        self.obstacle_velocities = None

        # set model parameters
        if dynamic:
            self.evasion_force_config = self.sfm_config['dynamic_obstacle_force']
        else:
            self.evasion_force_config = self.sfm_config['static_obstacle_force']
        self.lambda_weight = self.evasion_force_config.get('lambda', 2.0)
        self.A = self.evasion_force_config.get('A', 4.5)
        self.gamma = self.evasion_force_config.get('gamma', 0.35)
        self.n = self.evasion_force_config.get('n', 2.0)
        self.n_prime = self.evasion_force_config.get('n_prime', 3.0)
        self.epsilon = self.evasion_force_config.get('epsilon', 0.005)
        self.perception_threshold = self.evasion_force_config.get('perception_threshold', 20)

    def _get_force(self, peds):
        if self.obstacle_locs is None or self.obstacle_locs.size == 0:
            return np.zeros((peds.size(), 3))

        if self.obstacle_velocities is None:
            self.obstacle_velocities = np.zeros((len(self.obstacle_locs), 2))

        forces = []

        for ped in peds.state:
            loc = ped['loc'][:2]
            vel = ped['vel'][:2]

            # filter out obstacles that are outside the defined perception threshold
            distances = np.linalg.norm(loc - self.obstacle_locs, axis=-1)
            distance_filter = distances < self.perception_threshold
            close_borders = list(compress(self.obstacle_borders, distance_filter))
            close_obstacle_vel = np.array(self.obstacle_velocities)[distance_filter]

            # get the closest border point of each obstacle
            closest_i = [np.argmin(np.linalg.norm(loc - border, axis=-1)) for border in close_borders]
            closest_points = [border[i] for border, i in zip(close_borders, closest_i)]

            if closest_points:

                diff_direction, diff_length = stateutils.normalize(closest_points - loc)
                vel_diff = (vel - close_obstacle_vel)

                # subtract the radii of the pedestrians from the distances
                if self.use_ped_radius:
                    diff_length -= ped['radius']

                # compute interaction direction t_ij
                interaction_vec = self.lambda_weight * vel_diff + diff_direction
                interaction_direction, interaction_length = stateutils.normalize(interaction_vec)

                # compute n_ij (normal vector of t_ij orientated to the left)
                left_normal_direction = np.zeros(np.shape(interaction_direction))
                left_normal_direction[..., 0] = interaction_direction[..., 1] * -1
                left_normal_direction[..., 1] = interaction_direction[..., 0]

                # compute angle theta (between interaction and position difference direction)
                theta = stateutils.angle_diff_2d(diff_direction, interaction_direction)

                # compute model parameter B = gamma * ||D||
                B = self.gamma * interaction_length

                # apply bias to right-hand side for evasions
                theta += B * (-self.epsilon)

                # deceleration force along interaction direction t_ij
                f_v_value = (-1.0 * self.A
                             * np.exp(-1.0 * diff_length / B - np.square(self.n_prime * B * theta)))

                # force describing the directional change along n_ij (normal vector of t_ij orientated to the left)
                f_theta_value = (-1.0 * self.A * np.sign(theta)
                                 * np.exp(-1.0 * diff_length / B - np.square(self.n * B * theta)))

                # build force vectors from force value and direction
                f_v = np.expand_dims(f_v_value, -1) * interaction_direction
                f_theta = np.expand_dims(f_theta_value, -1) * left_normal_direction

                f = f_v + f_theta

                forces.append(np.sum(f, axis=0))

            else:
                forces.append(np.zeros(2))

        force = np.array(forces)

        # append z=0 to force vectors to make them 3D
        z_values = np.zeros(len(force))
        force = np.column_stack((force, z_values))

        return force

    def update_obstacles(self, obstacles):
        obstacle_locs, obstacle_borders = zip(*obstacles)
        self.obstacle_locs = np.array(obstacle_locs)
        self.obstacle_borders = obstacle_borders

    def update_obstacle_velocities(self, velocities):
        self.obstacle_velocities = np.array(velocities)
