import logging
from abc import ABC, abstractmethod

import numpy as np

import stateutils
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
        """Abstract class to get social forces
            return: an array of force vectors for each pedestrian
        """
        raise NotImplementedError

    def get_force(self, ped_state, debug=False):
        force = self._get_force(ped_state)
        if debug:
            logging.debug(f"{type(self).__name__}:\n {repr(force)}")
        return force


class GoalAttractiveForce(Force):
    """Goal attractive force based on the original paper "Social force model for pedestrian dynamics"
    from Helbing and Molnár (1995)"""

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
    """Ped to ped repulsive force based on the original paper "Social force model for pedestrian dynamics"
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
    """Obstacles to ped repulsive force based on the original paper "Social force model for pedestrian dynamics"
    from Helbing and Molnár (1995)"""

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
        return np.sum(F_aB, axis=1)


class PedestrianForce(Force):
    """Calculates the social force between pedestrians based on the paper "Experimental study of the behavioural
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
        theta = stateutils.angle_diff_2d(interaction_direction, diff_direction)

        # compute model parameter B = gamma * ||D||
        B = self.gamma * interaction_length

        # apply bias to right-hand side for evasions
        theta += B * self.epsilon

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


class ObstacleForce(Force):
    """Calculates the force between pedestrians and the nearest obstacle based on the paper "Experimental study of
    the behavioural mechanisms underlying self-organization in human crowds" form Moussaïd et. al (2009)
    """

    def __init__(self, step_length, sfm_config, obstacles):
        super().__init__(step_length, sfm_config)
        self.obstacles = obstacles

        self.obstacle_force_config = self.sfm_config['obstacle_force']
        self.a = self.obstacle_force_config.get('a', 3.0)
        self.b = self.obstacle_force_config.get('b', 0.1)

    def _get_force(self, peds):

        if not self.obstacles:
            return np.zeros((peds.size(), 3))

        ped_loc = np.expand_dims(peds.loc()[:, :2], 1)
        closest_i = [
            np.argmin(np.linalg.norm(ped_loc - np.expand_dims(boundary, 0), axis=-1), axis=1)
            for boundary in self.obstacles
        ]
        closest_points = np.swapaxes(
            np.stack([boundary[i] for boundary, i in zip(self.obstacles, closest_i)]),
            0, 1)  # index order: pedestrian, boundary, coordinates

        direction, distance = stateutils.normalize(ped_loc - closest_points)

        if self.use_ped_radius:
            distance = distance - np.expand_dims(peds.radius(), -1)

        force = direction * self.a * np.exp(-1.0 * np.expand_dims(distance, -1) / self.b)

        # append z=0 to force vectors to make them 3D
        z_values = np.zeros(force.shape[0:2])
        force = np.dstack((force, z_values))

        return np.sum(force, axis=1)
