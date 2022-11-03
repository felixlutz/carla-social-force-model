import logging
import random

import carla
import numpy as np

import stateutils
from path_planner import PedPathPlanner, GraphType
from ped_mode_state_machine import PedMode, PedModeStateMachine


class PedSpawnManager:
    """
    Class managing all pedestrian spawners.
    """

    def __init__(self, scenario_config, carla_sim, ped_sim):
        self.scenario_config = scenario_config
        self.carla_sim = carla_sim
        self.ped_sim = ped_sim

        self.spectator_focus = scenario_config.get('walker', {}).get('spectator_focus')
        self.ped_seed = scenario_config.get('walker', {}).get('pedestrian_seed', 2000)
        self.variate_speed = scenario_config.get('walker', {}).get('variate_speed', 0.0)
        waypoint_distance = scenario_config.get('walker', {}).get('waypoint_distance', 20)
        jaywalking_weight = scenario_config.get('walker', {}).get('jaywalking_weight', 2)

        # set pedestrian seed
        self.carla_sim.world.set_pedestrians_seed(self.ped_seed)

        # initialize pedestrian path planner
        self.path_planner = PedPathPlanner(self.carla_sim.carla_map, waypoint_distance, jaywalking_weight)

        # get pedestrian spawners from scenario config
        self.ped_spawners = self._extract_ped_info()

        self.ped_index = 0
        self.walker_dict = {}
        self.waypoint_dict = {}

    def tick(self, sim_time):
        """
        Check if pedestrian spawners are ready to spawn in this simulation step and let them spawn pedestrians
        if that's the case.
        :param sim_time:
        :return:
        """
        # filter out pedestrian spawner that don't have any pedestrians left to spawn
        self.ped_spawners[:] = [ped_spawner for ped_spawner in self.ped_spawners if ped_spawner.quantity > 0]

        for ped_spawner in self.ped_spawners:
            if ped_spawner.ready_to_spawn(sim_time):
                self._spawn_pedestrian(ped_spawner)
                ped_spawner.quantity -= 1

    def _extract_ped_info(self):
        """
        Extracts pedestrian spawner information from the configuration file and converts coordinates if necessary.
        """

        sumo_coordinates = self.scenario_config.get('map', {}).get('sumo_coordinates', False)
        sumo_offset = self.scenario_config.get('map', {}).get('sumo_offset')
        ped_spawner_config = self.scenario_config.get('walker', {}).get('ped_spawner')

        ped_spawners = []
        if ped_spawner_config is not None:
            for spawn_point in ped_spawner_config:

                spawn_location = np.array(spawn_point['spawn_location'])
                speed = spawn_point['speed']
                destination = np.array(spawn_point['destination'])

                # generate route or use manually defined waypoints depending on configuration
                generate_route = spawn_point.get('generate_route')
                if generate_route:
                    waypoint_tuples = self.path_planner.generate_route(spawn_location, destination,
                                                                       GraphType[generate_route])
                    waypoints = np.array([w[0] for w in waypoint_tuples])
                    crossing_road_bools = [w[1] for w in waypoint_tuples]
                else:
                    wp_list = spawn_point.get('waypoints', [])
                    wp_list.append(destination)
                    waypoints = np.array(wp_list)

                    crossing_road_bools = spawn_point.get('crossing_road_bools', [False] * len(waypoints))
                    if len(waypoints) != len(crossing_road_bools):
                        logging.warning('Length of waypoints and crossing_road_bools is not equal! '
                                        'Waypoints may get cut of!')

                quantity = spawn_point.get('quantity', 1)
                spawn_time = spawn_point.get('spawn_time', 0.0)
                spawn_interval = spawn_point.get('spawn_interval', 1.0)

                # convert coordinates if they are from SUMO simulator
                if sumo_coordinates and sumo_offset is not None:
                    spawn_location = stateutils.convert_coordinates(spawn_location, sumo_offset)
                    if waypoints.ndim > 1:
                        np.apply_along_axis(stateutils.convert_coordinates, 1, waypoints)
                    else:
                        waypoints = stateutils.convert_coordinates(waypoints, sumo_offset)

                ped_spawner = PedSpawner(spawn_location, waypoints, crossing_road_bools, speed, quantity, spawn_time,
                                         spawn_interval)
                ped_spawners.append(ped_spawner)

        return ped_spawners

    def _spawn_pedestrian(self, ped_spawner):
        """
        Spawn pedestrian in both Carla and pedestrian simulation.
        """
        spawn_point = ped_spawner.carla_spawn_point
        ped_name = self._generate_ped_name()

        random.seed(self.ped_seed)
        walker_bp = random.choice(self.carla_sim.walker_blueprints)

        if walker_bp.has_attribute('role_name'):
            walker_bp.set_attribute('role_name', ped_name)

        # spawn pedestrian in Carla
        actor_id = self.carla_sim.spawn_actor(walker_bp, spawn_point)

        if self.variate_speed != 0.0:
            ped_spawner.target_speed += random.uniform(-self.variate_speed, self.variate_speed)

        # increment ped seed so next walker has a different random blueprint and speed variation
        self.ped_seed += 1

        if actor_id == -1:
            logging.info(f'Failed to spawn pedestrian {ped_name}.')
        else:
            ped_radius = self.carla_sim.get_ped_radius(actor_id)
            initial_ped_state, remaining_waypoints = ped_spawner.generate_ped_state(ped_name, actor_id, ped_radius)

            # spawn pedestrian in pedestrian simulation
            self.ped_sim.spawn_pedestrian(initial_ped_state)

            self.walker_dict[ped_name] = actor_id
            # it's important to make a shallow copy of the remaining waypoints with [:], because otherwise waypoints
            # get removed for all peds of one spawner when the first pedestrian reaches the waypoint
            self.waypoint_dict[ped_name] = remaining_waypoints[:]

            # place spectator camera behind selected pedestrian
            if self.spectator_focus == ped_name:
                spectator = self.carla_sim.world.get_spectator()
                spectator_transform = carla.Transform()
                spectator_transform.location = spawn_point.transform(carla.Vector3D(-2.0, 0.0, 2.0))
                spectator_transform.rotation = spawn_point.rotation
                spectator.set_transform(spectator_transform)

            logging.info(f'Spawned pedestrian {ped_name}.')

    def _generate_ped_name(self):
        """
        Generate pedestrian name with incrementing index.
        """
        name = 'ped_' + str(self.ped_index)
        self.ped_index += 1

        return name


class PedSpawner:
    """
    Class containing all the information necessary to spawn one or multiple pedestrians from one spawn point.
    """

    def __init__(self, spawn_location, waypoints, crossing_road_bools, speed, quantity, spawn_time, spawn_interval):
        self.spawn_location = spawn_location
        self.target_speed = speed
        self.quantity = quantity
        self.spawn_interval = spawn_interval
        self.next_spawn_time = spawn_time

        if crossing_road_bools[0]:
            self.initial_mode = PedMode.CROSSING_ROAD
        else:
            self.initial_mode = PedMode.WALKING_SIDEWALK

        if waypoints.ndim > 1:
            self.first_waypoint = waypoints[0]
            self.remaining_waypoint_tuples = list(zip(waypoints[1:].tolist(), crossing_road_bools[1:]))
        else:
            self.first_waypoint = waypoints
            self.remaining_waypoint_tuples = []

        self.carla_spawn_point = self.generate_carla_spawn_point()
        direction = self.carla_spawn_point.get_forward_vector()
        self.velocity = np.array([direction.x, direction.y, direction.z]) * speed

    def ready_to_spawn(self, sim_time):
        """
        Check if pedestrian spawner is ready to spawn in this simulation time step.
        :param sim_time:
        :return: True if pedestrian spawner is ready to spawn and False if not
        """
        if self.next_spawn_time <= sim_time:
            self.next_spawn_time += self.spawn_interval
            return True
        else:
            return False

    def generate_ped_state(self, name, carla_id, radius):
        """
        Generate pedestrian state from spawn info.
        :param name:
        :param carla_id:
        :param radius:
        :return: initial pedestrian state and remaining waypoints
        """
        ped_mode = PedModeStateMachine(name, self.target_speed, self.initial_mode)
        ped_state = (name, carla_id, self.spawn_location, self.velocity, self.first_waypoint, ped_mode,
                     radius, self.target_speed)

        return ped_state, self.remaining_waypoint_tuples

    def generate_carla_spawn_point(self):
        """
        Generate spawn point for Carla simulator.
        :return: Carla spawn point
        """
        carla_spawn_point = carla.Transform()
        carla_spawn_point.location = carla.Location(self.spawn_location[0],
                                                    self.spawn_location[1],
                                                    self.spawn_location[2])
        # calculate spawn rotation
        direction = self.first_waypoint - self.spawn_location
        ref_axis = np.array([1.0, 0.0, 0.0])
        rotation_angle = stateutils.angle_diff_2d(direction, ref_axis)
        carla_spawn_point.rotation = carla.Rotation(0, np.degrees(rotation_angle), 0)

        return carla_spawn_point
