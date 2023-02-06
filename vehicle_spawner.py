import logging
import random

import carla
import numpy as np

from agents.navigation.extended_behavior_agent import ExtendedBehaviorAgent


class VehicleSpawnManager:
    """
    Class managing all vehicle spawners.
    """

    def __init__(self, scenario_config, carla_sim):
        self.scenario_config = scenario_config
        self.carla_sim = carla_sim

        self.vehicle_seed = scenario_config.get('vehicle', {}).get('vehicle_seed', 2000)
        self.variate_speed_factor = scenario_config.get('vehicle', {}).get('variate_speed_factor', 0.0)
        self.recommended_spawn_points = carla_sim.world.get_map().get_spawn_points()
        no_bikes = scenario_config.get('vehicle', {}).get('no_bikes', False)

        # get vehicle spawners from scenario config
        self.vehicle_spawners = self._extract_vehicle_info()

        blueprint_library = carla_sim.world.get_blueprint_library().filter('vehicle')
        if no_bikes:
            self.vehicle_blueprints = [x for x in blueprint_library if int(x.get_attribute('number_of_wheels')) == 4]
        else:
            self.vehicle_blueprints = blueprint_library

        self.SpawnActor = carla.command.SpawnActor
        self.SetAutopilot = carla.command.SetAutopilot
        self.FutureActor = carla.command.FutureActor

        self.traffic_manager = carla_sim.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(self.vehicle_seed)

        self.vehicle_list = []
        self.trajectory_dict = {}
        self.vehicle_agent_dict = {}

    def tick(self, sim_time):
        """
        Check if vehicle spawners are ready to spawn in this simulation step and let them spawn vehicles
        if that's the case.
        :param sim_time:
        :return:
        """
        # filter out vehicle spawner that don't have any vehicles left to spawn
        self.vehicle_spawners[:] = [spawner for spawner in self.vehicle_spawners if spawner.quantity > 0]

        for vehicle_spawner in self.vehicle_spawners:
            if vehicle_spawner.ready_to_spawn(sim_time):
                self._spawn_vehicle(vehicle_spawner)
                vehicle_spawner.quantity -= 1

    def _extract_vehicle_info(self):
        """
        Extracts vehicle spawner information from the configuration file.
        """

        vehicle_spawner_config = self.scenario_config.get('vehicle', {}).get('vehicle_spawner')

        vehicle_spawners = []
        if vehicle_spawner_config is not None:
            for spawner in vehicle_spawner_config:
                spawn_location = spawner.get('spawn_point')
                blueprint = spawner.get('blueprint')
                auto_pilot = spawner.get('auto_pilot', True)
                use_traffic_manager = spawner.get('use_traffic_manager', True)
                destination = spawner.get('destination')
                trajectory = spawner.get('trajectory', [])
                headings = spawner.get('headings', [])
                speeds = spawner.get('speeds', [])
                max_speed = spawner.get('max_speed', 50)
                speed_reduction_factor = spawner.get('speed_reduction_factor', 30)
                quantity = spawner.get('quantity', 1)
                spawn_time = spawner.get('spawn_time', 0.0)
                spawn_interval = spawner.get('spawn_interval', 5.0)
                ignore_walkers_percentage = spawner.get('ignore_walkers_percentage', 0)
                ignore_lights_percentage = spawner.get('ignore_lights_percentage', 0)

                vehicle_spawner = VehicleSpawner(spawn_location, blueprint, auto_pilot, use_traffic_manager,
                                                 destination, trajectory, headings, speeds, max_speed,
                                                 speed_reduction_factor, quantity, spawn_time, spawn_interval,
                                                 ignore_walkers_percentage, ignore_lights_percentage,
                                                 self.recommended_spawn_points)
                vehicle_spawners.append(vehicle_spawner)

        return vehicle_spawners

    def _spawn_vehicle(self, vehicle_spawner):
        """
        Spawn vehicle in Carla simulation.
        """
        spawn_transform = vehicle_spawner.carla_spawn_transform

        random.seed(self.vehicle_seed)

        if vehicle_spawner.blueprint:
            vehicle_bp = self.vehicle_blueprints.find(vehicle_spawner.blueprint)
        else:
            vehicle_bp = random.choice(self.vehicle_blueprints)

        # spawn vehicle in Carla
        tm_autopilot = vehicle_spawner.auto_pilot and vehicle_spawner.use_traffic_manager
        batch = [self.SpawnActor(vehicle_bp, spawn_transform)
                     .then(self.SetAutopilot(self.FutureActor, tm_autopilot, self.traffic_manager.get_port()))]
        actor_id = self.carla_sim.spawn_actor_with_batch(batch)

        if self.variate_speed_factor != 0.0:
            vehicle_spawner.speed_reduction_factor += random.uniform(-self.variate_speed_factor,
                                                                     self.variate_speed_factor)

        # increment vehicle seed so next walker has a different random blueprint and speed variation
        self.vehicle_seed += 1

        if actor_id != -1:

            vehicle = self.carla_sim.world.get_actor(actor_id)
            self.vehicle_list.append(actor_id)

            if vehicle_spawner.auto_pilot:
                if vehicle_spawner.use_traffic_manager:
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle,
                                                                             vehicle_spawner.speed_reduction_factor)
                    self.traffic_manager.ignore_walkers_percentage(vehicle, vehicle_spawner.ignore_walkers_percentage)
                    self.traffic_manager.ignore_lights_percentage(vehicle, vehicle_spawner.ignore_lights_percentage)
                else:
                    self.carla_sim.tick()
                    agent = ExtendedBehaviorAgent(vehicle, behavior='custom_speed',
                                                  max_speed=vehicle_spawner.max_speed)
                    if vehicle_spawner.destination:
                        agent.set_destination(vehicle_spawner.carla_destination_transform.location,
                                              vehicle_spawner.carla_spawn_transform.location)
                    agent.ignore_traffic_lights(vehicle_spawner.ignore_lights_percentage > 0)
                    agent.ignore_pedestrians(vehicle_spawner.ignore_walkers_percentage > 0)
                    self.vehicle_agent_dict[actor_id] = agent
            else:
                carla_trajectory = [generate_carla_transform(location, heading)
                                    for location, heading in zip(vehicle_spawner.trajectory, vehicle_spawner.headings)]
                self.trajectory_dict[actor_id] = {}
                self.trajectory_dict[actor_id]['trajectory'] = carla_trajectory
                self.trajectory_dict[actor_id]['speeds'] = vehicle_spawner.speeds

            logging.info(f'Spawned vehicle {actor_id} of type {vehicle.type_id}.')


class VehicleSpawner:
    """
    Class containing all the information necessary to spawn one or multiple vehicles from one spawn point.
    """

    def __init__(self, spawn_point, blueprint, auto_pilot, use_traffic_manager, destination, trajectory, headings,
                 speeds, max_speed, speed_reduction_factor, quantity, spawn_time, spawn_interval,
                 ignore_walkers_percentage, ignore_lights_percentage, recommended_spawn_points):
        self.spawn_point = spawn_point
        self.blueprint = blueprint
        self.auto_pilot = auto_pilot
        self.use_traffic_manager = use_traffic_manager
        self.destination = destination
        self.trajectory = trajectory
        self.headings = headings
        self.speeds = speeds[1:]
        self.max_speed = max_speed
        self.speed_reduction_factor = speed_reduction_factor
        self.quantity = quantity
        self.spawn_interval = spawn_interval
        self.next_spawn_time = spawn_time
        self.ignore_walkers_percentage = ignore_walkers_percentage
        self.ignore_lights_percentage = ignore_lights_percentage
        self.recommended_spawn_points = recommended_spawn_points

        self.carla_spawn_transform = self.generate_carla_spawn_transform()
        if self.destination:
            self.carla_destination_transform = self.recommended_spawn_points[self.destination]

    def ready_to_spawn(self, sim_time):
        """
        Check if vehicle spawner is ready to spawn in this simulation time step.
        :param sim_time:
        :return: True if vehicle spawner is ready to spawn and False if not
        """
        if self.next_spawn_time <= sim_time:
            self.next_spawn_time += self.spawn_interval
            return True
        else:
            return False

    def generate_carla_spawn_transform(self):
        """
        Generate spawn transform for Carla simulator.
        :return: Carla spawn point
        """
        if self.spawn_point:
            transform = self.recommended_spawn_points[self.spawn_point]
        else:
            spawn_loc = self.trajectory.pop(0)
            heading = self.headings.pop(0)

            transform = generate_carla_transform(spawn_loc, heading, 1.0)

        return transform


def generate_carla_transform(location, heading, z_loc=0.0):
    transform = carla.Transform()
    transform.location = carla.Location(location[0], location[1], z_loc)
    transform.rotation = carla.Rotation(0, np.degrees(heading), 0)

    return transform
