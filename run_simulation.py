import argparse
import logging
import time

import carla
import numpy as np
import tomli

import visualization
from carla_simulation import CarlaSimulation
from pedestrian_simulation import PedestrianSimulation
from pedestrian_spawner import PedSpawnManager
from sidewalk import extract_sidewalk
from stateutils import convert_coordinates


class SimulationRunner:
    """
    SimulationRunner class is responsible for the synchronization of the Social-Force and CARLA
    simulations.
    """

    def __init__(self, pedestrian_simulation, carla_simulation, ped_spawn_manager, scenario_config, args):

        self.ped_sim = pedestrian_simulation
        self.carla_sim = carla_simulation
        self.ped_spawn_manager = ped_spawn_manager
        self.scenario_config = scenario_config

        self.plot = args.plot
        self.animate = args.animate
        self.output_path = args.output
        self.step_length = args.step_length

        walker_config = scenario_config.get('walker', {})
        self.draw_bounding_boxes = walker_config.get('draw_bounding_boxes', False)
        self.despawn_on_arrival = walker_config.get('despawn_on_arrival', True)
        self.despawn_threshold = walker_config.get('despawn_threshold', 0.2)

        self.walker_dict = ped_spawn_manager.walker_dict
        self.waypoint_dict = ped_spawn_manager.waypoint_dict

    def tick(self):
        """
        Tick to simulation synchronization. One tick = one simulation step.
        """
        # spawn pedestrians that are supposed to spawn in this time step
        sim_time = self.carla_sim.get_sim_time()
        self.ped_spawn_manager.tick(sim_time)

        # get all pedestrians that arrived at their next waypoint and either assign a new waypoint or despawn
        # them if they reached their final destination
        arrived_peds = self.ped_sim.get_arrived_peds(self.despawn_threshold)
        for ped_name in arrived_peds:
            remaining_waypoints = self.waypoint_dict[ped_name]

            if remaining_waypoints:
                next_waypoint = remaining_waypoints.pop(0)
                self.ped_sim.peds.update_next_waypoint(ped_name, next_waypoint)
                self.waypoint_dict[ped_name] = remaining_waypoints

            elif not remaining_waypoints and self.despawn_on_arrival:
                self.ped_sim.destroy_pedestrian(ped_name)
                self.carla_sim.destroy_actor(self.walker_dict[ped_name])
                self.walker_dict.pop(ped_name)
                self.waypoint_dict.pop(ped_name)
                logging.info(f'Despawned pedestrian {ped_name}.')

        # Tick CARLA simulation and receive new location and velocity of every pedestrian and propagate
        # it to the pedestrian simulation
        self.carla_sim.tick()
        for actor_id in self.walker_dict.values():
            walker = self.carla_sim.get_actor(actor_id)
            carla_location = walker.get_location()
            carla_velocity = walker.get_velocity()

            location = np.array([carla_location.x, carla_location.y, carla_location.z])
            velocity = np.array([carla_velocity.x, carla_velocity.y, carla_velocity.z])

            self.ped_sim.update_ped_info(actor_id, location, velocity)

            if self.draw_bounding_boxes:
                self.carla_sim.draw_bounding_box(actor_id, self.step_length)

        # Tick pedestrian simulation and propagate new velocities resulting from social force model to CARLA
        self.ped_sim.tick()
        new_velocities = self.ped_sim.get_new_velocities()
        if new_velocities is not None:
            for velocity in new_velocities:
                walker_id = velocity['id'].item()
                new_velocity = velocity['vel']

                speed = np.linalg.norm(new_velocity)
                if speed != 0.0:
                    new_velocity = new_velocity / speed
                direction = carla.Vector3D(new_velocity[0], new_velocity[1], new_velocity[2])

                self.carla_sim.set_ped_velocity(walker_id, direction, speed)

    def close(self):
        """
        Cleans synchronization.
        """
        # Destroying synchronized actors.
        for carla_actor_id in self.walker_dict.values():
            self.carla_sim.destroy_actor(carla_actor_id)

        if self.plot:
            with visualization.SceneVisualizer(self.ped_sim, self.output_path, self.step_length) as sv:
                sv.plot()
        if self.animate:
            with visualization.SceneVisualizer(self.ped_sim, self.output_path, self.step_length) as sv:
                sv.animate()

        # Closing pedestrian simulation and CARLA client.
        self.carla_sim.close()
        self.ped_sim.close()


def simulation_loop(args):
    """
    Entry point for CARLA simulation with social force pedestrian model. Main simulation loop.
    """

    # load configs
    scenario_config = load_config(args.scenario_config)
    sfm_config = load_config(args.sfm_config)

    # initialize CARLA simulation
    carla_simulation = CarlaSimulation(args, scenario_config)

    # extract obstacle borders from scenario config
    obstacle_borders, carla_obstacle_borders = extract_obstacle_info(scenario_config)

    # extract sidewalk borders from map and add to other obstacle borders
    sidewalk_borders, carla_sidewalk_borders = extract_sidewalk(carla_simulation.carla_map, scenario_config)
    obstacle_borders.extend(sidewalk_borders)
    carla_obstacle_borders.extend(carla_sidewalk_borders)

    # draw obstacles
    if carla_simulation.draw_obstacles:
        for border in carla_obstacle_borders:
            carla_simulation.draw_points(border)

    # initialize pedestrian simulation
    pedestrian_simulation = PedestrianSimulation(obstacle_borders, sfm_config, args.step_length)

    # initialize pedestrian spawn manager
    ped_spawn_manager = PedSpawnManager(scenario_config, carla_simulation, pedestrian_simulation)

    # initialize simulation runner
    sim_runner = SimulationRunner(pedestrian_simulation, carla_simulation, ped_spawn_manager, scenario_config, args)

    # main simulation loop
    try:
        while True:
            start = time.time()

            sim_runner.tick()

            end = time.time()
            elapsed = end - start
            if elapsed < args.step_length:
                time.sleep(args.step_length - elapsed)

    except KeyboardInterrupt:
        logging.info('Cancelled by user.')

    finally:
        logging.info('Cleaning Simulation')

        sim_runner.close()


def load_config(config_path):
    """
    Loads configuration from file.
    :param config_path:
    :return:
    """
    with open(config_path, mode='rb') as fp:
        config = tomli.load(fp)
    return config


def extract_obstacle_info(scenario_config):
    sumo_coordinates = scenario_config['map']['sumo_coordinates']
    sumo_offset = scenario_config.get('map').get('sumo_offset')
    obstacle_config = scenario_config.get('obstacles')

    obstacles = []
    carla_obstacles = []
    if obstacle_config is not None:
        borders = obstacle_config.get('borders', [])
        obstacle_resolution = obstacle_config.get('resolution', 0.1)

        for border in borders:
            start_point = np.array(border['start_point'])
            end_point = np.array(border['end_point'])

            if sumo_coordinates:
                start_point = convert_coordinates(start_point, sumo_offset)
                end_point = convert_coordinates(end_point, sumo_offset)

            samples = int(np.linalg.norm(end_point - start_point) / obstacle_resolution)

            border_line = np.column_stack((np.linspace(start_point[0], end_point[0], samples),
                                           np.linspace(start_point[1], end_point[1], samples)))
            obstacles.append(border_line)

            carla_obstacles.append([carla.Vector3D(p[0], p[1], 0) for p in border_line])

    return obstacles, carla_obstacles


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--scenario-config',
                           default='config/scenarios/sidewalk_flow_scenario_config.toml',
                           type=str,
                           help='scenario configuration file')
    argparser.add_argument('--sfm-config',
                           default='config/sfm_config.toml',
                           type=str,
                           help='social force model configuration file')
    argparser.add_argument('--carla-host',
                           metavar='H',
                           default='127.0.0.1',
                           help='IP of the carla_sim host server (default: 127.0.0.1)')
    argparser.add_argument('--carla-port',
                           metavar='P',
                           default=2000,
                           type=int,
                           help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--step-length',
                           default=0.1,
                           type=float,
                           help='set fixed delta seconds (default: 0.1s)')
    argparser.add_argument('--plot', action='store_true', help='plot pedestrian trajectories')
    argparser.add_argument('--animate', action='store_true', help='animate pedestrian trajectories')
    argparser.add_argument('--output',
                           default='output/sim_run',
                           type=str,
                           help='path for output plot or animation')
    argparser.add_argument('--debug', action='store_true', help='enable debug messages')
    arguments = argparser.parse_args()

    if arguments.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    simulation_loop(arguments)
