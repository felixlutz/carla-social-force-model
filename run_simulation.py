import argparse
import logging
import random
import time

import carla
import numpy as np
import tomli

import visualization
from carla_simulation import CarlaSimulation
from pedestrian_simulation import PedestrianSimulation
from sidewalk import extract_sidewalk


class SimulationRunner:
    """
    SimulationRunner class is responsible for the synchronization of the Social-Force and CARLA
    simulations.
    """

    def __init__(self, pedestrian_simulation, carla_simulation, walker_dict, scenario_config, args):

        self.ped_sim = pedestrian_simulation
        self.carla_sim = carla_simulation
        self.walker_dict = walker_dict
        self.scenario_config = scenario_config
        self.draw_bounding_boxes = scenario_config['walker'].get('draw_bounding_boxes', False)

        self.plot = args.plot
        self.animate = args.animate
        self.output_path = args.output
        self.step_length = args.step_length

    def tick(self):
        """
        Tick to simulation synchronization. One tick = one simulation step.
        """

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

    # spawn initial pedestrians
    spawn_points, initial_ped_state = extract_ped_info(scenario_config)
    walker_dict = spawn_pedestrians(spawn_points, carla_simulation, scenario_config, initial_ped_state)

    # initialize pedestrian simulation
    pedestrian_simulation = PedestrianSimulation(initial_ped_state, obstacle_borders, sfm_config,
                                                 walker_dict, args.step_length)

    sim_runner = SimulationRunner(pedestrian_simulation, carla_simulation, walker_dict, scenario_config, args)

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


def convert_coordinates(coordinates, sumo_offset):
    """
    Converts SUMO coordinates to Carla coordinated by applying a map offset and inverting the y-axis.
    :param coordinates:
    :param sumo_offset:
    :return:
    """

    if len(coordinates) == 2:
        new_coordinates = coordinates - sumo_offset[0:2]
    else:
        new_coordinates = coordinates - sumo_offset

    new_coordinates[1] *= -1

    return new_coordinates


def extract_ped_info(scenario_config):
    """
    Extracts pedestrian spawn information from the configuration file and prepares it for both the CARLA client and the
    pedestrian simulation.
    :param scenario_config:
    :return:
    """

    sumo_coordinates = scenario_config['map']['sumo_coordinates']
    sumo_offset = scenario_config.get('map').get('sumo_offset')
    pedestrian_config = scenario_config['walker']['pedestrians']

    spawn_points = {}
    ped_state_data = []
    for pedestrian in pedestrian_config:

        role_name = pedestrian['role_name']
        spawn_location = np.array(pedestrian['spawn_location'])
        spawn_rotation = np.array(pedestrian['spawn_rotation'])
        speed = pedestrian['spawn_speed']
        destination = np.array(pedestrian['destination'])
        tau = pedestrian.get('tau', 0.5)

        # convert coordinates if they are from SUMO simulator
        if sumo_coordinates and sumo_offset is not None:
            spawn_location = convert_coordinates(spawn_location, sumo_offset)
            destination = convert_coordinates(destination, sumo_offset)

        spawn_point = carla.Transform()
        spawn_point.location = carla.Location(spawn_location[0], spawn_location[1], spawn_location[2])
        spawn_point.rotation = carla.Rotation(spawn_rotation[0], spawn_rotation[1], spawn_rotation[2])

        # spawn points for CARLA simulator
        spawn_points[role_name] = spawn_point

        direction = spawn_point.get_forward_vector()
        velocity = np.array([direction.x, direction.y, direction.z]) * speed

        # set invalid walker id and default radius that get replaced as soon as the walker is actually spawned
        walker_id = -1
        radius = 0.2

        ped_state_data.append((role_name, walker_id, spawn_location, velocity, destination, radius, tau))

    # initial pedestrian state matrix for social force simulator
    ped_state_dtype = [('name', 'U8'), ('id', 'i4'), ('loc', 'f8', (3,)), ('vel', 'f8', (3,)),
                       ('dest', 'f8', (3,)), ('radius', 'f8'), ('tau', 'f8')]
    initial_ped_state = np.array(ped_state_data, dtype=ped_state_dtype)

    return spawn_points, initial_ped_state


def spawn_pedestrians(spawn_points, carla_sim, scenario_config, initial_ped_state):
    """
    Spawns the pedestrians in CARLA and places the spectator camera.
    :param spawn_points:
    :param spawn_speeds:
    :param carla_sim:
    :param scenario_config:
    :return:
    """
    spectator_focus = scenario_config.get('walker').get('spectator_focus')
    ped_seed = scenario_config.get('walker').get('pedestrian_seed', 2000)

    walker_dict = {}

    # set pedestrian seed
    carla_sim.world.set_pedestrians_seed(ped_seed)
    random.seed(ped_seed)
    random_bps = random.choices(carla_sim.walker_blueprints, k=len(spawn_points))

    for i, (role_name, spawn_point) in enumerate(spawn_points.items()):

        # select random walker blueprint
        walker_bp = random_bps[i]

        if walker_bp.has_attribute('role_name'):
            walker_bp.set_attribute('role_name', role_name)

        actor_id = carla_sim.spawn_actor(walker_bp, spawn_point)

        if actor_id == -1:
            continue

        walker_dict[role_name] = actor_id

        walker_radius = carla_sim.get_ped_radius(actor_id)
        initial_ped_state['radius'][initial_ped_state['name'] == role_name] = walker_radius
        initial_ped_state['id'][initial_ped_state['name'] == role_name] = actor_id

        # place spectator camera behind selected walker
        if spectator_focus == role_name:
            spectator = carla_sim.world.get_spectator()
            spectator_transform = carla.Transform()
            spectator_transform.location = spawn_point.transform(carla.Vector3D(-2.0, 0.0, 2.0))
            spectator_transform.rotation = spawn_point.rotation
            spectator.set_transform(spectator_transform)

    logging.info('Spawned %d walkers.' % (len(walker_dict)))

    return walker_dict


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
                           default='config/scenarios/sidewalk_scenario_config.toml',
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
