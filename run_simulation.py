import argparse
import logging
import random
import time

import carla
import numpy as np
import tomli

from carla_simulation import CarlaSimulation
from pedestrian_simulation import PedestrianSimulation


class SimulationRunner(object):
    """
    SimulationRunner class is responsible for the synchronization of the Social-Force and CARLA
    simulations.
    """

    def __init__(self, pedestrian_simulation, carla_simulation, walker_dict):

        self.ped_sim = pedestrian_simulation
        self.carla_sim = carla_simulation
        self.walker_dict = walker_dict

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

        # Tick pedestrian simulation and propagate new velocities resulting from social force model to CARLA
        self.ped_sim.tick()
        new_velocities = self.ped_sim.get_new_velocities()
        for velocity in new_velocities:
            walker_id = velocity['id'].item()
            new_velocity = velocity['vel']

            speed = np.linalg.norm(new_velocity)
            new_velocity = new_velocity / speed
            direction = carla.Vector3D(new_velocity[0], new_velocity[1], new_velocity[2])

            self.carla_sim.set_ped_velocity(walker_id, direction, speed)

    def close(self):
        """
        Cleans synchronization.
        """
        # Configuring CARLA simulation in async mode.
        settings = self.carla_sim.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.carla_sim.world.apply_settings(settings)

        # Destroying synchronized actors.
        for carla_actor_id in self.walker_dict.values():
            self.carla_sim.destroy_actor(carla_actor_id)

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
    carla_simulation = CarlaSimulation(args.carla_host, args.carla_port, args.step_length, scenario_config)

    # spawn initial pedestrians
    spawn_points, spawn_speeds, initial_ped_state = extract_ped_info(scenario_config)
    walker_dict = spawn_pedestrians(spawn_points, spawn_speeds, carla_simulation, scenario_config)

    # extract obstacles for pedestrian simulation
    obstacles = extract_obstacle_info(scenario_config)

    # initialize pedestrian simulation
    pedestrian_simulation = PedestrianSimulation(initial_ped_state, obstacles, sfm_config,
                                                 walker_dict, args.step_length)

    sim_runner = SimulationRunner(pedestrian_simulation, carla_simulation, walker_dict)

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
    spawn_speeds = {}
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

        # spawn points and spawn speeds for CARLA simulator
        spawn_points[role_name] = spawn_point
        spawn_speeds[role_name] = speed

        direction = spawn_point.get_forward_vector()
        velocity = np.array([direction.x, direction.y, direction.z]) * speed
        walker_id = -1  # set invalid walker id that gets replaced as soon as the walker is actually spawned

        ped_state_data.append((role_name, walker_id, spawn_location, velocity, destination, tau))

    # initial pedestrian state matrix for social force simulator
    ped_state_dtype = [('name', 'U8'), ('id', 'i4'), ('loc', 'f8', (3,)), ('vel', 'f8', (3,)),
                       ('dest', 'f8', (3,)), ('tau', 'f8')]
    initial_ped_state = np.array(ped_state_data, dtype=ped_state_dtype)

    return spawn_points, spawn_speeds, initial_ped_state


def spawn_pedestrians(spawn_points, spawn_speeds, carla_sim, scenario_config):
    """
    Spawns the pedestrians in CARLA and places the spectator camera.
    :param spawn_points:
    :param spawn_speeds:
    :param carla_sim:
    :param scenario_config:
    :return:
    """
    spectator_focus = scenario_config.get('walker').get('spectator_focus')

    walker_dict = {}
    for role_name, spawn_point in spawn_points.items():

        # select random walker blueprint
        walker_bp = random.choice(carla_sim.walker_blueprints)

        if walker_bp.has_attribute('role_name'):
            walker_bp.set_attribute('role_name', role_name)

        actor_id = carla_sim.spawn_actor(walker_bp, spawn_point)

        if actor_id == -1:
            continue

        walker_dict[role_name] = actor_id

        # tick CARLA simulation to actually spawn walker
        # carla_sim.tick()
        #
        # transform = carla_sim.get_ped_transform(actor_id)
        # direction = transform.get_forward_vector()
        # speed = spawn_speeds[role_name]
        # carla_sim.set_ped_velocity(actor_id, direction, speed)

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

    if obstacle_config is not None:
        borders = obstacle_config.get('borders', [])
        obstacle_resolution = obstacle_config.get('resolution')

        obstacles = []
        for border in borders:
            start_point = np.array(border['start_point'])
            end_point = np.array(border['end_point'])

            if sumo_coordinates:
                start_point = convert_coordinates(start_point, sumo_offset)
                end_point = convert_coordinates(end_point, sumo_offset)

            samples = int(np.linalg.norm(end_point - start_point) * obstacle_resolution)

            border_line = np.column_stack((np.linspace(start_point[0], end_point[0], samples),
                                           np.linspace(start_point[1], end_point[1], samples)))
            obstacles.append(border_line)

        return obstacles

    else:
        return None


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--scenario-config',
                           default='config/minimal_scenario_config.toml',
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
    argparser.add_argument('--debug', action='store_true', help='enable debug messages')
    arguments = argparser.parse_args()

    if arguments.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    simulation_loop(arguments)
