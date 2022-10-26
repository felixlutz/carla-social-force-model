import copy
import logging
import random
import time

import carla
import numpy as np
import tomli
from libcarla.command import SpawnActor


def load_config():
    with open('config/scenarios/sidewalk_scenario_config.toml', mode='rb') as fp:
        config = tomli.load(fp)
    return config


def convert_coordinates(spawn_coordinates):
    """
    Converts SUMO coordinates to Carla coordinated by applying a map offset and inverting the y-axis.
    :param spawn_coordinates:
    :return:
    """
    offset_x = 0.06
    offset_y = 328.61
    map_offset = [offset_x, offset_y, 0.0]

    new_coordinates = spawn_coordinates - map_offset
    new_coordinates[1] *= -1

    return new_coordinates


def main():
    config = load_config()

    map_name = config['map']['map_name']
    map_path = config['map']['map_path']
    sumo_coordinates = config['map']['sumo_coordinates']
    ped_spawner = config['walker']['ped_spawner']
    ped_seed = config.get('walker').get('pedestrian_seed', 2000)
    spectator_focus = config.get('walker').get('spectator_focus')
    spectator_loc = config['map'].get('spectator_location')
    spectator_rot = config['map'].get('spectator_rotation')
    ai_controller = config['walker'].get('ai_controller', False)

    walker_dict = {}
    walkers_list = []
    all_id = []
    all_actors = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    try:
        world = client.get_world()

        carla_map = world.get_map()

        if carla_map.name != map_path + map_name:
            world = client.load_world(map_name)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        settings.deterministic_ragdolls = True
        world.apply_settings(settings)

        # set spectator

        if spectator_loc is not None and spectator_rot is not None:
            spectator = world.get_spectator()
            spectator_transform = carla.Transform()
            spectator_transform.location = carla.Location(spectator_loc[0], spectator_loc[1], spectator_loc[2])
            spectator_transform.rotation = carla.Rotation(spectator_rot[0], spectator_rot[1], spectator_rot[2])
            spectator.set_transform(spectator_transform)

        blueprint_library = world.get_blueprint_library()
        walker_blueprints = blueprint_library.filter('walker.pedestrian.*')

        # 1. get all spawn points from config
        spawn_points = []
        destinations = []
        role_names = []
        ped_speeds = {}
        ped_index = 0

        for pedestrian in ped_spawner:

            role_name = 'ped_' + str(ped_index)
            ped_index += 1
            role_names.append(role_name)

            speed = pedestrian['speed']
            ped_speeds[role_name] = speed

            spawn_location = np.array(pedestrian['spawn_location'])
            destination = np.array(pedestrian['destination'])

            spawn_point = carla.Transform()

            if sumo_coordinates:
                spawn_location = convert_coordinates(spawn_location)
                destination = convert_coordinates(destination)

            spawn_point.location = carla.Location(spawn_location[0], spawn_location[1], spawn_location[2])

            direction = destination - spawn_location
            axis = np.array([1.0, 0.0, 0.0])

            # get vector angles with arctan2(y, x)
            angle1 = np.arctan2(direction[1], direction[0])
            angle2 = np.arctan2(axis[1], axis[0])

            # compute angle diffs
            rotation = angle1 - angle2

            # normalize angles
            if rotation > np.pi:
                rotation -= 2 * np.pi
            elif rotation < - np.pi:
                rotation -= 2 * np.pi

            spawn_point.rotation = carla.Rotation(0, np.degrees(rotation), 0)

            destination_loc = carla.Location(destination[0], destination[1], destination[2])

            spawn_points.append(spawn_point)
            destinations.append(destination_loc)

            if spectator_focus == role_name:
                spectator = world.get_spectator()
                spectator_transform = carla.Transform()
                spectator_transform.location = spawn_point.transform(carla.Vector3D(-2.0, 0.0, 2.0))
                spectator_transform.rotation = spawn_point.rotation
                spectator.set_transform(spectator_transform)

        # 2. we spawn the walker object
        batch = []


        for i in range(len(spawn_points)):

            # set pedestrian seed
            world.set_pedestrians_seed(ped_seed)
            random.seed(ped_seed)
            walker_bp = random.choice(walker_blueprints)

            ped_seed += 1

            if walker_bp.has_attribute('role_name'):
                walker_bp.set_attribute('role_name', role_names[i])

            batch.append(SpawnActor(walker_bp, spawn_points[i]))

        results = client.apply_batch_sync(batch, True)

        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walker_id = results[i].actor_id
                actor = world.get_actor(walker_id)
                role_name = actor.attributes.get('role_name')
                walker_dict[role_name] = walker_id
                walkers_list.append({"id": walker_id})

        if ai_controller:
            world.set_pedestrians_cross_factor(0.0)
            batch = []
            walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
            for i in range(len(walkers_list)):
                batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
            results = client.apply_batch_sync(batch, True)
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    walkers_list[i]["con"] = results[i].actor_id
            # 4. we put together the walkers and controllers id to get the objects from their id
            for i in range(len(walkers_list)):
                all_id.append(walkers_list[i]["con"])
                all_id.append(walkers_list[i]["id"])
            all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        # world.wait_for_tick()
        world.tick()

        print('spawned %d walkers, press Ctrl+C to exit.' % (len(walker_dict)))
        print()

        if ai_controller:
            for i in range(0, len(all_id), 2):
                # start walker
                all_actors[i].start()
                # set walk to destination
                all_actors[i].go_to_location(destinations[int(i / 2)])
                # max speed
                all_actors[i].set_max_speed(1.0)
        else:
            for role_name, walker_id in walker_dict.items():
                walker = world.get_actor(walker_id)
                direction = walker.get_transform().get_forward_vector()
                speed = ped_speeds[role_name]
                walker_control = carla.WalkerControl(direction, speed, False)
                walker.apply_control(walker_control)

        while True:
            # world.wait_for_tick()

            start = time.time()

            world.tick()

            if ai_controller:
                check_for_arrived_walkers(walkers_list, destinations, world)

            end = time.time()
            elapsed = end - start
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)

    finally:
        # Configuring carla_sim simulation in async mode.
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walker_dict))
        client.apply_batch([carla.command.DestroyActor(x) for x in walker_dict.values()])
        client.apply_batch([carla.command.DestroyActor(x) for x in all_actors])

        time.sleep(0.5)


def check_for_arrived_walkers(walker_list, destinations, world):
    tmp_walker_list = copy.deepcopy(walker_list)
    for i in range(len(tmp_walker_list)):
        walker_id = tmp_walker_list[i]['id']
        controller_id = tmp_walker_list[i]['con']
        destination = destinations[i]
        walker = world.get_actor(walker_id)
        controller = world.get_actor(controller_id)

        location = walker.get_location()
        location_np = np.array([location.x, location.y])
        destination_np = np.array([destination.x, destination.y])

        diff = destination_np - location_np

        distance = np.linalg.norm(diff, axis=-1)

        if distance < 2:
            controller.stop()
            controller.destroy()
            walker.destroy()
            walker_list.pop(i)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
