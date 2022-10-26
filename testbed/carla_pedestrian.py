import logging
import random
import time

import carla
import numpy as np
from libcarla.command import SpawnActor

map_name = 'Town10HD_Opt'
map_path = 'Carla/Maps/'
random_spawn_points = True
spawn_coordinates = np.array([[122.11, 276.57], [71.40, 270.59]])
number_of_walkers = len(spawn_coordinates)


def convert_coordinates(spawn_coordinates):
    """
    Converts SUMO coordinates to Carla coordinated by applying a map offset and inverting the y-axis.
    :param spawn_coordinates:
    :return:
    """
    offset_x = 0.06
    offset_y = 328.61
    map_offset = [offset_x, offset_y]

    new_coordinates = spawn_coordinates - map_offset
    new_coordinates[:, 1] *= -1

    return new_coordinates


def main():
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

        world.unload_map_layer(carla.MapLayer.Buildings)

        blueprint_library = world.get_blueprint_library()
        walker_blueprints = blueprint_library.filter('walker.pedestrian.*')

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0  # how many pedestrians will run
        percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road

        # 1. take all the random locations to spawn
        spawn_points = []

        for i in range(number_of_walkers):
            spawn_point = carla.Transform()

            if random_spawn_points:
                loc = world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point.location = loc

            else:
                carla_coordinates = convert_coordinates(spawn_coordinates)
                x = carla_coordinates[i, 0]
                y = carla_coordinates[i, 1]
                spawn_point.location = carla.Location(x, y, 0.2)

            spawn_points.append(spawn_point)

        for spawn_point in spawn_points:
            walker = world.try_spawn_actor(random.choice(walker_blueprints), spawn_point)
            if walker is not None:
                walkers_list.append(walker)
            else:
                print('Spawn failed')

        spectator = world.get_spectator()
        spectator.set_transform(spawn_points[1])

        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(walker_blueprints)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2

        # 3. we spawn the walker controller
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
        world.wait_for_tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

        print('spawned %d walkers, press Ctrl+C to exit.' % (len(walkers_list)))
        print()

        while True:
            world.wait_for_tick()

    finally:
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_actors])

        time.sleep(0.5)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
