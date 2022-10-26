import random
import time

import carla

# spawn_points = [4, 5, 11, 12, 14, 13, 100, 120]
spawn_points = [89, 135, 126, 130, 138, 110, 111, 115, 103, 104, 95]
seed = 100

def main():
    step_length = 0.05

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.deterministic_ragdolls = True
        settings.fixed_delta_seconds = step_length
        settings.substepping = True
        settings.max_substep_delta_time = 0.005
        settings.max_substeps = int(step_length/settings.max_substep_delta_time)

        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library().filter('vehicle')
        cars = [x for x in blueprint_library if int(x.get_attribute('number_of_wheels')) == 4]


        possible_spawn_points = world.get_map().get_spawn_points()
        transforms = [possible_spawn_points[p] for p in spawn_points]

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        # traffic_manager.set_hybrid_physics_mode(True)

        batch = []
        vehicle_list = []
        for n, t in enumerate(transforms):
            random.seed(seed + n)
            bp = random.choice(cars)
            batch.append(SpawnActor(bp, t).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, False):
            if response.error:
                print(response.error)
            else:
                vehicle = world.get_actor(response.actor_id)
                print('created %s' % vehicle.type_id)
                vehicle_list.append(vehicle)
                traffic_manager.vehicle_percentage_speed_difference(vehicle, 85)
                traffic_manager.ignore_walkers_percentage(vehicle, 100)
                traffic_manager.ignore_lights_percentage(vehicle, 100)

        while True:
            start = time.time()

            # world.wait_for_tick()
            world.tick()

            throttle = vehicle.get_control().throttle
            brake = vehicle.get_control().brake
            print(f'Throttle: {throttle}    Brake: {brake}')

            end = time.time()
            elapsed = end - start
            # print(f'Elapsed time: {elapsed}')
            if elapsed < step_length:
                time.sleep(step_length - elapsed)

    finally:

        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])
        print('done.')
        world.apply_settings(original_settings)


def run_vehicles(client, world):
    blueprint_library = world.get_blueprint_library().filter('vehicle')
    cars = [x for x in blueprint_library if int(x.get_attribute('number_of_wheels')) == 4]

    possible_spawn_points = world.get_map().get_spawn_points()
    transforms = [possible_spawn_points[p] for p in spawn_points]

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    # traffic_manager.set_hybrid_physics_mode(True)

    batch = []
    vehicle_list = []
    for n, t in enumerate(transforms):
        random.seed(seed + n)
        bp = random.choice(cars)
        batch.append(SpawnActor(bp, t).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    for response in client.apply_batch_sync(batch, False):
        if response.error:
            print(response.error)
        else:
            vehicle = world.get_actor(response.actor_id)
            print('created %s' % vehicle.type_id)
            vehicle_list.append(vehicle)
            traffic_manager.vehicle_percentage_speed_difference(vehicle, 85)
            traffic_manager.ignore_walkers_percentage(vehicle, 100)
            traffic_manager.ignore_lights_percentage(vehicle, 100)

    return vehicle_list


if __name__ == '__main__':
    main()
