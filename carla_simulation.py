import logging
import math

import carla


class CarlaSimulation:

    def __init__(self, args, config):

        self.config = config
        self.map_config = self.config['map']
        self.map_name = self.map_config['map_name']
        self.map_path = self.map_config['map_path']
        self.unload_props = self.map_config.get('unload_props', False)
        self.draw_obstacles = self.map_config.get('draw_obstacles', False)

        # connect to CARLA server
        self.client = carla.Client(args.carla_host, args.carla_port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.start_time = self.world.get_snapshot().timestamp.elapsed_seconds

        # load configured map
        self.carla_map = self.world.get_map()
        if self.carla_map.name != self.map_path + self.map_name:
            self.world = self.client.load_world(self.map_name)
            # update map variable and start time after loading new map
            self.carla_map = self.world.get_map()
            self.start_time = self.world.get_snapshot().timestamp.elapsed_seconds

        # unload props
        if self.unload_props:
            self.world.unload_map_layer(carla.MapLayer.Props)
            self.world.unload_map_layer(carla.MapLayer.StreetLights)
            self.world.unload_map_layer(carla.MapLayer.Walls)
            self.world.unload_map_layer(carla.MapLayer.Foliage)

        # Configuring CARLA simulation in sync mode.
        self.original_settings = self.world.get_settings()
        self.step_length = args.step_length
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.deterministic_ragdolls = True
        settings.fixed_delta_seconds = self.step_length
        settings.substepping = True
        settings.max_substep_delta_time = args.sub_step_length
        settings.max_substeps = math.ceil(self.step_length / args.sub_step_length)
        self.world.apply_settings(settings)

        # set spectator
        spectator_loc = self.map_config.get('spectator_location')
        spectator_rot = self.map_config.get('spectator_rotation')

        if spectator_loc is not None and spectator_rot is not None:
            spectator = self.world.get_spectator()
            spectator_transform = carla.Transform()
            spectator_transform.location = carla.Location(spectator_loc[0], spectator_loc[1], spectator_loc[2])
            spectator_transform.rotation = carla.Rotation(spectator_rot[0], spectator_rot[1], spectator_rot[2])
            spectator.set_transform(spectator_transform)

        self.blueprint_library = self.world.get_blueprint_library()
        self.walker_blueprints = self.blueprint_library.filter('walker.pedestrian.*')

    def tick(self):
        """
        Tick to CARLA simulation.
        """
        self.world.tick()

    def get_actor(self, actor_id):
        """
        Accessor for CARLA actor.
        """
        return self.world.get_actor(actor_id)

    def spawn_actor(self, blueprint, transform):
        """
        Spawns a new actor.
        :param blueprint: blueprint of the actor to be spawned.
        :param transform: transform where the actor will be spawned.
        :return: actor id if the actor is successfully spawned. Otherwise, INVALID_ACTOR_ID.
        """
        transform = carla.Transform(transform.location, transform.rotation)
        batch = [carla.command.SpawnActor(blueprint, transform)]

        return self.spawn_actor_with_batch(batch)

    def spawn_actor_with_batch(self, batch):
        """
        Spawns a new actor with batch command.
        :param batch: batch of carla commands
        :return: actor id if the actor is successfully spawned. Otherwise, INVALID_ACTOR_ID.
        """
        response = self.client.apply_batch_sync(batch, False)[0]
        if response.error:
            logging.error('Spawn carla actor failed. %s', response.error)
            return -1

        return response.actor_id

    def destroy_actor(self, actor_id):
        """
        Destroys the given actor.
        """
        actor = self.world.get_actor(actor_id)
        if actor is not None:
            return actor.destroy()
        return False

    def set_ped_velocity(self, walker_id, direction, speed):
        walker = self.world.get_actor(walker_id)
        walker_control = carla.WalkerControl(direction, speed, False)
        walker.apply_control(walker_control)

    def get_ped_transform(self, walker_id):
        walker = self.world.get_actor(walker_id)
        transform = walker.get_transform()

        return transform

    def get_ped_radius(self, walker_id):
        walker = self.world.get_actor(walker_id)
        extent = walker.bounding_box.extent
        radius = max([extent.x, extent.y])
        return radius

    def get_sim_time(self):
        timestamp = self.world.get_snapshot().timestamp.elapsed_seconds
        sim_time = timestamp - self.start_time
        return sim_time

    def draw_lines(self, lines):
        for line in lines:
            self.world.debug.draw_line(line[0], line[1], color=carla.Color(0, 0, 0, 0), thickness=0.05, life_time=0)

    def draw_bounding_box(self, actor_id, step_length):
        actor = self.world.get_actor(actor_id)
        bb = carla.BoundingBox(actor.get_location(), actor.bounding_box.extent)
        self.world.debug.draw_box(bb, actor.get_transform().rotation, color=carla.Color(0, 0, 0, 0), thickness=0.01,
                                  life_time=step_length + 0.00000001)

    def draw_points(self, points, lt=0):
        for point in points:
            self.world.debug.draw_point(point, size=0.05, life_time=lt + 0.00000001)

    def close(self):
        # reset CARLA simulation settings
        self.world.apply_settings(self.original_settings)
