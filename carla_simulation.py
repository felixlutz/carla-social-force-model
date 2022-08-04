import logging

import carla


class CarlaSimulation:

    def __init__(self, host, port, step_length, config):

        self.config = config
        self.map_name = self.config['map']['map_name']
        self.map_path = self.config['map']['map_path']

        # connect to CARLA server
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Configuring CARLA simulation in sync mode.
        self.step_length = step_length
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.step_length
        self.world.apply_settings(settings)

        # load configured map
        self.carla_map = self.world.get_map()
        if self.carla_map.name != self.map_path + self.map_name:
            self.world = self.client.load_world(self.map_name)

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

        batch = [
            carla.command.SpawnActor(blueprint, transform)
        ]
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

    def close(self):
        pass
