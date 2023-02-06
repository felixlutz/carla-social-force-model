import argparse
import time

import carla
import numpy as np
import tomli

from path_planner import PedPathPlanner, GraphType, EdgeType

red = carla.Color(255, 0, 0)
green = carla.Color(0, 255, 0)
blue = carla.Color(47, 210, 231)
cyan = carla.Color(0, 255, 255)
yellow = carla.Color(255, 255, 0)
orange = carla.Color(255, 162, 0)
white = carla.Color(255, 255, 255)
black = carla.Color(0, 0, 0)


def draw_locations(debug, locations, z=0.5, lt=-1, color=red):
    for loc in locations:
        l = loc + carla.Location(z=z)
        debug.draw_point(l, size=0.15, life_time=lt, color=color)


def draw_loc_connection(debug, loc0, loc1, color=carla.Color(255, 0, 0), lt=5):
    debug.draw_line(
        loc0 + carla.Location(z=0.5),
        loc1 + carla.Location(z=0.5),
        thickness=0.05, color=color, life_time=lt, persistent_lines=False)
    debug.draw_point(loc0 + carla.Location(z=0.5), 0.1, color, lt, False)
    debug.draw_point(loc1 + carla.Location(z=0.5), 0.1, color, lt, False)


def draw_waypoints(debug, waypoints, z=0.5, lt=-1, col=red, arrow=False):
    for wpt in waypoints:
        wpt_t = wpt.transform
        begin = wpt_t.location + carla.Location(z=z)

        if arrow:
            angle = np.radians(wpt_t.rotation.yaw)
            end = begin + carla.Location(x=np.cos(angle), y=np.sin(angle))
            debug.draw_arrow(begin, end, arrow_size=0.3, color=col, life_time=lt)
        else:
            debug.draw_point(begin, size=0.1, color=col, life_time=lt)


def draw_transform(debug, trans, col=carla.Color(255, 0, 0), lt=-1):
    debug.draw_arrow(
        trans.location, trans.location + trans.get_forward_vector(),
        thickness=0.05, arrow_size=0.1, color=col, life_time=lt)


def draw_waypoint_union(debug, w0, w1, color=carla.Color(255, 0, 0), lt=5):
    debug.draw_line(
        w0.transform.location + carla.Location(z=0.5),
        w1.transform.location + carla.Location(z=0.5),
        thickness=0.05, color=color, life_time=lt, persistent_lines=False)
    debug.draw_point(w0.transform.location + carla.Location(z=0.5), 0.1, color, lt, False)
    debug.draw_point(w1.transform.location + carla.Location(z=0.5), 0.1, color, lt, False)


def load_config(path):
    with open(path, mode='rb') as fp:
        config = tomli.load(fp)
    return config


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--scenario-config',
        default='config/scenarios/routing2_scenario_config.toml',
        type=str,
        help='scenario configuration file')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-t', '--tick-time',
        metavar='T',
        default=10,
        type=float,
        help='Tick time between updates (forward velocity) (default: 0.2)')
    args = argparser.parse_args()

    config = load_config(args.scenario_config)

    map_name = config['map']['map_name']
    map_path = config['map']['map_path']
    ped_spawner = config['walker']['ped_spawner']
    waypoint_distance = config.get('walker', {}).get('waypoint_distance', 5)
    jaywalking_weight = config.get('walker', {}).get('jaywalking_weight', 2)

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        world = client.get_world()
        carla_map = world.get_map()
        if carla_map.name != map_path + map_name:
            world = client.load_world(map_name)
            # update map variable and start time after loading new map
            carla_map = world.get_map()

        debug = world.debug

        path_planner = PedPathPlanner(carla_map, waypoint_distance, jaywalking_weight)

        graph_type = GraphType.JAYWALKING
        graph = path_planner.graph_dict[graph_type]

        routes = []
        for spawner in ped_spawner:
            start = np.array(spawner['spawn_location'])
            destination = np.array(spawner['destination'])
            generate_route = spawner.get('generate_route')

            if generate_route:
                r = path_planner.generate_route(start, destination, GraphType[generate_route], with_origin=True,
                                                carla_loc=True)
                routes.append(r)


        # main loop
        while True:

            for n1, n2, data in graph.edges.data():
                # draw_waypoint_union(debug, data['entry_waypoint'], data['exit_waypoint'], cyan, lt=args.tick_time)

                if data['type'] == EdgeType.JAYWALKING_JUNCTION:
                    draw_waypoint_union(debug, data['entry_waypoint'], data['exit_waypoint'], yellow, lt=args.tick_time)
                elif data['type'] == EdgeType.SIDEWALK_TO_ROAD:
                    draw_waypoint_union(debug, data['entry_waypoint'], data['exit_waypoint'], green, lt=args.tick_time)
                elif data['type'] == EdgeType.JAYWALKING:
                    draw_waypoint_union(debug, data['entry_waypoint'], data['exit_waypoint'], red, lt=args.tick_time)
                elif data['type'] in [EdgeType.SIDEWALK, EdgeType.CROSSWALK]:
                    draw_waypoint_union(debug, data['entry_waypoint'], data['exit_waypoint'], cyan, lt=args.tick_time)

            # for route in routes:
            #     for i in range(len(route)-1):
            #         crossing_road = route[i+1][1]
            #         color = red
            #         # if crossing_road:
            #         #     color = green
            #         # else:
            #         #     color = red
            #         draw_loc_connection(debug, route[i][0], route[i+1][0], color=color, lt=args.tick_time)
            #
            #     draw_locations(debug, [route[0][0]], color=green, lt=args.tick_time)
            #     draw_locations(debug, [route[len(route) - 1][0]], color=black, lt=args.tick_time)

            time.sleep(args.tick_time)

    finally:
        pass


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nExit by user.')
    finally:
        print('\nExit.')
