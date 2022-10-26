import argparse
import itertools
import time

import carla
import numpy as np
from path_planner import PedPathPlanner, GraphType, EdgeType

red = carla.Color(255, 0, 0)
green = carla.Color(0, 255, 0)
blue = carla.Color(47, 210, 231)
cyan = carla.Color(0, 255, 255)
yellow = carla.Color(255, 255, 0)
orange = carla.Color(255, 162, 0)
white = carla.Color(255, 255, 255)


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


def draw_bounding_box(debug, bb, color=red, life_time=-1):
    debug.draw_box(bb, bb.rotation, color=color, thickness=0.01, life_time=life_time)


def main():
    argparser = argparse.ArgumentParser()
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
        default=20,
        type=float,
        help='Tick time between updates (forward velocity) (default: 0.2)')
    args = argparser.parse_args()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        map_name = 'Town05_Opt'
        map_path = 'Carla/Maps/'
        ellipse_shape = True
        resolution = 0.2

        world = client.get_world()
        carla_map = world.get_map()
        if carla_map.name != map_path + map_name:
            world = client.load_world(map_name)
            # update map variable and start time after loading new map
            carla_map = world.get_map()

        debug = world.debug
        env_objects_full = []
        env_objects_full.extend(world.get_environment_objects(carla.CityObjectLabel.Static))
        env_objects_full.extend(world.get_environment_objects(carla.CityObjectLabel.Dynamic))
        env_objects_full.extend(world.get_environment_objects(carla.CityObjectLabel.Poles))
        env_objects_full.extend(world.get_environment_objects(carla.CityObjectLabel.Walls))
        env_objects_full.extend(world.get_environment_objects(carla.CityObjectLabel.Vehicles))

        # main loop
        while True:

            vehicles = world.get_actors().filter("*vehicle*")

            all_objects = env_objects_full[:]
            all_objects.extend(vehicles)

            for o in all_objects:
                bb = o.bounding_box
                draw_bounding_box(debug, bb, life_time=args.tick_time)

                border_points = []

                if ellipse_shape:
                    vertices = bb.get_local_vertices()[::2]
                    if vertices[0].z > 0.3:
                        continue

                    ellipse = []
                    if isinstance(o, carla.Actor):
                        transform = o.get_transform()
                    else:
                        tolerance = (bb.location - o.transform.location) * 0.1
                        test_point = o.transform.location + tolerance
                        rot = carla.Rotation(-bb.rotation.pitch, -bb.rotation.yaw, -bb.rotation.roll)
                        if bb_contains(bb, test_point, carla.Transform(rotation=rot))\
                                and o.type is not carla.CityObjectLabel.Walls:
                            transform = o.transform
                        else:
                            loc = carla.Location(bb.location.x, bb.location.y, vertices[0].z)
                            transform = carla.Transform(loc, bb.rotation)

                    if not isinstance(o, carla.Actor) and o.type == carla.CityObjectLabel.Poles:
                        extent = min([bb.extent.x, bb.extent.y])
                        extent_x = extent
                        extent_y = extent

                    else:
                        extent_x = bb.extent.x
                        extent_y = bb.extent.y

                    circumference = 2 * extent_x + 2 * extent_y
                    samples = max([6, int(circumference / resolution)])

                    for theta in (np.pi * 2 * i / samples for i in range(samples)):
                        point = [extent_x * np.cos(theta) * np.sqrt(2), extent_y * np.sin(theta) * np.sqrt(2), 0.0]

                        point_loc = transform.transform(carla.Location(point[0], point[1], point[2]))

                        ellipse.append(point_loc)

                    border_points.extend(ellipse)

                else:

                    vertices = bb.get_local_vertices()[::2]
                    vertices = [v for v in vertices if v.z < 0.3]

                    borders = []
                    border_lengths = []

                    combinations = itertools.combinations(vertices, 2)

                    for c in combinations:
                        start = c[0]
                        end = c[1]
                        start_point = np.array([start.x, start.y])
                        end_point = np.array([end.x, end.y])
                        length = np.linalg.norm(end_point - start_point)
                        border_lengths.append(length)
                        samples = max(2, int(length / resolution))

                        border_line = np.column_stack((np.linspace(start_point[0], end_point[0], samples),
                                                       np.linspace(start_point[1], end_point[1], samples)))
                        borders.append(border_line)
                    if len(vertices) == 4:
                        indices = np.argpartition(border_lengths, 4)[:4]
                        borders = [borders[i] for i in indices]
                    else:
                        continue

                    border_points.extend(
                        [carla.Location(p[0], p[1], vertices[0].z) for border_line in borders for p in border_line])

                # vertices.append(o.transform.location)
                draw_locations(debug, border_points, z=0, lt=args.tick_time)

            time.sleep(args.tick_time)

    finally:
        pass


def bb_contains(bb, loc, transform):
    diff = bb.location - loc

    diff = transform.transform(diff)

    x_cond = abs(diff.x) < bb.extent.x
    y_cond = abs(diff.y) < bb.extent.y
    z_cond = abs(diff.z) < bb.extent.z

    if x_cond and y_cond and z_cond:
        return True
    else:
        return False


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nExit by user.')
    finally:
        print('\nExit.')
