import glob
import hashlib
import itertools
import logging
import os
import time

import carla
import numpy as np


def extract_sidewalk(carla_map, scenario_config):
    """
    Extracts sidewalk borders form Carla map (based on the underlying OpenDRIVE map) as a list of points.
    A cache system is used, so if the OpenDrive content of a Carla town and the obstacle resolution has not changed,
    it will read and use the stored sidewalk borders that were extracted in a previous simulation run.
    :param carla_map:
    :param scenario_config:
    :return: numpy_borders (list of numpy arrays), carla_borders (list of Carla Vector3d)
    """

    logging.info('Start extracting sidewalks.')
    start = time.time()
    # distance between extracted border points
    resolution = scenario_config.get('obstacles', {}).get('resolution', 0.1)

    # Load OpenDrive content
    opendrive_content = carla_map.to_opendrive()

    # Get hash based on content
    hash_func = hashlib.sha1()
    hash_func.update(opendrive_content.encode('UTF-8'))
    opendrive_hash = str(hash_func.hexdigest())

    # Build path for saving or loading the cached rendered map
    filename = carla_map.name.split('/')[-1] + '_' + str(resolution) + '_' + opendrive_hash + '.npz'
    dirname = os.path.join('cache', 'sidewalk_borders')
    full_path = str(os.path.join(dirname, filename))

    if os.path.isfile(full_path):
        # Load sidewalk borders from cache file
        logging.info('Using cached sidewalk borders.')
        loaded_file = np.load(full_path, allow_pickle=True)
        numpy_borders = loaded_file['borders']
        section_info = loaded_file['section_info']
        carla_borders = [[carla.Vector3D(p[0], p[1], 0.5) for p in border] for border in numpy_borders]

    else:
        carla_borders, section_info = extract_sidewalk_borders(carla_map, resolution)

        # convert carla Vector3d to numpy arrays
        numpy_borders = [np.array([[vec.x, vec.y] for vec in border]) for border in carla_borders]

        # If folders path does not exist, create it
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # Remove files if selected town had a previous version saved
        list_filenames = glob.glob(os.path.join(dirname, carla_map.name.split('/')[-1]) + "*")
        for town_filename in list_filenames:
            os.remove(town_filename)

        # Save sidewalk borders and section info for next executions of same map
        np.savez(full_path, borders=numpy_borders, section_info=section_info)

    end = time.time()
    logging.info('Finished extracting sidewalks. Time: ' + str(end - start))

    return numpy_borders, section_info, carla_borders


def extract_sidewalk_borders(carla_map, resolution):
    # get topology (minimal graph of OpenDRIVE map), which consists of a list of tuples of
    # waypoints (start of road, end of road)
    carla_topology = carla_map.get_topology()

    # use only the start waypoints
    topology = [x[0] for x in carla_topology]

    # the sidewalks in junctions must be extracted separately, because they are not always attached to a lane of
    # type driving

    # get all junctions and filter out duplicates
    junctions = [w.get_junction() for w in topology if w.is_junction]
    junction_ids = set()
    filtered_junctions = []
    for j in junctions:
        if j.id not in junction_ids:
            filtered_junctions.append(j)
            junction_ids.add(j.id)

    # get all start waypoints of type sidewalk that are located in a junction
    junction_waypoints = []
    for junction in filtered_junctions:
        waypoint_tuples = junction.get_waypoints(carla.LaneType.Sidewalk)
        start_waypoints = [x[0] for x in waypoint_tuples]
        junction_waypoints.extend(start_waypoints)

    # filter out all duplicates and all waypoints that are part of a junction to avoid duplicates when merging with
    # separately extracted junction waypoints
    topology = [w for w in topology if w.is_junction is False]
    waypoint_ids = set()
    filtered_waypoints = []
    for w in topology:
        if w.id not in waypoint_ids:
            filtered_waypoints.append(w)
            waypoint_ids.add(w.id)

    # merge start waypoints inside and outside of junctions
    filtered_waypoints.extend(junction_waypoints)

    carla_borders = []
    section_info = []
    for waypoint in filtered_waypoints:
        waypoints = [waypoint]

        # Generate waypoints of a road id. Stop when road id differs
        nxt = waypoint.next(resolution)
        if len(nxt) > 0:
            nxt = nxt[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(resolution)
                if len(nxt) > 0:
                    nxt = nxt[0]
                else:
                    break

        middle_wp = waypoints[len(waypoints) // 2]
        middle_loc = np.array([middle_wp.transform.location.x, middle_wp.transform.location.y])
        section_length = len(waypoints) * resolution

        sidewalk_waypoints = []
        for w in waypoints:
            if w.lane_type == carla.LaneType.Sidewalk:
                sidewalk_waypoints.append(w)

            # Check for sidewalk lane type until there are no waypoints by going left
            l = w.get_left_lane()
            while l and l.lane_type != carla.LaneType.Driving:

                if l.lane_type == carla.LaneType.Sidewalk:
                    sidewalk_waypoints.append(l)

                l = l.get_left_lane()

            # Check for sidewalk lane type until there are no waypoints by going right
            r = w.get_right_lane()
            while r and r.lane_type != carla.LaneType.Driving:

                if r.lane_type == carla.LaneType.Sidewalk:
                    sidewalk_waypoints.append(r)

                r = r.get_right_lane()

        if sidewalk_waypoints:
            border_left = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in sidewalk_waypoints]
            border_right = [lateral_shift(w.transform, w.lane_width * 0.5) for w in sidewalk_waypoints]

            carla_borders.append(border_left)
            section_info.append([middle_loc, section_length])

            carla_borders.append(border_right)
            section_info.append([middle_loc, section_length])

    return carla_borders, section_info


def lateral_shift(transform, shift):
    """Makes a lateral shift of the forward vector of a transform"""
    transform.rotation.yaw += 90
    transform.location.z = 0.5
    return transform.location + shift * transform.get_forward_vector()


def extract_obstacles(carla_world, scenario_config):
    """
    Extracts obstacles from Carla world map and generates border points around its bounding box.
    Depending on the configuration, the border points are either in the shap of a rectangle or of an ellipse.
    """

    resolution = scenario_config.get('obstacles', {}).get('resolution', 0.1)
    ellipse_shape = scenario_config.get('obstacles', {}).get('ellipse_shape', True)
    max_obstacle_z_pos = scenario_config.get('obstacles', {}).get('max_obstacle_z_pos', 0.3)

    # Get objects from Carla word map
    env_objects = carla_world.get_environment_objects(carla.CityObjectLabel.Static)
    env_objects.extend(carla_world.get_environment_objects(carla.CityObjectLabel.Dynamic))
    env_objects.extend(carla_world.get_environment_objects(carla.CityObjectLabel.Poles))
    env_objects.extend(carla_world.get_environment_objects(carla.CityObjectLabel.Walls))
    env_objects.extend(carla_world.get_environment_objects(carla.CityObjectLabel.Vehicles))

    carla_obstacle_borders = []
    obstacle_positions = []

    for o in env_objects:
        bb = o.bounding_box
        vertices = bb.get_local_vertices()[::2]

        # filter out obstacles that are above a defined z-position (e.g. traffic light hanging above a road)
        if vertices[0].z > max_obstacle_z_pos:
            continue

        border_points = []

        if ellipse_shape:
            ellipse = []

            # In Carla the object location isn't always the same as the center of the objects bounding box.
            # (e.g. the location of a streetlight is where its pole is placed on the ground, but its bounding box center
            # is somewhere else because the streetlight is bent towards the road)
            # Here it is decided which of these two points is taken as the center for generating the border ellipse
            tolerance = (bb.location - o.transform.location) * 0.1
            object_loc = o.transform.location + tolerance
            rot = carla.Rotation(-bb.rotation.pitch, -bb.rotation.yaw, -bb.rotation.roll)
            if bb_contains(bb, object_loc, carla.Transform(rotation=rot)) and o.type is not carla.CityObjectLabel.Walls:
                transform = o.transform
            else:
                loc = carla.Location(bb.location.x, bb.location.y, vertices[0].z)
                transform = carla.Transform(loc, bb.rotation)

            center = np.array([transform.location.x, transform.location.y])

            # For pole objects the longer extent of the bounding box is ignored
            if o.type is carla.CityObjectLabel.Poles:
                extent = min([bb.extent.x, bb.extent.y])
                extent_x = extent
                extent_y = extent
            else:
                extent_x = bb.extent.x
                extent_y = bb.extent.y

            # Generate border points in shape of ellipse
            circumference = 2 * extent_x + 2 * extent_y
            samples = max([6, int(circumference / resolution)])
            for theta in (np.pi * 2 * i / samples for i in range(samples)):
                point = [extent_x * np.cos(theta) * np.sqrt(2), extent_y * np.sin(theta) * np.sqrt(2), 0.0]
                point_loc = transform.transform(carla.Location(point[0], point[1], point[2]))

                ellipse.append(point_loc)

            border_points.extend(ellipse)

        else:
            borders = []
            border_lengths = []

            # Generate border points based on the bounding box.
            # Because the vertices from the bounding box are not sorted, every vertex is connected with every other
            # vertex and only the shortest 4 connections (= borders) are used.
            combinations = itertools.combinations(vertices, 2)
            center = np.array([bb.location.x, bb.location.y])
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

        carla_obstacle_borders.append(border_points)
        obstacle_positions.append(center)

    numpy_obstacle_borders = [np.array([[vec.x, vec.y] for vec in border]) for border in carla_obstacle_borders]

    return obstacle_positions, numpy_obstacle_borders, carla_obstacle_borders


def bb_contains(bounding_box, location, transform):
    """Check if a location is within a bounding box"""

    diff = bounding_box.location - location
    diff = transform.transform(diff)

    x_cond = abs(diff.x) < bounding_box.extent.x
    y_cond = abs(diff.y) < bounding_box.extent.y
    z_cond = abs(diff.z) < bounding_box.extent.z

    return x_cond and y_cond and z_cond
