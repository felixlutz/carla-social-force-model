import carla
import numpy as np


def extract_sidewalk(carla_map, scenario_config):
    """
    Extracts sidewalk borders form Carla map (based on the underlying OpenDRIVE map) as a list of points.
    :param carla_map:
    :param scenario_config:
    :return: borders (list of numpy arrays), carla_borders (list of Carla Vector3d)
    """

    # distance between extracted border points
    resolution = scenario_config.get('obstacles', {}).get('resolution', 0.1)

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
            carla_borders.append(border_right)

    # convert carla Vector3d to numpy arrays
    borders = [np.array([[vec.x, vec.y] for vec in border]) for border in carla_borders]

    return borders, carla_borders


def lateral_shift(transform, shift):
    """Makes a lateral shift of the forward vector of a transform"""
    transform.rotation.yaw += 90
    transform.location.z = 0.5
    return transform.location + shift * transform.get_forward_vector()
