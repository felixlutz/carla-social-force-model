import numpy as np
from shapely.geometry import LineString, Point

from stateutils import normalize


def check_traffic(ped, vehicles, vehicle_velocities, vehicle_extents):
    """
    Check if the given pedestrian can cross the road safely by checking if any vehicles are crossing the trajectory
    of the pedestrian within the time period needed by the pedestrian to cross the road (+ individual safety margin).
    :param ped: pedestrian state with location, next waypoint and crossing speed
    :param vehicles: list of all vehicles in the simulation (dynamic obstacle tuples with position + border)
    :param vehicle_velocities: list of velocities of all vehicles
    :param vehicle_extents: list of extents (relative x-, y-extent from vehicle center) of all vehicles
    :return: True if pedestrian can cross road safely and False if not
    """
    safe_to_cross = True

    ped_loc = ped['loc'][:2]
    ped_goal = ped['next_waypoint'][:2]
    ped_speed = ped['mode'].crossing_speed
    safety_margin = ped['mode'].crossing_safety_margin

    # if safety margin is negative, the pedestrian crosses without checking traffic
    if safety_margin >= 0:

        # calculate time needed by pedestrian to cross the road
        distance = np.linalg.norm(ped_goal - ped_loc)
        time_ped = distance / ped_speed

        ped_trajectory = LineString([ped_loc, ped_goal])

        # calculate position of vehicle fronts
        vehicle_locs, _ = zip(*vehicles)
        vehicle_directions, _ = normalize(vehicle_velocities)
        vehicle_fronts = vehicle_locs + vehicle_directions * vehicle_extents[:][0]

        for veh_front, veh_vel, veh_extent in zip(vehicle_fronts, vehicle_velocities, vehicle_extents):
            # calculate linear vehicle trajectory for the time period that the pedestrian needs to cross the road
            veh_goal = veh_front + veh_vel * time_ped
            veh_trajectory = LineString([veh_front, veh_goal])

            # check if trajectory of the pedestrian intersects with the vehicle trajectory
            intersection_point = ped_trajectory.intersection(veh_trajectory)

            if not intersection_point.is_empty:
                veh_speed = np.linalg.norm(veh_vel)
                if veh_speed != 0:
                    # calculate time to intersection point (+ safety margin)
                    tti_ped = intersection_point.distance(Point(ped_loc)) / ped_speed + safety_margin
                    tti_veh = intersection_point.distance(Point(veh_front)) / veh_speed

                    # if the pedestrian can't pass the intersection point before the vehicle arrives it has to wait
                    if tti_ped > tti_veh:
                        safe_to_cross = False

    return safe_to_cross

