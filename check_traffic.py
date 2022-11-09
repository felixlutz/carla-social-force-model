import numpy as np
from shapely.geometry import LineString

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

    ped_loc = ped['loc'][:2]
    ped_goal = ped['next_waypoint'][:2]
    ped_speed = ped['mode'].crossing_speed
    safety_margin = ped['mode'].crossing_safety_margin

    # if safety margin is negative, the pedestrian crosses without checking traffic
    if safety_margin >= 0:

        # calculate time needed by pedestrian to cross the road (+ safety margin)
        distance = np.linalg.norm(ped_goal - ped_loc)
        time_ped = distance / ped_speed + safety_margin

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
            if ped_trajectory.intersects(veh_trajectory):
                return False

    return True

