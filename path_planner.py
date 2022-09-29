import itertools
from enum import Enum

import carla
import networkx as nx
import numpy as np


class EdgeType(Enum):
    VOID = -1
    SIDEWALK = 1
    CROSSWALK = 2
    JAYWALKING = 3
    JAYWALKING_JUNCTION = 4


class GraphType(Enum):
    NO_JAYWALKING = 1
    JAYWALKING_AT_JUNCTION = 2
    JAYWALKING = 3


class PedPathPlanner:
    """
    This class generates a navigation graph for pedestrians based on the Carla map (with its underlying OpenDrive map).
    This graph can be used to determine the shortest path for a pedestrian fom A to B on the map.
    """

    def __init__(self, carla_map, waypoint_distance=20, jaywalking_weight_factor=2):
        self.waypoint_distance = waypoint_distance
        self.jaywalking_weight_factor = jaywalking_weight_factor
        self.carla_map = carla_map
        self.ped_topology = None
        self.graph = None
        self.graph_dict = None
        self.id_map = None
        self.road_id_to_edge = None

        # Build the graph
        self._build_topology()
        self._build_graph()
        self._extract_subgraphs()

    def generate_route(self, origin, destination, graph_type, with_origin=False, carla_loc=False):
        """
        Generate pedestrian route from origin to destination. Depending on the specified graph type the route may
        include jaywalking. The method returns a list of tuples where each tuple contains a waypoint in the specified
        format and a boolean that indicates if a road is being crossed when heading to the waypoint
        :param origin: carla.Location or numpy array of start position
        :param destination: carla.Location or numpy array of end position
        :param graph_type: graph type for routing:
                            GraphType.NO_JAYWALKING -> no jaywalking allowed, routing only over sidewalks and crosswalks
                            GraphType.JAYWALKING_AT_JUNCTION -> jaywalking only allowed at junctions
                            GraphTYpe.JAYWALKING -> jaywalking is allowed everywhere
        :param with_origin: boolean to determine weather the origin shall be included in the returned route
        :param carla_loc:   boolean to specify if the returned route shall consist of carla.Location objects or numpy
                            arrays
        :return: list of tuples (carla.Location, bool) or list of tuples (numpy array, bool)
        """

        # get graph based on graph type
        graph = self.graph_dict[graph_type]

        # convert origin and destination if necessary
        if not isinstance(origin, carla.Location):
            origin = carla.Location(origin[0], origin[1], origin[2])
        if not isinstance(destination, carla.Location):
            destination = carla.Location(destination[0], destination[1], destination[2])

        # find route
        route_node_ids = self._path_search(graph, origin, destination)

        route = []
        if with_origin:
            route.append((origin, False))

        crossing_road_successor = False

        # get waypoint locations from graph using the node ids
        for i in range(len(route_node_ids) - 1):

            # append boolean to waypoint to indicate if a road is being crossed in order to reach waypoint
            # (the successor of a waypoint, where crossing_road is set to true, is also set to true to guarantee that
            # the pedestrian is back on a sidewalk before resetting the crossing_road boolean to false again)
            crossing_road = False
            edge = graph.edges[(route_node_ids[i], route_node_ids[i + 1])]
            edge_type = edge['type']
            if edge_type in [EdgeType.CROSSWALK, EdgeType.JAYWALKING, EdgeType.JAYWALKING_JUNCTION]:
                crossing_road = True
                crossing_road_successor = True
            if not crossing_road and crossing_road_successor:
                crossing_road = True
                crossing_road_successor = False

            if i == 0:
                first_waypoint = graph.nodes[route_node_ids[i]]['waypoint']
                route.append((first_waypoint.transform.location, False))
            next_waypoint = graph.nodes[route_node_ids[i + 1]]['waypoint']

            route.append((next_waypoint.transform.location, crossing_road))

        route.append((destination, False))

        # convert route waypoints if specified
        if not carla_loc:
            route = [(np.array([loc.x, loc.y, loc.z]), c) for loc, c in route]

        return route

    def _path_search(self, graph, origin, destination):
        """
        This method finds the shortest path connecting origin and destination using A* search with distance heuristic.
        :param graph: networkx graph used for path search
        :param origin: carla.Location object of start position
        :param destination: carla.Location object of end position
        :return: path as list of node ids (as int) of the graph connecting origin and destination
        """
        start, end = self._find_closest_node_id(origin), self._find_closest_node_id(destination)

        route = nx.astar_path(graph, source=start, target=end, heuristic=self._distance_heuristic, weight='length')

        self._remove_unnecessary_start_end_nodes(route, origin, destination)

        return route

    def _find_closest_node_id(self, location):
        """
        This function finds the sidewalk segment that a given location is part of, returning the closest node in the
        routing graph.
        """
        waypoint = self.carla_map.get_waypoint(location, lane_type=carla.LaneType.Sidewalk)
        closest_node = None
        try:
            # get all edges of the sidewalk segment which the given location is part of
            edges = self.road_id_to_edge[waypoint.road_id][waypoint.section_id][waypoint.lane_id]
            min_distance = float('inf')

            for edge in edges:
                wp1 = self.graph.edges[edge]['entry_waypoint']
                wp2 = self.graph.edges[edge]['exit_waypoint']

                for i, wp in enumerate([wp1, wp2]):
                    distance = waypoint.transform.location.distance(wp.transform.location)
                    if distance < min_distance:
                        min_distance = distance
                        closest_node = edge[i]
        except KeyError:
            pass

        return closest_node

    def _distance_heuristic(self, n1, n2):
        """Distance heuristic calculator for path searching in graph"""

        wp_1 = self.graph.nodes[n1]['waypoint']
        wp_2 = self.graph.nodes[n2]['waypoint']
        distance = wp_1.transform.location.distance(wp_2.transform.location)

        return distance

    def _remove_unnecessary_start_end_nodes(self, route, origin_loc, destination_loc):
        """Remove unnecessary start and end nodes from route if they would cause a detour"""

        if len(route) > 1:
            first = self.graph.nodes[route[0]]['waypoint'].transform.location
            second = self.graph.nodes[route[1]]['waypoint'].transform.location
            distance_first_second = first.distance(second)
            distance_origin_second = origin_loc.distance(second)

            last = self.graph.nodes[route[-1]]['waypoint'].transform.location
            second_to_last = self.graph.nodes[route[-2]]['waypoint'].transform.location
            distance_last_second_to_last = last.distance(second_to_last)
            distance_destination_second_to_last = destination_loc.distance(second_to_last)

            if distance_first_second > distance_origin_second:
                del route[0]

            if distance_last_second_to_last > distance_destination_second_to_last:
                del route[-1]

    def _build_topology(self):
        """
        This method retrieves the carla topology from the server as a list of road segments as pairs of waypoint
        objects and processes into a pedestrian topology represented as a list of dictionary objects with the following
        attributes:

        - entry (carla.Waypoint): waypoint of entry point of edge
        - entry_xyz (tuple): (x,y,z) of entry point of edge
        - exit (carla.Waypoint): waypoint of exit point of edge
        - exit_xyz (tuple): (x,y,z) of exit point of edge
        - length (float): length of the edge in [m]
        """

        carla_topology = self.carla_map.get_topology()
        self.ped_topology = []

        # construct detailed pedestrian topology from carla road topology
        for segment in carla_topology:
            sidewalk_waypoints = self._generate_sidewalk_waypoints(segment)
            for side in sidewalk_waypoints:
                if side:
                    for lane in side.values():
                        sidewalk_edges = self._generate_edge_dicts(lane, EdgeType.SIDEWALK)
                        self.ped_topology.extend(sidewalk_edges)

        # generate junction edges
        junction_edges = self._generate_junction_edges(carla_topology)
        self.ped_topology.extend(junction_edges)

        # generate crosswalk edges and connection edges to other edges of the pedestrian topology
        crosswalk_edges = self._generate_crosswalk_edges()
        connection_edges = self._generate_edges_to_crosswalks(crosswalk_edges, connection_radius=10)
        self.ped_topology.extend(connection_edges)
        self.ped_topology.extend(crosswalk_edges)

    def _generate_sidewalk_waypoints(self, segment):
        """Generate and return sidewalk waypoints along a Carla topology segment"""

        sidewalk_waypoints_left = {}
        sidewalk_waypoints_right = {}

        wp_start = segment[0]
        segment_wps = [wp_start]
        if not wp_start.is_junction:
            segment_wps.extend(wp_start.next_until_lane_end(self.waypoint_distance))

        for w in segment_wps:
            # Check for sidewalk lane type until there are no waypoints by going left
            left_lane = w.get_left_lane()
            while left_lane and left_lane.lane_type != carla.LaneType.Driving:
                if left_lane.lane_type == carla.LaneType.Sidewalk:
                    if left_lane.lane_id not in sidewalk_waypoints_left:
                        sidewalk_waypoints_left[left_lane.lane_id] = []
                    sidewalk_waypoints_left[left_lane.lane_id].append(left_lane)
                left_lane = left_lane.get_left_lane()

            # Check for sidewalk lane type until there are no waypoints by going right
            right_lane = w.get_right_lane()
            while right_lane and right_lane.lane_type != carla.LaneType.Driving:
                if right_lane.lane_type == carla.LaneType.Sidewalk:
                    if right_lane.lane_id not in sidewalk_waypoints_right:
                        sidewalk_waypoints_right[right_lane.lane_id] = []
                    sidewalk_waypoints_right[right_lane.lane_id].append(right_lane)
                right_lane = right_lane.get_right_lane()

        return [sidewalk_waypoints_left, sidewalk_waypoints_right]

    def _generate_junction_edges(self, carla_topology):
        """Generates and returns all edges within junctions."""

        junctions = self._get_all_junctions(carla_topology)
        junction_edges = []
        junction_straight_edges = []

        # get all waypoints of type sidewalk that are located in a junction
        for junction in junctions:
            waypoint_tuples = junction.get_waypoints(carla.LaneType.Sidewalk)

            junction_corners = []
            for segment in waypoint_tuples:
                wp_start = segment[0]
                wp_end = segment[1]

                # differentiate between segments that are junction corners and junction straights (e.g. T-junction)
                # (corner segments have no neighboring lanes of type driving)
                is_corner = True

                # Check for driving lane type until there are no waypoints by going left
                left_lane = wp_start.get_left_lane()
                while left_lane and is_corner:
                    if left_lane.lane_type == carla.LaneType.Driving:
                        is_corner = False
                    left_lane = left_lane.get_left_lane()

                # Check for driving lane type until there are no waypoints by going right
                right_lane = wp_start.get_right_lane()
                while right_lane and is_corner:
                    if right_lane.lane_type == carla.LaneType.Driving:
                        is_corner = False
                    right_lane = right_lane.get_right_lane()

                # if segment is a corner, get middle waypoint of corner segment (= junction corner waypoint)
                if is_corner:
                    corner_wps = wp_start.next_until_lane_end(0.5)

                    middle_index = len(corner_wps) // 2
                    middle_wp = corner_wps[middle_index]

                    junction_corner_waypoints = [wp_start, middle_wp, wp_end]

                    edges = self._generate_edge_dicts(junction_corner_waypoints, EdgeType.SIDEWALK)
                    junction_edges.extend(edges)
                    junction_corners.append(middle_wp)

                # if segment is a straight, estimate junction corner waypoint with sidewalk width
                else:
                    wp_1 = wp_start.next(wp_start.lane_width)[0]
                    wp_2 = wp_end.previous(wp_start.lane_width)[0]

                    junction_straight_waypoints = [wp_start, wp_1, wp_2, wp_end]
                    edges = self._generate_edge_dicts(junction_straight_waypoints, EdgeType.SIDEWALK)
                    junction_straight_edges.extend(edges)
                    junction_corners.extend([wp_1, wp_2])

            # generate connection edges between junction corners for jaywalking
            corner_connection_edges = self._generate_junction_corner_connection_edges(junction_corners)
            junction_edges.extend(corner_connection_edges)

            # junction straight edges must be appended after the corner connection edges in order to have the correct
            # edge type (one of the corner connection edge is equivalent to the straight edge and the edge that gets
            # appended last overrides the edge type of the first one)
            junction_edges.extend(junction_straight_edges)

        return junction_edges

    def _get_all_junctions(self, carla_topology):
        """Returns all Carla junction objects from the given Carla topology"""

        # get all junctions and filter out duplicates
        junctions = [w[0].get_junction() for w in carla_topology if w[0].is_junction]
        junction_ids = set()
        filtered_junctions = []
        for j in junctions:
            if j.id not in junction_ids:
                filtered_junctions.append(j)
                junction_ids.add(j.id)

        return filtered_junctions

    def _generate_junction_corner_connection_edges(self, corners):
        """Generates and returns connection edges between junction corners for jaywalking"""

        corner_connection_edges = []
        edge_lengths = []
        combinations = itertools.combinations(corners, 2)

        # create edges between all possible combinations of corners
        for c in combinations:
            connection_edge = self._generate_edge_dicts(c, EdgeType.JAYWALKING_JUNCTION)
            length = connection_edge[0]['length']
            edge_lengths.append(length)
            corner_connection_edges.extend(connection_edge)

        # typically a junction has 4 corners which results in 6 edges
        # the diagonal edges are not wanted and are therefore filtered out by only picking the four shortest edges
        if len(corners) == 4:
            indices = np.argpartition(edge_lengths, 4)[:4]
            corner_connection_edges = [corner_connection_edges[i] for i in indices]

        return corner_connection_edges

    def _generate_crosswalk_edges(self):
        """Generates and returns all crosswalk edges"""

        # get all crosswalk corners as a list
        # every crosswalk is represented by 5 points (4 corners + repetition of first corner)
        # in rare cases a crosswalk consists of 7 points (4 corners + 2 middle point + repetition of first corner)
        crosswalk_corners = self.carla_map.get_crosswalks()
        filtered_crosswalk_corners = []
        crosswalk = []

        # filter out unnecessary points, so only 4 corners per crosswalk remain
        for point in crosswalk_corners:
            if point not in crosswalk:
                crosswalk.append(point)
            else:
                if len(crosswalk) == 4:
                    filtered_crosswalk_corners.extend(crosswalk)
                elif len(crosswalk) == 6:
                    del crosswalk[4]
                    del crosswalk[1]
                    filtered_crosswalk_corners.extend(crosswalk)
                crosswalk = []

        crosswalks_np = np.array([np.array([p.x, p.y, p.z]) for p in filtered_crosswalk_corners])
        sorted_crosswalks = np.reshape(crosswalks_np, (-1, 2, 2, 3))

        # extract edges from crosswalks
        crosswalk_edges = []
        for crosswalk in sorted_crosswalks:
            crosswalk_waypoints = []
            for side in crosswalk:
                middle = (side[0] + side[1]) / 2
                middle_loc = carla.Location(x=middle[0], y=middle[1], z=middle[2])
                middle_waypoint = self.carla_map.get_waypoint(middle_loc, lane_type=carla.LaneType.Shoulder)
                if middle_waypoint is not None:
                    crosswalk_waypoints.append(middle_waypoint)

            sub_segment = self._generate_edge_dicts(crosswalk_waypoints, EdgeType.CROSSWALK)
            crosswalk_edges.extend(sub_segment)

        return crosswalk_edges

    def _generate_edges_to_crosswalks(self, crosswalk_edges, connection_radius):
        """Generate and return edges that connect the crosswalk edges with the other edges of the ped topology"""

        topology_waypoints = self._get_all_waypoints_from_topology(self.ped_topology)
        connection_edges = []

        for crosswalk in crosswalk_edges:
            wp_1 = crosswalk['entry']
            wp_2 = crosswalk['exit']

            for wp in wp_1, wp_2:
                loc = wp.transform.location
                neighboring_waypoints = [w for w in topology_waypoints if w.road_id == wp.road_id
                                         and loc.distance(w.transform.location) < connection_radius]
                for n in neighboring_waypoints:
                    edge = self._generate_edge_dicts([wp, n], EdgeType.SIDEWALK)
                    connection_edges.extend(edge)

        return connection_edges

    def _generate_edge_dicts(self, waypoint_list, edge_type=EdgeType.SIDEWALK):
        """
        Generate and return edge dictionaries from a list of waypoints consecutive in preparation for generating
        routing graph edges. The dictionaries have the following attributes:

        - entry (carla.Waypoint): waypoint of entry point of edge
        - entry_xyz (tuple): (x,y,z) of entry point of edge
        - exit (carla.Waypoint): waypoint of exit point of edge
        - exit_xyz (tuple): (x,y,z) of exit point of edge
        - length (float): length of the edge in [m]
        """

        waypoint_locs = [w.transform.location for w in waypoint_list]
        # rounding off to avoid floating point imprecision
        waypoints_xyz = [tuple(np.round([loc.x, loc.y, loc.z], 0)) for loc in waypoint_locs]

        edges = []
        for i in range(len(waypoint_list) - 1):
            current_wp, next_wp = waypoint_list[i], waypoint_list[i + 1]
            current_xyz, next_xyz = waypoints_xyz[i], waypoints_xyz[i + 1]

            length = current_wp.transform.location.distance(next_wp.transform.location)

            edge_dict = {'entry': current_wp, 'exit': next_wp, 'entry_xyz': current_xyz, 'exit_xyz': next_xyz,
                         'length': length, 'edge_type': edge_type}
            edges.append(edge_dict)

        return edges

    def _build_graph(self):
        """
        This function builds a networkx graph representation of ped_topology, creating several class attributes:
        - graph (networkx.Graph): networkx graph representing the world map, with:
            Node properties:
                xyz: (x,y,z) position in world map
            Edge properties:
                entry_vector: unit vector along tangent at entry point
                exit_vector: unit vector along tangent at exit point
                net_vector: unit vector of the chord from entry to exit
                intersection: boolean indicating if the edge belongs to an  intersection
        - id_map (dictionary): mapping from (x,y,z) to node id
        - road_id_to_edge (dictionary): map from road id to edge in the graph
        """

        self.graph = nx.Graph()
        self.id_map = {}  # dict with structure {(x,y,z): id, ... }
        self.road_id_to_edge = {}  # dict with structure {road_id: {lane_id: edge, ... }, ... }

        for edge in self.ped_topology:
            entry_xyz, exit_xyz = edge['entry_xyz'], edge['exit_xyz']
            entry_wp, exit_wp = edge['entry'], edge['exit']
            length = edge['length']
            edge_type = edge['edge_type']

            # multiply length of jaywalking edges with weight factor to make routing over them more expensive
            if edge_type is EdgeType.JAYWALKING or edge_type is EdgeType.JAYWALKING_JUNCTION:
                length *= self.jaywalking_weight_factor

            intersection = entry_wp.is_junction
            road_id, section_id, lane_id = entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id

            for xyz, waypoint in zip([entry_xyz, exit_xyz], [entry_wp, exit_wp]):
                # Adding unique nodes and populating id_map
                if xyz not in self.id_map:
                    new_id = len(self.id_map)
                    self.id_map[xyz] = new_id
                    self.graph.add_node(new_id, xyz=xyz, waypoint=waypoint)
            n1 = self.id_map[entry_xyz]
            n2 = self.id_map[exit_xyz]
            if road_id not in self.road_id_to_edge:
                self.road_id_to_edge[road_id] = dict()
            if section_id not in self.road_id_to_edge[road_id]:
                self.road_id_to_edge[road_id][section_id] = dict()
            if lane_id not in self.road_id_to_edge[road_id][section_id]:
                self.road_id_to_edge[road_id][section_id][lane_id] = [(n1, n2)]
            else:
                self.road_id_to_edge[road_id][section_id][lane_id].append([n1, n2])

            # Adding edge with attributes
            self.graph.add_edge(
                n1, n2,
                length=length,
                entry_waypoint=entry_wp, exit_waypoint=exit_wp,
                intersection=intersection, type=edge_type)

        # generate and add jaywaling edges to graph
        self._generate_jaywalking_graph_edges()

    def _generate_jaywalking_graph_edges(self):
        """Generate jaywalking edges and add them directly to the routing graph"""

        topology_waypoints = self._get_all_waypoints_from_topology(self.ped_topology)
        topology_xyz = self._get_all_xyz_nodes_from_topology(self.ped_topology)

        # search for an opposing waypoint on the other side of the road for every waypoint in the pedestrian topology
        for wp, xyz in zip(topology_waypoints, topology_xyz):
            if wp.lane_type is not carla.LaneType.Sidewalk:
                continue

            opposite_waypoint = None
            lane_id_sign = np.sign(wp.lane_id)

            # Check for sidewalk lane type until there are no waypoints by going left
            left_lane = wp.get_left_lane()
            while left_lane and not opposite_waypoint:
                if left_lane.lane_type == carla.LaneType.Sidewalk:
                    opposite_waypoint = left_lane

                # the direction of the lanes change when crossing the mid_line (= sign change of lane_id) and therefore
                # left and right also changes
                if np.sign(left_lane.lane_id) == lane_id_sign:
                    left_lane = left_lane.get_left_lane()
                else:
                    left_lane = left_lane.get_right_lane()

            # Check for sidewalk lane type until there are no waypoints by going right
            r = wp.get_right_lane()
            while r and not opposite_waypoint:
                if r.lane_type == carla.LaneType.Sidewalk:
                    opposite_waypoint = r

                # the direction of the lanes change when crossing the mid_line (= sign change of lane_id) and therefore
                # left and right also changes
                if np.sign(r.lane_id) == lane_id_sign:
                    r = r.get_right_lane()
                else:
                    r = r.get_left_lane()

            # if opposite waypoint exists, add a jaywalking edge across the road to the graph
            if opposite_waypoint:
                current_id = self.id_map[xyz]
                opposite_id = self._find_closest_node_id(opposite_waypoint.transform.location)
                if opposite_id:
                    exit_w = self.graph.nodes[opposite_id]['waypoint']
                    distance = wp.transform.location.distance(exit_w.transform.location)

                    self.graph.add_edge(
                        current_id, opposite_id,
                        length=distance * self.jaywalking_weight_factor,
                        entry_waypoint=wp, exit_waypoint=exit_w,
                        intersection=wp.is_junction, type=EdgeType.JAYWALKING)

    def _extract_subgraphs(self):
        """Extract sub-graphs from the main graph to allow routing with and without jaywalking"""

        graph_jaywalking_at_junction = self._filter_graph_edges(self.graph, 'type', EdgeType.JAYWALKING)
        graph_no_jaywalking = self._filter_graph_edges(graph_jaywalking_at_junction, 'type',
                                                       EdgeType.JAYWALKING_JUNCTION)

        self.graph_dict = {GraphType.NO_JAYWALKING: graph_no_jaywalking,
                           GraphType.JAYWALKING_AT_JUNCTION: graph_jaywalking_at_junction,
                           GraphType.JAYWALKING: self.graph}

    def _filter_graph_edges(self, graph, edge_attribute, filter_val):
        """
        Creates a sub-graph by filtering out edges based on a specified attribute
        :param graph: navigation graph
        :param edge_attribute: edge attribute that is used for filtering
        :param filter_val: value of the edge attribute that shall be filtered out
        :return: sub-graph without the edges that have the specified attribute
        """

        filtered_edges = [(n1, n2) for n1, n2, data in graph.edges.data() if data[edge_attribute] != filter_val]
        filtered_graph = graph.edge_subgraph(filtered_edges)

        return filtered_graph

    def _get_all_waypoints_from_topology(self, topology):
        
        all_waypoints = []
        for segment in topology:
            all_waypoints.append(segment['entry'])
            all_waypoints.append(segment['exit'])

        return all_waypoints

    def _get_all_xyz_nodes_from_topology(self, topology):
        
        all_nodes = []
        for segment in topology:
            all_nodes.append(segment['entry_xyz'])
            all_nodes.append(segment['exit_xyz'])

        return all_nodes
