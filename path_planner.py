from enum import Enum

import carla
import networkx as nx
import numpy as np


class EdgeType(Enum):
    VOID = -1
    SIDEWALK = 1
    CROSSWALK = 2
    JAYWALK = 3


class PedPathPlanner:

    def __init__(self, carla_map, waypoint_distance):
        self.waypoint_distance = waypoint_distance
        self.carla_map = carla_map
        self.topology = None
        self.graph = None
        self.id_map = None
        self.road_id_to_edge = None

        # Build the graph
        self.build_topology()
        self.build_graph()

    def get_all_junctions(self, carla_topology):
        # get all junctions and filter out duplicates
        junctions = [w[0].get_junction() for w in carla_topology if w[0].is_junction]
        junction_ids = set()
        filtered_junctions = []
        for j in junctions:
            if j.id not in junction_ids:
                filtered_junctions.append(j)
                junction_ids.add(j.id)

        return filtered_junctions

    def get_all_junction_corners(self, carla_topology):
        junctions = self.get_all_junctions(carla_topology)

        # get all waypoints of type sidewalk that are located in a junction
        junction_corner_sub_segments = []
        for junction in junctions:
            waypoint_tuples = junction.get_waypoints(carla.LaneType.Sidewalk)

            for segment in waypoint_tuples:
                w = segment[0]

                # filter out segments that are no corners (corner segments have no neighboring lanes of type driving)
                is_corner = True
                # Check for driving lane type until there are no waypoints by going left
                l = w.get_left_lane()
                while l and is_corner:
                    if l.lane_type == carla.LaneType.Driving:
                        is_corner = False
                    l = l.get_left_lane()

                # Check for driving lane type until there are no waypoints by going right
                r = w.get_right_lane()
                while r and is_corner:
                    if r.lane_type == carla.LaneType.Driving:
                        is_corner = False
                    r = r.get_right_lane()

                # get middle waypoint of corner segment (= junction corner)
                if is_corner:
                    w_list = w.next_until_lane_end(0.5)

                    middle_index = len(w_list) // 2
                    middle_w = w_list[middle_index]

                    junction_corner_waypoints = [w, middle_w, segment[1]]

                    sub_segments = self.generate_sub_segment_dicts(junction_corner_waypoints)
                    junction_corner_sub_segments.extend(sub_segments)

        return junction_corner_sub_segments

    def get_all_crosswalks(self):
        crosswalk_corners = self.carla_map.get_crosswalks()
        # every crosswalk is represented by 5 points (4 corners + repetition of first corner) -> delete every 5th element
        del crosswalk_corners[4::5]
        crosswalks_np = np.array([np.array([p.x, p.y, p.z]) for p in crosswalk_corners])
        sorted_crosswalks = np.reshape(crosswalks_np, (-1, 2, 2, 3))

        crosswalk_sub_segments = []

        for crosswalk in sorted_crosswalks:
            crosswalk_waypoints = []
            for side in crosswalk:
                middle = (side[0] + side[1]) / 2
                middle_loc = carla.Location(x=middle[0], y=middle[1])
                middle_waypoint = self.carla_map.get_waypoint(middle_loc, lane_type=carla.LaneType.Shoulder)
                if middle_waypoint is not None:
                    crosswalk_waypoints.append(middle_waypoint)

            sub_segment = self.generate_sub_segment_dicts(crosswalk_waypoints)
            crosswalk_sub_segments.extend(sub_segment)

        return crosswalk_sub_segments

    def get_sidewalk_waypoints(self, segment):
        sidewalk_waypoints_right = []
        sidewalk_waypoints_left = []
        start = segment[0]
        w_list = [start]
        w_list.extend(start.next_until_lane_end(self.waypoint_distance))
        for w in w_list:
            # Check for sidewalk lane type until there are no waypoints by going left
            l = w.get_left_lane()
            while l and l.lane_type != carla.LaneType.Driving:
                if l.lane_type == carla.LaneType.Sidewalk:
                    sidewalk_waypoints_left.append(l)
                l = l.get_left_lane()

            # Check for sidewalk lane type until there are no waypoints by going right
            r = w.get_right_lane()
            while r and r.lane_type != carla.LaneType.Driving:
                if r.lane_type == carla.LaneType.Sidewalk:
                    sidewalk_waypoints_right.append(r)
                r = r.get_right_lane()

        return [sidewalk_waypoints_left, sidewalk_waypoints_right]

    def generate_sub_segment_dicts(self, waypoint_list):
        waypoint_locs = [w.transform.location for w in waypoint_list]
        waypoints_xyz = [tuple(np.round([loc.x, loc.y, loc.z], 0)) for loc in waypoint_locs]

        sub_segments = []
        for i in range(len(waypoint_list) - 1):
            current_wp, next_wp = waypoint_list[i], waypoint_list[i + 1]
            current_xyz, next_xyz = waypoints_xyz[i], waypoints_xyz[i + 1]

            sub_seg_dict = {'entry': current_wp, 'exit': next_wp, 'entry_xyz': current_xyz, 'exit_xyz': next_xyz}
            sub_segments.append(sub_seg_dict)

        return sub_segments

    def get_all_waypoints_from_topology(self, topology):
        all_waypoints = []

        for segment in topology:
            all_waypoints.append(segment['entry'])
            all_waypoints.append(segment['exit'])

        return all_waypoints

    def get_connections_to_crosswalks(self, crosswalk_sub_segments):
        topology_waypoints = self.get_all_waypoints_from_topology(self.topology)
        connections = []

        for crosswalk in crosswalk_sub_segments:
            wp1 = crosswalk['entry']
            wp2 = crosswalk['exit']

            for wp in wp1, wp2:
                loc = wp.transform.location
                neighboring_waypoints = [w for w in topology_waypoints
                                         if w.road_id == wp.road_id and loc.distance(w.transform.location) < 10]
                for n in neighboring_waypoints:
                    sub_segment = self.generate_sub_segment_dicts([wp, n])
                    connections.extend(sub_segment)

        return connections

    def build_topology(self):
        """
        This function retrieves topology from the server as a list of
        road segments as pairs of waypoint objects, and processes the
        topology into a list of dictionary objects with the following attributes

        - entry (carla.Waypoint): waypoint of entry point of road segment
        - entryxyz (tuple): (x,y,z) of entry point of road segment
        - exit (carla.Waypoint): waypoint of exit point of road segment
        - exitxyz (tuple): (x,y,z) of exit point of road segment
        - path (list of carla.Waypoint):  list of waypoints between entry to exit, separated by the resolution
        """
        carla_topology = self.carla_map.get_topology()
        self.topology = []
        # Retrieving waypoints to construct a detailed topology
        for segment in carla_topology:

            sidewalk_waypoints = self.get_sidewalk_waypoints(segment)

            for side in sidewalk_waypoints:
                if side:
                    sub_segments = self.generate_sub_segment_dicts(side)
                    self.topology.extend(sub_segments)

        junction_sub_segments = self.get_all_junction_corners(carla_topology)
        self.topology.extend(junction_sub_segments)

        crosswalk_sub_segments = self.get_all_crosswalks()
        connections = self.get_connections_to_crosswalks(crosswalk_sub_segments)
        self.topology.extend(connections)
        self.topology.extend(crosswalk_sub_segments)

    def build_graph(self):
        """
        This function builds a networkx graph representation of topology, creating several class attributes:
        - graph (networkx.DiGraph): networkx graph representing the world map, with:
            Node properties:
                vertex: (x,y,z) position in world map
            Edge properties:
                entry_vector: unit vector along tangent at entry point
                exit_vector: unit vector along tangent at exit point
                net_vector: unit vector of the chord from entry to exit
                intersection: boolean indicating if the edge belongs to an  intersection
        - id_map (dictionary): mapping from (x,y,z) to node id
        - road_id_to_edge (dictionary): map from road id to edge in the graph
        """

        self.graph = nx.Graph()
        self.id_map = dict()  # Map with structure {(x,y,z): id, ... }
        self.road_id_to_edge = dict()  # Map with structure {road_id: {lane_id: edge, ... }, ... }

        for segment in self.topology:
            entry_xyz, exit_xyz = segment['entry_xyz'], segment['exit_xyz']
            entry_wp, exit_wp = segment['entry'], segment['exit']
            intersection = entry_wp.is_junction
            road_id, section_id, lane_id = entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id

            for vertex in entry_xyz, exit_xyz:
                # Adding unique nodes and populating id_map
                if vertex not in self.id_map:
                    new_id = len(self.id_map)
                    self.id_map[vertex] = new_id
                    self.graph.add_node(new_id, vertex=vertex)
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

            if entry_wp.lane_type == carla.LaneType.Shoulder and exit_wp.lane_type == carla.LaneType.Shoulder:
                edge_type = EdgeType.CROSSWALK
            else:
                edge_type = EdgeType.SIDEWALK

            length = entry_wp.transform.location.distance(exit_wp.transform.location)

            # Adding edge with attributes
            self.graph.add_edge(
                n1, n2,
                length=length,
                entry_waypoint=entry_wp, exit_waypoint=exit_wp,
                intersection=intersection, type=edge_type)

    def localize(self, location):
        """
        This function finds the road segment that a given location
        is part of, returning the edge it belongs to
        """
        waypoint = self.carla_map.get_waypoint(location, lane_type=carla.LaneType.Sidewalk)
        closest_node = None
        try:
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

    def distance_heuristic(self, n1, n2):
        """
        Distance heuristic calculator for path searching
        in self._graph
        """
        l1 = np.array(self.graph.nodes[n1]['vertex'])
        l2 = np.array(self.graph.nodes[n2]['vertex'])
        return np.linalg.norm(l1 - l2)

    def path_search(self, origin, destination):
        """
        This function finds the shortest path connecting origin and destination
        using A* search with distance heuristic.
        origin      :   carla.Location object of start position
        destination :   carla.Location object of end position
        return      :   path as list of node ids (as int) of the graph self._graph
        connecting origin and destination
        """
        start, end = self.localize(origin), self.localize(destination)

        route = nx.astar_path(
            self.graph, source=start, target=end,
            heuristic=self.distance_heuristic, weight='length')
        return route
