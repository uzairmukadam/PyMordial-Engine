"""Generic Waypoint Navigation Graph & Points of Interest (POI) Registry.

Provides domain-agnostic spatial routing, A* pathfinding, lane connectivity,
and designated Points of Interest for open-world games and AI pathfinding.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import heapq
import math
from typing import Any


@dataclass(slots=True)
class WaypointNode:
    """A spatial navigation node."""
    id: int
    position: tuple[float, float, float]
    tags: set[str] = field(default_factory=set)


@dataclass(slots=True)
class WaypointEdge:
    """A directed navigation edge connecting two waypoint nodes."""
    target_id: int
    distance: float
    speed_limit: float = 50.0  # Speed limit in m/s (~110 mph)
    lane_width: float = 3.5    # Width in meters
    is_alley: bool = False     # Narrow evasion alleyway flag


@dataclass(slots=True)
class PointOfInterest:
    """A landmark or gameplay objective trigger area."""
    name: str
    poi_type: str              # e.g. "safehouse", "respray", "bank", "warehouse", "dropoff"
    position: tuple[float, float, float]
    radius: float = 6.0        # Trigger radius in meters
    heading_deg: float = 0.0   # Preferred vehicle arrival/departure heading
    metadata: dict[str, Any] = field(default_factory=dict)


class WaypointGraph:
    """Directed spatial graph with fast nearest-neighbor lookups and A* routing."""

    __slots__ = (
        "_nodes",
        "_edges",
        "_pois",
        "_pois_by_type",
        "_next_node_id",
    )

    def __init__(self) -> None:
        self._nodes: dict[int, WaypointNode] = {}
        self._edges: dict[int, list[WaypointEdge]] = {}
        self._pois: dict[str, PointOfInterest] = {}
        self._pois_by_type: dict[str, list[PointOfInterest]] = {}
        self._next_node_id: int = 0

    @property
    def nodes(self) -> dict[int, WaypointNode]:
        return self._nodes

    @property
    def edges(self) -> dict[int, list[WaypointEdge]]:
        return self._edges

    @property
    def pois(self) -> dict[str, PointOfInterest]:
        return self._pois

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    # ---------------- Graph Node Management ----------------

    def add_node(
        self,
        position: tuple[float, float, float] | WaypointNode,
        tags: set[str] | list[str] | None = None,
        node_id: int | None = None,
    ) -> int:
        """Adds a waypoint node and returns its unique ID."""
        if isinstance(position, WaypointNode):
            node = position
            node_id = node.id
            self._next_node_id = max(self._next_node_id, node_id + 1)
            self._nodes[node_id] = node
            if node_id not in self._edges:
                self._edges[node_id] = []
            return node_id

        if node_id is None:
            node_id = self._next_node_id
            self._next_node_id += 1
        else:
            self._next_node_id = max(self._next_node_id, node_id + 1)

        tag_set = set(tags) if tags else set()
        node = WaypointNode(id=node_id, position=position, tags=tag_set)
        self._nodes[node_id] = node
        if node_id not in self._edges:
            self._edges[node_id] = []
        return node_id

    def get_node(self, node_id: int) -> WaypointNode | None:
        """Fetches a node by ID."""
        return self._nodes.get(node_id, None)

    # ---------------- Edge & Connectivity Management ----------------

    def add_edge(
        self,
        from_id: int,
        to_id: int,
        is_bidirectional: bool = True,
        speed_limit: float = 50.0,
        lane_width: float = 3.5,
        is_alley: bool = False,
    ) -> None:
        """Connects two nodes with directed or bidirectional edges."""
        if from_id not in self._nodes or to_id not in self._nodes:
            return

        p1 = self._nodes[from_id].position
        p2 = self._nodes[to_id].position
        dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

        edge = WaypointEdge(
            target_id=to_id,
            distance=dist,
            speed_limit=speed_limit,
            lane_width=lane_width,
            is_alley=is_alley,
        )
        self._edges[from_id].append(edge)

        if is_bidirectional:
            rev_edge = WaypointEdge(
                target_id=from_id,
                distance=dist,
                speed_limit=speed_limit,
                lane_width=lane_width,
                is_alley=is_alley,
            )
            self._edges[to_id].append(rev_edge)

    def get_connected_nodes(self, node_id: int) -> list[WaypointEdge]:
        """Returns all outgoing edges from a given node."""
        return self._edges.get(node_id, [])

    # ---------------- Spatial Queries & A* Pathfinding ----------------

    def find_nearest_node(
        self,
        pos: tuple[float, float, float],
        max_dist: float = 1000.0,
        required_tag: str | None = None,
    ) -> int | None:
        """Finds the closest node to a given world position."""
        best_id: int | None = None
        best_sq = max_dist * max_dist
        px, py, pz = pos[0], pos[1], pos[2]

        for nid, node in self._nodes.items():
            if required_tag and required_tag not in node.tags:
                continue
            nx, ny, nz = node.position
            dx = nx - px
            dy = ny - py
            dz = nz - pz
            d_sq = dx * dx + dy * dy + dz * dz
            if d_sq < best_sq:
                best_sq = d_sq
                best_id = nid

        return best_id

    def find_path(self, start_id: int, goal_id: int) -> list[int]:
        """Computes shortest path between two nodes using A* with Euclidean distance heuristic.

        Returns:
            List of node IDs from start_id to goal_id inclusive, or [] if unreachable.
        """
        if start_id not in self._nodes or goal_id not in self._nodes:
            return []

        if start_id == goal_id:
            return [start_id]

        goal_pos = self._nodes[goal_id].position

        def heuristic(nid: int) -> float:
            p = self._nodes[nid].position
            return math.sqrt(
                (p[0] - goal_pos[0]) ** 2
                + (p[1] - goal_pos[1]) ** 2
                + (p[2] - goal_pos[2]) ** 2
            )

        # Priority queue: (f_score, current_node_id)
        open_set: list[tuple[float, int]] = [(heuristic(start_id), start_id)]
        came_from: dict[int, int] = {}
        g_score: dict[int, float] = {start_id: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_id:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            cur_g = g_score[current]
            for edge in self._edges.get(current, []):
                neighbor = edge.target_id
                tentative_g = cur_g + edge.distance

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, neighbor))

        return []

    # ---------------- Points of Interest (POIs) ----------------

    def add_poi(
        self,
        name: str | PointOfInterest,
        poi_type: str | None = None,
        position: tuple[float, float, float] | None = None,
        radius: float = 6.0,
        heading_deg: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> PointOfInterest:
        """Registers a named Point of Interest on the map."""
        if isinstance(name, PointOfInterest):
            poi = name
        else:
            if poi_type is None or position is None:
                raise ValueError("poi_type and position are required when passing POI name as string")
            poi = PointOfInterest(
                name=name,
                poi_type=poi_type.lower(),
                position=position,
                radius=radius,
                heading_deg=heading_deg,
                metadata=metadata if metadata is not None else {},
            )
        self._pois[poi.name] = poi
        if poi.poi_type not in self._pois_by_type:
            self._pois_by_type[poi.poi_type] = []
        self._pois_by_type[poi.poi_type].append(poi)
        return poi

    def get_poi(self, name: str) -> PointOfInterest | None:
        """Retrieves a POI by unique name."""
        return self._pois.get(name, None)

    def get_pois_by_type(self, poi_type: str) -> list[PointOfInterest]:
        """Retrieves all POIs of a specified type (e.g. 'safehouse', 'warehouse')."""
        return self._pois_by_type.get(poi_type.lower(), [])

    def find_nearest_poi(
        self,
        pos: tuple[float, float, float],
        poi_type: str | None = None,
    ) -> PointOfInterest | None:
        """Finds the closest POI to a position, optionally filtered by type."""
        candidates = self._pois_by_type.get(poi_type.lower(), []) if poi_type else list(self._pois.values())
        if not candidates:
            return None

        best_poi: PointOfInterest | None = None
        best_sq = float("inf")
        px, py, pz = pos[0], pos[1], pos[2]

        for poi in candidates:
            x, y, z = poi.position
            d_sq = (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2
            if d_sq < best_sq:
                best_sq = d_sq
                best_poi = poi

        return best_poi
