"""Automated Unit Tests for WaypointGraph & Point of Interest System."""

import pytest
from engine.world.waypoint_graph import WaypointGraph, PointOfInterest


def test_waypoint_graph_nodes_and_edges():
    """Verifies node creation and bidirectional edge linking."""
    graph = WaypointGraph()

    n0 = graph.add_node((0.0, 0.0, 0.0), tags={"road", "intersection"})
    n1 = graph.add_node((0.0, 0.0, -50.0), tags={"road"})
    n2 = graph.add_node((50.0, 0.0, 0.0), tags={"road"})

    assert graph.node_count == 3
    node0 = graph.get_node(n0)
    assert node0 is not None
    assert "intersection" in node0.tags

    graph.add_edge(n0, n1, is_bidirectional=True)
    graph.add_edge(n0, n2, is_bidirectional=False)

    edges_0 = graph.get_connected_nodes(n0)
    assert len(edges_0) == 2
    assert edges_0[0].target_id == n1
    assert edges_0[0].distance == pytest.approx(50.0)

    # n1 should have edge back to n0 (bidirectional)
    edges_1 = graph.get_connected_nodes(n1)
    assert len(edges_1) == 1
    assert edges_1[0].target_id == n0

    # n2 should have NO edge back to n0 (unidirectional)
    edges_2 = graph.get_connected_nodes(n2)
    assert len(edges_2) == 0


def test_waypoint_graph_nearest_node():
    """Verifies spatial nearest-neighbor lookups with distance and tag filters."""
    graph = WaypointGraph()
    n0 = graph.add_node((0.0, 0.0, 0.0), tags={"avenue"})
    n1 = graph.add_node((100.0, 0.0, 0.0), tags={"avenue"})
    n2 = graph.add_node((10.0, 0.0, 5.0), tags={"alley"})

    # Closest to (8.0, 0.0, 4.0) should be n2
    nearest = graph.find_nearest_node((8.0, 0.0, 4.0))
    assert nearest == n2

    # Filtered by tag "avenue" should return n0
    nearest_avenue = graph.find_nearest_node((8.0, 0.0, 4.0), required_tag="avenue")
    assert nearest_avenue == n0


def test_waypoint_graph_astar_pathfinding():
    """Verifies A* shortest path computation."""
    graph = WaypointGraph()
    # Build a simple grid:
    # (0,0) --- (50,0) --- (100,0)
    #   |                     |
    # (0,50) -------------- (100,50)
    n_a = graph.add_node((0.0, 0.0, 0.0))
    n_b = graph.add_node((50.0, 0.0, 0.0))
    n_c = graph.add_node((100.0, 0.0, 0.0))
    n_d = graph.add_node((0.0, 0.0, 50.0))
    n_e = graph.add_node((100.0, 0.0, 50.0))

    graph.add_edge(n_a, n_b)
    graph.add_edge(n_b, n_c)
    graph.add_edge(n_a, n_d)
    graph.add_edge(n_d, n_e)
    graph.add_edge(n_c, n_e)

    # Shortest path A -> C should be [A, B, C]
    path_a_c = graph.find_path(n_a, n_c)
    assert path_a_c == [n_a, n_b, n_c]

    # Path A -> E can go [A, B, C, E] (150m) or [A, D, E] (150m)
    path_a_e = graph.find_path(n_a, n_e)
    assert len(path_a_e) in (3, 4)
    assert path_a_e[0] == n_a
    assert path_a_e[-1] == n_e


def test_points_of_interest_registry():
    """Verifies POI creation, retrieval by type, and proximity search."""
    graph = WaypointGraph()

    p_safehouse = graph.add_poi(
        name="Safehouse_Garage",
        poi_type="safehouse",
        position=(-80.0, 0.0, 40.0),
        heading_deg=90.0,
        metadata={"garage_tier": 1},
    )
    p_bank = graph.add_poi(
        name="Metropolitan_Bank",
        poi_type="heist_target",
        position=(60.0, 0.0, -80.0),
        metadata={"payout": 250000},
    )
    p_respray = graph.add_poi(
        name="Underground_Respray",
        poi_type="respray",
        position=(10.0, 0.0, 15.0),
    )

    assert graph.get_poi("Safehouse_Garage") == p_safehouse
    assert graph.get_poi("Nonexistent") is None

    heist_targets = graph.get_pois_by_type("heist_target")
    assert len(heist_targets) == 1
    assert heist_targets[0].name == "Metropolitan_Bank"

    # Nearest POI to (0, 0, 10) should be Underground_Respray
    nearest = graph.find_nearest_poi((0.0, 0.0, 10.0))
    assert nearest is not None
    assert nearest.name == "Underground_Respray"
