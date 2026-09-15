from __future__ import annotations

from .audit_pursuit_local_factorization import bounded_local_graph, jaccard_distance


def test_bounded_local_graph_is_predictable_local_and_capped() -> None:
    positions = ((0, 0), (1, 0), (0, 2), (2, 2), (8, 8))
    graph, raw_degrees = bounded_local_graph(
        positions, radius=2, max_neighbors=2
    )
    assert raw_degrees == (3, 3, 3, 3, 0)
    assert graph == (
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 3),
        (3, 2),
        (3, 1),
    )


def test_jaccard_distance_handles_empty_and_changed_graphs() -> None:
    assert jaccard_distance((), ()) == 0.0
    assert jaccard_distance(((0, 1),), ((0, 1),)) == 0.0
    assert jaccard_distance(((0, 1),), ((1, 0),)) == 1.0
    assert jaccard_distance(((0, 1), (1, 0)), ((0, 1),)) == 0.5
