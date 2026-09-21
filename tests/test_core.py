import subprocess
import sys
from dataclasses import replace

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from ricci import (
    BatchGroup,
    Edge,
    Graph,
    Index,
    IndexSpace,
    NodeSpec,
    Port,
    RicciError,
    SpaceKind,
    Variance,
)
from ricci.backends.torch import Node, contract
from ricci.core import lower


def matrix_graph(size=3, label="i"):
    space = IndexSpace("V", size)
    up, down = Index(label, space, Variance.UP), Index(label, space, Variance.DOWN)
    return Graph(
        (NodeSpec("A", (up, down)), NodeSpec("v", (up,))),
        (Edge(Port("A", 1), Port("v", 0)),),
        outputs=(Port("A", 0),),
    )


@given(size=st.integers(1, 8), label=st.text(min_size=1, max_size=12))
@settings(deadline=None)
def test_matrix_vector_and_label_independence(size, label):
    graph = matrix_graph(size, label)
    a = torch.arange(size * size, dtype=torch.float64).reshape(size, size)
    v = torch.arange(size, dtype=torch.float64)
    actual = contract(graph, {"A": a, "v": v})
    torch.testing.assert_close(actual.data, a @ v)
    assert lower(graph) == lower(matrix_graph(size, "other label"))


def test_trace_and_multiple_edges():
    graph = matrix_graph()
    matrix = graph.nodes[0]
    a = torch.arange(9, dtype=torch.float64).reshape(3, 3)
    trace = Graph((matrix,), (Edge(Port("A", 0), Port("A", 1)),))
    torch.testing.assert_close(contract(trace, {"A": a}).data, torch.trace(a))
    other = NodeSpec("B", tuple(i.dual() for i in matrix.indices))
    double = Graph((matrix, other), tuple(Edge(Port("A", i), Port("B", i)) for i in range(2)))
    torch.testing.assert_close(contract(double, {"A": a, "B": a}).data, a.square().sum())


def test_outer_product_has_no_implicit_connections():
    v = matrix_graph().nodes[1]
    w = replace(v, name="w")
    graph = Graph((v, w), outputs=(Port("w", 0), Port("v", 0)))
    x, y = torch.arange(3, dtype=torch.float64), torch.arange(3, dtype=torch.float64) + 1
    out = contract(graph, {"v": x, "w": y})
    torch.testing.assert_close(out.data, torch.outer(y, x))


def test_scalar_nodes_and_permutation():
    scalar = Graph((NodeSpec("s", ()),))
    x = torch.tensor(2.0, requires_grad=True)
    out = contract(scalar, {"s": x})
    out.data.backward()
    assert x.grad == 1
    a = matrix_graph().nodes[0]
    graph = Graph((a,), outputs=(Port("A", 1), Port("A", 0)))
    data = torch.arange(9, dtype=torch.float64).reshape(3, 3)
    torch.testing.assert_close(contract(graph, {"A": data}).data, data.T)


def test_core_import_has_no_backend_side_effects():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import ricci.core, sys; assert 'torch' not in sys.modules; "
            "assert 'pydantic' not in sys.modules; assert 'mcp' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("size", [0, -1, True, 1.2])
def test_bad_space_size(size):
    with pytest.raises(RicciError, match="positive integer"):
        IndexSpace("V", size)


def test_invalid_metadata_and_node():
    with pytest.raises(RicciError):
        IndexSpace("V", 2, "geometric")
    with pytest.raises(RicciError):
        Index("", IndexSpace("V", 2), Variance.UP)
    with pytest.raises(RicciError):
        Index("i", IndexSpace("V", 2))
    with pytest.raises(RicciError):
        Index("b", IndexSpace("batch", 2, SpaceKind.BATCH), Variance.UP)
    with pytest.raises(RicciError):
        NodeSpec("", ())
    with pytest.raises(RicciError):
        Node("x", torch.ones(2), ())
    with pytest.raises(RicciError):
        Node("x", [1, 2], ())
    assert Index("i", IndexSpace("c", 2, SpaceKind.COMPONENT)).dual().variance is Variance.NEUTRAL


@pytest.mark.parametrize("change", ["space", "basis", "fiber", "variance", "size"])
def test_invalid_geometric_connections(change):
    graph = matrix_graph()
    v = graph.nodes[1]
    index = v.indices[0]
    if change == "variance":
        index = index.dual()
    else:
        changes = {
            "space": {"id": "W"},
            "basis": {"basis": "other"},
            "fiber": {"fiber": "q"},
            "size": {"size": 4},
        }
        index = replace(index, space=replace(index.space, **changes[change]))
    with pytest.raises(RicciError):
        replace(graph, nodes=(graph.nodes[0], replace(v, indices=(index,))))


@pytest.mark.parametrize(
    "edges,outputs",
    [
        ((Edge(Port("A", 1), Port("v", -1)),), (Port("A", 0),)),
        ((Edge(Port("A", 1), Port("missing", 0)),), (Port("A", 0),)),
        ((Edge(Port("A", 1), Port("v", 0)),), ()),
        ((Edge(Port("A", 1), Port("v", 0)),), (Port("A", 0), Port("A", 1))),
        ((Edge(Port("A", 1), Port("v", 0)),), (Port("A", 0), Port("A", 0))),
        ((Edge(Port("A", 1), Port("v", 0)),) * 2, (Port("A", 0),)),
        ((Edge(Port("A", 1), Port("A", 1)),), (Port("A", 0), Port("v", 0))),
    ],
)
def test_invalid_topology(edges, outputs):
    with pytest.raises(RicciError):
        replace(matrix_graph(), edges=edges, outputs=outputs)


def test_empty_duplicate_graphs_and_symbol_limit():
    with pytest.raises(RicciError):
        Graph(())
    with pytest.raises(RicciError):
        Graph((NodeSpec("s", ()), NodeSpec("s", ())))
    i = Index("i", IndexSpace("c", 2, SpaceKind.COMPONENT))
    nodes = tuple(NodeSpec(str(n), (i,)) for n in range(53))
    graph = Graph(nodes, outputs=tuple(Port(n.name, 0) for n in nodes))
    with pytest.raises(RicciError, match="52"):
        lower(graph)


def test_batch_hyperedge_retains_one_axis():
    batch = Index("b", IndexSpace("batch", 3, SpaceKind.BATCH))
    nodes = tuple(NodeSpec(n, (batch,)) for n in ("x", "y", "z"))
    ports = tuple(Port(n.name, 0) for n in nodes)
    graph = Graph(nodes, batches=(BatchGroup(ports),), outputs=(ports[1],))
    x = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(contract(graph, {n.name: x for n in nodes}).data, x**3)
    for outputs in ((), ports):
        with pytest.raises(RicciError):
            replace(graph, outputs=outputs)
    with pytest.raises(RicciError):
        replace(graph, batches=(BatchGroup((ports[0], ports[0])),))
    with pytest.raises(RicciError):
        Graph(nodes[:2], edges=(Edge(ports[0], ports[1]),))


def test_invalid_batch_spaces_and_occurrences():
    batch = Index("b", IndexSpace("batch", 2, SpaceKind.BATCH))
    component = Index("c", IndexSpace("c", 2, SpaceKind.COMPONENT))
    for other in (component, replace(batch, space=IndexSpace("other", 2, SpaceKind.BATCH))):
        with pytest.raises(RicciError):
            Graph(
                (NodeSpec("a", (batch,)), NodeSpec("b", (other,))),
                batches=(BatchGroup((Port("a", 0), Port("b", 0))),),
                outputs=(Port("a", 0),),
            )
    with pytest.raises(RicciError):
        Graph(
            (NodeSpec("a", (batch, batch)),),
            batches=(BatchGroup((Port("a", 0), Port("a", 1))),),
            outputs=(Port("a", 0),),
        )
