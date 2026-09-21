"""Explicit application of an input covariant metric/bilinear form."""

from ricci import Edge, Graph, Port, RicciError, Variance
from ricci.backends.torch import Node, contract


def metric_pairing(metric: Node, u: Node, v: Node) -> Node:
    """Compute g_ij u^i v^j without implicit Euclidean identification.

    The caller supplies g. Symmetry, nondegeneracy and signature are assumptions
    of calling this a metric; this helper evaluates any compatible bilinear form.
    No conjugation is applied, even for complex data.
    """
    if tuple(len(n.indices) for n in (metric, u, v)) != (2, 1, 1):
        raise RicciError("INVALID_ARGUMENT", "Expected an order-2 form and two vectors")
    if any(index.variance is not Variance.DOWN for index in metric.indices):
        raise RicciError("INVALID_VARIANCE", "The covariant form must have two DOWN legs")
    if u.indices[0].variance is not Variance.UP or v.indices[0].variance is not Variance.UP:
        raise RicciError("INVALID_VARIANCE", "Vectors must have UP legs")
    # Local names allow the same vector object to occupy two operand positions.
    g = Node("g", metric.data, metric.indices)
    left, right = Node("u", u.data, u.indices), Node("v", v.data, v.indices)
    graph = Graph(
        (g.spec, left.spec, right.spec),
        edges=(Edge(Port("g", 0), Port("u", 0)), Edge(Port("g", 1), Port("v", 0))),
    )
    return contract(graph, {"g": g.data, "u": left.data, "v": right.data})
