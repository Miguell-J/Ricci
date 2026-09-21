"""Finite-dimensional state operations in a declared orthonormal basis."""

from ricci import Edge, Graph, Port, RicciError, SpaceKind
from ricci.backends.torch import Node, contract


def inner(left: Node, right: Node) -> Node:
    """Return <left|right>, antilinear in left, for component-space state tensors.

    Every axis is a physical factor in an orthonormal product basis. This is not
    automatic metric raising/lowering in a general complex geometric basis.
    """
    if not left.indices or len(left.indices) != len(right.indices):
        raise RicciError("INVALID_ARGUMENT", "State tensors must have equal nonzero order")
    if any(i.space.kind is not SpaceKind.COMPONENT for i in (*left.indices, *right.indices)):
        raise RicciError("INVALID_DOMAIN", "inner requires orthonormal component spaces")
    bra = Node("bra", left.data.conj(), left.indices)
    ket = Node("ket", right.data, right.indices)
    graph = Graph(
        (bra.spec, ket.spec),
        edges=tuple(Edge(Port("bra", i), Port("ket", i)) for i in range(len(left.indices))),
    )
    return contract(graph, {"bra": bra.data, "ket": ket.data})
