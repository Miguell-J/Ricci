"""Attention scores with explicit preserved batch/head axes."""

from math import sqrt

from ricci import BatchGroup, Edge, Graph, Port, RicciError, SpaceKind
from ricci.backends.torch import Node, contract


def attention_scores(query: Node, key: Node, *, scaled: bool = True) -> Node:
    """Compute real Q[b,h,i,d] K[b,h,j,d] -> scores[b,h,i,j].

    Batch/head legs use batch spaces; token/feature legs use component spaces.
    This returns scores only. Masks, softmax and value aggregation remain explicit.
    """
    if len(query.indices) != 4 or len(key.indices) != 4:
        raise RicciError(
            "INVALID_ARGUMENT", "Expected query/key with four legs: batch, head, token, d"
        )
    if query.data.is_complex() or key.data.is_complex():
        raise RicciError(
            "INVALID_DOMAIN", "Real attention only; define complex semantics explicitly"
        )
    for node in (query, key):
        if any(i.space.kind is not SpaceKind.COMPONENT for i in node.indices[2:]):
            raise RicciError("INVALID_DOMAIN", "Token and feature legs must use component spaces")
    q, k = Node("q", query.data, query.indices), Node("k", key.data, key.indices)
    graph = Graph(
        (q.spec, k.spec),
        edges=(Edge(Port("q", 3), Port("k", 3)),),
        batches=tuple(BatchGroup((Port("q", i), Port("k", i))) for i in (0, 1)),
        outputs=(Port("q", 0), Port("q", 1), Port("q", 2), Port("k", 2)),
    )
    scores = contract(graph, {"q": q.data, "k": k.data})
    data = scores.data / sqrt(q.indices[3].size) if scaled else scores.data
    return Node("scores", data, scores.indices)
