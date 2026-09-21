import pytest
import torch

from ricci import Edge, Graph, Index, IndexSpace, NodeSpec, Port, SpaceKind
from ricci.backends.torch import contract


@pytest.mark.parametrize("strategy", ["direct", "greedy", "auto"])
def test_mps_chain_values_and_tied_parameter_gradients(strategy):
    physical = Index("p", IndexSpace("physical", 2, SpaceKind.COMPONENT))
    bond = Index("a", IndexSpace("bond", 3, SpaceKind.COMPONENT))
    graph = Graph(
        (
            NodeSpec("left", (physical, bond)),
            NodeSpec("middle", (bond, physical, bond)),
            NodeSpec("right", (bond, physical)),
        ),
        edges=(Edge(Port("left", 1), Port("middle", 0)), Edge(Port("middle", 2), Port("right", 0))),
        outputs=(Port("left", 0), Port("middle", 1), Port("right", 1)),
    )
    a = torch.randn(2, 3, dtype=torch.complex128, requires_grad=True)
    b = torch.randn(3, 2, 3, dtype=torch.complex128, requires_grad=True)
    # Reuse one trainable tensor at both ends; the right operand is a live view.
    actual = contract(graph, {"left": a, "middle": b, "right": a.T}, strategy=strategy).data
    reference = torch.stack([a @ b[:, j, :] @ a.T for j in range(2)], dim=1)
    torch.testing.assert_close(actual, reference)
    actual_gradients = torch.autograd.grad(actual.abs().square().sum(), (a, b))
    expected_gradients = torch.autograd.grad(reference.abs().square().sum(), (a, b))
    for got, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(got, expected)
