"""Bell-state normalization and a small MPS contracted without a dense input state."""

import torch

from ricci import Edge, Graph, Index, IndexSpace, NodeSpec, Port, SpaceKind
from ricci.backends.torch import Node, contract
from ricci.quantum import inner


def main() -> None:
    qubit = IndexSpace("qubit", 2, SpaceKind.COMPONENT, basis="computational")
    bond = IndexSpace("bond", 2, SpaceKind.COMPONENT)
    p, a = Index("physical", qubit), Index("bond", bond)
    graph = Graph(
        (NodeSpec("left", (p, a)), NodeSpec("right", (a, p))),
        edges=(Edge(Port("left", 1), Port("right", 0)),),
        outputs=(Port("left", 0), Port("right", 1)),
    )
    left = (torch.eye(2, dtype=torch.complex128) / 2**0.5).requires_grad_()
    right = torch.eye(2, dtype=torch.complex128)
    state = contract(graph, {"left": left, "right": right})
    norm = inner(state, state)
    torch.testing.assert_close(norm.data, torch.tensor(1.0, dtype=torch.complex128))
    # Z on the first qubit in the declared product basis.
    z_state = Node("Zpsi", torch.tensor([1.0, -1.0])[:, None] * state.data, state.indices)
    expectation = inner(state, z_state)
    torch.testing.assert_close(expectation.data, torch.tensor(0.0, dtype=torch.complex128))
    norm.data.real.backward()
    print("Bell state:", state.data, "norm:", norm.data.item(), "<Z ⊗ I>:", expectation.data.item())


if __name__ == "__main__":
    main()
