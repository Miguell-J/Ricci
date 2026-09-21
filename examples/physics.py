"""Minkowski pairing and continuum strain energy, both with parameter gradients."""

import torch

from ricci import Edge, Graph, Index, IndexSpace, NodeSpec, Port, Variance
from ricci.backends.torch import Node, contract
from ricci.physics import metric_pairing


def main() -> None:
    spacetime = IndexSpace("spacetime", 4, basis="inertial", fiber="p")
    down = Index("mu", spacetime, Variance.DOWN)
    up = Index("mu", spacetime, Variance.UP)
    g = Node(
        "eta", torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64)), (down, down)
    )
    velocity = torch.tensor([2.0, 1.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    u = Node("u", velocity, (up,))
    norm = metric_pairing(g, u, u)
    norm.data.backward()
    torch.testing.assert_close(norm.data, torch.tensor(-3.0, dtype=torch.float64))
    print("Minkowski g(u,u):", norm.data.item(), "gradient:", velocity.grad)

    space = IndexSpace("material", 3)
    lo, hi = Index("i", space, Variance.DOWN), Index("i", space, Variance.UP)
    # W = 1/2 C^{ijkl} epsilon_ij epsilon_kl, supplied stiffness components.
    graph = Graph(
        (NodeSpec("C", (hi, hi, hi, hi)), NodeSpec("e1", (lo, lo)), NodeSpec("e2", (lo, lo))),
        edges=tuple(Edge(Port("C", a), Port("e1" if a < 2 else "e2", a % 2)) for a in range(4)),
    )
    identity = torch.eye(3, dtype=torch.float64)
    stiffness = torch.einsum("ik,jl->ijkl", identity, identity)
    strain = (identity * 0.01).requires_grad_()
    energy = 0.5 * contract(graph, {"C": stiffness, "e1": strain, "e2": strain}).data
    energy.backward()
    torch.testing.assert_close(strain.grad, strain)
    print("Strain energy:", energy.item())


if __name__ == "__main__":
    main()
