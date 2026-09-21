"""Trainable attention composition; softmax is explicitly outside the contraction IR."""

import torch
from torch import Tensor, nn

from ricci import Index, IndexSpace, SpaceKind
from ricci.backends.torch import Node
from ricci.nn import attention_scores


class DiagramAttention(nn.Module):
    """Single-head demonstration, not a fused production attention kernel."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        batch = Index("b", IndexSpace("batch", b, SpaceKind.BATCH))
        head = Index("h", IndexSpace("head", 1, SpaceKind.BATCH))
        token = Index("token", IndexSpace("token", t, SpaceKind.COMPONENT))
        feature = Index("d", IndexSpace("feature", d, SpaceKind.COMPONENT))
        indices = (batch, head, token, feature)
        q = Node("q", self.q(x).unsqueeze(1), indices)
        k = Node("k", self.k(x).unsqueeze(1), indices)
        scores = attention_scores(q, k).data.squeeze(1)
        return scores.softmax(dim=-1) @ self.v(x)


def main() -> None:
    torch.manual_seed(7)
    model = DiagramAttention(8).double()
    x = torch.randn(2, 4, 8, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(3):
        optimizer.zero_grad()
        loss = (model(x) - x).square().mean()
        loss.backward()
        optimizer.step()
    assert all(p.grad is not None for p in model.parameters())
    print("Attention output:", tuple(model(x).shape), "final loss:", loss.item())


if __name__ == "__main__":
    main()
