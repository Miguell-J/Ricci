"""Concrete JSON models for future adapters; not an alternative MCP envelope."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from ricci.core import (
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

Name = Annotated[str, Field(min_length=1, max_length=256)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SpaceDocument(Model):
    id: Name
    size: Annotated[StrictInt, Field(gt=0)]
    kind: SpaceKind
    basis: Name = "default"
    fiber: Name | None = None


class IndexDocument(Model):
    label: Name
    space: Name
    variance: Variance


class NodeDocument(Model):
    name: Name
    indices: Annotated[tuple[IndexDocument, ...], Field(max_length=32)]


class PortDocument(Model):
    node: Name
    axis: Annotated[StrictInt, Field(ge=0, le=31)]

    def to_port(self) -> Port:
        return Port(self.node, self.axis)


class EdgeDocument(Model):
    left: PortDocument
    right: PortDocument


class BatchDocument(Model):
    ports: Annotated[tuple[PortDocument, ...], Field(min_length=2, max_length=128)]


class GraphDocument(Model):
    """JSON syntax validation is followed by semantic validation via to_graph().

    This is a contraction specification, not tensor storage. A future service
    resolves separate operand references and executes forward/backward in one job.
    """

    schema_version: Literal["ricci.graph.v1"] = "ricci.graph.v1"
    spaces: Annotated[tuple[SpaceDocument, ...], Field(max_length=4096)]
    nodes: Annotated[tuple[NodeDocument, ...], Field(min_length=1, max_length=128)]
    edges: Annotated[tuple[EdgeDocument, ...], Field(max_length=2048)] = ()
    batches: Annotated[tuple[BatchDocument, ...], Field(max_length=52)] = ()
    outputs: Annotated[tuple[PortDocument, ...], Field(max_length=32)] = ()

    def to_graph(self) -> Graph:
        spaces = {s.id: IndexSpace(s.id, s.size, s.kind, s.basis, s.fiber) for s in self.spaces}
        if len(spaces) != len(self.spaces):
            raise RicciError("SPACE_CONFLICT", "Space ids must be unique")
        try:
            nodes = tuple(
                NodeSpec(
                    n.name, tuple(Index(i.label, spaces[i.space], i.variance) for i in n.indices)
                )
                for n in self.nodes
            )
        except KeyError as exc:
            raise RicciError("SPACE_MISMATCH", f"Undeclared space: {exc.args[0]}") from exc
        return Graph(
            nodes,
            tuple(Edge(e.left.to_port(), e.right.to_port()) for e in self.edges),
            tuple(BatchGroup(tuple(p.to_port() for p in b.ports)) for b in self.batches),
            tuple(p.to_port() for p in self.outputs),
        )

    @classmethod
    def from_graph(cls, graph: Graph) -> "GraphDocument":
        graph.validate()
        spaces = {i.space.id: i.space for n in graph.nodes for i in n.indices}

        def port(p: Port) -> PortDocument:
            return PortDocument(node=p.node, axis=p.axis)

        return cls(
            spaces=tuple(
                SpaceDocument(id=s.id, size=s.size, kind=s.kind, basis=s.basis, fiber=s.fiber)
                for s in sorted(spaces.values(), key=lambda s: s.id)
            ),
            nodes=tuple(
                NodeDocument(
                    name=n.name,
                    indices=tuple(
                        IndexDocument(label=i.label, space=i.space.id, variance=i.variance)
                        for i in n.indices
                    ),
                )
                for n in graph.nodes
            ),
            edges=tuple(EdgeDocument(left=port(e.left), right=port(e.right)) for e in graph.edges),
            batches=tuple(
                BatchDocument(ports=tuple(port(p) for p in b.ports)) for b in graph.batches
            ),
            outputs=tuple(port(p) for p in graph.outputs),
        )
