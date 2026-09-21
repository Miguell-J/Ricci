"""Immutable contraction IR. No torch import or tensor storage lives here."""

from dataclasses import dataclass

from ricci.core.errors import RicciError
from ricci.core.index import Index, IndexSpace, SpaceKind


@dataclass(frozen=True, slots=True)
class NodeSpec:
    name: str
    indices: tuple[Index, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", tuple(self.indices))
        if not self.name:
            raise RicciError("INVALID_NODE", "Node name must be nonempty")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(index.size for index in self.indices)


@dataclass(frozen=True, slots=True)
class Port:
    node: str
    axis: int


@dataclass(frozen=True, slots=True)
class Edge:
    left: Port
    right: Port


@dataclass(frozen=True, slots=True)
class BatchGroup:
    """Aligned occurrences retained once in the output, never implicitly summed."""

    ports: tuple[Port, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", tuple(self.ports))


@dataclass(frozen=True, slots=True)
class Graph:
    nodes: tuple[NodeSpec, ...]
    edges: tuple[Edge, ...] = ()
    batches: tuple[BatchGroup, ...] = ()
    outputs: tuple[Port, ...] = ()

    def __post_init__(self) -> None:
        for name in ("nodes", "edges", "batches", "outputs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        self.validate()

    def index(self, port: Port) -> Index:
        for node in self.nodes:
            if node.name == port.node:
                if type(port.axis) is int and 0 <= port.axis < len(node.indices):
                    return node.indices[port.axis]
                break
        raise RicciError("INVALID_PORT", f"Unknown port {port}")

    def validate(self) -> None:
        if not self.nodes or len({node.name for node in self.nodes}) != len(self.nodes):
            raise RicciError("INVALID_GRAPH", "A graph requires distinct, nonempty node names")
        spaces: dict[str, IndexSpace] = {}
        all_ports: set[Port] = set()
        for node in self.nodes:
            for axis, index in enumerate(node.indices):
                previous = spaces.setdefault(index.space.id, index.space)
                if previous != index.space:
                    raise RicciError("SPACE_CONFLICT", f"Conflicting space id {index.space.id}")
                all_ports.add(Port(node.name, axis))
        used: set[Port] = set()
        for edge in self.edges:
            a, b = self.index(edge.left), self.index(edge.right)
            if edge.left == edge.right or edge.left in used or edge.right in used:
                raise RicciError("PORT_REUSED", "A port can belong to only one connection")
            if a.space != b.space:
                raise RicciError("SPACE_MISMATCH", "Contraction requires the same space and basis")
            if a.space.kind is SpaceKind.BATCH:
                raise RicciError("BATCH_REDUCTION", "Batch legs must be retained, not contracted")
            if a.variance != b.variance.dual():
                raise RicciError("VARIANCE_MISMATCH", "Contraction requires opposite variances")
            used.update((edge.left, edge.right))
        groups: list[set[Port]] = []
        for batch in self.batches:
            ports = set(batch.ports)
            if len(ports) < 2 or len(ports) != len(batch.ports) or ports & used:
                raise RicciError("INVALID_BATCH", "Batch group requires distinct, unused ports")
            if len({p.node for p in ports}) != len(ports):
                raise RicciError("INVALID_BATCH", "A batch group has at most one leg per node")
            indices = [self.index(port) for port in batch.ports]
            if any(i.space.kind is not SpaceKind.BATCH for i in indices):
                raise RicciError("INVALID_BATCH", "Batch alignment requires batch spaces")
            if any(i.space != indices[0].space for i in indices):
                raise RicciError("SPACE_MISMATCH", "Batch sizes and identities must match exactly")
            groups.append(ports)
            used.update(ports)
        outputs = set(self.outputs)
        for port in self.outputs:
            self.index(port)
        if len(outputs) != len(self.outputs):
            raise RicciError("INVALID_OUTPUT", "Output ports must be unique")
        free = all_ports - used
        if not free <= outputs:
            raise RicciError("INVALID_OUTPUT", "Every free leg must appear in the output")
        permitted = set(free)
        for group in groups:
            if len(group & outputs) != 1:
                raise RicciError("INVALID_OUTPUT", "Retain exactly one representative per batch")
            permitted.update(group)
        if not outputs <= permitted:
            raise RicciError("INVALID_OUTPUT", "A contracted leg cannot appear in the output")

    @property
    def output_indices(self) -> tuple[Index, ...]:
        return tuple(self.index(port) for port in self.outputs)
