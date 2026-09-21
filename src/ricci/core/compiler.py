"""Lower validated diagrams to Einstein notation with explicit output order."""

from dataclasses import dataclass
from string import ascii_letters

from ricci.core.errors import RicciError
from ricci.core.graph import Graph, Port


@dataclass(frozen=True, slots=True)
class ContractionIR:
    equation: str
    input_names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]


def lower(graph: Graph) -> ContractionIR:
    """Names of indices never participate in symbol assignment."""
    graph.validate()
    representative: dict[Port, Port] = {}
    for edge in graph.edges:
        representative[edge.right] = edge.left
    for batch in graph.batches:
        for port in batch.ports:
            representative[port] = batch.ports[0]
    symbols: dict[Port, str] = {}

    def symbol(port: Port) -> str:
        root = representative.get(port, port)
        if root not in symbols:
            if len(symbols) == len(ascii_letters):
                raise RicciError(
                    "UNSUPPORTED_OPERATION", "The v0.1 executor supports at most 52 wire labels"
                )
            symbols[root] = ascii_letters[len(symbols)]
        return symbols[root]

    inputs = [
        "".join(symbol(Port(node.name, axis)) for axis in range(len(node.indices)))
        for node in graph.nodes
    ]
    output = "".join(symbol(port) for port in graph.outputs)
    return ContractionIR(
        ",".join(inputs) + "->" + output,
        tuple(node.name for node in graph.nodes),
        tuple(node.shape for node in graph.nodes),
        tuple(index.size for index in graph.output_indices),
    )
