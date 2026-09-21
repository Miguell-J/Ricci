import json

import pytest
from pydantic import ValidationError

from ricci import BatchGroup, Edge, Graph, Index, IndexSpace, NodeSpec, Port, RicciError, SpaceKind
from ricci.interop import GraphDocument


def document():
    batch = Index("b", IndexSpace("batch", 2, SpaceKind.BATCH))
    component = Index("i", IndexSpace("c", 3, SpaceKind.COMPONENT))
    graph = Graph(
        (NodeSpec("a", (batch, component)), NodeSpec("b", (batch, component))),
        edges=(Edge(Port("a", 1), Port("b", 1)),),
        batches=(BatchGroup((Port("a", 0), Port("b", 0))),),
        outputs=(Port("a", 0),),
    )
    return GraphDocument.from_graph(graph)


def test_json_roundtrip_preserves_topology_and_metadata():
    original = document()
    restored = GraphDocument.model_validate_json(original.model_dump_json())
    assert restored.to_graph() == original.to_graph()
    assert restored.model_dump_json() == original.model_dump_json()
    assert "torch" not in original.model_dump_json()


@pytest.mark.parametrize("mutation", ["version", "unknown", "size", "axis", "missing_node"])
def test_invalid_document_syntax(mutation):
    data = json.loads(document().model_dump_json())
    if mutation == "version":
        data["schema_version"] = "ricci.graph.v2"
    elif mutation == "unknown":
        data["execute_python"] = "print(1)"
    elif mutation == "size":
        data["spaces"][0]["size"] = True
    elif mutation == "axis":
        data["outputs"][0]["axis"] = -1
    else:
        del data["nodes"]
    with pytest.raises(ValidationError):
        GraphDocument.model_validate(data)


def test_invalid_document_semantics():
    data = json.loads(document().model_dump_json())
    data["nodes"][0]["indices"][0]["space"] = "undeclared"
    with pytest.raises(RicciError):
        GraphDocument.model_validate(data).to_graph()
    data = json.loads(document().model_dump_json())
    data["spaces"].append(data["spaces"][0])
    with pytest.raises(RicciError):
        GraphDocument.model_validate(data).to_graph()
