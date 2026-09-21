"""Generate the single authoritative graph JSON Schema, or detect drift."""

import argparse
import json
from pathlib import Path

from ricci.interop import GraphDocument


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "schemas" / "ricci.graph.v1.schema.json"
    content = json.dumps(GraphDocument.model_json_schema(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not path.exists() or path.read_text() != content:
            raise SystemExit("Schema drift: run python scripts/export_schema.py")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


if __name__ == "__main__":
    main()
