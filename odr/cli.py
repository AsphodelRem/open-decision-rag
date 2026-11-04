"""Offline build and saved-index search CLI using the SDK."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from odr.corpus.facade import CorpusHierarchy
from odr.corpus.ingestion import load_documents
from odr.evaluation.metrics import evaluate
from odr.models.mock import MockDecisionModel
from odr.schema import SearchArguments, StorageArguments
from odr.search_engine.engine import DecisionSearchEngine


def main() -> None:
    """Run explicit offline commands without a cloud fallback."""
    parser = argparse.ArgumentParser(
        description="DecisionRAG offline CLI (deterministic mock scoring)"
    )
    parser.add_argument("command", choices=("build", "load", "search", "evaluate"))
    parser.add_argument("--index", required=True)
    parser.add_argument("--input")
    parser.add_argument("--query")
    parser.add_argument("--cutoff", type=int, default=10)
    asyncio.run(execute(parser.parse_args()))


async def execute(arguments: argparse.Namespace) -> None:
    """Construct the same facade and engine used by Python callers.

    :param arguments: Parsed CLI arguments.
    """
    if arguments.command == "build":
        hierarchy = CorpusHierarchy(
            storage_settings=StorageArguments(storage_folder=arguments.index)
        )
        await hierarchy.build_from_docs(load_documents(arguments.input))
        assert hierarchy.index is not None
        emit(json.dumps({"revision": hierarchy.index.revision}))
        return
    hierarchy = await CorpusHierarchy.open(arguments.index)
    if arguments.command == "load":
        assert hierarchy.index is not None
        emit(json.dumps({"revision": hierarchy.index.revision}))
        return
    engine = DecisionSearchEngine(
        hierarchy,
        MockDecisionModel(),
        search_settings=SearchArguments(
            return_top_k=arguments.cutoff if arguments.command == "evaluate" else 5
        ),
    )
    if arguments.command == "evaluate":
        records = json.loads(Path(arguments.input).read_text())
        emit(json.dumps(await evaluate(engine, records, cutoff=arguments.cutoff)))
    else:
        emit((await engine.search(arguments.query)).model_dump_json())


def emit(message: str) -> None:
    """Write one machine-readable CLI record.

    :param message: One machine-readable output record.
    """
    sys.stdout.write(f"{message}\n")


if __name__ == "__main__":
    main()
