# L88/scripts/ingest.py
"""Bulk ingestion script for local files."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.api.deps import get_chunker, get_document_store, get_retrieval_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest text files into local index")
    parser.add_argument("input_dir", type=Path, help="Directory with .txt files")
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.txt"))
    docs = [{"id": f.stem, "text": f.read_text(encoding="utf-8")} for f in files]

    store = get_document_store()
    chunker = get_chunker()
    engine = get_retrieval_engine()

    store.overwrite(docs)
    chunks = []
    for doc in docs:
        chunks.extend(chunker.chunk(doc["text"]))
    engine.build_indexes(chunks)

    print(f"Ingested {len(docs)} docs and {len(chunks)} chunks")


if __name__ == "__main__":
    main()
