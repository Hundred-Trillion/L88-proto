# L88/src/storage/document_store.py
"""Persistent document store using JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.utils.config import get_settings
from src.utils.logger import ensure_directory


class DocumentStore:
    """Store and load source documents and chunks."""

    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        self.path = path or settings.documents_path
        ensure_directory(self.path.parent)

    def add_documents(self, docs: List[Dict[str, str]]) -> None:
        """Append documents to storage."""
        with self.path.open("a", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    def list_documents(self) -> List[Dict[str, str]]:
        """Read all documents."""
        if not self.path.exists():
            return []
        output: List[Dict[str, str]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    output.append(json.loads(line))
        return output

    def overwrite(self, docs: List[Dict[str, str]]) -> None:
        """Replace content with given docs."""
        with self.path.open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
