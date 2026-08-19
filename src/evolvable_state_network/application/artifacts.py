"""Small, reusable persistence primitives for application artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from time import sleep
from typing import Mapping
from uuid import uuid4


def write_json_atomically(
    path: Path,
    document: Mapping[str, object],
    *,
    lock: Lock,
    replace_attempts: int = 6,
) -> None:
    """Serialize and atomically replace a JSON artifact under a shared lock.

    A unique sibling temporary file prevents concurrent writers from sharing
    partial output. The bounded retry handles transient Windows file locks from
    virus scanners and readers without weakening the atomic replacement.
    """
    temporary = path.with_name(f"{path.stem}.{uuid4().hex}.json.tmp")
    with lock:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
        )
        try:
            # Confirm that the serialized artifact can be read before replacing
            # a previously valid document.
            json.loads(temporary.read_text(encoding="utf-8"))
            for attempt in range(replace_attempts):
                try:
                    temporary.replace(path)
                    break
                except PermissionError:
                    if attempt == replace_attempts - 1:
                        raise
                    sleep(0.05 * (attempt + 1))
        finally:
            temporary.unlink(missing_ok=True)
