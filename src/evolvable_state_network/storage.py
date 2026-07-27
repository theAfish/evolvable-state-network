"""Conventional application storage paths.

Interactive runs belong to application data, not to an arbitrary dashboard
export directory supplied on every launch.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from platformdirs import user_data_path


def application_data_dir() -> Path:
    override = os.environ.get("ESN_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return user_data_path(
        "EvolvableStateNetwork",
        appauthor="StateNetworkLab",
        ensure_exists=False,
    )


def new_run_directory(category: str, root: Path | None = None) -> Path:
    directory = (root or application_data_dir()) / category / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    return directory
