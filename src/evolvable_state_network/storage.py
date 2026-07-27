"""Project-local storage paths for generated experiment artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

def application_data_dir() -> Path:
    override = os.environ.get("ESN_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    # Keep generated runs beside the working project by default.  This makes
    # results inspectable and removable without writing into a user-profile
    # application-data directory.
    return (Path.cwd() / ".outputs").resolve()


def new_run_directory(category: str, root: Path | None = None) -> Path:
    directory = (root or application_data_dir()) / category / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    return directory
