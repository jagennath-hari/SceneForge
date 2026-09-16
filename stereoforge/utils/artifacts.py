"""Publish a complete run without overwriting previous outputs."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@contextmanager
def staged_output(destination: Path) -> Generator[Path, None, None]:
    """Publish by same-filesystem rename; failures clean up only our temporary files."""
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"Output must be new or empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        yield staging
        # rename refuses to replace a nonempty directory, including one filled
        # by another process while this run was executing.
        staging.rename(destination)
