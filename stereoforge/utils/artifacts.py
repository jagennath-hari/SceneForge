"""Write JSON metadata for reconstruction outputs."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
