"""
Shared helpers for loading asset-specific assumption schema YAML files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from engine.asset_registry import get_asset

_ASSUMPTION_SCHEMAS_DIR = Path(__file__).parent / "assumption_schemas"


def load_assumption_schema_entries(asset_type: str) -> List[Dict[str, Any]]:
    """Load raw assumption schema entries for an asset type from YAML."""
    asset = get_asset(asset_type)
    schema_file = asset.get("assumption_schema_file")
    if not schema_file:
        raise ValueError(
            f"Asset registry entry for '{asset_type}' is missing "
            f"'assumption_schema_file'"
        )

    path = _ASSUMPTION_SCHEMAS_DIR / schema_file
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    assumptions = raw.get("assumptions", [])
    if not isinstance(assumptions, list):
        raise ValueError(
            f"Assumption schema file '{path}' must contain a top-level "
            f"'assumptions' list"
        )

    return assumptions
