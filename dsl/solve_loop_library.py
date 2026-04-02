"""
Shared helpers for loading asset-specific solve-loop YAML files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from engine.asset_registry import get_asset

_SOLVE_LOOPS_DIR = Path(__file__).parent / "solve_loops"


def load_solve_loop_entries(asset_type: str) -> List[Dict[str, Any]]:
    """Load raw solve-loop entries for an asset type from YAML."""
    asset = get_asset(asset_type)
    loop_file = asset.get("solve_loop_file")
    if not loop_file:
        raise ValueError(
            f"Asset registry entry for '{asset_type}' is missing "
            f"'solve_loop_file'"
        )

    path = _SOLVE_LOOPS_DIR / loop_file
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    solve_loops = raw.get("solve_loops", [])
    if not isinstance(solve_loops, list):
        raise ValueError(
            f"Solve-loop file '{path}' must contain a top-level "
            f"'solve_loops' list"
        )

    return solve_loops
