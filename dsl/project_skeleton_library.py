"""
Shared helpers for loading asset-specific project skeleton YAML files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from engine.asset_registry import get_asset

_PROJECT_SKELETONS_DIR = Path(__file__).parent / "project_skeletons"


def load_project_skeleton_template(asset_type: str) -> Dict[str, Any]:
    """Load raw project skeleton template data for an asset type from YAML."""
    asset = get_asset(asset_type)
    skeleton_file = asset.get("project_skeleton_file")
    if not skeleton_file:
        raise ValueError(
            f"Asset registry entry for '{asset_type}' is missing "
            f"'project_skeleton_file'"
        )

    path = _PROJECT_SKELETONS_DIR / skeleton_file
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    skeleton = raw.get("project_skeleton")
    if not isinstance(skeleton, dict):
        raise ValueError(
            f"Project skeleton file '{path}' must contain a top-level "
            f"'project_skeleton' mapping"
        )

    return skeleton
