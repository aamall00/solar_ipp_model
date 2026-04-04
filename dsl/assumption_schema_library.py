"""
Shared helpers for loading asset-specific assumption schema YAML files.

Base assumptions come from dsl/assumption_schemas/<asset>.yaml.
User-added block YAMLs may declare an assumption_extensions section to
introduce new assumptions without modifying the base schema file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from engine.asset_registry import get_asset

_ASSUMPTION_SCHEMAS_DIR = Path(__file__).parent / "assumption_schemas"
_BLOCKS_DIR = Path(__file__).parent.parent / "blocks"
_CORE_MODULE_DIR = Path(__file__).parent.parent / "core_module"


def load_assumption_schema_entries(asset_type: str) -> List[Dict[str, Any]]:
    """
    Load assumption schema entries for an asset type.

    Base entries come from dsl/assumption_schemas/<asset>.yaml.
    Extensions declared in assumption_extensions sections of block YAML files
    are merged in, allowing user-added blocks to introduce new assumptions
    without modifying the base schema file.
    """
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

    extensions = _collect_block_extensions(asset_type)
    existing_names = {a["name"] for a in assumptions}
    for name, ext_entry in extensions.items():
        if name not in existing_names:
            assumptions.append(ext_entry)

    return assumptions


def _collect_block_extensions(asset_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Scan block YAML files for assumption_extensions sections.

    Searches blocks/<asset_type>_ipp/ and core_module/. First declaration
    wins if two blocks define the same assumption name.
    """
    block_library = get_asset(asset_type).get("block_library", f"{asset_type}_ipp")
    search_dirs = [
        _BLOCKS_DIR / block_library,
        _CORE_MODULE_DIR,
    ]

    extensions: Dict[str, Dict[str, Any]] = {}
    for directory in search_dirs:
        if not directory.exists():
            continue
        for yaml_file in sorted(directory.glob("*.yaml")):
            with open(yaml_file, "r", encoding="utf-8") as fh:
                block_data = yaml.safe_load(fh) or {}
            for name, meta in block_data.get("assumption_extensions", {}).items():
                if name not in extensions:
                    extensions[name] = {"name": name, **meta}

    return extensions
