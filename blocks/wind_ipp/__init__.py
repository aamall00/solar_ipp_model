"""
blocks/wind_ipp — CalculationBlock YAML definitions for wind IPP models.

Wind-specific blocks (generation, revenue) live in this directory.
All shared blocks (construction, debt, opex, tax, cashflow, waterfall, etc.)
are loaded from core_module/ — they contain no asset-specific logic.

Block loading order matches the topological evaluation order of the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

from dsl.types import CalculationBlock

_WIND_BLOCK_DIR = Path(__file__).parent
_CORE_DIR = _WIND_BLOCK_DIR.parent.parent / "core_module"

# Blocks that have wind-specific implementations (live in blocks/wind_ipp/)
_WIND_SPECIFIC: frozenset[str] = frozenset({"generation.yaml", "revenue.yaml"})

# Full ordered list: wind-specific first, then all shared core blocks
BLOCK_FILES: list[str] = sorted(_WIND_SPECIFIC) + sorted(
    p.name for p in _CORE_DIR.glob("*.yaml")
)


def load_block_raw(filename: str) -> dict:
    """
    Load a single wind IPP block YAML file as a raw dict.
    Wind-specific blocks are loaded from this directory;
    shared blocks are loaded from core_module/.
    """
    if filename in _WIND_SPECIFIC:
        path = _WIND_BLOCK_DIR / filename
    else:
        path = _CORE_DIR / filename
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_block(filename: str) -> CalculationBlock:
    """Load a single block YAML and return a validated CalculationBlock."""
    return CalculationBlock.model_validate(load_block_raw(filename))


def load_all_blocks_raw() -> Dict[str, dict]:
    """
    Load all wind IPP blocks as raw dicts, in topological order.
    Returns dict mapping block_id → raw block dict.
    """
    blocks: Dict[str, dict] = {}
    for fname in BLOCK_FILES:
        block = load_block_raw(fname)
        blocks[block["block_id"]] = block
    return blocks


def load_all_blocks() -> Dict[str, CalculationBlock]:
    """Load all wind IPP blocks as validated CalculationBlock objects."""
    blocks: Dict[str, CalculationBlock] = {}
    for fname in BLOCK_FILES:
        block = load_block(fname)
        blocks[block.block_id] = block
    return blocks
