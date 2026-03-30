"""
blocks/wind_ipp — CalculationBlock YAML definitions for wind IPP models.

Wind-specific blocks (generation, revenue) live in this directory.
All other blocks (construction, debt, opex, tax, cashflow, waterfall, returns, etc.)
are shared directly from blocks/solar_ipp — they contain no solar-specific logic.

Block loading order matches the topological evaluation order of the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

from blocks.solar_ipp import BLOCK_FILES as _SOLAR_BLOCK_FILES
from blocks.solar_ipp import _BLOCK_DIR as _SOLAR_BLOCK_DIR
from dsl.types import CalculationBlock

_WIND_BLOCK_DIR = Path(__file__).parent

# Blocks that have wind-specific implementations (live in blocks/wind_ipp/)
_WIND_SPECIFIC = {"generation.yaml", "revenue.yaml"}

# Full ordered list: wind-specific first two, then all solar shared blocks
# (skip solar generation.yaml and revenue.yaml — replaced by wind versions)
BLOCK_FILES = ["generation.yaml", "revenue.yaml"] + [
    f for f in _SOLAR_BLOCK_FILES if f not in _WIND_SPECIFIC
]


def load_block_raw(filename: str) -> dict:
    """
    Load a single wind IPP block YAML file as a raw dict.
    Wind-specific blocks are loaded from this package directory;
    shared blocks are loaded from blocks/solar_ipp/.
    """
    if filename in _WIND_SPECIFIC:
        path = _WIND_BLOCK_DIR / filename
    else:
        path = _SOLAR_BLOCK_DIR / filename
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
