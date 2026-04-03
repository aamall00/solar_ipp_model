"""
blocks/solar_ipp — Solar-specific CalculationBlock YAML definitions.

Solar-specific blocks (generation, revenue) live in this directory.
All shared blocks (construction, debt, opex, tax, cashflow, waterfall, etc.)
live in core_module/ and are loaded from there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

from dsl.types import CalculationBlock

_BLOCK_DIR = Path(__file__).parent
_CORE_DIR = _BLOCK_DIR.parent.parent / "core_module"

# Solar-specific block filenames (generation.yaml, revenue.yaml)
_SOLAR_FILES: frozenset[str] = frozenset(p.name for p in _BLOCK_DIR.glob("*.yaml"))

# All block filenames: solar-specific + shared core
BLOCK_FILES: list[str] = sorted(_SOLAR_FILES) + sorted(
    p.name for p in _CORE_DIR.glob("*.yaml")
)


def load_block_raw(filename: str) -> dict:
    """
    Load a single block YAML file and return the raw dict.
    Solar-specific blocks are loaded from this directory; shared blocks
    are loaded from core_module/.

    Parameters
    ----------
    filename : str
        Filename (e.g. "generation.yaml") relative to the block directories,
        or an absolute path string.

    Returns
    -------
    dict
    """
    path = Path(filename)
    if path.is_absolute():
        pass
    elif filename in _SOLAR_FILES:
        path = _BLOCK_DIR / filename
    else:
        path = _CORE_DIR / filename
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_block(filename: str) -> CalculationBlock:
    """
    Load a single block YAML file and return a validated CalculationBlock.

    Parameters
    ----------
    filename : str
        Filename (e.g. "generation.yaml") relative to the block directories,
        or an absolute path string.

    Returns
    -------
    CalculationBlock
    """
    return CalculationBlock.model_validate(load_block_raw(filename))


def load_all_blocks_raw() -> Dict[str, dict]:
    """
    Load all solar IPP block YAML files as raw dicts.
    Use this when blocks need to be mutated before assembly (e.g. in BlueprintAgent).

    Returns
    -------
    dict mapping block_id → raw block dict, in BLOCK_FILES order
    """
    blocks: Dict[str, dict] = {}
    for fname in BLOCK_FILES:
        block = load_block_raw(fname)
        blocks[block["block_id"]] = block
    return blocks


def load_all_blocks() -> Dict[str, CalculationBlock]:
    """
    Load all solar IPP block YAML files.

    Returns
    -------
    dict mapping block_id → CalculationBlock
    """
    blocks: Dict[str, CalculationBlock] = {}
    for fname in BLOCK_FILES:
        block = load_block(fname)
        blocks[block.block_id] = block
    return blocks
