"""
blocks/solar_ipp — Individual CalculationBlock YAML definitions for the
Karnataka Solar IPP model.

Each YAML file in this directory defines one CalculationBlock that can be
loaded with `load_block(path)` and composed into a full ModelDefinition via
the template in dsl/templates/solar_ipp_base.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

from dsl.types import CalculationBlock

_BLOCK_DIR = Path(__file__).parent

# Block filenames — derived from YAML files on disk so the list stays in sync
# with the block library without manual maintenance.
BLOCK_FILES: list[str] = sorted(p.name for p in _BLOCK_DIR.glob("*.yaml"))


def load_block_raw(filename: str) -> dict:
    """
    Load a single block YAML file and return the raw dict.
    Use this when the block needs to be mutated before assembly (e.g. in BlueprintAgent).

    Parameters
    ----------
    filename : str
        Filename (e.g. "generation.yaml") relative to this package directory,
        or an absolute path string.

    Returns
    -------
    dict
    """
    path = Path(filename)
    if not path.is_absolute():
        path = _BLOCK_DIR / filename
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_block(filename: str) -> CalculationBlock:
    """
    Load a single block YAML file and return a validated CalculationBlock.

    Parameters
    ----------
    filename : str
        Filename (e.g. "generation.yaml") relative to this package directory,
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
