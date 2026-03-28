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

# Ordered list of block filenames (defines evaluation order documentation)
BLOCK_FILES = [
    "generation.yaml",
    "revenue.yaml",
    "construction.yaml",
    "idc.yaml",
    "debt_sizing.yaml",
    "debt_drawdown.yaml",
    "debt_service.yaml",
    "opex.yaml",
    "dsra.yaml",
    "depreciation.yaml",
    "tax.yaml",
    "cashflow.yaml",
    "waterfall.yaml",
    "returns.yaml",
]


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
    path = Path(filename)
    if not path.is_absolute():
        path = _BLOCK_DIR / filename
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return CalculationBlock.model_validate(data)


def load_all_blocks() -> Dict[str, CalculationBlock]:
    """
    Load all 14 solar IPP block YAML files.

    Returns
    -------
    dict mapping block_id → CalculationBlock
    """
    blocks: Dict[str, CalculationBlock] = {}
    for fname in BLOCK_FILES:
        block = load_block(fname)
        blocks[block.block_id] = block
    return blocks
