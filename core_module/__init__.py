"""
core_module — Shared CalculationBlock YAML definitions used by all asset types.

These blocks contain no asset-specific logic and are reused by both solar_ipp
and wind_ipp models:
  cashflow, construction, debt_drawdown, debt_service, depreciation,
  dsra, idc, income_statement, opex, revenue_subsidy, waterfall.
"""

from pathlib import Path

_CORE_DIR = Path(__file__).parent

BLOCK_FILES: list[str] = sorted(p.name for p in _CORE_DIR.glob("*.yaml"))
