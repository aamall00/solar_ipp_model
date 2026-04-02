"""
engine/asset_registry.py — Registry mapping asset types to block libraries,
assumption contexts, and display metadata.

Adding a new asset type:
  1. Add an entry to ASSET_REGISTRY.
  2. Create blocks/<asset_type>/ with at minimum generation.yaml and revenue.yaml.
  3. Add a canonical assumption schema in assumption_agent.py.
  4. Add the block_library key to DSLParser._resolve_block_import.
"""

from __future__ import annotations

from typing import Dict, Any

ASSET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "solar": {
        "block_library":           "solar_ipp",
        "assumption_schema_file":  "solar_ipp.yaml",
        "solve_loop_file":         "shared_ipp.yaml",
        "assumption_context":      "solar IPP in India",
        "display_name":            "Solar IPP",
        "project_type":            "solar_ipp",
        "capex_per_mw_default":    450.0,   # INR Lakhs/MW
        "opex_per_mw_pa_default":  8.0,     # INR Lakhs/MW/yr
        "cuf_or_cf_key":           "cuf",
        "cuf_or_cf_default":       0.22,
    },
    "wind": {
        "block_library":           "wind_ipp",
        "assumption_schema_file":  "wind_ipp.yaml",
        "solve_loop_file":         "shared_ipp.yaml",
        "assumption_context":      "wind IPP in India",
        "display_name":            "Wind IPP",
        "project_type":            "wind_ipp",
        "capex_per_mw_default":    700.0,   # INR Lakhs/MW
        "opex_per_mw_pa_default":  20.0,    # INR Lakhs/MW/yr
        "cuf_or_cf_key":           "capacity_factor",
        "cuf_or_cf_default":       0.30,
    },
}


def get_asset(asset_type: str) -> Dict[str, Any]:
    """Return registry entry for an asset type, raising KeyError if unknown."""
    asset_type = asset_type.lower().strip()
    if asset_type not in ASSET_REGISTRY:
        raise KeyError(
            f"Unknown asset type '{asset_type}'. "
            f"Supported: {sorted(ASSET_REGISTRY.keys())}"
        )
    return ASSET_REGISTRY[asset_type]


SUPPORTED_ASSET_TYPES = sorted(ASSET_REGISTRY.keys())
