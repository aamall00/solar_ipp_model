"""
agents/assumption_agent.py — Layer 3: Assumption Ingestion Agent.

Two-stage pipeline
------------------
1. AI stage (Claude tool_use)
   Sends the user's free-form project description to Claude together with a
   ``set_assumptions`` tool.  Claude calls the tool once, populating every
   assumption it can identify, and calls ``note_inferences`` for values it had
   to infer rather than read explicitly.

2. Python validation stage (no AI)
   a. Apply schema defaults for any assumption not touched by Claude.
   b. Derive cross-assumptions (construction_periods, operations_periods,
      debt_maturity) from the filled values.
   c. Constraint checks  (min/max bounds, enum membership).
   d. Cross-consistency checks (debt_tenor <= PPA_tenor, capex_schedule sums
      to 1.0, moratorium < debt tenor, interest_rate > 0, etc.).

Returns IngestionResult — a typed container with:
  filled_assumptions   dict[str, Any]  — all 20+ canonical assumptions
  missing_required     list[str]       — assumptions with no value and no default
  inferred_assumptions dict[str, InferredAssumption]  — AI-derived values + notes
  validation           ValidationResult — errors + warnings from Python checks

Usage
-----
    agent = AssumptionAgent()
    result = agent.extract(
        "100 MW solar plant near Tumkur, Karnataka.  "
        "CUF 21%, ₹2.65/kWh PPA for 25 years, 70% debt at 9.75%, "
        "capex ₹430 Lakh/MW, 4-quarter construction."
    )
    if result.validation.valid:
        model_def, _ = BlueprintAgent().generate(result)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import anthropic

from dsl.types import ValidationResult, ValidationWarning

# ---------------------------------------------------------------------------
# Canonical assumption schema
# ---------------------------------------------------------------------------
# Each entry:
#   type        : "scalar" | "schedule"
#   unit        : human-readable unit string
#   default     : default value (None = required, must be supplied)
#   aliases     : alternative names Claude might use
#   constraints : min/max bounds (None = unbounded)
#   sensitivity : sensitivity analysis config
#   description : shown to Claude in the tool schema

_CANONICAL: Dict[str, Dict[str, Any]] = {
    # ---- Generation ----
    "capacity_mw": {
        "type": "scalar", "unit": "MW",
        "default": None,  # required
        "aliases": ["installed_capacity_mw", "plant_capacity", "dc_capacity_mw",
                    "ac_capacity_mw", "capacity", "size_mw", "mw"],
        "constraints": {"min": 1.0, "max": 5000.0},
        "description": "Installed AC capacity in megawatts",
    },
    "cuf": {
        "type": "scalar", "unit": "ratio (decimal)",
        "default": 0.22,
        "aliases": ["capacity_utilisation_factor", "capacity_factor", "cuf_p50",
                    "plant_load_factor", "plf", "yield", "annual_yield"],
        "constraints": {"min": 0.10, "max": 0.40},
        "sensitivity": {"vary": True, "range_pct": 0.10, "distribution": "triangular"},
        "description": "P50 capacity utilisation factor as a decimal (e.g. 0.22 for 22%)",
    },
    "degradation_rate": {
        "type": "scalar", "unit": "per_year (decimal)",
        "default": 0.005,
        "aliases": ["annual_degradation", "panel_degradation", "module_degradation"],
        "constraints": {"min": 0.0, "max": 0.02},
        "description": "Annual generation degradation rate as a decimal (e.g. 0.005 for 0.5%/yr)",
    },
    "auxiliary_consumption": {
        "type": "scalar", "unit": "ratio (decimal)",
        "default": 0.005,
        "aliases": ["aux_consumption", "auxiliary_losses", "transformer_losses",
                    "internal_consumption"],
        "constraints": {"min": 0.0, "max": 0.05},
        "description": "Fraction of gross generation consumed internally",
    },
    # ---- Revenue ----
    "tariff": {
        "type": "scalar", "unit": "INR per kWh",
        "default": None,  # required
        "aliases": ["ppa_tariff", "tariff_per_kwh", "ppa_rate", "feed_in_tariff",
                    "fit", "contracted_tariff", "levelised_tariff"],
        "constraints": {"min": 0.50, "max": 15.0},
        "sensitivity": {"vary": True, "range_pct": 0.10, "distribution": "triangular"},
        "description": "PPA tariff in INR per kWh",
    },
    "tariff_escalation": {
        "type": "scalar", "unit": "per_year (decimal)",
        "default": 0.0,
        "aliases": ["ppa_escalation", "tariff_escalation_rate", "annual_tariff_increase"],
        "constraints": {"min": 0.0, "max": 0.10},
        "description": "Annual PPA tariff escalation rate as a decimal (0 = fixed tariff)",
    },
    "ppa_tenor_years": {
        "type": "scalar", "unit": "years",
        "default": 25.0,
        "aliases": ["ppa_duration", "ppa_life", "contract_duration", "offtake_term"],
        "constraints": {"min": 10.0, "max": 35.0},
        "description": "Duration of the PPA in years (determines operations_periods)",
    },
    # ---- Capex ----
    "capex_per_mw": {
        "type": "scalar", "unit": "INR Lakhs per MW",
        "default": None,  # required
        "aliases": ["capex_per_mw_lakh", "specific_capex", "unit_cost",
                    "epc_cost_per_mw", "cost_per_mw"],
        "constraints": {"min": 150.0, "max": 1000.0},
        "sensitivity": {"vary": True, "range_pct": 0.10, "distribution": "triangular"},
        "description": (
            "Total project capex per MW in INR Lakhs/MW. "
            "If user gives total_capex in Lakhs, divide by capacity_mw. "
            "If user gives total_capex in Crores, convert: 1 Cr = 100 Lakh."
        ),
    },
    "capex_schedule": {
        "type": "schedule", "unit": "fractions summing to 1.0",
        "default": None,  # derived from construction_periods if not given
        "aliases": ["construction_drawdown", "capex_drawdown_schedule",
                    "construction_schedule", "drawdown_schedule"],
        "description": (
            "List of fractions per construction quarter summing to 1.0. "
            "E.g. [0.25, 0.25, 0.25, 0.25] for 4 equal quarterly draws. "
            "Length determines construction_periods."
        ),
    },
    # ---- O&M ----
    "opex_per_mw_pa": {
        "type": "scalar", "unit": "INR Lakhs per MW per year",
        "default": 8.0,
        "aliases": ["om_cost_per_mw", "annual_opex_per_mw", "maintenance_cost",
                    "opex_lakh_per_mw", "amc_per_mw"],
        "constraints": {"min": 1.0, "max": 50.0},
        "sensitivity": {"vary": True, "range_pct": 0.20, "distribution": "triangular"},
        "description": "Annual O&M cost per MW in INR Lakhs",
    },
    "opex_escalation": {
        "type": "scalar", "unit": "per_year (decimal)",
        "default": 0.03,
        "aliases": ["om_escalation", "opex_inflation", "annual_opex_increase"],
        "constraints": {"min": 0.0, "max": 0.10},
        "description": "Annual O&M cost escalation rate as a decimal",
    },
    # ---- Debt ----
    "debt_pct": {
        "type": "scalar", "unit": "ratio (decimal)",
        "default": 0.70,
        "aliases": ["debt_percent", "leverage", "debt_equity_ratio",
                    "loan_to_cost", "ltc", "debt_fraction"],
        "constraints": {"min": 0.30, "max": 0.85},
        "description": (
            "Senior debt as a fraction of total project cost (e.g. 0.70 for 70:30 D:E). "
            "If user states D:E as 70:30 this is 0.70."
        ),
    },
    "interest_rate": {
        "type": "scalar", "unit": "per_year (decimal)",
        "default": None,  # required
        "aliases": ["loan_interest_rate", "coupon_rate", "borrowing_rate",
                    "interest_pa", "cost_of_debt"],
        "constraints": {"min": 0.04, "max": 0.25},
        "sensitivity": {"vary": True, "range_pct": 0.15, "distribution": "triangular"},
        "description": "Annual interest rate on senior debt as a decimal (e.g. 0.0975 for 9.75%)",
    },
    "debt_tenor_years": {
        "type": "scalar", "unit": "years",
        "default": 18.0,
        "aliases": ["loan_tenor", "debt_maturity_years", "repayment_period",
                    "loan_life", "loan_term"],
        "constraints": {"min": 5.0, "max": 25.0},
        "description": "Debt tenor in years from COD (determines debt_maturity milestone)",
    },
    "moratorium_periods": {
        "type": "scalar", "unit": "quarters",
        "default": 0.0,
        "aliases": ["repayment_holiday", "grace_periods", "principal_holiday",
                    "moratorium_quarters"],
        "constraints": {"min": 0.0, "max": 8.0},
        "description": "Number of quarterly periods after COD before principal repayment begins",
    },
    "dscr_target": {
        "type": "scalar", "unit": "ratio (×)",
        "default": 1.20,
        "aliases": ["minimum_dscr", "dscr_covenant", "target_dscr",
                    "lender_dscr", "sculpting_target"],
        "constraints": {"min": 1.0, "max": 2.5},
        "description": "Target DSCR for debt sculpting; also minimum covenant threshold",
    },
    "debt_sizing_mode": {
        "type": "scalar", "unit": "flag (0.0 or 1.0)",
        "default": 0.0,
        "aliases": ["sculpting_mode", "debt_structure"],
        "constraints": {"min": 0.0, "max": 1.0},
        "description": (
            "0.0 = cost-based (equal principal), 1.0 = CFADS-based sculpted. "
            "Set to 1.0 if user mentions 'sculpted debt service' or 'DSCR-shaped repayment'."
        ),
    },
    "dsra_months": {
        "type": "scalar", "unit": "months",
        "default": 6.0,
        "aliases": ["dsra", "debt_service_reserve", "reserve_months",
                    "dsra_cover", "debt_reserve_months"],
        "constraints": {"min": 0.0, "max": 12.0},
        "description": "DSRA target expressed as number of months of forward debt service",
    },
    "cash_sweep_rate": {
        "type": "scalar", "unit": "ratio (decimal)",
        "default": 0.0,
        "aliases": ["sweep_rate", "cash_sweep", "excess_cash_sweep",
                    "debt_sweep_rate", "mandatory_prepayment_rate"],
        "constraints": {"min": 0.0, "max": 1.0},
        "description": (
            "Fraction of post-DSRA excess cash swept to accelerate debt repayment. "
            "0.0 = no sweep (all residual to equity); 1.0 = full sweep until debt retired."
        ),
    },
    # ---- Tax & Depreciation ----
    "tax_rate": {
        "type": "scalar", "unit": "ratio (decimal)",
        "default": 0.25,
        "aliases": ["corporate_tax", "income_tax_rate", "effective_tax_rate",
                    "tax_percent"],
        "constraints": {"min": 0.0, "max": 0.40},
        "description": "Corporate income tax rate as a decimal (e.g. 0.25 for 25%)",
    },
    "depreciation_rate": {
        "type": "scalar", "unit": "per_year (decimal)",
        "default": 0.05,
        "aliases": ["slm_rate", "depreciation_percent", "book_depreciation",
                    "useful_life_years"],  # if years given, convert: 1/years
        "constraints": {"min": 0.02, "max": 0.40},
        "description": (
            "SLM depreciation rate as a decimal (e.g. 0.05 for 5%/yr = 20-yr life). "
            "If user gives useful life in years, convert: rate = 1/life."
        ),
    },
    # ---- IDC capitalisation ----
    "idc_capitalised": {
        "type": "scalar", "unit": "flag (0.0 or 1.0)",
        "default": 0.0,
        "aliases": ["capitalise_idc", "idc_into_debt", "idc_cap"],
        "constraints": {"min": 0.0, "max": 1.0},
        "description": (
            "0.0 = IDC funded by equity (default). "
            "1.0 = IDC capitalised into the debt base (adds a goal_seek solve loop). "
            "Set to 1.0 if user mentions 'capitalised IDC' or 'IDC into debt'."
        ),
    },
    # ---- Equity ----
    "equity_irr_target": {
        "type": "scalar", "unit": "per_year (decimal)",
        "default": 0.14,
        "aliases": ["hurdle_rate", "equity_hurdle", "required_irr",
                    "minimum_equity_irr", "irr_target"],
        "constraints": {"min": 0.08, "max": 0.35},
        "description": "Equity IRR hurdle rate used for NPV discounting and bankability tests",
    },
    # ---- O&M sub-components ----
    "insurance_percent_of_capex": {
        "type": "scalar", "unit": "ratio (decimal, e.g. 0.005 for 0.5% p.a.)",
        "default": 0.005,
        "aliases": ["insurance_rate", "insurance_percent", "property_insurance",
                    "insurance_of_capex", "annual_insurance"],
        "constraints": {"min": 0.0, "max": 0.02},
        "description": (
            "Annual insurance premium as a fraction of total project capex "
            "(e.g. 0.005 for 0.5%/yr — typical for Indian solar IPP)."
        ),
    },
    "land_lease_lakhs_pa": {
        "type": "scalar", "unit": "INR Lakhs per year",
        "default": 0.0,
        "aliases": ["land_lease", "land_rent", "ground_rent", "land_cost_pa",
                    "annual_land_lease", "lease_rentals"],
        "constraints": {"min": 0.0, "max": 500.0},
        "description": (
            "Annual land lease payment in INR Lakhs. "
            "Use 0 if land is owned outright (common for large Karnataka projects)."
        ),
    },
    "maintenance_capex_pct": {
        "type": "scalar", "unit": "ratio (decimal, e.g. 0.005 for 0.5% p.a.)",
        "default": 0.0,
        "aliases": ["capex_during_ops", "capex_during_ops_pct", "maintenance_capex",
                    "major_maintenance_capex", "replacement_capex_pct",
                    "inverter_replacement_pct", "capex_maintenance_rate"],
        "constraints": {"min": 0.0, "max": 0.10},
        "description": (
            "Annual capital expenditure during operations as a fraction of total project capex "
            "(e.g. 0.005 for 0.5%/yr to provision for inverter and component replacement). "
            "Deducted from CFADS: CFADS = EBITDA − Tax − capex_during_ops."
        ),
    },
    # ---- Depreciation method ----
    "depreciation_method": {
        "type": "enum", "unit": "string (slm or wdv)",
        "default": "slm",
        "aliases": ["dep_method", "depreciation_type", "book_dep_method",
                    "depreciation_basis"],
        "constraints": {},
        "enum_values": ["slm", "wdv"],
        "description": (
            "Depreciation method for P&L and tax: "
            "'slm' = straight-line method (default, 5% p.a. over 20 yrs), "
            "'wdv' = written-down value / declining balance (40% p.a. — Indian solar benefit)."
        ),
    },
    "wdv_rate": {
        "type": "scalar", "unit": "per_year (decimal)",
        "default": 0.40,
        "aliases": ["wdv_depreciation_rate", "written_down_value_rate", "block_rate",
                    "declining_balance_rate", "wdv_percent"],
        "constraints": {"min": 0.05, "max": 0.80},
        "description": (
            "Annual WDV depreciation rate as a decimal "
            "(e.g. 0.40 for 40% p.a. — standard Indian Income Tax Act rate for solar plants). "
            "Only used when depreciation_method = 'wdv'."
        ),
    },
    "use_wdv": {
        "type": "scalar", "unit": "flag (0.0 or 1.0)",
        "default": 0.0,
        "aliases": [],
        "constraints": {"min": 0.0, "max": 1.0},
        "description": (
            "Numeric flag derived from depreciation_method: 0.0 = SLM, 1.0 = WDV. "
            "Set automatically by _derive_cross_assumptions; do not set directly."
        ),
    },
}

_REQUIRED_ASSUMPTIONS = [k for k, v in _CANONICAL.items() if v["default"] is None]

# ---------------------------------------------------------------------------
# Wind IPP canonical assumption schema
# ---------------------------------------------------------------------------
# Replaces cuf + degradation_rate with capacity_factor + availability_factor.
# All financial assumptions (debt, opex, tax, etc.) are shared with solar.

_WIND_CANONICAL: Dict[str, Dict[str, Any]] = {
    # ---- Generation (wind-specific) ----
    "capacity_mw": _CANONICAL["capacity_mw"],
    "capacity_factor": {
        "type": "scalar", "unit": "ratio (decimal)",
        "default": 0.30,
        "aliases": ["cf", "wind_cf", "wind_capacity_factor", "annual_yield",
                    "plant_load_factor", "plf", "cuf"],
        "constraints": {"min": 0.15, "max": 0.55},
        "sensitivity": {"vary": True, "range_pct": 0.10, "distribution": "triangular"},
        "description": "P50 annual capacity factor as a decimal (e.g. 0.30 for 30%)",
    },
    "availability_factor": {
        "type": "scalar", "unit": "ratio (decimal)",
        "default": 0.95,
        "aliases": ["turbine_availability", "machine_availability", "availability",
                    "pa", "plant_availability"],
        "constraints": {"min": 0.80, "max": 0.99},
        "description": "Annual turbine availability factor as a decimal (e.g. 0.95 for 95%)",
    },
    "auxiliary_consumption": _CANONICAL["auxiliary_consumption"],
    # ---- Revenue ----
    "tariff":             _CANONICAL["tariff"],
    "tariff_escalation":  _CANONICAL["tariff_escalation"],
    "ppa_tenor_years":    _CANONICAL["ppa_tenor_years"],
    # ---- Capex (wind defaults differ) ----
    "capex_per_mw": {
        **_CANONICAL["capex_per_mw"],
        "default": 700.0,
        "constraints": {"min": 300.0, "max": 1500.0},
        "description": "Total project capex per MW in INR Lakhs/MW (wind: typically 600–900 Lakh/MW)",
    },
    "capex_schedule": _CANONICAL["capex_schedule"],
    # ---- O&M (wind defaults differ) ----
    "opex_per_mw_pa": {
        **_CANONICAL["opex_per_mw_pa"],
        "default": 20.0,
        "description": "Annual O&M cost per MW in INR Lakhs (wind: typically 15–30 Lakh/MW/yr)",
    },
    "opex_escalation":          _CANONICAL["opex_escalation"],
    # ---- Debt ----
    "debt_pct":                 _CANONICAL["debt_pct"],
    "interest_rate":            _CANONICAL["interest_rate"],
    "debt_tenor_years":         _CANONICAL["debt_tenor_years"],
    "moratorium_periods":       _CANONICAL["moratorium_periods"],
    "dscr_target":              _CANONICAL["dscr_target"],
    "debt_sizing_mode":         _CANONICAL["debt_sizing_mode"],
    "dsra_months":              _CANONICAL["dsra_months"],
    "cash_sweep_rate":          _CANONICAL["cash_sweep_rate"],
    # ---- Tax & Depreciation ----
    "tax_rate":                 _CANONICAL["tax_rate"],
    "depreciation_rate":        _CANONICAL["depreciation_rate"],
    # ---- O&M sub-components ----
    "insurance_percent_of_capex": _CANONICAL["insurance_percent_of_capex"],
    "land_lease_lakhs_pa":        _CANONICAL["land_lease_lakhs_pa"],
    "maintenance_capex_pct":      _CANONICAL["maintenance_capex_pct"],
    # ---- Depreciation method ----
    "depreciation_method":      _CANONICAL["depreciation_method"],
    "wdv_rate":                 _CANONICAL["wdv_rate"],
    "use_wdv":                  _CANONICAL["use_wdv"],
    # ---- IDC capitalisation ----
    "idc_capitalised":          _CANONICAL["idc_capitalised"],
    # ---- Equity ----
    "equity_irr_target":        _CANONICAL["equity_irr_target"],
}

_WIND_REQUIRED_ASSUMPTIONS = [k for k, v in _WIND_CANONICAL.items() if v["default"] is None]

# Map asset_type → (canonical dict, required list)
_SCHEMA_BY_ASSET: Dict[str, tuple] = {
    "solar": (_CANONICAL, _REQUIRED_ASSUMPTIONS),
    "wind":  (_WIND_CANONICAL, _WIND_REQUIRED_ASSUMPTIONS),
}

# ---------------------------------------------------------------------------
# Assumption contract — the explicit bridge between Layer 3 and Layer 4
# ---------------------------------------------------------------------------
#
# CANONICAL_NAMES is the authoritative set of assumption keys produced by this
# agent.  Every name here MUST appear as an assumption_schema entry in the YAML
# template consumed by the executor (DSLParser._check_assumption_sources enforces
# this at parse time).
#
# Conversely, every assumption.X source in any YAML block input MUST have a
# corresponding entry here so the ingestion agent can populate it.
#
# The two directions are checked by audit_template_coverage() below.
#
CANONICAL_NAMES: frozenset = frozenset(_CANONICAL.keys())


def audit_template_coverage(template_schema_names: set) -> dict:
    """
    Cross-check the agent's canonical assumption keys against the names declared
    in a YAML model's assumption_schema.

    Parameters
    ----------
    template_schema_names : set
        Set of assumption names from the YAML template's assumption_schema section
        (e.g. {a.name for a in model.assumption_schema.assumptions}).

    Returns
    -------
    dict with two keys:
        "agent_not_in_template"   : canonical keys the agent knows about but the
                                    template does not declare — executor will
                                    silently use 0 for these if ever referenced.
        "template_not_in_agent"   : template schema names the agent does not know
                                    about — user text cannot populate these; they
                                    must rely on template defaults.
    """
    return {
        "agent_not_in_template": sorted(CANONICAL_NAMES - template_schema_names),
        "template_not_in_agent": sorted(template_schema_names - CANONICAL_NAMES),
    }


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class InferredAssumption:
    """An assumption value inferred by the AI with a derivation note."""
    value: Any
    derivation_note: str


@dataclass
class IngestionResult:
    """
    Output of AssumptionAgent.extract().

    filled_assumptions   : All canonical assumptions including defaults.
                           Safe to pass directly to BlueprintAgent.generate().
    missing_required     : Names of required assumptions with no value and no
                           default — model cannot run until these are supplied.
    inferred_assumptions : Assumptions whose values Claude inferred rather than
                           read directly from the input text.
    validation           : Errors (blocking) and warnings (advisory).
    """
    filled_assumptions: Dict[str, Any]
    missing_required: List[str]
    inferred_assumptions: Dict[str, InferredAssumption]
    validation: ValidationResult


_NOTE_INFERENCES_TOOL: Dict[str, Any] = {
    "name": "note_inferences",
    "description": (
        "Record assumptions you inferred or derived rather than reading directly "
        "from the user's text.  Provide a brief derivation note so the user can "
        "review and correct each inference."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "inferences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "assumption": {"type": "string"},
                        "value": {},
                        "derivation_note": {"type": "string"},
                    },
                    "required": ["assumption", "value", "derivation_note"],
                },
            }
        },
        "required": ["inferences"],
    },
}

_SYSTEM_PROMPT = f"""You are a project finance analyst specialising in solar IPP projects in India.
Your task is to extract all financial and technical assumptions from the user's project description
and map them to the canonical solar IPP assumption schema.

Rules:
1. Always call set_assumptions first with every value you can determine.
2. Optionally call note_inferences for values you had to derive or calculate (e.g. capex_per_mw
   computed from total_capex ÷ capacity_mw, or debt_sizing_mode inferred from "sculpted debt").
3. All rates and ratios in DECIMAL form (e.g. 9.75% → 0.0975, 70% → 0.70).
4. capex_per_mw in INR Lakhs/MW. If the user gives total capex in Crores, convert: 1 Cr = 100 Lakh.
5. capex_schedule: list of fractions summing to 1.0 per construction quarter.
   If user says "4 quarters" without a schedule, infer [0.25, 0.25, 0.25, 0.25].
   If user says "3 quarters" without a schedule, infer [0.30, 0.40, 0.30].
6. ppa_tenor_years: extract from "25-year PPA", "25 yr PPA" etc.
7. debt_tenor_years: extract from "18-year loan", "18 yr tenor" etc.
8. debt_sizing_mode: set to 1.0 only if user explicitly mentions sculpted/DSCR-shaped repayment.
9. Karnataka-specific defaults you may apply if not stated: CUF ≈ 0.22, tariff ≈ 2.65 INR/kWh.
10. Do NOT hallucinate values. If you are unsure, omit the assumption.
11. insurance_percent_of_capex: annual insurance as a fraction of capex (e.g. 0.005 for 0.5% p.a.).
    Extract from phrases like "0.5% insurance on capex" or "property insurance 0.5%".
12. land_lease_lakhs_pa: annual land lease in INR Lakhs. Use 0 if user says land is owned.
    Extract from "land lease of ₹X Lakh/year" or "ground rent X Lakh p.a.".
13. depreciation_method: 'slm' for straight-line, 'wdv' for written-down value / declining balance.
    Set 'wdv' only if user explicitly mentions WDV, declining balance, or 40% block depreciation.
    Default: 'slm'.
14. wdv_rate: WDV annual rate as decimal (default 0.40). Only relevant when depreciation_method='wdv'.
    Indian Income Tax Act specifies 40% for solar plants.
15. maintenance_capex_pct: annual capex-during-ops as fraction of total capex (default 0.0).
    Extract from "inverter replacement at 0.5% capex/yr", "maintenance capex 1%", "capex during ops".
    Deducted from CFADS: CFADS = EBITDA − Tax − capex_during_ops.
"""


# ---------------------------------------------------------------------------
# AssumptionAgent
# ---------------------------------------------------------------------------


class AssumptionAgent:
    """
    Extracts, validates, and defaults IPP assumptions from free-form text.

    Parameters
    ----------
    asset_type : "solar" (default) or "wind" — selects the canonical schema.
    model      : Anthropic model name.
    client     : Pre-configured anthropic.Anthropic client (uses env var if None).
    """

    def __init__(
        self,
        asset_type: str = "solar",
        model: str = "claude-sonnet-4-6",
        client: Optional[anthropic.Anthropic] = None,
    ) -> None:
        self.asset_type = asset_type.lower().strip()
        if self.asset_type not in _SCHEMA_BY_ASSET:
            raise ValueError(
                f"Unknown asset_type '{asset_type}'. "
                f"Supported: {sorted(_SCHEMA_BY_ASSET.keys())}"
            )
        self._canonical, self._required = _SCHEMA_BY_ASSET[self.asset_type]
        self.model = model
        self.client = client or anthropic.Anthropic()
        self._set_tool = self._build_tool()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, user_text: str) -> IngestionResult:
        """
        Extract and validate assumptions from a free-form project description.

        Parameters
        ----------
        user_text : Plain-English description of the solar IPP project.

        Returns
        -------
        IngestionResult with validated assumptions, warnings, and missing list.
        """
        explicit, inferred = self._extract_with_claude(user_text)
        filled, missing = self._apply_defaults(explicit)
        filled = self._derive_cross_assumptions(filled)
        validation = self._validate(filled)
        return IngestionResult(
            filled_assumptions=filled,
            missing_required=missing,
            inferred_assumptions=inferred,
            validation=validation,
        )

    def assumption_names(self) -> List[str]:
        """Return all canonical assumption names for this asset type."""
        return list(self._canonical.keys())

    def required_names(self) -> List[str]:
        """Return assumption names that have no default and must be supplied."""
        return list(self._required)

    def _build_tool(self) -> Dict[str, Any]:
        """Build the set_assumptions tool schema from this agent's canonical dict."""
        properties: Dict[str, Any] = {}
        for name, meta in self._canonical.items():
            prop: Dict[str, Any] = {
                "description": (
                    f"{meta['description']}  "
                    f"[unit: {meta['unit']}]  "
                    f"[aliases: {', '.join(meta.get('aliases', [])[:4])}]"
                )
            }
            if meta["type"] == "schedule":
                prop["type"] = "array"
                prop["items"] = {"type": "number"}
            elif meta["type"] == "enum":
                prop["type"] = "string"
                enum_vals = meta.get("enum_values", [])
                if enum_vals:
                    prop["enum"] = enum_vals
            else:
                prop["type"] = "number"
            properties[name] = prop

        asset_label = self.asset_type.upper().replace("_", " ")
        return {
            "name": "set_assumptions",
            "description": (
                f"Record the {asset_label} IPP assumption values identified in the user's text. "
                "Include every assumption you can determine. Omit assumptions you cannot determine — "
                "defaults will be applied.  Do NOT invent values; if uncertain, omit."
            ),
            "input_schema": {"type": "object", "properties": properties},
        }

    # ------------------------------------------------------------------
    # Private: Claude extraction
    # ------------------------------------------------------------------

    def _extract_with_claude(
        self, user_text: str
    ) -> Tuple[Dict[str, Any], Dict[str, InferredAssumption]]:
        """
        Send user_text to Claude and collect set_assumptions + note_inferences calls.
        Returns (explicit_values, inferred_values).
        """
        messages = [{"role": "user", "content": user_text}]
        tools = [self._set_tool, _NOTE_INFERENCES_TOOL]

        # Allow multiple tool calls in one response
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            tools=tools,
            tool_choice={"type": "auto"},
            messages=messages,
        )

        explicit: Dict[str, Any] = {}
        inferred: Dict[str, InferredAssumption] = {}

        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "set_assumptions":
                for k, v in block.input.items():
                    if k in self._canonical:
                        explicit[k] = v
            elif block.name == "note_inferences":
                for entry in block.input.get("inferences", []):
                    name = entry.get("assumption", "")
                    if name in self._canonical:
                        inferred[name] = InferredAssumption(
                            value=entry["value"],
                            derivation_note=entry["derivation_note"],
                        )
                        if name not in explicit:
                            explicit[name] = entry["value"]

        return explicit, inferred

    # ------------------------------------------------------------------
    # Private: defaults + cross-derivation
    # ------------------------------------------------------------------

    def _apply_defaults(
        self, extracted: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Fill in schema defaults; return (filled_dict, missing_required_names)."""
        filled = dict(extracted)
        missing: List[str] = []

        for name, meta in self._canonical.items():
            if name in filled:
                continue
            if meta["default"] is not None:
                filled[name] = meta["default"]
            else:
                missing.append(name)

        # Default capex_schedule if not set
        if "capex_schedule" not in filled:
            # Infer from construction_periods if derivable, else assume 4 quarters
            n_construct = len(filled.get("capex_schedule") or []) or 4
            filled["capex_schedule"] = [round(1.0 / n_construct, 6)] * n_construct

        return filled, missing

    def _derive_cross_assumptions(self, assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute model-skeleton quantities from filled assumptions.
        Adds: construction_periods, operations_periods, debt_tenor_periods.
        """
        a = dict(assumptions)

        schedule = a.get("capex_schedule", [0.25, 0.25, 0.25, 0.25])
        a["construction_periods"] = len(schedule)

        ppa_years = float(a.get("ppa_tenor_years", 25.0))
        a["operations_periods"] = int(round(ppa_years * 4))

        debt_years = float(a.get("debt_tenor_years", 18.0))
        a["debt_tenor_periods"] = int(round(debt_years * 4))

        # Derive numeric WDV flag from depreciation_method enum
        dep_method = a.get("depreciation_method", "slm")
        a["use_wdv"] = 1.0 if dep_method == "wdv" else 0.0

        return a

    # ------------------------------------------------------------------
    # Private: validation (pure Python — no AI)
    # ------------------------------------------------------------------

    def _validate(self, assumptions: Dict[str, Any]) -> ValidationResult:
        """
        Constraint validation + cross-consistency checks.
        Returns ValidationResult with errors (blocking) and warnings (advisory).
        """
        vr = ValidationResult(valid=True)

        # --- Constraint bounds ---
        for name, meta in self._canonical.items():
            if name not in assumptions:
                continue
            val = assumptions[name]
            if not isinstance(val, (int, float)):
                continue
            c = meta.get("constraints", {})
            lo, hi = c.get("min"), c.get("max")
            if lo is not None and val < lo:
                vr.add_error(
                    f"'{name}' = {val} is below minimum {lo} [{meta['unit']}]"
                )
            if hi is not None and val > hi:
                vr.add_error(
                    f"'{name}' = {val} exceeds maximum {hi} [{meta['unit']}]"
                )

        # --- Schedule: fractions must sum to 1.0 ---
        schedule = assumptions.get("capex_schedule")
        if isinstance(schedule, list):
            total = sum(float(x) for x in schedule)
            if abs(total - 1.0) > 0.005:
                vr.add_error(
                    f"capex_schedule fractions sum to {total:.4f}, must equal 1.0"
                )

        # --- Cross-assumption consistency ---
        debt_years = float(assumptions.get("debt_tenor_years", 18.0))
        ppa_years  = float(assumptions.get("ppa_tenor_years", 25.0))
        if debt_years > ppa_years:
            vr.add_warning(
                code="DEBT_TENOR_EXCEEDS_PPA",
                msg=(
                    f"debt_tenor_years ({debt_years:.0f}) > ppa_tenor_years ({ppa_years:.0f}). "
                    "Senior lenders typically require debt repayment within the PPA term."
                ),
            )

        moratorium_q = float(assumptions.get("moratorium_periods", 0.0))
        debt_q = assumptions.get("debt_tenor_periods", debt_years * 4)
        if moratorium_q >= debt_q:
            vr.add_error(
                f"moratorium_periods ({moratorium_q:.0f}) >= debt tenor in quarters "
                f"({debt_q:.0f}). No repayment periods would remain."
            )

        interest = assumptions.get("interest_rate")
        if interest is not None and float(interest) <= 0:
            vr.add_error("interest_rate must be positive")

        cuf = assumptions.get("cuf")
        if cuf is not None and float(cuf) > 0.35:
            vr.add_warning(
                code="HIGH_CUF",
                msg=(
                    f"cuf = {float(cuf):.2%} is above typical Karnataka range (19–26%). "
                    "Verify with site-specific irradiance data."
                ),
            )

        tariff = assumptions.get("tariff")
        if tariff is not None and float(tariff) <= 0:
            vr.add_error("tariff must be positive")

        capex = assumptions.get("capex_per_mw")
        if capex is not None and float(capex) < 200:
            vr.add_warning(
                code="LOW_CAPEX",
                msg=(
                    f"capex_per_mw = ₹{float(capex):.0f} Lakh/MW seems low for India "
                    "(typical range ₹300–600 Lakh/MW). Verify the figure."
                ),
            )

        debt_pct = assumptions.get("debt_pct")
        if debt_pct is not None and float(debt_pct) > 0.80:
            vr.add_warning(
                code="HIGH_LEVERAGE",
                msg=(
                    f"debt_pct = {float(debt_pct):.0%} is above typical lender ceiling "
                    "(75–80%). DSCR breaches are likely."
                ),
            )

        # --- Depreciation method enum validation ---
        dep_method = assumptions.get("depreciation_method")
        if dep_method is not None:
            if dep_method not in ("slm", "wdv"):
                vr.add_error(
                    f"depreciation_method must be 'slm' or 'wdv', got '{dep_method}'"
                )
            elif dep_method == "wdv":
                wdv = assumptions.get("wdv_rate")
                if wdv is None or float(wdv) <= 0:
                    vr.add_warning(
                        code="WDV_RATE_MISSING",
                        msg=(
                            "depreciation_method='wdv' but wdv_rate is not set or zero. "
                            "Defaulting to 0.40 (40% p.a. — Indian Income Tax Act rate for solar)."
                        ),
                    )

        # --- PPA tenor vs debt tenor cross-check ---
        ppa_years = float(assumptions.get("ppa_tenor_years", 25.0))
        if ppa_years <= 0:
            vr.add_error("ppa_tenor_years must be positive")

        return vr
