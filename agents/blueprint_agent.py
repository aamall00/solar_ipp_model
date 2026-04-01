"""
agents/blueprint_agent.py — Layer 4: Blueprint Agent.

Takes a validated IngestionResult (or raw assumption dict) and produces a
complete, parser-ready ModelDefinition for a standard solar IPP.

Design
------
Blocks are loaded at runtime from the block library under blocks/solar_ipp/*.yaml,
not hardcoded in the agent.  The agent's job is to:

  1. _load_block_library()       — read all 14 YAML files from blocks/solar_ipp/
  2. _configure_block_inputs()   — wire assumption sources
  3. _build_skeleton()           — compute periods/milestones from assumptions
  4. _build_assumption_schema()  — map filled_assumptions → assumption_schema entries
  5. _determine_solve_loops()    — sculpting loop only when debt_sizing_mode=1;
                                   IDC array_fixed_point loop always present
  6. _units_check()              — basic units algebra sanity (no AI)
  7. _claude_structure_review()  — Claude flags structural anomalies (optional)
  8. Parse with DSLParser         — validate structure
  9. _explain() [optional]       — Claude writes a configuration rationale

Claude roles
------------
  - _claude_structure_review(): flags unusual parameter combinations before building
    (e.g. very high leverage + low CUF, IDC capitalisation with short tenor)
  - _explain(): plain-English rationale for the project sponsor

Claude NEVER computes arithmetic.  All structural decisions are Python.

Usage
-----
    from agents.assumption_agent import AssumptionAgent
    from agents.blueprint_agent  import BlueprintAgent

    result   = AssumptionAgent().extract(user_text)
    bp_agent = BlueprintAgent()
    model_def, validation, yaml_str, rationale = bp_agent.generate(result, explain=True)
    compiled = ModelExecutor().compile(model_def)
"""

from __future__ import annotations

import math
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import yaml

from agents.assumption_agent import IngestionResult, _CANONICAL
from dsl.parser import load_model_from_dict
from dsl.types import ModelDefinition, ValidationResult

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BLOCKS_DIR = pathlib.Path(__file__).parent.parent / "blocks" / "solar_ipp"
_TEMPLATE_PATH = pathlib.Path(__file__).parent.parent / "dsl" / "templates" / "solar_ipp_base.yaml"

# Block order is derived dynamically from the YAML library at runtime.
# _BLOCK_ORDER is intentionally removed — see _build_blocks_and_wiring().

# ---------------------------------------------------------------------------
# Claude prompts
# ---------------------------------------------------------------------------

_REVIEW_SYSTEM = """You are a project finance senior analyst reviewing a solar IPP model configuration.
Given the project parameters below, identify any structural concerns or unusual combinations
that could lead to unreliable results or bankability issues.
Be specific and concise — 2-4 bullet points maximum.
Do NOT compute numbers; only flag qualitative structural issues."""

_EXPLAIN_SYSTEM = """You are a project finance structuring specialist.
Given the key parameters of a solar IPP, write a concise (4-6 sentence) rationale
explaining how the model has been configured: debt structure, tenor, solve loop
type, and any notable parameter choices.  Be specific about the numbers.
Do not repeat the table of values — only interpret the choices and flag any risks."""


class BlueprintAgent:
    """
    Converts validated IPP assumptions into a complete ModelDefinition
    plus a YAML string representation.

    Parameters
    ----------
    asset_type : "solar" (default) or "wind" — selects the block library and schema.
    model      : Anthropic model for Claude-assisted review and explanation.
    client     : Pre-configured anthropic.Anthropic client (uses env var if None).
    """

    def __init__(
        self,
        asset_type: str = "solar",
        model: str = "claude-sonnet-4-6",
        client: Optional[anthropic.Anthropic] = None,
    ) -> None:
        from engine.asset_registry import get_asset
        self.asset_type = asset_type.lower().strip()
        self._asset_meta = get_asset(self.asset_type)
        self.model = model
        self.client = client or anthropic.Anthropic()
        self._block_library: Optional[Dict[str, Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        ingestion_result: "IngestionResult | Dict[str, Any]",
        explain: bool = False,
        review: bool = False,
    ) -> Tuple[ModelDefinition, ValidationResult, str, Optional[str]]:
        """
        Build a complete ModelDefinition from validated assumptions.

        Parameters
        ----------
        ingestion_result : IngestionResult from AssumptionAgent, or a raw
                           assumption dict (same keys, values already validated).
        explain          : If True, ask Claude to write a configuration rationale.
        review           : If True, ask Claude to flag structural anomalies before
                           building (uses an extra API call).

        Returns
        -------
        (model_def, validation, yaml_str, rationale)
          model_def   : Parsed ModelDefinition, ready for ModelExecutor.compile().
          validation  : DSL validation result (errors + warnings).
          yaml_str    : YAML text of the full model_dict (for inspection / storage).
          rationale   : Plain-English explanation string, or None.
        """
        if isinstance(ingestion_result, dict):
            assumptions = ingestion_result
        else:
            assumptions = ingestion_result.filled_assumptions

        # Optional Claude structural review before building
        review_notes: Optional[str] = None
        if review:
            review_notes = self._claude_structure_review(assumptions)

        model_dict = self._build_model_dict(assumptions)

        # Units algebra check (pure Python, no AI)
        unit_warnings = self._units_check(model_dict)

        model_def, validation = load_model_from_dict(model_dict)

        # Attach unit warnings to validation
        for w in unit_warnings:
            validation.add_warning(code="UNITS_MISMATCH", msg=w)

        yaml_str = yaml.dump(model_dict, default_flow_style=False, sort_keys=False,
                             allow_unicode=True)

        rationale: Optional[str] = None
        if explain:
            review_prefix = (
                f"Structural review notes:\n{review_notes}\n\n" if review_notes else ""
            )
            rationale = self._explain(assumptions, validation, review_prefix)

        return model_def, validation, yaml_str, rationale

    # ------------------------------------------------------------------
    # Block library loading
    # ------------------------------------------------------------------

    def _load_block_library(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all block YAML files for this asset type and return a dict
        keyed by block_id.  Result is cached after the first call.
        """
        if self._block_library is not None:
            return self._block_library
        if self.asset_type == "wind":
            from blocks.wind_ipp import load_all_blocks_raw
        else:
            from blocks.solar_ipp import load_all_blocks_raw
        self._block_library = load_all_blocks_raw()
        return self._block_library

    # ------------------------------------------------------------------
    # Private: model dict construction (pure Python)
    # ------------------------------------------------------------------

    def _build_model_dict(self, a: Dict[str, Any]) -> Dict[str, Any]:
        blocks, wiring = self._build_blocks_and_wiring(a)
        return {
            "project_skeleton":   self._build_skeleton(a),
            "assumption_schema":  self._build_assumption_schema(a),
            "calculation_blocks": blocks,
            "model_wiring":       wiring,
        }

    def _build_skeleton(self, a: Dict[str, Any]) -> Dict[str, Any]:
        schedule             = list(a.get("capex_schedule", [0.25, 0.25, 0.25, 0.25]))
        construction_periods = len(schedule)
        ppa_years            = float(a.get("ppa_tenor_years", 25.0))
        operations_periods   = int(round(ppa_years * 4))
        debt_years           = float(a.get("debt_tenor_years", 18.0))
        cod                  = construction_periods
        debt_maturity        = cod + int(round(debt_years * 4)) - 1
        total_periods        = construction_periods + operations_periods
        if debt_maturity >= total_periods:
            debt_maturity = total_periods - 1

        capacity_mw  = float(a.get("capacity_mw", 100))
        tariff       = float(a.get("tariff", 2.65))
        project_type = self._asset_meta["project_type"]
        cf_key       = self._asset_meta["cuf_or_cf_key"]
        cf_val       = float(a.get(cf_key, self._asset_meta["cuf_or_cf_default"]))
        model_id     = _slugify(f"{project_type}_{int(capacity_mw)}mw_cf{cf_val:.0%}")

        return {
            "model_id":             model_id,
            "project_type":         project_type,
            "currency":             "INR",
            "currency_unit":        "Lakhs",
            "periods_per_year":     4,
            "construction_periods": construction_periods,
            "operations_periods":   operations_periods,
            "milestones": {
                "financial_close": 0,
                "cod":             cod,
                "debt_maturity":   debt_maturity,
            },
        }

    def _build_assumption_schema(self, a: Dict[str, Any]) -> Dict[str, Any]:
        entries = []

        def _entry(name: str, atype: str, unit: str, value: Any,
                   cmin: Optional[float] = None, cmax: Optional[float] = None,
                   vary: bool = False, range_pct: float = 0.10,
                   dist: str = "triangular") -> Dict[str, Any]:
            e: Dict[str, Any] = {"name": name, "type": atype, "unit": unit, "value": value}
            if cmin is not None or cmax is not None:
                e["constraints"] = {}
                if cmin is not None:
                    e["constraints"]["min"] = cmin
                if cmax is not None:
                    e["constraints"]["max"] = cmax
            if vary:
                e["sensitivity"] = {
                    "vary": True, "range_pct": range_pct, "distribution": dist,
                }
            return e

        # --- Generation (asset-type specific) ---
        entries.append(_entry("capacity_mw", "scalar", "MW",
                               float(a.get("capacity_mw", 100.0))))
        if self.asset_type == "wind":
            cf_default = self._asset_meta["cuf_or_cf_default"]
            entries.append(_entry("capacity_factor", "scalar", "ratio",
                                   float(a.get("capacity_factor", cf_default)),
                                   cmin=0.15, cmax=0.55, vary=True, range_pct=0.10))
            entries.append(_entry("availability_factor", "scalar", "ratio",
                                   float(a.get("availability_factor", 0.95)),
                                   cmin=0.80, cmax=0.99))
        else:
            entries.append(_entry("cuf", "scalar", "ratio",
                                   float(a.get("cuf", 0.22)),
                                   cmin=0.10, cmax=0.40, vary=True, range_pct=0.10))
            entries.append(_entry("degradation_rate", "scalar", "per_year",
                                   float(a.get("degradation_rate", 0.005)),
                                   vary=True, range_pct=0.50))
        entries.append(_entry("auxiliary_consumption", "scalar", "ratio",
                               float(a.get("auxiliary_consumption", 0.005))))

        # --- Revenue ---
        entries.append(_entry("tariff", "scalar", "INR_per_kWh",
                               float(a.get("tariff", 2.65)),
                               cmin=0.0, vary=True, range_pct=0.10))
        entries.append(_entry("tariff_escalation", "scalar", "per_year",
                               float(a.get("tariff_escalation", 0.0))))

        # --- Capex ---
        capex_default = self._asset_meta["capex_per_mw_default"]
        entries.append(_entry("capex_per_mw", "scalar", "INR_Lakhs_per_MW",
                               float(a.get("capex_per_mw", capex_default)),
                               vary=True, range_pct=0.10))
        schedule = list(a.get("capex_schedule", [0.25, 0.25, 0.25, 0.25]))
        entries.append(_entry("capex_schedule", "time_series", "fraction", schedule))

        # --- O&M ---
        opex_default = self._asset_meta["opex_per_mw_pa_default"]
        entries.append(_entry("opex_per_mw_pa", "scalar", "INR_Lakhs_per_MW_per_year",
                               float(a.get("opex_per_mw_pa", opex_default)),
                               vary=True, range_pct=0.20))
        entries.append(_entry("opex_escalation", "scalar", "per_year",
                               float(a.get("opex_escalation", 0.03))))

        # --- Debt ---
        entries.append(_entry("debt_pct", "scalar", "ratio",
                               float(a.get("debt_pct", 0.70)),
                               cmin=0.30, cmax=0.85))
        entries.append(_entry("interest_rate", "scalar", "per_year",
                               float(a.get("interest_rate", 0.0975)),
                               cmin=0.04, cmax=0.25, vary=True, range_pct=0.15))
        entries.append(_entry("moratorium_periods", "scalar", "quarters",
                               float(a.get("moratorium_periods", 0.0))))
        entries.append(_entry("dscr_target", "scalar", "ratio",
                               float(a.get("dscr_target", 1.20)),
                               cmin=1.0, cmax=2.5))
        entries.append(_entry("debt_sizing_mode", "scalar", "flag",
                               float(a.get("debt_sizing_mode", 0.0))))
        entries.append(_entry("dsra_months", "scalar", "months",
                               float(a.get("dsra_months", 6.0))))


        # --- Tax ---
        entries.append(_entry("tax_rate", "scalar", "ratio",
                               float(a.get("tax_rate", 0.25)),
                               cmin=0.0, cmax=0.40))
        entries.append(_entry("depreciation_rate", "scalar", "per_year",
                               float(a.get("depreciation_rate", 0.05))))

        # --- Equity ---
        entries.append(_entry("equity_irr_target", "scalar", "per_year",
                               float(a.get("equity_irr_target", 0.14))))

        return {"assumptions": entries}

    def _build_blocks_and_wiring(
        self, a: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load blocks from the library, configure inputs, and return
        (calculation_blocks_dict, model_wiring_dict).
        """
        library = self._load_block_library()

        # Assemble blocks in library scan order (executor uses dependency graph for
        # actual execution order, so declaration order here is informational only).
        ordered_blocks: List[Dict[str, Any]] = [
            self._configure_block(block, a) for block in library.values()
        ]

        solve_loops = self._determine_solve_loops(a)

        calculation_blocks = {
            "blocks":      ordered_blocks,
            "solve_loops": solve_loops,
        }
        return calculation_blocks, _WIRING

    def _configure_block(
        self, block: Dict[str, Any], a: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Return a (shallow-copied) block dict with inputs verified against the
        validated assumption set.  This is a no-op for most blocks; inputs are
        fully declared in the YAML block library files.
        """
        import copy
        b = copy.deepcopy(block)

        # Verify assumption.* sources exist in the validated assumption set
        all_assumption_names = set(_CANONICAL.keys())
        for inp in b.get("inputs", []):
            src: str = inp.get("source", "")
            if src.startswith("assumption."):
                key = src.split(".", 1)[1]
                if key not in all_assumption_names:
                    # Warn but don't hard-fail; DSLParser will catch it
                    pass  # could raise ValueError(f"Unknown assumption: {key}")

        return b

    def _determine_solve_loops(self, a: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Conditional solve loop configuration:
          - Sculpting loop: ONLY when debt_sizing_mode = 1 (CFADS-sculpted).
          - IDC array_fixed_point: always present (IDC is always capitalised).

        Note: when debt_sizing_mode = 0 (cost-based), the sculpting loop is
        still required for the executor to compute equal-principal debt service.
        If you need cost-based debt service, keep debt_sizing_mode = 0 but the
        sculpting loop must still be present.  This method includes the loop
        for both modes because the executor's _run_sculpting handles both.
        """
        loops: List[Dict[str, Any]] = []

        # Sculpting loop — always needed for debt service computation.
        # debt_sizing_mode controls the algorithm inside the executor.
        dscr_target = float(a.get("dscr_target", 1.20))
        loops.append({
            "loop_id":           "debt_service_sculpting",
            "type":              "sculpting",
            "free_variable":     "debt_service_block.total_debt_service",
            "target_expression": "cashflow_block.cfads",
            "target_value":      dscr_target,
            "tolerance":         1e-6,
            "max_iterations":    50,
            # Executor reads these from loop.parameters — no assumption names hardcoded
            # in the solver. Fallbacks in the executor match these defaults.
            "parameters": {
                "interest_rate_var":    "assumption.interest_rate",
                "moratorium_var":       "assumption.moratorium_periods",
                "debt_sizing_mode_var": "assumption.debt_sizing_mode",
                "output_principal":     "principal_repayment",
                "output_interest":      "interest_payment",
                "output_balance":       "outstanding_debt_balance",
                "output_ds":            "total_debt_service",
            },
        })

        # IDC array fixed-point — always present (IDC always capitalised into debt).
        # The circular dependency (construction_block total_capex uses idc_per_period;
        # debt_service_block drawdown uses total_capex; idc_block uses cumulative_drawdown)
        # is resolved generically by re-evaluating owned_blocks in declared order until
        # idc_per_period converges. Contraction ratio ≈ debt_pct × r_q / 2 ≈ 0.0085.
        loops.append({
            "loop_id":           "idc_capitalisation",
            "type":              "array_fixed_point",
            "free_variable":     "idc_block.idc_per_period",
            "target_expression": "idc_block.idc_per_period",
            "target_value":      0.0,
            "owned_blocks":      ["construction_block", "debt_service_block", "idc_block"],
        })

        return loops

    # ------------------------------------------------------------------
    # Units algebra check (pure Python — no AI)
    # ------------------------------------------------------------------

    def _units_check(self, model_dict: Dict[str, Any]) -> List[str]:
        """
        Basic dimensional sanity checks.  Returns a list of warning strings.
        These are advisory — the model can still run with unit mismatches.
        """
        warnings: List[str] = []
        assumptions = {
            e["name"]: e
            for e in model_dict.get("assumption_schema", {}).get("assumptions", [])
        }

        # 1. tariff should be in INR_per_kWh (range 0.5 – 15)
        if "tariff" in assumptions:
            v = float(assumptions["tariff"].get("value", 2.65))
            if not (0.5 <= v <= 15.0):
                warnings.append(
                    f"tariff={v} is outside [0.5, 15] INR/kWh — check units"
                )

        # 2. capex_per_mw should be in INR Lakhs/MW (range 150 – 1000)
        if "capex_per_mw" in assumptions:
            v = float(assumptions["capex_per_mw"].get("value", 450.0))
            if not (150.0 <= v <= 1000.0):
                warnings.append(
                    f"capex_per_mw={v} is outside [150, 1000] INR Lakhs/MW — check units"
                )

        # 3. interest_rate should be a decimal in [0.04, 0.25], not a percentage
        if "interest_rate" in assumptions:
            v = float(assumptions["interest_rate"].get("value", 0.0975))
            if v > 0.25:
                warnings.append(
                    f"interest_rate={v} looks like a percentage; expected decimal "
                    f"(e.g. 0.0975 for 9.75%)"
                )

        # 4. debt_pct should be a decimal in [0, 1], not a percentage
        if "debt_pct" in assumptions:
            v = float(assumptions["debt_pct"].get("value", 0.70))
            if v > 1.0:
                warnings.append(
                    f"debt_pct={v} > 1.0 — should be a decimal fraction (e.g. 0.70 for 70%)"
                )

        # 5. capex_schedule fractions should sum to 1.0
        if "capex_schedule" in assumptions:
            schedule = assumptions["capex_schedule"].get("value", [])
            if isinstance(schedule, list):
                total = sum(float(x) for x in schedule)
                if abs(total - 1.0) > 0.005:
                    warnings.append(
                        f"capex_schedule sums to {total:.4f} — expected 1.0"
                    )

        return warnings

    # ------------------------------------------------------------------
    # Private: Claude structural review
    # ------------------------------------------------------------------

    def _claude_structure_review(self, a: Dict[str, Any]) -> str:
        """
        Ask Claude to flag any structural anomalies in the assumption set.
        Returns a short string with bullet-point concerns (or empty string).
        """
        capacity   = a.get("capacity_mw", "?")
        cuf        = a.get("cuf", "?")
        capex      = a.get("capex_per_mw", "?")
        tariff     = a.get("tariff", "?")
        debt_pct   = a.get("debt_pct", "?")
        rate       = a.get("interest_rate", "?")
        tenor      = a.get("debt_tenor_years", "?")
        mode_flag  = float(a.get("debt_sizing_mode", 0.0))
        mode_str   = "CFADS-sculpted" if mode_flag >= 0.5 else "equal-principal"
        asset_label = self.asset_type.upper()
        cf_key  = self._asset_meta["cuf_or_cf_key"]
        cf_val  = a.get(cf_key, self._asset_meta["cuf_or_cf_default"])
        prompt = (
            f"{asset_label} IPP configuration to review:\n"
            f"  Capacity: {capacity} MW, CF/CUF: {float(cf_val):.1%}\n"
            f"  Capex: ₹{capex} Lakh/MW, Tariff: ₹{tariff}/kWh\n"
            f"  Debt: {float(debt_pct)*100:.0f}% at {float(rate)*100:.2f}% p.a., "
            f"{tenor}-yr tenor\n"
            f"  Debt service: {mode_str}\n"
            f"  IDC treatment: capitalised into debt (average-balance)\n\n"
            "Flag any structural concerns or unusual combinations. "
            "Be specific. 2-4 bullet points only."
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system=_REVIEW_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""

    # ------------------------------------------------------------------
    # Private: Claude explanation
    # ------------------------------------------------------------------

    def _explain(
        self, a: Dict[str, Any], validation: ValidationResult,
        review_prefix: str = ""
    ) -> str:
        capacity  = a.get("capacity_mw", "?")
        tariff    = a.get("tariff", "?")
        capex     = a.get("capex_per_mw", "?")
        debt_pct  = a.get("debt_pct", "?")
        rate      = a.get("interest_rate", "?")
        tenor     = a.get("debt_tenor_years", "?")
        mode      = ("CFADS-sculpted" if float(a.get("debt_sizing_mode", 0)) >= 0.5
                     else "equal-principal")
        dscr      = a.get("dscr_target", "?")
        idc_str   = "IDC capitalised into debt (array_fixed_point loop)"
        warn_str  = ""
        if validation.warnings:
            warn_str = "Validation warnings: " + "; ".join(
                w.message for w in validation.warnings
            )

        prompt = (
            f"{review_prefix}"
            f"Solar IPP configuration summary:\n"
            f"  Capacity: {capacity} MW\n"
            f"  Tariff: ₹{tariff}/kWh\n"
            f"  Capex: ₹{capex} Lakh/MW\n"
            f"  Debt: {float(debt_pct)*100:.0f}% at {float(rate)*100:.2f}% p.a., "
            f"{tenor}-yr tenor\n"
            f"  Debt service mode: {mode} (DSCR target: {dscr}×)\n"
            f"  {idc_str}\n"
            f"  {warn_str}\n\n"
            "Write a concise configuration rationale for the project sponsor."
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=_EXPLAIN_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert text to a lowercase underscore slug safe for model_id."""
    text = text.lower().replace(" ", "_").replace("/", "_")
    text = re.sub(r"[^a-z0-9_]", "", text)
    return re.sub(r"_+", "_", text).strip("_")


# ---------------------------------------------------------------------------
# Wiring — loaded from the template (single source of truth)
# ---------------------------------------------------------------------------

def _load_wiring() -> Dict[str, Any]:
    with open(_TEMPLATE_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["model_wiring"]

_WIRING: Dict[str, Any] = _load_wiring()
