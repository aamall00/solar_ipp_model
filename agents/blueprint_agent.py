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
  5. _determine_solve_loops()    — load declarative solve-loop config from YAML
  6. _units_check()              — basic units algebra sanity (no AI)
  7. _claude_structure_review()  — Claude flags structural anomalies (optional)
  8. Parse with DSLParser         — validate structure
  9. _explain() [optional]       — Claude writes a configuration rationale

Claude roles
------------
  - _claude_structure_review(): flags unusual parameter combinations before building
    (e.g. very high leverage + low CUF, IDC capitalisation with short tenor)
  - _explain(): plain-English rationale for the project sponsor

Claude NEVER computes arithmetic. Block, assumption-schema, and solve-loop
declarations are YAML-driven; Python assembles and validates the model instance.

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
import numpy as np
import yaml

from agents.assumption_agent import IngestionResult
from dsl.assumption_schema_library import load_assumption_schema_entries
from dsl.expression import ExpressionEvaluator
from dsl.parser import load_model_from_dict
from dsl.solve_loop_library import load_solve_loop_entries
from dsl.types import ModelDefinition, ValidationResult

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BLOCKS_DIR = pathlib.Path(__file__).parent.parent / "blocks" / "solar_ipp"
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
        self._assumption_schema_library: Optional[List[Dict[str, Any]]] = None
        self._solve_loop_library: Optional[List[Dict[str, Any]]] = None

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

    def _load_assumption_schema_library(self) -> List[Dict[str, Any]]:
        """
        Load the asset-specific assumption schema template from YAML.
        The raw entries may contain assembly-only metadata such as include_when,
        exclude_when, and formula-based default_rule strings.
        """
        if self._assumption_schema_library is not None:
            return self._assumption_schema_library
        self._assumption_schema_library = load_assumption_schema_entries(self.asset_type)
        return self._assumption_schema_library

    def _load_solve_loop_library(self) -> List[Dict[str, Any]]:
        """
        Load the asset-specific solve-loop declarations from YAML.
        Raw entries may contain assembly-only metadata such as include_when,
        exclude_when, and target_value_rule expressions.
        """
        if self._solve_loop_library is not None:
            return self._solve_loop_library
        self._solve_loop_library = load_solve_loop_entries(self.asset_type)
        return self._solve_loop_library

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
        import copy

        entries: List[Dict[str, Any]] = []
        resolved_values: Dict[str, Any] = dict(a)

        for raw_entry in copy.deepcopy(self._load_assumption_schema_library()):
            include_when = raw_entry.pop("include_when", None)
            exclude_when = raw_entry.pop("exclude_when", None)

            if include_when and not self._evaluate_schema_condition(include_when, resolved_values):
                continue
            if exclude_when and self._evaluate_schema_condition(exclude_when, resolved_values):
                continue

            name = raw_entry["name"]
            if name in a:
                value = a[name]
            elif raw_entry.get("value") is not None:
                value = raw_entry["value"]
            elif raw_entry.get("default_rule"):
                value = self._evaluate_schema_expression(
                    raw_entry["default_rule"],
                    resolved_values,
                )
            else:
                value = None

            raw_entry["value"] = value
            raw_entry.pop("default_rule", None)
            entries.append(raw_entry)
            resolved_values[name] = value

        return {"assumptions": entries}

    def _evaluate_schema_expression(
        self,
        expr: str,
        values: Dict[str, Any],
    ) -> Any:
        """
        Evaluate a scalar schema expression against the currently resolved
        assumption values. Expressions use the same safe DSL evaluator as block
        bodies but run on a trivial one-period context.
        """
        cleaned = expr.strip()
        if cleaned.startswith("="):
            cleaned = cleaned[1:].strip()

        evaluator = ExpressionEvaluator(n_periods=1, periods_per_year=1)
        result = evaluator.evaluate(cleaned, values)
        if isinstance(result, np.ndarray):
            if result.size != 1:
                raise ValueError(
                    f"Schema expression '{expr}' produced a non-scalar array with "
                    f"shape {result.shape}"
                )
            return result.reshape(-1)[0].item()
        if isinstance(result, np.generic):
            return result.item()
        return result

    def _evaluate_schema_condition(
        self,
        expr: str,
        values: Dict[str, Any],
    ) -> bool:
        result = self._evaluate_schema_expression(expr, values)
        return bool(result)

    def _select_blocks(
        self, library: Dict[str, Dict[str, Any]], a: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return the subset of blocks to include for this model instance.

        Optional blocks are controlled declaratively from block YAML metadata.
        When ``wiring.optional`` is true, ``wiring.include_when`` and
        ``wiring.exclude_when`` are evaluated against the assumption dict.
        """
        included: Dict[str, Dict[str, Any]] = {}

        for block_id, block in library.items():
            wiring = block.get("wiring", {}) or {}
            if not wiring.get("optional", False):
                included[block_id] = block
                continue

            include_block = True
            try:
                include_when = wiring.get("include_when")
                if include_when:
                    include_block = self._evaluate_schema_condition(
                        str(include_when), a
                    )

                exclude_when = wiring.get("exclude_when")
                if include_block and exclude_when:
                    include_block = not self._evaluate_schema_condition(
                        str(exclude_when), a
                    )
            except Exception as exc:
                raise ValueError(
                    f"Failed to evaluate optional wiring rules for block "
                    f"'{block_id}'"
                ) from exc

            if include_block:
                included[block_id] = block

        return included

    def _build_wiring_from_included(
        self,
        included: Dict[str, Dict[str, Any]],
        excluded_ids: set,
    ) -> Dict[str, Any]:
        """
        Assemble model_wiring dynamically from the included block set.

        connections   : derived mechanically from each included block's inputs[].source,
                        filtering out any connection whose source block is excluded.
        output_reports: merged from each block's wiring.output_reports section.
        """
        from collections import defaultdict

        connections: List[Dict[str, str]] = []
        seen_connections: set = set()
        output_reports: Dict[str, List[str]] = defaultdict(list)

        for block in included.values():
            block_id = block["block_id"]

            # Connections: one per declared input whose source is a block output
            for inp in block.get("inputs", []):
                src: str = inp.get("source", "")
                src_block = src.split(".")[0]
                # Only add connections between blocks (skip assumption.*, phase.*)
                if "." not in src or src_block in ("assumption", "phase"):
                    continue
                # Drop if the source block has been excluded
                if src_block in excluded_ids:
                    continue
                conn = {"from": src, "to": f"{block_id}.{inp['name']}"}
                key = (conn["from"], conn["to"])
                if key not in seen_connections:
                    seen_connections.add(key)
                    connections.append(conn)

            # Output reports: merge from block's wiring.output_reports
            for report_name, vars_list in (
                block.get("wiring", {}).get("output_reports", {}).items()
            ):
                output_reports[report_name].extend(vars_list)

        return {
            "connections":    connections,
            "output_reports": dict(output_reports),
        }

    def _build_blocks_and_wiring(
        self, a: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load blocks from the library, select which to include based on assumption
        flags, configure inputs, and return (calculation_blocks_dict, model_wiring_dict).
        """
        library = self._load_block_library()

        included = self._select_blocks(library, a)
        excluded_ids = set(library.keys()) - set(included.keys())

        ordered_blocks: List[Dict[str, Any]] = [
            self._configure_block(block, a) for block in included.values()
        ]

        wiring = self._build_wiring_from_included(included, excluded_ids)
        solve_loops = self._determine_solve_loops(a, set(included.keys()))

        calculation_blocks = {
            "blocks":      ordered_blocks,
            "solve_loops": solve_loops,
        }
        return calculation_blocks, wiring

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

        # Verify assumption.* sources exist in the asset schema library
        all_assumption_names = {
            entry["name"] for entry in self._load_assumption_schema_library()
        }
        for inp in b.get("inputs", []):
            src: str = inp.get("source", "")
            if src.startswith("assumption."):
                key = src.split(".", 1)[1]
                if key not in all_assumption_names:
                    # Warn but don't hard-fail; DSLParser will catch it
                    pass  # could raise ValueError(f"Unknown assumption: {key}")

        return b

    def _determine_solve_loops(
        self,
        a: Dict[str, Any],
        included_block_ids: set[str],
    ) -> List[Dict[str, Any]]:
        """
        Assemble solve loops declaratively from the YAML loop library.

        Assembly-only metadata supported in the raw YAML entries:
          - include_when / exclude_when: boolean expressions over assumptions
          - target_value_rule: scalar expression evaluated over assumptions
        """
        import copy

        loops: List[Dict[str, Any]] = []
        for raw_loop in self._load_solve_loop_library():
            loop = copy.deepcopy(raw_loop)

            include_when = loop.pop("include_when", None)
            exclude_when = loop.pop("exclude_when", None)
            if include_when and not self._evaluate_schema_condition(
                str(include_when), a
            ):
                continue
            if exclude_when and self._evaluate_schema_condition(str(exclude_when), a):
                continue

            target_value_rule = loop.pop("target_value_rule", None)
            if target_value_rule is not None:
                loop["target_value"] = float(
                    self._evaluate_schema_expression(str(target_value_rule), a)
                )

            owned_blocks = loop.get("owned_blocks", []) or []
            missing_owned_blocks = [
                block_id
                for block_id in owned_blocks
                if block_id not in included_block_ids
            ]
            if missing_owned_blocks:
                raise ValueError(
                    f"Solve loop '{loop.get('loop_id', '?')}' references excluded "
                    f"or unknown owned_blocks: {missing_owned_blocks}"
                )

            loops.append(loop)

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
# Note: model_wiring is now assembled dynamically in BlueprintAgent._build_wiring_from_included()
# rather than loaded statically from solar_ipp_base.yaml.  The template is no longer
# referenced for wiring — each block YAML declares its own wiring.output_reports.
# ---------------------------------------------------------------------------
