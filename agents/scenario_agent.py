"""
agents/scenario_agent.py — Layer 5b: Scenario Construction Agent.

Three capabilities
------------------
1. propose_scenarios(base_results, compiled, n_scenarios)
   Uses Claude to propose N correlated, named scenarios based on the base-case
   KPIs and known Indian solar IPP risk factors.  Always includes the 7
   standard scenarios listed in the spec; additional AI-proposed scenarios are
   appended when n_scenarios > 7.

2. run_all_scenarios(compiled, scenarios)
   Executes every scenario in a list (or the default STANDARD_SCENARIOS) via
   run_batch() and returns a dict of {scenario_name → ModelResults}.

3. run(scenario_description)   [legacy single-run mode]
   Accepts a plain-English description of a single scenario, uses Claude
   tool_use to extract assumption overrides, runs the model, and optionally
   generates a short narrative.

The Claude layer NEVER computes arithmetic.  It only identifies which
assumptions to change and by how much.  All calculations go through the
ModelExecutor / NumPy pipeline.

Usage
-----
    from dsl.parser import load_model
    from engine.executor import ModelExecutor
    from agents.scenario_agent import ScenarioAgent, STANDARD_SCENARIOS

    model_def, _ = load_model("dsl/templates/solar_ipp_base.yaml")
    executor     = ModelExecutor()
    compiled     = executor.compile(model_def)
    agent        = ScenarioAgent(compiled=compiled, executor=executor)

    # Propose + run 7 standard scenarios
    scenarios    = agent.propose_scenarios(base_results=executor.run(compiled, {}))
    all_results  = agent.run_all_scenarios(compiled, scenarios)

    # Single NL query
    result, narrative = agent.run(
        "What if CUF drops to 18% and tariff rises to 3.10?", explain=True
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import anthropic

from engine.executor import CompiledModel, ModelExecutor, ModelResults

# ---------------------------------------------------------------------------
# Scenario container
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """
    A named scenario consisting of assumption overrides relative to the base case.

    Attributes
    ----------
    name          : Human-readable scenario name (e.g. "Construction Stress").
    rationale     : Brief explanation of why this scenario is plausible.
    overrides     : Dict mapping assumption names to override values.
    scenario_type : One of: base | upside | downside | stress | combined.
    """
    name: str
    rationale: str
    overrides: Dict[str, Any]
    scenario_type: str = "downside"


# ---------------------------------------------------------------------------
# Standard 7 scenarios (deterministic — no AI needed)
# ---------------------------------------------------------------------------

STANDARD_SCENARIOS: List[Scenario] = [
    Scenario(
        name="Base Case",
        rationale="Schema defaults — no overrides applied.",
        overrides={},
        scenario_type="base",
    ),
    Scenario(
        name="Management Case",
        rationale=(
            "Slightly optimistic assumptions: CUF at P60 (0.23), "
            "on-time delivery, capex at lower end of range."
        ),
        overrides={"cuf": 0.23, "capex_per_mw": 430.0},
        scenario_type="upside",
    ),
    Scenario(
        name="Downside Case",
        rationale=(
            "Mild operational underperformance: CUF −5% relative, "
            "O&M costs +10%, effective tariff −3% via curtailment."
        ),
        overrides={
            "cuf": 0.209,        # −5% relative to 0.22
            "opex_per_mw_pa": 8.8,  # +10%
            "tariff": 2.57,      # −3%
        },
        scenario_type="downside",
    ),
    Scenario(
        name="Construction Stress",
        rationale=(
            "Two-quarter construction delay pushes COD back, combined with "
            "12% capex overrun due to EPC cost inflation."
        ),
        overrides={
            "capex_per_mw": 504.0,   # +12% on ₹450 base
            "capex_schedule": [0.15, 0.20, 0.25, 0.25, 0.15],  # 5 quarters
        },
        scenario_type="stress",
    ),
    Scenario(
        name="Interest Rate Stress",
        rationale=(
            "RBI monetary tightening scenario: interest rate rises 150bps "
            "above base case (9.75% → 11.25% p.a.)."
        ),
        overrides={"interest_rate": 0.1125},
        scenario_type="stress",
    ),
    Scenario(
        name="Combined Stress",
        rationale=(
            "Simultaneous adverse events: 2-quarter delay + 12% capex overrun "
            "+ 150bps rate rise + CUF −5%. Tests project resilience."
        ),
        overrides={
            "capex_per_mw": 504.0,
            "capex_schedule": [0.15, 0.20, 0.25, 0.25, 0.15],
            "interest_rate": 0.1125,
            "cuf": 0.209,
        },
        scenario_type="combined",
    ),
    Scenario(
        name="Upside Case",
        rationale=(
            "Favourable conditions: CUF at P90 (0.25), lower capex (₹420/MW), "
            "accelerated depreciation benefit modelled via higher depreciation rate."
        ),
        overrides={
            "cuf": 0.25,
            "capex_per_mw": 420.0,
            "depreciation_rate": 0.10,  # proxy for accelerated depreciation
        },
        scenario_type="upside",
    ),
]


# ---------------------------------------------------------------------------
# Tool definition for single-run NL mode
# ---------------------------------------------------------------------------

_RUN_MODEL_TOOL: Dict[str, Any] = {
    "name": "run_model",
    "description": (
        "Run the solar IPP financial model with the given assumption overrides. "
        "Call this tool once you have identified which assumptions the user wants "
        "to change and what their new values should be. "
        "Common assumption names: capacity_mw, cuf, capex_per_mw, tariff, "
        "tariff_escalation, debt_pct, interest_rate, dscr_target, opex_per_mw_pa, "
        "opex_escalation, depreciation_rate, tax_rate, moratorium_periods, "
        "dsra_months, equity_irr_target, debt_sizing_mode."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "assumptions": {
                "type": "object",
                "description": (
                    "Dictionary mapping assumption names to their new numeric values. "
                    "Only include assumptions that differ from the base case."
                ),
                "additionalProperties": {"type": "number"},
            }
        },
        "required": ["assumptions"],
    },
}

_SINGLE_RUN_SYSTEM = """You are a financial-model assistant for a solar IPP project.
Translate the user's scenario description into assumption overrides and call run_model once.
Use decimal form for rates (e.g. 0.22 for 22% CUF, 0.0975 for 9.75% interest).
Only override assumptions that the user explicitly wants to change.
Do NOT perform any arithmetic — just identify the overrides."""

# Tool for proposing scenarios
_PROPOSE_SCENARIOS_TOOL: Dict[str, Any] = {
    "name": "propose_scenarios",
    "description": (
        "Propose additional financially coherent scenarios beyond the 7 standard ones. "
        "Each scenario must have correlated assumption changes that tell a plausible story."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "scenarios": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":          {"type": "string"},
                        "rationale":     {"type": "string"},
                        "scenario_type": {"type": "string",
                                          "enum": ["base", "upside", "downside",
                                                   "stress", "combined"]},
                        "overrides": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                        },
                    },
                    "required": ["name", "rationale", "scenario_type", "overrides"],
                },
            }
        },
        "required": ["scenarios"],
    },
}

_PROPOSE_SYSTEM = """You are a senior project finance analyst specialising in Indian solar IPP projects.
Propose additional stress and upside scenarios beyond the 7 standard ones already included.
Each scenario must:
- Have correlated assumption changes that tell a coherent story (don't move just one assumption)
- Be relevant to Karnataka solar: grid curtailment, off-taker credit, GST changes, etc.
- Use decimal form for all rates and ratios
- NOT repeat any of the 7 standard scenarios
Call propose_scenarios with the new scenarios list."""


# ---------------------------------------------------------------------------
# ScenarioAgent
# ---------------------------------------------------------------------------


class ScenarioAgent:
    """
    Agent for scenario construction, proposal, and execution.

    Parameters
    ----------
    compiled         : CompiledModel from ModelExecutor.compile().
    executor         : ModelExecutor instance.
    base_assumptions : Default overrides applied before any scenario changes.
    model            : Anthropic model name.
    client           : Pre-configured anthropic.Anthropic client (uses env var if None).
    """

    def __init__(
        self,
        compiled: CompiledModel,
        executor: ModelExecutor,
        base_assumptions: Optional[Dict[str, Any]] = None,
        model: str = "claude-sonnet-4-6",
        client: Optional[anthropic.Anthropic] = None,
    ) -> None:
        self.compiled = compiled
        self.executor = executor
        self.base_assumptions: Dict[str, Any] = base_assumptions or {}
        self.model = model
        self.client = client or anthropic.Anthropic()

    # ------------------------------------------------------------------
    # 1. Scenario proposal
    # ------------------------------------------------------------------

    def propose_scenarios(
        self,
        base_results: ModelResults,
        n_scenarios: int = 7,
    ) -> List[Scenario]:
        """
        Return a list of named scenarios for stress-testing the model.

        The first 7 entries are always the STANDARD_SCENARIOS constants.
        If n_scenarios > 7, Claude is asked to propose (n_scenarios − 7)
        additional correlated scenarios tailored to the base-case KPIs.

        Parameters
        ----------
        base_results : ModelResults from the base-case run.
        n_scenarios  : Total number of scenarios to return (minimum 7).

        Returns
        -------
        List[Scenario], length = max(n_scenarios, 7).
        """
        scenarios = list(STANDARD_SCENARIOS)

        extra_needed = max(0, n_scenarios - len(STANDARD_SCENARIOS))
        if extra_needed > 0:
            ai_scenarios = self._propose_with_claude(base_results, extra_needed)
            scenarios.extend(ai_scenarios)

        return scenarios

    def _propose_with_claude(
        self, base_results: ModelResults, n_extra: int
    ) -> List[Scenario]:
        """Ask Claude to propose n_extra additional scenarios."""
        kpis = base_results.kpis
        kpi_str = (
            f"equity_irr={kpis.equity_irr:.2%}" if kpis.equity_irr else ""
            + f", min_dscr={kpis.min_dscr:.3f}x" if kpis.min_dscr else ""
            + f", debt_payback={kpis.debt_payback_period:.1f}yr" if kpis.debt_payback_period else ""
        )
        assumptions_used = base_results.assumptions_used
        capacity = assumptions_used.get("capacity_mw", "100")
        tariff   = assumptions_used.get("tariff", "2.65")

        prompt = (
            f"Project: {capacity} MW Karnataka solar IPP, ₹{tariff}/kWh PPA.\n"
            f"Base-case KPIs: {kpi_str or 'not available'}.\n\n"
            f"The 7 standard scenarios (Base, Management, Downside, Construction Stress, "
            f"Interest Rate Stress, Combined Stress, Upside) are already included.\n"
            f"Propose {n_extra} additional financially coherent scenario(s) relevant to "
            f"Indian solar IPP risk factors."
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=_PROPOSE_SYSTEM,
            tools=[_PROPOSE_SCENARIOS_TOOL],
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": prompt}],
        )

        ai_scenarios: List[Scenario] = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "propose_scenarios":
                for s in block.input.get("scenarios", []):
                    ai_scenarios.append(Scenario(
                        name=s["name"],
                        rationale=s["rationale"],
                        overrides={k: float(v) for k, v in s.get("overrides", {}).items()},
                        scenario_type=s.get("scenario_type", "downside"),
                    ))
        return ai_scenarios

    # ------------------------------------------------------------------
    # 2. Batch scenario execution
    # ------------------------------------------------------------------

    def run_all_scenarios(
        self,
        compiled: Optional[CompiledModel] = None,
        scenarios: Optional[List[Scenario]] = None,
    ) -> Dict[str, ModelResults]:
        """
        Run every scenario and return a mapping of name → ModelResults.

        Parameters
        ----------
        compiled  : CompiledModel to use (defaults to self.compiled).
        scenarios : List of Scenario objects.  Defaults to STANDARD_SCENARIOS.

        Returns
        -------
        Dict[str, ModelResults] — key is scenario name, value is run output.
        """
        compiled  = compiled  or self.compiled
        scenarios = scenarios or list(STANDARD_SCENARIOS)

        assumptions_batch = [
            {**self.base_assumptions, **s.overrides} for s in scenarios
        ]
        results_list = self.executor.run_batch(compiled, assumptions_batch)

        return {s.name: r for s, r in zip(scenarios, results_list)}

    # ------------------------------------------------------------------
    # 3. Single NL → run (legacy mode, kept for backwards compatibility)
    # ------------------------------------------------------------------

    def run(
        self,
        scenario_description: str,
        explain: bool = False,
    ) -> Tuple[ModelResults, Optional[str]]:
        """
        Parse a natural-language scenario description, run the model, and
        optionally generate a short narrative.

        Parameters
        ----------
        scenario_description : Plain-English description of the scenario.
        explain              : If True, generate a brief narrative via Claude.

        Returns
        -------
        (ModelResults, narrative_str | None)
        """
        overrides = self._extract_overrides(scenario_description)
        assumptions = {**self.base_assumptions, **overrides}
        results = self.executor.run(self.compiled, assumptions)

        narrative: Optional[str] = None
        if explain:
            narrative = self._generate_narrative(scenario_description, overrides, results)

        return results, narrative

    def assumption_names(self) -> List[str]:
        """Return the list of assumption names from the model schema."""
        return [a.name for a in self.compiled.model_def.assumption_schema.assumptions]

    # ------------------------------------------------------------------
    # Private: single-run NL extraction
    # ------------------------------------------------------------------

    def _extract_overrides(self, scenario_description: str) -> Dict[str, Any]:
        messages = [{"role": "user", "content": scenario_description}]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SINGLE_RUN_SYSTEM,
            tools=[_RUN_MODEL_TOOL],
            tool_choice={"type": "auto"},
            messages=messages,
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "run_model":
                return {k: float(v) for k, v in block.input.get("assumptions", {}).items()}
        return {}

    def _generate_narrative(
        self,
        original_query: str,
        overrides: Dict[str, Any],
        results: ModelResults,
    ) -> str:
        kpis = results.kpis
        overrides_str = json.dumps(overrides, indent=2) if overrides else "(base case)"
        kpi_lines = _format_kpis_brief(kpis)

        prompt = (
            f'User asked: "{original_query}"\n'
            f"Assumption overrides: {overrides_str}\n\n"
            f"KPI results:\n{kpi_lines}\n\n"
            "Write 3–5 sentences interpreting these results for the project sponsor. "
            "Highlight bankability concerns if DSCR < 1.20× or equity IRR < 14%."
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _format_kpis_brief(kpis: "KPIResult") -> str:
    lines = []
    if kpis.equity_irr   is not None: lines.append(f"  Equity IRR:  {kpis.equity_irr:.2%}")
    if kpis.project_irr  is not None: lines.append(f"  Project IRR: {kpis.project_irr:.2%}")
    if kpis.min_dscr     is not None: lines.append(f"  Min DSCR:    {kpis.min_dscr:.3f}x")
    if kpis.llcr         is not None: lines.append(f"  LLCR:        {kpis.llcr:.3f}x")
    if kpis.npv_equity   is not None: lines.append(f"  NPV equity:  {kpis.npv_equity:,.0f} INR Lakhs")
    if kpis.debt_payback_period is not None:
        lines.append(f"  Debt payback: {kpis.debt_payback_period:.1f} years")
    return "\n".join(lines) if lines else "  (no KPIs computed)"
