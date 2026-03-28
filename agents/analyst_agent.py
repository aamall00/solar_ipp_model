"""
agents/analyst_agent.py — Narrative financial analysis agent.

AnalystAgent takes a ModelResults object (and optionally sensitivity /
Monte Carlo results) and generates a structured financial analysis report
using the Claude API.

The agent does NOT perform any arithmetic — it formats the pre-computed
numeric results into a structured prompt and asks Claude to write the
interpretation.

Usage
-----
    from agents.analyst_agent import AnalystAgent
    from engine.executor import ModelExecutor

    executor = ModelExecutor()
    compiled = executor.compile(model_def)
    results  = executor.run(compiled, {})

    agent = AnalystAgent()
    report = agent.analyse(results)
    print(report)

    # With sensitivity data
    sensitivity = executor.run_sensitivity(compiled, {}, sweep_config)
    report = agent.analyse(results, sensitivity_results=sensitivity)
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

import anthropic

from engine.executor import ModelExecutor, ModelResults, MonteCarloResults, SensitivityResults

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a senior infrastructure finance analyst specialising in solar IPP projects in India.
You write clear, concise financial reports for project sponsors, lenders, and equity investors.

Your reports must:
- Be factual and grounded in the numbers provided.
- Highlight bankability concerns when DSCR < 1.20× or equity IRR < 14%.
- Use plain business English — avoid excessive jargon.
- Be structured with clear sections.
- Never invent numbers not present in the input data.
"""

_ANALYSIS_SECTIONS = [
    "Executive Summary",
    "Returns Assessment",
    "Debt Service & Coverage",
    "Cash Flow Profile",
    "Key Risks & Mitigants",
]


class AnalystAgent:
    """
    Agent that generates a structured financial analysis narrative from model results.

    Parameters
    ----------
    model  : Anthropic model to use.
    client : Pre-configured anthropic.Anthropic client.
             If None, one is created using the ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        client: Optional[anthropic.Anthropic] = None,
    ) -> None:
        self.model = model
        self.client = client or anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        results: ModelResults,
        sensitivity_results: Optional[SensitivityResults] = None,
        mc_results: Optional[MonteCarloResults] = None,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate a full financial analysis report.

        Parameters
        ----------
        results              : ModelResults from a single base-case run.
        sensitivity_results  : Optional SensitivityResults for tornado commentary.
        mc_results           : Optional MonteCarloResults for probability commentary.
        max_tokens           : Maximum tokens in the generated report.

        Returns
        -------
        Formatted report string.
        """
        prompt = self._build_prompt(results, sensitivity_results, mc_results)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text if response.content else ""

    def kpi_summary(self, results: ModelResults) -> str:
        """Return a short (≤ 10 line) plain-text KPI table."""
        return _format_kpi_table(results.kpis)

    def bankability_verdict(self, results: ModelResults) -> str:
        """
        Return a one-line bankability verdict based on min_dscr and equity_irr.
        """
        kpis = results.kpis
        issues: List[str] = []

        if kpis.min_dscr is not None and kpis.min_dscr < 1.10:
            issues.append(f"min_dscr = {kpis.min_dscr:.3f}× (below 1.10×)")
        if kpis.equity_irr is not None and kpis.equity_irr < 0.14:
            issues.append(f"equity_irr = {kpis.equity_irr:.2%} (below 14% hurdle)")

        if not issues:
            return "BANKABLE — KPIs within acceptable ranges."
        return "CONCERNS: " + "; ".join(issues)

    # ------------------------------------------------------------------
    # Private: prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        results: ModelResults,
        sensitivity_results: Optional[SensitivityResults],
        mc_results: Optional[MonteCarloResults],
    ) -> str:
        sections: List[str] = []

        # --- Model metadata ---
        sections.append(f"## Model: {results.model_id}")

        # --- Assumptions used ---
        assumptions_str = _format_assumptions(results.assumptions_used)
        sections.append(f"### Key Assumptions\n{assumptions_str}")

        # --- KPI table ---
        sections.append(f"### KPI Results\n{_format_kpi_table(results.kpis)}")

        # --- Solve-loop convergence ---
        if results.convergence:
            conv_lines = [
                f"  {m.loop_id}: converged={m.converged}, "
                f"iterations={m.iterations}, residual={m.final_residual:.2e}"
                for m in results.convergence
            ]
            sections.append("### Solve-Loop Convergence\n" + "\n".join(conv_lines))

        # --- Warnings ---
        if results.warnings:
            warn_lines = [f"  ⚠ {w}" for w in results.warnings]
            sections.append("### Model Warnings\n" + "\n".join(warn_lines))

        # --- Sensitivity ---
        if sensitivity_results is not None:
            sections.append(
                "### Sensitivity (OAT Tornado — equity_irr impact)\n"
                + _format_tornado(sensitivity_results)
            )

        # --- Monte Carlo ---
        if mc_results is not None:
            sections.append(
                f"### Monte Carlo ({mc_results.n_iterations} iterations)\n"
                + _format_mc(mc_results)
            )

        # --- Instruction ---
        sections.append(
            "Based on the data above, write a structured financial analysis report "
            f"covering the following sections: {', '.join(_ANALYSIS_SECTIONS)}. "
            "Be concise and quantitative. Flag any bankability concerns clearly."
        )

        return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_kpi_table(kpis: "KPIResult") -> str:
    rows = []
    _add_row(rows, "Equity IRR",             kpis.equity_irr,             fmt=".2%")
    _add_row(rows, "Project IRR",            kpis.project_irr,            fmt=".2%")
    _add_row(rows, "Min DSCR",               kpis.min_dscr,               fmt=".3f", suffix="×")
    _add_row(rows, "Avg DSCR",               kpis.avg_dscr,               fmt=".3f", suffix="×")
    _add_row(rows, "LLCR",                   kpis.llcr,                   fmt=".3f", suffix="×")
    _add_row(rows, "PLCR",                   kpis.plcr,                   fmt=".3f", suffix="×")
    _add_row(rows, "NPV (equity)",           kpis.npv_equity,             fmt=",.0f", suffix=" INR Lakhs")
    _add_row(rows, "Debt payback",           kpis.debt_payback_period,    fmt=".1f",  suffix=" years")
    _add_row(rows, "Peak debt outstanding",  kpis.peak_debt_outstanding,  fmt=",.0f", suffix=" INR Lakhs")
    return "\n".join(rows) if rows else "  (no KPIs computed)"


def _add_row(
    rows: List[str],
    label: str,
    value: Optional[float],
    fmt: str = ".4f",
    suffix: str = "",
) -> None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return
    formatted = format(value, fmt)
    rows.append(f"  {label:<28} {formatted}{suffix}")


def _format_assumptions(assumptions: Dict[str, Any]) -> str:
    _SHOW_KEYS = (
        "capacity_mw", "cuf", "capex_per_mw", "tariff", "debt_pct",
        "interest_rate", "dscr_target", "opex_per_mw_pa", "tax_rate",
        "depreciation_rate", "equity_irr_target",
    )
    lines = []
    for key in _SHOW_KEYS:
        if key in assumptions:
            val = assumptions[key]
            if isinstance(val, float):
                lines.append(f"  {key}: {val}")
            else:
                lines.append(f"  {key}: {val}")
    return "\n".join(lines) if lines else "  (defaults)"


def _format_tornado(sensitivity_results: SensitivityResults) -> str:
    items = sorted(
        sensitivity_results.tornado_data.items(),
        key=lambda kv: abs(kv[1][1] - kv[1][0]),
        reverse=True,
    )
    lines = []
    for name, (lo, hi) in items:
        lines.append(
            f"  {name:<20}  low: {lo:+.2%}   high: {hi:+.2%}   spread: {hi - lo:.2%}"
        )
    return "\n".join(lines) if lines else "  (no data)"


def _format_mc(mc: MonteCarloResults) -> str:
    lines = [
        f"  {'KPI':<16} {'P10':>8} {'P50':>8} {'P90':>8}",
        f"  {'-'*46}",
    ]
    kpi_names = ["equity_irr", "min_dscr", "llcr", "npv_equity"]
    fmt_map = {
        "equity_irr": ".2%",
        "min_dscr":   ".3f",
        "llcr":       ".3f",
        "npv_equity": ",.0f",
    }
    for name in kpi_names:
        p10 = mc.p10.get(name)
        p50 = mc.p50.get(name)
        p90 = mc.p90.get(name)
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (p10, p50, p90)):
            continue
        f = fmt_map.get(name, ".4f")
        lines.append(
            f"  {name:<16}  {format(p10, f):>8}  {format(p50, f):>8}  {format(p90, f):>8}"
        )
    lines.append(f"  Prob(DSCR < 1.0):       {mc.prob_dscr_below_1:.1%}")
    lines.append(f"  Prob(IRR < hurdle):     {mc.prob_irr_below_hurdle:.1%}")
    return "\n".join(lines)
