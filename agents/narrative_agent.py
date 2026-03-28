"""
agents/narrative_agent.py — Output Layer: Structured Bankability Report.

Produces a six-section financial analysis report:

  Section 1 — Executive Summary          (Claude — qualitative interpretation)
  Section 2 — Key Metrics Table          (pure Python — multi-scenario comparison)
  Section 3 — Binding Constraint         (pure Python — period-level DSCR analysis)
  Section 4 — Sensitivity Narrative      (pure Python — tornado data formatted to prose)
  Section 5 — Risk Flags                 (Claude — qualitative risk interpretation)
  Section 6 — Structural Recommendations (Claude — advisory based on flags)

Sections 2–4 are fully deterministic.  Sections 1, 5, 6 use Claude for
interpretation only — no arithmetic.

Usage
-----
    from agents.narrative_agent import NarrativeAgent

    agent   = NarrativeAgent()
    report  = agent.report(
        base_results=base_results,
        scenario_results=all_scenario_results,   # dict name → ModelResults
        sensitivity_results=sensitivity,
        mc_results=mc,
        compiled=compiled,
    )
    print(report.full_text)
    print(report.binding_constraint)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import anthropic

from engine.executor import (
    CompiledModel,
    ModelResults,
    MonteCarloResults,
    SensitivityResults,
)

# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------


@dataclass
class NarrativeReport:
    """
    Structured output of NarrativeAgent.report().

    All sections are plain text (markdown-friendly).  full_text is the
    concatenation of all six sections with headers.
    """
    executive_summary:      str = ""
    metrics_table:          str = ""
    binding_constraint:     str = ""
    sensitivity_narrative:  str = ""
    risk_flags:             str = ""
    recommendations:        str = ""
    full_text:              str = ""


# ---------------------------------------------------------------------------
# Claude system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a senior infrastructure finance analyst writing a bankability assessment
for a solar IPP project in Karnataka, India.  Your audience is the project sponsor and prospective
senior lenders.

Guidelines:
- Be specific: always cite the exact KPI numbers provided.
- Flag any scenario where min DSCR < 1.10× or equity IRR < 14% as a bankability concern.
- Do NOT invent numbers not present in the input data.
- Write in clear, professional business English.
- Each section you produce should be 3–6 sentences unless otherwise instructed.
"""

# KPI display names and format strings
_KPI_DISPLAY: List[Tuple[str, str, str]] = [
    # (attr, label, format)
    ("equity_irr",           "Equity IRR",            ".2%"),
    ("project_irr",          "Project IRR",           ".2%"),
    ("min_dscr",             "Min DSCR",              ".3f"),
    ("avg_dscr",             "Avg DSCR",              ".3f"),
    ("llcr",                 "LLCR",                  ".3f"),
    ("plcr",                 "PLCR",                  ".3f"),
    ("npv_equity",           "NPV Equity (₹ Lakh)",   ",.0f"),
    ("debt_payback_period",  "Debt Payback (yr)",     ".1f"),
    ("peak_debt_outstanding","Peak Debt (₹ Lakh)",    ",.0f"),
]


class NarrativeAgent:
    """
    Generates a structured six-section bankability report.

    Parameters
    ----------
    model  : Anthropic model to use for narrative sections.
    client : Pre-configured anthropic.Anthropic client (uses env var if None).
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

    def report(
        self,
        base_results: ModelResults,
        scenario_results: Optional[Dict[str, ModelResults]] = None,
        sensitivity_results: Optional[SensitivityResults] = None,
        mc_results: Optional[MonteCarloResults] = None,
        compiled: Optional[CompiledModel] = None,
        max_tokens: int = 3000,
    ) -> NarrativeReport:
        """
        Generate the full six-section bankability report.

        Parameters
        ----------
        base_results         : ModelResults from the base-case run.
        scenario_results     : Optional dict of {scenario_name → ModelResults}.
        sensitivity_results  : Optional OAT sensitivity output.
        mc_results           : Optional Monte Carlo output.
        compiled             : CompiledModel (needed for period metadata).
        max_tokens           : Max tokens for the Claude narrative sections.

        Returns
        -------
        NarrativeReport with each section populated and full_text assembled.
        """
        # --- Deterministic sections (Sections 2-4) ---
        s2_table       = self._build_metrics_table(base_results, scenario_results)
        s3_constraint  = self._identify_binding_constraint(base_results, compiled)
        s4_sensitivity = self._build_sensitivity_narrative(sensitivity_results, base_results)

        # --- Claude sections (1, 5, 6) ---
        context = self._build_claude_context(
            base_results, scenario_results, sensitivity_results,
            mc_results, s3_constraint, s4_sensitivity,
        )
        s1_exec, s5_risks, s6_recs = self._generate_ai_sections(context, max_tokens)

        # --- Assemble ---
        full_text = _assemble(
            s1_exec, s2_table, s3_constraint,
            s4_sensitivity, s5_risks, s6_recs,
        )

        return NarrativeReport(
            executive_summary=s1_exec,
            metrics_table=s2_table,
            binding_constraint=s3_constraint,
            sensitivity_narrative=s4_sensitivity,
            risk_flags=s5_risks,
            recommendations=s6_recs,
            full_text=full_text,
        )

    # ------------------------------------------------------------------
    # Section 2: Metrics table (pure Python)
    # ------------------------------------------------------------------

    def _build_metrics_table(
        self,
        base_results: ModelResults,
        scenario_results: Optional[Dict[str, ModelResults]],
    ) -> str:
        """Build a markdown table with base case + all scenarios."""
        # Collect all results in order
        all_results: List[Tuple[str, ModelResults]] = [("Base Case", base_results)]
        if scenario_results:
            for name, res in scenario_results.items():
                if name != "Base Case":
                    all_results.append((name, res))

        # Header
        col_names  = [name for name, _ in all_results]
        header_pad = 30
        col_width  = max(12, max(len(n) for n in col_names) + 2)

        lines: List[str] = []
        # Top header row
        header = f"{'KPI':<{header_pad}}" + "".join(f"{n:>{col_width}}" for n in col_names)
        lines.append(header)
        lines.append("-" * len(header))

        for attr, label, fmt in _KPI_DISPLAY:
            row = f"{label:<{header_pad}}"
            for _, res in all_results:
                val = getattr(res.kpis, attr, None)
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    cell = "—"
                else:
                    cell = format(val, fmt)
                    # Add × suffix for coverage ratios
                    if attr in ("min_dscr", "avg_dscr", "llcr", "plcr"):
                        cell += "×"
                row += f"{cell:>{col_width}}"
            lines.append(row)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Section 3: Binding constraint (pure Python)
    # ------------------------------------------------------------------

    def _identify_binding_constraint(
        self,
        base_results: ModelResults,
        compiled: Optional[CompiledModel],
    ) -> str:
        """
        Find when and where the minimum DSCR occurs and characterise the
        binding constraint (debt service coverage, free cashflow, or equity IRR).
        Returns a 2–3 sentence plain-text description.
        """
        ppy   = compiled.periods_per_year if compiled else 4
        cod   = compiled.cod_period       if compiled else 4

        cfads = base_results.variables.get("cashflow_block.cfads")
        ds    = base_results.variables.get("debt_service_block.total_debt_service")
        bal   = base_results.variables.get("debt_service_block.outstanding_debt_balance")
        fc    = base_results.variables.get("cashflow_block.free_cashflow")

        if cfads is None or ds is None:
            return "Binding constraint analysis unavailable: CFADS or debt service arrays not found."

        parts: List[str] = []

        # --- Minimum DSCR ---
        debt_periods = np.where(ds > 0.01)[0]
        if len(debt_periods) > 0:
            dscr_arr = np.where(ds > 0.01, cfads / np.where(ds > 0.01, ds, np.nan), np.nan)
            min_idx  = int(np.nanargmin(dscr_arr))
            min_val  = float(np.nanmin(dscr_arr))
            period_from_cod = min_idx - cod
            year_from_cod   = period_from_cod / ppy
            q_in_year       = (period_from_cod % ppy) + 1
            year_label      = f"Year {int(math.ceil(year_from_cod))}, Q{q_in_year}"
            parts.append(
                f"Minimum DSCR of {min_val:.3f}× occurs in period {min_idx} "
                f"({year_label} post-COD)."
            )
            if min_val < 1.10:
                parts.append(
                    f"This is below the typical lender covenant of 1.10× and represents "
                    f"a hard bankability constraint that must be resolved."
                )
            elif min_val < 1.20:
                parts.append(
                    f"This is below the DSCR sculpting target of 1.20× — consider "
                    f"switching to CFADS-based debt sizing (debt_sizing_mode = 1.0)."
                )
        else:
            parts.append("No debt service periods found in the model output.")

        # --- Negative free cashflow ---
        if fc is not None:
            neg_periods = int(np.sum(fc < 0))
            if neg_periods > 0:
                parts.append(
                    f"Free cashflow is negative in {neg_periods} period(s), "
                    f"implying equity injection or reserve drawdown would be required."
                )

        # --- Equity IRR vs target ---
        hurdle = float(base_results.assumptions_used.get("equity_irr_target", 0.14))
        irr    = base_results.kpis.equity_irr
        if irr is not None:
            if irr < hurdle:
                parts.append(
                    f"Equity IRR of {irr:.2%} is below the {hurdle:.0%} hurdle rate — "
                    f"the project does not clear the sponsor's return threshold."
                )

        return "  ".join(parts) if parts else "No binding constraint identified."

    # ------------------------------------------------------------------
    # Section 4: Sensitivity narrative (pure Python)
    # ------------------------------------------------------------------

    def _build_sensitivity_narrative(
        self,
        sensitivity_results: Optional[SensitivityResults],
        base_results: ModelResults,
    ) -> str:
        """
        Convert tornado chart data to 3–5 quantitative sentences.
        Ranks assumptions by absolute impact (spread) on equity_irr.
        """
        if sensitivity_results is None or not sensitivity_results.tornado_data:
            return "Sensitivity analysis was not performed."

        base_irr = sensitivity_results.base_kpis.equity_irr
        if base_irr is None:
            return "Sensitivity analysis available but base equity IRR could not be computed."

        # Rank by absolute spread
        ranked = sorted(
            sensitivity_results.tornado_data.items(),
            key=lambda kv: abs(kv[1][1] - kv[1][0]),
            reverse=True,
        )

        lines: List[str] = [
            f"Equity IRR sensitivity analysis (base case: {base_irr:.2%}):"
        ]

        for i, (name, (lo_delta, hi_delta)) in enumerate(ranked[:5]):
            lo_val = base_irr + lo_delta
            hi_val = base_irr + hi_delta
            spread_bps = abs(hi_delta - lo_delta) * 10000
            direction  = "increases" if hi_delta > lo_delta else "decreases"
            lines.append(
                f"  {i+1}. {name}: IRR range {lo_val:.2%} – {hi_val:.2%} "
                f"(spread {spread_bps:.0f}bps; higher {name} {direction} IRR)."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Claude narrative sections (1, 5, 6)
    # ------------------------------------------------------------------

    def _build_claude_context(
        self,
        base_results: ModelResults,
        scenario_results: Optional[Dict[str, ModelResults]],
        sensitivity_results: Optional[SensitivityResults],
        mc_results: Optional[MonteCarloResults],
        s3_constraint: str,
        s4_sensitivity: str,
    ) -> str:
        """Assemble the full context block sent to Claude."""
        parts: List[str] = []

        # Model identity
        parts.append(f"## Model: {base_results.model_id}")
        parts.append(f"### Base-Case Assumptions\n{_format_assumptions(base_results.assumptions_used)}")

        # Base-case KPIs
        parts.append(f"### Base-Case KPIs\n{_format_kpi_block(base_results.kpis)}")

        # Scenario table (abbreviated)
        if scenario_results:
            rows = ["Scenario | Equity IRR | Min DSCR | NPV Equity"]
            rows.append("---|---|---|---")
            for name, res in scenario_results.items():
                irr  = f"{res.kpis.equity_irr:.2%}" if res.kpis.equity_irr is not None else "—"
                dscr = f"{res.kpis.min_dscr:.3f}×"  if res.kpis.min_dscr   is not None else "—"
                npv  = f"{res.kpis.npv_equity:,.0f}" if res.kpis.npv_equity is not None else "—"
                rows.append(f"{name} | {irr} | {dscr} | {npv}")
            parts.append("### Scenario Summary\n" + "\n".join(rows))

        # Binding constraint and sensitivity (deterministic output)
        parts.append(f"### Binding Constraint\n{s3_constraint}")
        parts.append(f"### Sensitivity Summary\n{s4_sensitivity}")

        # Convergence warnings
        if base_results.warnings:
            parts.append("### Model Warnings\n" + "\n".join(f"- {w}" for w in base_results.warnings))

        # Monte Carlo if available
        if mc_results is not None:
            mc_block = (
                f"Monte Carlo ({mc_results.n_iterations} iterations):\n"
                f"  Equity IRR P10/P50/P90: {mc_results.p10.get('equity_irr', float('nan')):.2%} / "
                f"{mc_results.p50.get('equity_irr', float('nan')):.2%} / "
                f"{mc_results.p90.get('equity_irr', float('nan')):.2%}\n"
                f"  Min DSCR  P10/P50/P90: {mc_results.p10.get('min_dscr', float('nan')):.3f}× / "
                f"{mc_results.p50.get('min_dscr', float('nan')):.3f}× / "
                f"{mc_results.p90.get('min_dscr', float('nan')):.3f}×\n"
                f"  Prob(DSCR < 1.0×): {mc_results.prob_dscr_below_1:.1%}\n"
                f"  Prob(IRR < hurdle): {mc_results.prob_irr_below_hurdle:.1%}"
            )
            parts.append(f"### Monte Carlo Results\n{mc_block}")

        return "\n\n".join(parts)

    def _generate_ai_sections(
        self,
        context: str,
        max_tokens: int,
    ) -> Tuple[str, str, str]:
        """
        Ask Claude to write Sections 1, 5, and 6 using a single API call.
        Returns (executive_summary, risk_flags, recommendations).
        """
        prompt = (
            f"{context}\n\n"
            "Using the data above, write the following three sections separated by "
            "exactly these headers (do not add any other headers):\n\n"
            "### EXECUTIVE SUMMARY\n"
            "(3–4 sentences: overall bankability verdict, key KPI highlights, "
            "most critical risk)\n\n"
            "### RISK FLAGS\n"
            "(Bullet-point list of scenarios or conditions where DSCR < 1.10× or "
            "equity IRR < 14%; each flag should cite the specific scenario name and value)\n\n"
            "### STRUCTURAL RECOMMENDATIONS\n"
            "(3–5 actionable recommendations to improve bankability if risks were flagged, "
            "or to confirm robustness if the project is bankable; be specific and quantitative)"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text if response.content else ""
        return _parse_ai_response(raw)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_ai_response(raw: str) -> Tuple[str, str, str]:
    """Split Claude's response into the three sections."""
    exec_summary = ""
    risk_flags   = ""
    recs         = ""

    current = None
    buffer: List[str] = []

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped == "### EXECUTIVE SUMMARY":
            if current == "risk":   risk_flags   = "\n".join(buffer).strip()
            elif current == "exec": exec_summary = "\n".join(buffer).strip()
            buffer  = []
            current = "exec"
        elif stripped == "### RISK FLAGS":
            if current == "exec":   exec_summary = "\n".join(buffer).strip()
            buffer  = []
            current = "risk"
        elif stripped == "### STRUCTURAL RECOMMENDATIONS":
            if current == "exec":   exec_summary = "\n".join(buffer).strip()
            elif current == "risk": risk_flags   = "\n".join(buffer).strip()
            buffer  = []
            current = "recs"
        else:
            buffer.append(line)

    # Flush last buffer
    if current == "exec":   exec_summary = "\n".join(buffer).strip()
    elif current == "risk": risk_flags   = "\n".join(buffer).strip()
    elif current == "recs": recs         = "\n".join(buffer).strip()

    return exec_summary, risk_flags, recs


def _assemble(s1: str, s2: str, s3: str, s4: str, s5: str, s6: str) -> str:
    sections = [
        ("1. Executive Summary",            s1),
        ("2. Key Metrics Table",            s2),
        ("3. Binding Constraint",           s3),
        ("4. Sensitivity Narrative",        s4),
        ("5. Risk Flags",                   s5),
        ("6. Structural Recommendations",   s6),
    ]
    parts = []
    for title, content in sections:
        if content.strip():
            parts.append(f"## {title}\n\n{content.strip()}")
    return "\n\n---\n\n".join(parts)


def _format_kpi_block(kpis: "KPIResult") -> str:
    rows = []
    for attr, label, fmt in _KPI_DISPLAY:
        val = getattr(kpis, attr, None)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        suffix = "×" if attr in ("min_dscr", "avg_dscr", "llcr", "plcr") else ""
        rows.append(f"  {label:<28} {format(val, fmt)}{suffix}")
    return "\n".join(rows) if rows else "  (no KPIs computed)"


def _format_assumptions(assumptions: Dict[str, Any]) -> str:
    _SHOW = ("capacity_mw", "cuf", "capex_per_mw", "tariff", "debt_pct",
             "interest_rate", "dscr_target", "opex_per_mw_pa", "tax_rate",
             "equity_irr_target", "debt_sizing_mode")
    lines = []
    for k in _SHOW:
        if k in assumptions:
            v = assumptions[k]
            lines.append(f"  {k}: {v}")
    return "\n".join(lines) if lines else "  (defaults)"
