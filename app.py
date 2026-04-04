"""
app.py — Streamlit UI for the IPP Portfolio Financial Model.

Run with:
    cd solar_ipp_model
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from env_loader import load_local_env
load_local_env()

from agents.portfolio_agent import PortfolioAgent
from agents.dashboard_agent import DashboardAgent
from engine.portfolio_runner import PortfolioRunner
from engine.excel_exporter import export_portfolio_to_excel


# ---------------------------------------------------------------------------
# Model context builder
# ---------------------------------------------------------------------------

def _build_model_context(ar) -> Dict[str, Any]:
    """
    Build a compact quarterly model context dict (~5 KB) for Claude.
    Quarterly series are sliced at COD (operations periods only), except
    equity_cashflow and debt_drawdown which span the full project life.
    """
    skel = ar.model_def.project_skeleton
    ppy  = skel.periods_per_year
    cod  = skel.milestones.cod
    vars_ = ar.model_results.variables

    def _ops(key: str) -> List[float]:
        arr = vars_.get(key)
        if arr is None:
            return []
        return [round(float(x), 1) for x in arr[cod:]]

    def _full(key: str) -> List[float]:
        arr = vars_.get(key)
        if arr is None:
            return []
        return [round(float(x), 1) for x in arr]

    # DSCR — avoid division by zero
    cfads_arr = vars_.get("cashflow_block.cfads", np.zeros(skel.total_periods))
    ds_arr    = vars_.get("debt_service_block.total_debt_service", np.zeros(skel.total_periods))
    ops_cfads = cfads_arr[cod:]
    ops_ds    = ds_arr[cod:]
    with np.errstate(divide="ignore", invalid="ignore"):
        dscr_arr = np.where(ops_ds > 0, ops_cfads / ops_ds, None)
    dscr_list = [round(float(x), 3) if x is not None else None for x in dscr_arr]

    k = ar.model_results.kpis
    assumptions_used = ar.model_results.assumptions_used or {}
    _ASSUMPTION_KEYS = {
        "capacity_mw", "tariff", "debt_pct", "interest_rate", "debt_tenor_years",
        "cuf", "capex_per_mw", "ppa_tenor_years", "dsra_months", "opex_per_mw_pa",
    }

    return {
        "asset": {
            "name": ar.spec.name,
            "type": ar.spec.asset_type,
            "spv":  ar.spec.spv_name,
        },
        "assumptions": {
            k_: round(float(v), 4) if isinstance(v, float) else v
            for k_, v in assumptions_used.items()
            if k_ in _ASSUMPTION_KEYS
        },
        "kpis": {
            "equity_irr_pct":     round(k.equity_irr * 100, 2)       if k.equity_irr            else None,
            "project_irr_pct":    round(k.project_irr * 100, 2)      if k.project_irr           else None,
            "min_dscr":           round(k.min_dscr, 3)               if k.min_dscr              else None,
            "avg_dscr":           round(k.avg_dscr, 3)               if k.avg_dscr              else None,
            "llcr":               round(k.llcr, 3)                   if k.llcr                  else None,
            "npv_equity_lakhs":   round(k.npv_equity, 1)             if k.npv_equity            else None,
            "peak_debt_lakhs":    round(k.peak_debt_outstanding, 1)  if k.peak_debt_outstanding else None,
            "debt_payback_years": round(k.debt_payback_period, 1)    if k.debt_payback_period   else None,
        },
        "periods": {
            "construction": skel.construction_periods,
            "operations":   skel.operations_periods,
            "ppy":          ppy,
            "cod":          cod,
            "total":        skel.total_periods,
        },
        "quarterly_series": {
            "revenue":          _ops("revenue_block.revenue"),
            "total_opex":       _ops("opex_block.total_opex"),
            "ebitda":           _ops("income_statement_block.ebitda"),
            "cfads":            _ops("cashflow_block.cfads"),
            "debt_service":     _ops("debt_service_block.total_debt_service"),
            "dscr":             dscr_list,
            "debt_balance":     _ops("debt_service_block.outstanding_debt_balance"),
            "principal":        _ops("debt_service_block.principal_repayment"),
            "interest":         _ops("debt_service_block.interest_payment"),
            "dsra_balance":     _ops("dsra_block.closing_balance"),
            "equity_dist":      _ops("waterfall_block.equity_distribution"),
            "tax":              _ops("income_statement_block.tax"),
            "free_cashflow":    _ops("cashflow_block.free_cashflow"),
            # Full project life (construction + operations)
            "equity_cashflow":  _full("cashflow_block.equity_cashflow"),
            "debt_drawdown":    _full("debt_drawdown_block.drawdown"),
        },
    }


# ---------------------------------------------------------------------------
# Standard chart renderer
# ---------------------------------------------------------------------------

def _render_standard_charts(ar, ctx: Dict[str, Any]) -> None:
    """Render the 7 hardcoded standard financial charts for an asset."""
    qs   = ctx["quarterly_series"]
    skel = ar.model_def.project_skeleton
    n_ops  = len(qs["cfads"])
    n_full = skel.total_periods
    ops_idx  = list(range(1, n_ops + 1))
    full_idx = list(range(1, n_full + 1))

    # Row 1 — CFADS vs Debt Service | DSCR
    col1, col2 = st.columns(2, gap="large")
    with col1:
        _render_chart_heading(
            "CFADS vs Debt Service",
            "Operating cash available versus scheduled lender claims.",
        )
        df = pd.DataFrame(
            {"CFADS": qs["cfads"], "Debt Service": qs["debt_service"]},
            index=ops_idx,
        )
        st.line_chart(df)

    with col2:
        _render_chart_heading(
            "DSCR by Quarter",
            "Coverage trend across the operating life.",
        )
        dscr = [x if x is not None else float("nan") for x in qs["dscr"]]
        df2 = pd.DataFrame({"DSCR": dscr}, index=ops_idx)
        st.line_chart(df2)
        k = ar.model_results.kpis
        if k.min_dscr and k.avg_dscr:
            st.caption(f"Min {k.min_dscr:.3f}x  |  Avg {k.avg_dscr:.3f}x")

    # Row 2 — Revenue / OPEX / EBITDA | Debt Balance
    col3, col4 = st.columns(2, gap="large")
    with col3:
        _render_chart_heading(
            "Revenue, OPEX and EBITDA",
            "Top-line build, operating cost load, and earnings profile.",
        )
        df3 = pd.DataFrame(
            {"Revenue": qs["revenue"], "OPEX": qs["total_opex"], "EBITDA": qs["ebitda"]},
            index=ops_idx,
        )
        st.line_chart(df3)

    with col4:
        _render_chart_heading(
            "Outstanding Debt Balance",
            "How quickly the debt stack amortizes over time.",
        )
        df4 = pd.DataFrame({"Debt Balance": qs["debt_balance"]}, index=ops_idx)
        st.area_chart(df4)

    # Row 3 — Principal + Interest | DSRA
    col5, col6 = st.columns(2, gap="large")
    with col5:
        _render_chart_heading(
            "Debt Service Components",
            "Principal and interest split through the repayment schedule.",
        )
        df5 = pd.DataFrame(
            {"Principal": qs["principal"], "Interest": qs["interest"]},
            index=ops_idx,
        )
        st.bar_chart(df5, stack=True)

    with col6:
        dsra = qs["dsra_balance"]
        if any(v is not None and v > 0 for v in dsra):
            _render_chart_heading(
                "DSRA Balance",
                "Reserve build-up and release across debt periods.",
            )
            df6 = pd.DataFrame({"DSRA": dsra}, index=ops_idx)
            st.area_chart(df6)
        else:
            _render_chart_heading(
                "DSRA Balance",
                "This asset does not maintain a DSRA in the current configuration.",
            )
            st.info("DSRA is inactive for this asset.")

    # Row 4 — Equity J-curve (full width)
    eq_cf = qs["equity_cashflow"]
    if eq_cf:
        _render_chart_heading(
            "Equity Cashflow J-Curve",
            "Construction drag followed by cumulative equity recovery.",
        )
        cumulative_eq = list(np.cumsum(eq_cf))
        df7 = pd.DataFrame({"Cumulative Equity CF": cumulative_eq}, index=full_idx)
        st.line_chart(df7)


# ---------------------------------------------------------------------------
# AI-suggested chart renderer
# ---------------------------------------------------------------------------

def _render_ai_charts(ctx: Dict[str, Any], ai_specs: dict) -> None:
    """Render the AI-proposed additional charts in pairs."""
    charts  = ai_specs.get("charts", [])
    series  = ctx["quarterly_series"]
    n_ops   = len(series["cfads"])
    n_full  = int(ctx.get("periods", {}).get("total", n_ops))
    ops_idx = list(range(1, n_ops + 1))
    full_idx = list(range(1, n_full + 1))

    pairs = [charts[i:i + 2] for i in range(0, len(charts), 2)]
    for pair in pairs:
        cols = st.columns(len(pair), gap="large")
        for col, spec in zip(cols, pair):
            with col:
                _render_chart_heading(
                    spec["title"],
                    spec.get("insight"),
                )
                keys = [s for s in spec.get("series", []) if s in series]
                if not keys:
                    st.caption("(data not available)")
                    continue
                raw_data = {s: list(series[s]) for s in keys}
                lengths = {len(v) for v in raw_data.values()}

                # AI-suggested charts can mix operations-only series (length = n_ops)
                # with full-life series such as equity_cashflow or debt_drawdown
                # (length = n_full). Align mixed charts to the operations window so
                # pandas receives uniformly sized columns and the x-axis remains coherent.
                if len(lengths) == 1:
                    target_len = next(iter(lengths))
                elif n_ops in lengths:
                    target_len = n_ops
                else:
                    target_len = min(lengths)

                normalized: Dict[str, List[Any]] = {}
                for name, values in raw_data.items():
                    if len(values) == target_len:
                        normalized[name] = values
                    elif len(values) > target_len:
                        normalized[name] = values[-target_len:]
                    else:
                        normalized[name] = values + [float("nan")] * (target_len - len(values))

                index = full_idx[:target_len] if target_len == n_full else ops_idx[:target_len]
                df = pd.DataFrame(normalized, index=index)
                ctype = spec.get("chart_type", "line")
                if ctype == "stacked_bar":
                    st.bar_chart(df, stack=True)
                elif ctype == "bar":
                    st.bar_chart(df)
                elif ctype == "area":
                    st.area_chart(df)
                else:
                    st.line_chart(df)


def _asset_ui_key(ar, idx: int) -> str:
    """Return a stable, unique per-run key for asset-scoped UI state."""
    return f"{idx}_{ar.spec.spv_name}_{ar.spec.asset_type}_{ar.spec.name}"


def _render_asset_summary_card(ar) -> None:
    """Render a compact, polished summary card for one asset."""
    k = ar.model_results.kpis

    def _fmt_pct(value) -> str:
        return f"{value * 100:.2f}%" if value else "n/a"

    def _fmt_ratio(value) -> str:
        return f"{value:.3f}x" if value else "n/a"

    def _fmt_money(value) -> str:
        return f"Rs {value:,.1f} L" if value else "n/a"

    def _fmt_years(value) -> str:
        return f"{value:.1f} yrs" if value else "n/a"

    metrics = [
        ("Equity IRR", _fmt_pct(k.equity_irr)),
        ("Project IRR", _fmt_pct(k.project_irr)),
        ("Min DSCR", _fmt_ratio(k.min_dscr)),
        ("Avg DSCR", _fmt_ratio(k.avg_dscr)),
        ("LLCR", _fmt_ratio(k.llcr)),
        ("NPV (Equity)", _fmt_money(k.npv_equity)),
        ("Peak Debt", _fmt_money(k.peak_debt_outstanding)),
        ("Debt Payback", _fmt_years(k.debt_payback_period)),
    ]

    metric_rows = "".join(
        (
            '<div class="metric-row">'
            f'<span class="metric-label">{label}</span>'
            f'<span class="metric-value">{value}</span>'
            "</div>"
        )
        for label, value in metrics
    )

    warning_rows = ""
    if ar.validation.warnings:
        warning_rows = "".join(
            f'<div class="warning-item">{w.message}</div>'
            for w in ar.validation.warnings
        )
        warning_rows = f'<div class="warning-stack">{warning_rows}</div>'

    st.markdown(
        (
            '<div class="summary-card">'
            f'<div class="summary-card-header">{ar.spec.name}</div>'
            f'<div class="summary-card-subtitle">{ar.spec.asset_type.upper()}  |  SPV: {ar.spec.spv_name}</div>'
            f'<div class="metric-grid">{metric_rows}</div>'
            f"{warning_rows}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _format_assumption_chip(value, suffix: str = "", decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}{suffix}"
    return str(value)


def _render_asset_dashboard_intro(ar, ctx: Dict[str, Any]) -> None:
    assumptions = ctx.get("assumptions", {})
    chips = [
        ("Capacity", _format_assumption_chip(assumptions.get("capacity_mw"), " MW", 0)),
        ("Tariff", _format_assumption_chip(assumptions.get("tariff"), " Rs/kWh", 2)),
        ("Debt", _format_assumption_chip(
            assumptions.get("debt_pct") * 100 if assumptions.get("debt_pct") is not None else None,
            "%", 0
        )),
        ("Rate", _format_assumption_chip(
            assumptions.get("interest_rate") * 100 if assumptions.get("interest_rate") is not None else None,
            "%", 2
        )),
        ("DSRA", _format_assumption_chip(assumptions.get("dsra_months"), " months", 0)),
    ]
    chip_html = "".join(
        (
            '<div class="asset-chip">'
            f'<span class="asset-chip-label">{label}</span>'
            f'<span class="asset-chip-value">{value}</span>'
            "</div>"
        )
        for label, value in chips
    )

    st.markdown(
        (
            '<div class="asset-dashboard-hero">'
            f'<div class="asset-dashboard-title">{ar.spec.name}</div>'
            f'<div class="asset-dashboard-copy">{ar.spec.asset_type.title()} asset in {ar.spec.spv_name}. '
            'Use the charts below to inspect operating performance, debt pressure, reserve behavior, and follow-up questions.</div>'
            f'<div class="asset-chip-row">{chip_html}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def _render_chart_heading(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="chart-subtitle">{subtitle}</div>', unsafe_allow_html=True)


@st.dialog("Asset QA", width="large")
def _render_asset_qa_dialog(asset_key: str, asset_name: str, ctx: Dict[str, Any]) -> None:
    chat_key = f"chat_{asset_key}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    st.markdown(
        f'<div class="panel-title">Ask about {asset_name}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel-subtitle">Past questions remain available below and are reused as context for follow-up analysis.</div>',
        unsafe_allow_html=True,
    )

    history = st.session_state[chat_key]
    history_box = st.container(height=360, border=False)
    with history_box:
        if not history:
            st.markdown(
                """
                <div class="qa-empty-state">
                    Ask about DSCR pressure, cashflow shape, debt behavior, reserve movements,
                    or what is driving a specific KPI.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for msg in history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

    if question := st.chat_input(
        "e.g. What drives the DSCR dip in Year 6?",
        key=f"qa_dialog_input_{asset_key}",
    ):
        st.session_state[chat_key].append({"role": "user", "content": question})
        with st.spinner("Thinking through the model context..."):
            answer = DashboardAgent().answer_question(
                ctx,
                st.session_state[chat_key][:-1],
                question,
            )
        st.session_state[chat_key].append({"role": "assistant", "content": answer})
        st.rerun(scope="fragment")

    done_col, _ = st.columns([1, 4])
    with done_col:
        if st.button("Done", key=f"qa_done_{asset_key}", use_container_width=True):
            st.rerun()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IPP Portfolio Financial Model",
    page_icon="⚡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #050505;
        --panel: rgba(20, 20, 20, 0.84);
        --panel-strong: rgba(28, 28, 28, 0.96);
        --ink: #f5f3ee;
        --muted: #b4b0a8;
        --line: rgba(255, 255, 255, 0.10);
        --accent: #6ce0b2;
        --accent-soft: rgba(108, 224, 178, 0.10);
        --warm: #ffb06b;
        --warning-bg: rgba(131, 88, 34, 0.20);
    }

    html, body, [class*="css"] {
        font-family: "Manrope", sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(255, 176, 107, 0.10), transparent 22%),
            radial-gradient(circle at top right, rgba(108, 224, 178, 0.12), transparent 24%),
            linear-gradient(180deg, #090909 0%, #030303 100%);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1360px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero-panel {
        background: linear-gradient(145deg, rgba(20,20,20,0.96), rgba(11,11,11,0.90));
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 2rem 2rem 1.35rem 2rem;
        box-shadow: 0 22px 60px rgba(0, 0, 0, 0.38);
        margin-bottom: 1.15rem;
    }

    .eyebrow {
        display: inline-block;
        padding: 0.38rem 0.7rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: clamp(2.2rem, 4vw, 4.2rem);
        line-height: 0.98;
        font-weight: 800;
        max-width: 900px;
        color: var(--ink);
        margin-bottom: 0.8rem;
    }

    .hero-subtitle {
        max-width: 780px;
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.7;
        margin-bottom: 1rem;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
    }

    .chip {
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.45rem 0.78rem;
        background: rgba(255,255,255,0.04);
        font-size: 0.84rem;
        color: var(--ink);
    }

    .section-title {
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.35rem;
    }

    .panel-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 0.25rem;
    }

    .panel-subtitle {
        font-size: 0.96rem;
        color: var(--muted);
        margin-bottom: 0.85rem;
        line-height: 1.6;
    }

    .mini-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1rem 1rem 0.9rem 1rem;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.24);
    }

    .mini-card-title {
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 0.3rem;
    }

    .mini-card-copy {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.6;
    }

    .example-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.95rem 1rem 0.85rem 1rem;
        margin-bottom: 0.55rem;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
    }

    .example-title {
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.45rem;
    }

    .example-copy {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.55;
    }

    .subtle-note {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.65;
        padding-top: 0.3rem;
    }

    .landing-spacer-sm {
        height: 0.65rem;
    }

    .landing-spacer-md {
        height: 1.15rem;
    }

    .header-action-spacer {
        height: 4.8rem;
    }

    .qa-fab {
        position: fixed;
        right: 1.5rem;
        bottom: 2.5rem;
        z-index: 9999;
    }

    .qa-fab button {
        background: linear-gradient(135deg, #2a2a2a, #1a1a1a) !important;
        border: 1px solid var(--line) !important;
        border-radius: 24px !important;
        color: var(--ink) !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.1rem !important;
        box-shadow: 0 4px 18px rgba(0,0,0,0.45) !important;
        white-space: nowrap !important;
        cursor: pointer !important;
        transition: box-shadow 0.18s ease, border-color 0.18s ease !important;
    }

    .qa-fab button:hover {
        border-color: var(--accent) !important;
        box-shadow: 0 6px 24px rgba(0,0,0,0.6) !important;
    }

    .qa-launch-spacer {
        height: 1rem;
    }

    .qa-empty-state {
        border: 1px dashed var(--line);
        background: rgba(255,255,255,0.03);
        color: var(--muted);
        border-radius: 18px;
        padding: 1rem 1rem;
        line-height: 1.6;
    }

    .summary-card {
        background: linear-gradient(160deg, rgba(26,26,26,0.98), rgba(15,15,15,0.92));
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 1.2rem;
        min-height: 100%;
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.28);
    }

    .summary-card-header {
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 0.2rem;
    }

    .summary-card-subtitle {
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 1rem;
    }

    .metric-grid {
        display: grid;
        gap: 0.5rem;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 0.8rem;
        border-bottom: 1px dashed rgba(255, 255, 255, 0.10);
        padding-bottom: 0.42rem;
    }

    .metric-label {
        font-size: 0.82rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .metric-value {
        font-size: 1rem;
        font-weight: 800;
        color: var(--ink);
        text-align: right;
    }

    .warning-stack {
        margin-top: 0.9rem;
        display: grid;
        gap: 0.5rem;
    }

    .warning-item {
        background: var(--warning-bg);
        border: 1px solid rgba(255, 176, 107, 0.22);
        color: #ffd9b5;
        border-radius: 14px;
        padding: 0.65rem 0.75rem;
        font-size: 0.86rem;
        line-height: 1.45;
    }

    .asset-dashboard-hero {
        background: linear-gradient(145deg, rgba(21,21,21,0.95), rgba(10,10,10,0.88));
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.15rem 1.2rem 1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
    }

    .asset-dashboard-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 0.25rem;
    }

    .asset-dashboard-copy {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.6;
        margin-bottom: 0.85rem;
    }

    .asset-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
    }

    .asset-chip {
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.55rem 0.7rem;
        min-width: 112px;
    }

    .asset-chip-label {
        display: block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.16rem;
    }

    .asset-chip-value {
        display: block;
        font-size: 0.95rem;
        font-weight: 800;
        color: var(--ink);
    }

    .chart-title {
        font-size: 1rem;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 0.15rem;
    }

    .chart-subtitle {
        color: var(--muted);
        font-size: 0.84rem;
        line-height: 1.5;
        margin-bottom: 0.45rem;
    }

    div[data-testid="stTextArea"] textarea {
        min-height: 180px;
        border-radius: 20px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.03);
        color: var(--ink);
        font-size: 0.98rem;
        padding: 1rem 1rem 1.1rem 1rem;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: rgba(245, 243, 238, 0.44);
    }

    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stCaptionContainer"],
    label p,
    .st-emotion-cache-10trblm,
    .st-emotion-cache-16idsys {
        color: var(--ink);
    }

    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        border-radius: 999px;
        border: 1px solid var(--line);
        min-height: 2.8rem;
        font-weight: 700;
        background: rgba(255,255,255,0.04);
        color: var(--ink);
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.24);
    }

    div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stDownloadButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #6ce0b2, #2bb98b);
        color: #03140e;
        border-color: transparent;
        box-shadow: 0 16px 30px rgba(108, 224, 178, 0.22);
    }

    div[data-testid="stTabs"] button {
        border-radius: 999px;
        color: var(--muted);
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--ink);
        background: rgba(255,255,255,0.06);
    }

    div[data-testid="stChatInput"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.2rem 0.35rem;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.24);
    }

    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] input {
        color: var(--ink);
    }

    div[data-testid="stStatusWidget"],
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--line);
        border-radius: 18px;
    }

    div[data-testid="stDialog"] div[role="dialog"] {
        background: linear-gradient(160deg, rgba(16,16,16,0.88), rgba(7,7,7,0.82));
        border: 1px solid rgba(255,255,255,0.10);
        backdrop-filter: blur(18px);
        box-shadow: 0 30px 80px rgba(0, 0, 0, 0.46);
    }

    div[data-testid="stDialog"] div[role="dialog"] [data-testid="stVerticalBlock"] {
        gap: 0.8rem;
    }

    div[data-testid="stInfo"] {
        background: rgba(108, 224, 178, 0.08);
        color: var(--ink);
        border: 1px solid rgba(108, 224, 178, 0.14);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-panel">
        <div class="hero-title">Project Finance Workbench</div>
        <div class="chip-row">
            <span class="chip">Solar and wind assets</span>
            <span class="chip">Compiled model execution</span>
            <span class="chip">Debt and reserve logic</span>
            <span class="chip">Excel-ready outputs</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Example prompts
# ---------------------------------------------------------------------------
EXAMPLES = [
    "I want to create a project finance model for two assets - one wind and one solar. "
    "The solar model will not have any DSRA requirements and will have revenue support of 1 rs/kwh.",
    "Two assets: 100 MW solar SPV-A at Rs. 2.65/kWh with 70% debt, and 50 MW wind SPV-B at Rs. 3.20/kWh with 75% debt.",
    "Single 150 MW solar asset, 25-year PPA at Rs. 2.80/kWh, 72% debt, 12-month DSRA.",
]

outer_left, center_band, outer_right = st.columns([0.07, 0.86, 0.07], gap="small")

with center_band:
    intro_col, side_col = st.columns([2.3, 1], gap="large")

    with intro_col:
        st.markdown('<div class="section-title">Portfolio Prompt</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-title">Describe the assets, and let the app wire the rest.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="panel-subtitle">Include capacity, tariff, leverage, debt tenor, DSRA, subsidy support, or any project-specific notes you already have.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="landing-spacer-sm"></div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Prompt input
    # -----------------------------------------------------------------------
    prompt_col, notes_col = st.columns([2.2, 1], gap="large")

    with prompt_col:
        prompt = st.text_area(
            "Portfolio description",
            value=st.session_state.get("prompt_text", ""),
            height=190,
            placeholder=(
                "Example:\n"
                "Two assets: 100 MW solar in SPV-A at Rs. 2.65/kWh with 70% debt,\n"
                "and 50 MW wind in SPV-B at Rs. 3.20/kWh with 75% debt and 6-month DSRA."
            ),
            label_visibility="collapsed",
        )

        st.markdown('<div class="landing-spacer-sm"></div>', unsafe_allow_html=True)

        action_col, note_col = st.columns([1, 2.7], gap="medium")
        with action_col:
            run_btn = st.button("Run Model", type="primary", use_container_width=True)
        with note_col:
            st.markdown(
                '<div class="subtle-note">The app parses the portfolio, extracts assumptions for each asset, runs the finance model, and then prepares charts and a downloadable workbook.</div>',
                unsafe_allow_html=True,
            )

    with notes_col:
        st.markdown('<div class="landing-spacer-md"></div>', unsafe_allow_html=True)
        for i, ex in enumerate(EXAMPLES, start=1):
            if st.button(f"Use example {i}", key=f"ex_{i}", use_container_width=True):
                st.session_state["prompt_text"] = ex
                st.rerun()

if "latest_results" not in st.session_state:
    st.session_state["latest_results"] = None
if "latest_excel_bytes" not in st.session_state:
    st.session_state["latest_excel_bytes"] = None

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if run_btn:
    prompt = prompt.strip()
    if not prompt:
        st.warning("Please enter a portfolio description before running.")
        st.stop()

    # -- Step 1: Parse portfolio
    with st.status("Parsing portfolio...", expanded=True) as status:
        try:
            port_agent = PortfolioAgent()
            specs = port_agent.parse(prompt)
            st.write(f"Identified **{len(specs)}** asset(s): " +
                     ", ".join(f"{s.name} [{s.asset_type}]" for s in specs))
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"Portfolio parsing failed: {e}")
            st.stop()

        # -- Step 2: Collect assumptions
        st.write("Collecting assumptions for each asset...")
        try:
            ingestions = port_agent.collect_assumptions(specs, prompt_fn=None)
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"Assumption collection failed: {e}")
            st.stop()

        # -- Step 3: Run financial models
        st.write("Running financial models...")
        try:
            runner = PortfolioRunner()
            results = runner.run(ingestions)
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"Financial model run failed: {e}")
            st.stop()

        status.update(label="Model complete!", state="complete", expanded=False)

    # -- Step 4: Export to Excel in memory
    output_dir = _HERE / "output"
    output_dir.mkdir(exist_ok=True)
    export_path = output_dir / "portfolio_streamlit.xlsx"
    export_portfolio_to_excel(results, path=export_path)

    with open(export_path, "rb") as fh:
        excel_bytes = fh.read()

    st.session_state["latest_results"] = results
    st.session_state["latest_excel_bytes"] = excel_bytes
    # Invalidate cached dashboard context so it is rebuilt for the new run
    st.session_state.pop("model_context", None)
    st.session_state.pop("chart_specs", None)

results = st.session_state.get("latest_results")
excel_bytes = st.session_state.get("latest_excel_bytes")

if results:
    # -----------------------------------------------------------------------
    # Results display
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Portfolio Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-subtitle">A compact view of the base-case outputs for each asset in the current run.</div>',
        unsafe_allow_html=True,
    )

    # KPI cards per asset
    cols = st.columns(len(results))
    for col, ar in zip(cols, results):
        with col:
            _render_asset_summary_card(ar)

    # -----------------------------------------------------------------------
    # Build dashboard context + AI chart specs (cached per run)
    # -----------------------------------------------------------------------
    ctx_cache  = st.session_state.setdefault("model_context", {})
    spec_cache = st.session_state.setdefault("chart_specs", {})

    for idx, ar in enumerate(results):
        asset_key = _asset_ui_key(ar, idx)
        if asset_key not in ctx_cache:
            ctx_cache[asset_key] = _build_model_context(ar)

    if any(_asset_ui_key(ar, idx) not in spec_cache for idx, ar in enumerate(results)):
        with st.status("Selecting AI charts...", expanded=False) as dash_status:
            agent = DashboardAgent()
            for idx, ar in enumerate(results):
                asset_key = _asset_ui_key(ar, idx)
                if asset_key not in spec_cache:
                    spec_cache[asset_key] = agent.select_charts(ctx_cache[asset_key])
            dash_status.update(label="Dashboard ready", state="complete", expanded=False)

    # -----------------------------------------------------------------------
    # Financial Dashboard
    # -----------------------------------------------------------------------
    st.divider()
    diag_copy_col, diag_action_col = st.columns([4.2, 1.2], gap="large")
    with diag_copy_col:
        st.markdown('<div class="section-title">Diagnostics</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Financial Dashboard</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-subtitle">Explore operating performance, debt behavior, and asset-level follow-up questions.</div>',
            unsafe_allow_html=True,
        )
    with diag_action_col:
        st.markdown('<div class="header-action-spacer"></div>', unsafe_allow_html=True)
        st.download_button(
            label="Download Excel Model",
            data=excel_bytes,
            file_name="ipp_portfolio_model.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )

    asset_tabs = st.tabs([ar.spec.name for ar in results])
    for idx, (tab, ar) in enumerate(zip(asset_tabs, results)):
        with tab:
            asset_key = _asset_ui_key(ar, idx)
            ctx      = ctx_cache[asset_key]
            ai_specs = spec_cache.get(asset_key, {"charts": []})
            _render_asset_dashboard_intro(ar, ctx)

            st.markdown('<div class="qa-fab">', unsafe_allow_html=True)
            if st.button("Asset Q&A", key=f"open_qa_{asset_key}"):
                _render_asset_qa_dialog(asset_key, ar.spec.name, ctx)
            st.markdown('</div>', unsafe_allow_html=True)

            # 1. Standard charts
            _render_standard_charts(ar, ctx)

            # 2. AI-suggested additional charts
            extra_charts = ai_specs.get("charts", [])
            if extra_charts:
                st.markdown('<div class="section-title">Extended View</div>', unsafe_allow_html=True)
                st.markdown('<div class="panel-title">Additional Relevant Charts</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="panel-subtitle">Extra diagnostics suggested from the asset context and KPI profile.</div>',
                    unsafe_allow_html=True,
                )
                _render_ai_charts(ctx, ai_specs)
