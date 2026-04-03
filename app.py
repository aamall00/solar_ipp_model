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
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**CFADS vs Debt Service** (INR Lakhs)")
        df = pd.DataFrame(
            {"CFADS": qs["cfads"], "Debt Service": qs["debt_service"]},
            index=ops_idx,
        )
        st.line_chart(df)

    with col2:
        st.markdown("**DSCR per Quarter**")
        dscr = [x if x is not None else float("nan") for x in qs["dscr"]]
        df2 = pd.DataFrame({"DSCR": dscr}, index=ops_idx)
        st.line_chart(df2)
        k = ar.model_results.kpis
        if k.min_dscr and k.avg_dscr:
            st.caption(f"Min {k.min_dscr:.3f}×  |  Avg {k.avg_dscr:.3f}×")

    # Row 2 — Revenue / OPEX / EBITDA | Debt Balance
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Revenue, OPEX & EBITDA** (INR Lakhs)")
        df3 = pd.DataFrame(
            {"Revenue": qs["revenue"], "OPEX": qs["total_opex"], "EBITDA": qs["ebitda"]},
            index=ops_idx,
        )
        st.line_chart(df3)

    with col4:
        st.markdown("**Outstanding Debt Balance** (INR Lakhs)")
        df4 = pd.DataFrame({"Debt Balance": qs["debt_balance"]}, index=ops_idx)
        st.area_chart(df4)

    # Row 3 — Principal + Interest | DSRA
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**Debt Service Components** (INR Lakhs)")
        df5 = pd.DataFrame(
            {"Principal": qs["principal"], "Interest": qs["interest"]},
            index=ops_idx,
        )
        st.bar_chart(df5, stack=True)

    with col6:
        dsra = qs["dsra_balance"]
        if any(v is not None and v > 0 for v in dsra):
            st.markdown("**DSRA Balance** (INR Lakhs)")
            df6 = pd.DataFrame({"DSRA": dsra}, index=ops_idx)
            st.area_chart(df6)

    # Row 4 — Equity J-curve (full width)
    eq_cf = qs["equity_cashflow"]
    if eq_cf:
        st.markdown("**Equity Cashflow J-Curve** (cumulative, INR Lakhs)")
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
        cols = st.columns(len(pair))
        for col, spec in zip(cols, pair):
            with col:
                st.markdown(f"**{spec['title']}**")
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
                if spec.get("insight"):
                    st.caption(spec["insight"])


def _asset_ui_key(ar, idx: int) -> str:
    """Return a stable, unique per-run key for asset-scoped UI state."""
    return f"{idx}_{ar.spec.spv_name}_{ar.spec.asset_type}_{ar.spec.name}"

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
    .main-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-title  { font-size: 1rem; color: #555; margin-bottom: 1.5rem; }
    .kpi-card   { background: #f8f9fa; border-radius: 8px; padding: 1rem; text-align: center; }
    .kpi-label  { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value  { font-size: 1.4rem; font-weight: 700; color: #1a1a2e; }
    .warn-box   { background: #fff8e1; border-left: 4px solid #ffc107; padding: 0.6rem 1rem;
                  border-radius: 4px; margin-top: 0.5rem; font-size: 0.85rem; }
    .asset-header { font-size: 1.1rem; font-weight: 600; color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">⚡ IPP Portfolio Financial Model</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Describe your portfolio in plain English — the model does the rest.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Example prompts
# ---------------------------------------------------------------------------
EXAMPLES = [
    "I want to create a project finance model for two assets - one wind and one solar. "
    "The solar model will not have any DSRA requirements and will have revenue support of 1 rs/kwh.",
    "Two assets: 100 MW solar SPV-A at Rs. 2.65/kWh with 70% debt, and 50 MW wind SPV-B at Rs. 3.20/kWh with 75% debt.",
    "Single 150 MW solar asset, 25-year PPA at Rs. 2.80/kWh, 72% debt, 12-month DSRA.",
]

with st.expander("Show example prompts", expanded=False):
    for i, ex in enumerate(EXAMPLES, 1):
        if st.button(f"Use example {i}", key=f"ex_{i}"):
            st.session_state["prompt_text"] = ex

# ---------------------------------------------------------------------------
# Prompt input
# ---------------------------------------------------------------------------
prompt = st.text_area(
    "Portfolio description",
    value=st.session_state.get("prompt_text", ""),
    height=120,
    placeholder=(
        "Describe the assets you want to model. Include details like:\n"
        "  • Asset type (solar / wind)\n"
        "  • Capacity (MW)\n"
        "  • Tariff (Rs./kWh)\n"
        "  • Debt ratio, tenor, DSRA requirements, revenue support, etc."
    ),
)

run_col, _ = st.columns([1, 5])
with run_col:
    run_btn = st.button("Run Model", type="primary", use_container_width=True)

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
    st.subheader("Portfolio Summary")

    # KPI cards per asset
    cols = st.columns(len(results))
    for col, ar in zip(cols, results):
        k = ar.model_results.kpis

        irr_str  = f"{k.equity_irr * 100:.2f}%" if k.equity_irr else "n/a"
        pirr_str = f"{k.project_irr * 100:.2f}%" if k.project_irr else "n/a"
        dscr_str = f"{k.min_dscr:.3f}×" if k.min_dscr else "n/a"
        adscr_str = f"{k.avg_dscr:.3f}×" if k.avg_dscr else "n/a"
        llcr_str = f"{k.llcr:.3f}×" if k.llcr else "n/a"
        npv_str  = f"₹ {k.npv_equity:,.1f} L" if k.npv_equity else "n/a"
        debt_str = f"₹ {k.peak_debt_outstanding:,.1f} L" if k.peak_debt_outstanding else "n/a"
        payback_str = f"{k.debt_payback_period:.1f} yrs" if k.debt_payback_period else "n/a"

        with col:
            st.markdown(f'<p class="asset-header">{ar.spec.name}</p>', unsafe_allow_html=True)
            st.caption(f"{ar.spec.asset_type.upper()} · SPV: {ar.spec.spv_name}")

            for label, value in [
                ("Equity IRR", irr_str),
                ("Project IRR", pirr_str),
                ("Min DSCR", dscr_str),
                ("Avg DSCR", adscr_str),
                ("LLCR", llcr_str),
                ("NPV (Equity)", npv_str),
                ("Peak Debt", debt_str),
                ("Debt Payback", payback_str),
            ]:
                st.markdown(
                    f'<div class="kpi-card">'
                    f'<div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.write("")  # spacing

            if ar.validation.warnings:
                for w in ar.validation.warnings:
                    st.markdown(
                        f'<div class="warn-box">⚠ {w.message}</div>',
                        unsafe_allow_html=True,
                    )

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
    st.subheader("Financial Dashboard")

    asset_tabs = st.tabs([ar.spec.name for ar in results])
    for idx, (tab, ar) in enumerate(zip(asset_tabs, results)):
        with tab:
            asset_key = _asset_ui_key(ar, idx)
            ctx      = ctx_cache[asset_key]
            ai_specs = spec_cache.get(asset_key, {"charts": []})

            # 1. Standard charts
            _render_standard_charts(ar, ctx)

            # 2. AI-suggested additional charts
            extra_charts = ai_specs.get("charts", [])
            if extra_charts:
                st.markdown("#### Additional Relevant Charts")
                _render_ai_charts(ctx, ai_specs)

            st.divider()

            # 3. Follow-up chat
            st.markdown("**Ask a question about this asset**")
            chat_key = f"chat_{asset_key}"
            if chat_key not in st.session_state:
                st.session_state[chat_key] = []

            for msg in st.session_state[chat_key]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            if question := st.chat_input(
                "e.g. What drives the DSCR dip in Year 6?", key=f"inp_{asset_key}"
            ):
                st.session_state[chat_key].append({"role": "user", "content": question})
                with st.chat_message("assistant"):
                    answer = DashboardAgent().answer_question(
                        ctx,
                        st.session_state[chat_key][:-1],
                        question,
                    )
                    st.write(answer)
                st.session_state[chat_key].append({"role": "assistant", "content": answer})

    # Download button
    st.divider()
    st.download_button(
        label="Download Excel Model",
        data=excel_bytes,
        file_name="ipp_portfolio_model.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
