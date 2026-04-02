"""
app.py — Streamlit UI for the IPP Portfolio Financial Model.

Run with:
    cd solar_ipp_model
    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

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
from engine.portfolio_runner import PortfolioRunner
from engine.excel_exporter import export_portfolio_to_excel

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

    # Download button
    st.divider()
    st.download_button(
        label="Download Excel Model",
        data=excel_bytes,
        file_name="ipp_portfolio_model.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
