"""
ui/app.py — Streamlit web interface for the Solar IPP Project Finance Model.

Run from the solar_ipp_model/ directory:
    streamlit run ui/app.py

This file is a pure UI wrapper — it imports from the existing engine and DSL
layers without modifying any core code.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — ensure engine/dsl packages are importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dsl.parser import DSLParser
from engine.executor import ModelExecutor, ModelResults, SensitivityResults

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Solar IPP — Project Finance Model",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---------- global — pure black base ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #000000 !important;
    color: #e0e0e0 !important;
}
.stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: #000000 !important;
}
[data-testid="stMain"] { background-color: #000000 !important; }
section[data-testid="stMainBlockContainer"] { background-color: #000000 !important; }
.block-container { background-color: #000000 !important; padding-top: 1.5rem !important; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background-color: #0a0a0a !important;
    border-right: 1px solid #1f1f1f !important;
    padding-top: 0.5rem;
}
[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
[data-testid="stSidebar"] label { font-size: 0.78rem !important; font-weight: 500; letter-spacing: 0.02em; color: #8fadc8 !important; }
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stSelectbox select,
[data-testid="stSidebar"] .stTextInput input {
    background: #0b1120 !important;
    border: 1px solid #1a2540 !important;
    border-radius: 6px !important;
    color: #d0e8ff !important;
}
/* slider — thin track */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div[class] {
    height: 2px !important;
    border-radius: 2px !important;
    background: #1e3a5f !important;
}
/* slider — filled (active) portion */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] [role="progressbar"] {
    height: 2px !important;
    border-radius: 2px !important;
    background: #4da6ff !important;
}
/* slider thumb */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    width: 14px !important;
    height: 14px !important;
    background: #4da6ff !important;
    border: 2px solid #7ec8ff !important;
    border-radius: 50% !important;
    box-shadow: 0 0 6px #4da6ff66 !important;
}
/* slider min/max tick labels */
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMax"],
[data-testid="stSidebar"] [data-testid="stSlider"] small {
    color: #3a6080 !important;
    font-size: 0.68rem !important;
}
/* slider current value label */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
    color: #90c8ff !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
}
/* radio buttons */
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stRadio div {
    color: #a0c4ff !important;
}
/* number input +/- buttons */
[data-testid="stSidebar"] .stNumberInput button {
    background: #0d1628 !important;
    border: 1px solid #1a2540 !important;
    color: #7eb8ff !important;
}
/* expander container */
[data-testid="stSidebar"] .stExpander {
    background: #0b1120 !important;
    border: 1px solid #1a2540 !important;
    border-radius: 8px !important;
    margin-bottom: 6px !important;
}
/* expander header row (the clickable summary bar) */
[data-testid="stSidebar"] .stExpander summary,
[data-testid="stSidebar"] .stExpander [data-testid="stExpanderToggleIcon"],
[data-testid="stSidebar"] details summary,
[data-testid="stSidebar"] details > summary {
    background: #0d1628 !important;
    border-radius: 8px !important;
    color: #7eb8ff !important;
}
[data-testid="stSidebar"] .stExpander summary:hover,
[data-testid="stSidebar"] details > summary:hover {
    background: #111e38 !important;
}
/* expander header text */
[data-testid="stSidebar"] .stExpander summary p,
[data-testid="stSidebar"] .stExpander summary span,
[data-testid="stSidebar"] details summary p,
[data-testid="stSidebar"] details summary span {
    color: #7eb8ff !important;
    font-weight: 600 !important;
}
/* expander arrow/chevron icon */
[data-testid="stSidebar"] .stExpander svg,
[data-testid="stSidebar"] details summary svg {
    fill: #7eb8ff !important;
    stroke: #7eb8ff !important;
}
/* expander body (content area) */
[data-testid="stSidebar"] .stExpander > div,
[data-testid="stSidebar"] details > div {
    background: #0b1120 !important;
    border-top: 1px solid #1a2540 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #00c896 !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.2rem !important;
    width: 100% !important;
    margin-top: 4px !important;
    letter-spacing: 0.03em;
    transition: opacity 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover { opacity: 0.85; }

/* ---------- KPI cards ---------- */
.kpi-card {
    background: #0d0d0d;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 10px 10px 0 0;
}
.kpi-card.green::before  { background: #00c896; }
.kpi-card.blue::before   { background: #4da6ff; }
.kpi-card.orange::before { background: #ffaa00; }
.kpi-card.purple::before { background: #a78bfa; }
.kpi-card.red::before    { background: #ff5555; }
.kpi-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #7aa8cc !important;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff !important;
    line-height: 1;
    letter-spacing: -0.02em;
}
.kpi-sub {
    font-size: 0.7rem;
    color: #6090b0 !important;
    margin-top: 5px;
}
.kpi-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.62rem;
    font-weight: 700;
    margin-top: 7px;
    letter-spacing: 0.06em;
}
.badge-pass { background: #00c89620; color: #00c896 !important; border: 1px solid #00c89640; }
.badge-warn { background: #ffaa0020; color: #ffaa00 !important; border: 1px solid #ffaa0040; }
.badge-fail { background: #ff555520; color: #ff5555 !important; border: 1px solid #ff555540; }

/* ---------- page header ---------- */
.page-header {
    background: #0d0d0d;
    border: 1px solid #1e1e1e;
    border-left: 3px solid #00c896;
    border-radius: 10px;
    padding: 20px 26px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.page-header h1 { font-size: 1.5rem; font-weight: 800; color: #fff !important; margin: 0; letter-spacing: -0.02em; }
.page-header p  { font-size: 0.82rem; color: #7aa8cc !important; margin: 5px 0 0 0; }
.sun-icon { font-size: 2.2rem; }

/* ---------- section label ---------- */
.section-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5a8ab0;
    margin: 16px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #1a2a3a;
}

/* ---------- warning banner ---------- */
.warn-banner {
    background: #ffaa0010;
    border: 1px solid #ffaa0030;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.8rem;
    color: #ffaa00;
    margin-bottom: 14px;
}

/* ---------- convergence pills ---------- */
.conv-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 10px; border-radius: 4px;
    font-size: 0.7rem; font-weight: 600;
    margin-right: 6px; margin-bottom: 10px;
}
.conv-ok   { background: #00c89615; color: #00c896; border: 1px solid #00c89630; }
.conv-fail { background: #ff555515; color: #ff5555; border: 1px solid #ff555530; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: #0d0d0d;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 6px 18px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #7aa8cc !important;
}
.stTabs [aria-selected="true"] {
    background: #00c89618 !important;
    color: #00c896 !important;
    border: 1px solid #00c89630 !important;
}

/* ---------- main run button (non-sidebar) ---------- */
.stButton > button {
    background: #141414 !important;
    color: #00c896 !important;
    border: 1px solid #00c89640 !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #00c89615 !important;
    border-color: #00c896 !important;
}

/* ---------- scenario / sensitivity table ---------- */
.scenario-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    margin-top: 16px;
}
.scenario-table th {
    background: #0d0d0d;
    color: #5a8ab0;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 10px 14px;
    text-align: right;
    border-bottom: 1px solid #1e1e1e;
}
.scenario-table th:first-child { text-align: left; }
.scenario-table td {
    padding: 9px 14px;
    text-align: right;
    border-bottom: 1px solid #141414;
    color: #c0d8f0;
}
.scenario-table td:first-child { text-align: left; font-weight: 600; color: #ddd; }
.scenario-table tr:hover td { background: #0f0f0f; }
.scenario-table .base-row td { color: #00c896 !important; }
.flag-dscr { color: #ff5555 !important; font-size: 0.62rem; font-weight: 700; }

/* ---------- sidebar logo ---------- */
.sidebar-logo {
    text-align: center;
    padding: 16px 0 18px 0;
    border-bottom: 1px solid #1a1a1a;
    margin-bottom: 14px;
}
.sidebar-logo .logo-icon { font-size: 2rem; }
.sidebar-logo h2 { font-size: 0.95rem; font-weight: 800; color: #fff !important; margin: 6px 0 2px 0; }
.sidebar-logo p  { font-size: 0.65rem; color: #5a8ab0 !important; margin: 0; letter-spacing: 0.08em; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers — formatting
# ---------------------------------------------------------------------------

def _pct(v, d=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v * 100:.{d}f}%"

def _x(v, d=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{d}f}x"

def _f(v, d=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:,.{d}f}"


# ---------------------------------------------------------------------------
# Cache model compilation (expensive — only recompile when template changes)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Compiling model…")
def _load_and_compile():
    template = _ROOT / "dsl" / "templates" / "solar_ipp_base.yaml"
    model_def, validation = DSLParser().load_file(template)
    executor = ModelExecutor()
    compiled = executor.compile(model_def)
    return executor, compiled, validation


# ---------------------------------------------------------------------------
# Scenarios & sensitivity config (mirrors main.py exactly — no changes)
# ---------------------------------------------------------------------------

_SCENARIOS = [
    {"name": "Base Case",            "overrides": {}},
    {"name": "Management Case",      "overrides": {"cuf": 0.235, "capex_per_mw": 430.0}},
    {"name": "Downside Case",        "overrides": {"cuf": 0.209, "opex_escalation": 0.033, "tariff": 2.57}},
    {"name": "Construction Stress",  "overrides": {"capex_per_mw": 504.0, "moratorium_periods": 4}},
    {"name": "Interest Rate Stress", "overrides": {"interest_rate": 0.1125}},
    {"name": "Combined Stress",      "overrides": {"cuf": 0.209, "capex_per_mw": 504.0,
                                                    "interest_rate": 0.1125, "moratorium_periods": 4}},
    {"name": "Upside Case",          "overrides": {"cuf": 0.245, "capex_per_mw": 420.0, "degradation_rate": 0.004}},
]

_SWEEP = [
    {"assumption": "cuf",              "low": 0.18,   "high": 0.26},
    {"assumption": "tariff",           "low": 2.20,   "high": 3.10},
    {"assumption": "capex_per_mw",     "low": 380.0,  "high": 520.0},
    {"assumption": "interest_rate",    "low": 0.085,  "high": 0.115},
    {"assumption": "opex_per_mw_pa",   "low": 6.0,    "high": 12.0},
    {"assumption": "degradation_rate", "low": 0.003,  "high": 0.008},
    {"assumption": "debt_pct",         "low": 0.60,   "high": 0.80},
]

_ASSUMPTION_LABELS = {
    "cuf":              "CUF",
    "tariff":           "PPA Tariff",
    "capex_per_mw":     "CapEx / MW",
    "interest_rate":    "Interest Rate",
    "opex_per_mw_pa":   "OpEx / MW / yr",
    "degradation_rate": "Degradation Rate",
    "debt_pct":         "Debt %",
}


# ---------------------------------------------------------------------------
# Chart colour palette
# ---------------------------------------------------------------------------

_C = {
    "green":   "#00c896",
    "teal":    "#00a87a",
    "blue":    "#4da6ff",
    "navy":    "#1e7ed4",
    "orange":  "#ffaa00",
    "amber":   "#ff8800",
    "red":     "#ff5555",
    "purple":  "#a78bfa",
    "gray":    "#555555",
    "bg":      "#000000",
    "bg2":     "#0d0d0d",
    "grid":    "rgba(255,255,255,0.05)",
    "text":    "#666666",
}

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0a0a0a",
    font=dict(color="#888888", family="Inter, Segoe UI, sans-serif", size=11),
    xaxis=dict(gridcolor=_C["grid"], zerolinecolor="#1a1a1a", linecolor="#1a1a1a"),
    yaxis=dict(gridcolor=_C["grid"], zerolinecolor="#1a1a1a", linecolor="#1a1a1a"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e1e1e", borderwidth=1,
                font=dict(color="#888")),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
)

def _apply_layout(fig, **extra):
    import plotly.graph_objects as go
    layout = {**_PLOTLY_LAYOUT, **extra}
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridwidth=1)
    fig.update_yaxes(showgrid=True, gridwidth=1)
    return fig


# ---------------------------------------------------------------------------
# Build time axis (quarter labels)
# ---------------------------------------------------------------------------

def _quarters(n_periods: int, cod: int = 4):
    labels = []
    for i in range(n_periods):
        if i < cod:
            labels.append(f"C{i+1}")
        else:
            q = i - cod + 1
            yr = (q - 1) // 4 + 1
            qt = (q - 1) % 4 + 1
            labels.append(f"Y{yr}Q{qt}")
    return labels


# ---------------------------------------------------------------------------
# Chart factories
# ---------------------------------------------------------------------------

def chart_generation_revenue(vars_: dict, n: int, cod: int):
    import plotly.graph_objects as go
    qs = _quarters(n, cod)
    ops = slice(cod, n)

    rev   = vars_.get("revenue_block.revenue",                np.zeros(n))
    gen   = vars_.get("generation_block.net_generation_kwh",   np.zeros(n))

    fig = go.Figure()
    fig.add_vrect(x0=0, x1=cod - 0.5,
                  fillcolor="rgba(255,255,255,0.03)", line_width=0,
                  annotation_text="Construction", annotation_position="top left",
                  annotation_font_color=_C["gray"])
    fig.add_trace(go.Bar(
        x=qs[cod:], y=rev[ops] / 1e2,
        name="Revenue (Cr)",
        marker_color=_C["teal"],
        opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=qs[cod:], y=gen[ops] / 1e6,
        name="Net Gen (MU)",
        yaxis="y2", line=dict(color=_C["orange"], width=2),
        mode="lines",
    ))
    _apply_layout(fig,
        title=dict(text="Revenue & Net Generation", font=dict(size=13, color="#fff")),
        yaxis=dict(title="Revenue (INR Cr)", gridcolor=_C["grid"]),
        yaxis2=dict(title="Net Generation (MU)", overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)"),
        barmode="group",
    )
    return fig


def chart_cashflow(vars_: dict, n: int, cod: int):
    import plotly.graph_objects as go
    qs = _quarters(n, cod)
    ops = slice(cod, n)

    ebitda = vars_.get("cashflow_block.ebitda",        np.zeros(n))
    cfads  = vars_.get("cashflow_block.cfads",         np.zeros(n))
    fcf    = vars_.get("cashflow_block.free_cashflow", np.zeros(n))
    tax    = vars_.get("tax_block.tax",                np.zeros(n))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qs[cod:], y=ebitda[ops], name="EBITDA",
                              line=dict(color=_C["blue"], width=2), fill="tonexty",
                              fillcolor="rgba(109,213,237,0.08)"))
    fig.add_trace(go.Scatter(x=qs[cod:], y=cfads[ops], name="CFADS",
                              line=dict(color=_C["green"], width=2.5)))
    fig.add_trace(go.Scatter(x=qs[cod:], y=fcf[ops], name="Free Cash Flow",
                              line=dict(color=_C["orange"], width=2, dash="dot")))
    fig.add_trace(go.Bar(x=qs[cod:], y=-tax[ops], name="Tax (–)",
                          marker_color=_C["red"], opacity=0.5))
    _apply_layout(fig,
        title=dict(text="Cash Flow Summary (INR Lakhs)", font=dict(size=13, color="#fff")),
        yaxis=dict(title="INR Lakhs", gridcolor=_C["grid"]),
        barmode="overlay",
    )
    return fig


def chart_debt_schedule(vars_: dict, n: int, cod: int):
    import plotly.graph_objects as go
    qs = _quarters(n, cod)
    ops = slice(cod, n)

    bal  = vars_.get("debt_service_block.outstanding_debt_balance", np.zeros(n))
    prin = vars_.get("debt_service_block.principal_repayment",       np.zeros(n))
    intr = vars_.get("debt_service_block.interest_payment",          np.zeros(n))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=qs[cod:], y=prin[ops], name="Principal",
                          marker_color=_C["navy"], opacity=0.85))
    fig.add_trace(go.Bar(x=qs[cod:], y=intr[ops], name="Interest",
                          marker_color=_C["purple"], opacity=0.85))
    fig.add_trace(go.Scatter(x=qs[cod:], y=bal[ops], name="Outstanding Balance",
                              yaxis="y2", line=dict(color=_C["orange"], width=2.5)))
    _apply_layout(fig,
        title=dict(text="Debt Schedule (INR Lakhs)", font=dict(size=13, color="#fff")),
        yaxis=dict(title="Debt Service (INR Lakhs)", gridcolor=_C["grid"]),
        yaxis2=dict(title="Outstanding Balance (INR Lakhs)", overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)"),
        barmode="stack",
    )
    return fig


def chart_dscr(vars_: dict, n: int, cod: int, dscr_target: float = 1.20):
    import plotly.graph_objects as go
    qs = _quarters(n, cod)
    ops = slice(cod, n)

    cfads = vars_.get("cashflow_block.cfads",                      np.zeros(n))
    ds    = vars_.get("debt_service_block.total_debt_service",     np.zeros(n))

    with np.errstate(divide="ignore", invalid="ignore"):
        dscr = np.where(ds[ops] > 1e-6, cfads[ops] / ds[ops], np.nan)

    fig = go.Figure()
    fig.add_hrect(y0=0, y1=1.0, fillcolor="rgba(233,108,108,0.07)", line_width=0)
    fig.add_hrect(y0=1.0, y1=dscr_target, fillcolor="rgba(255,210,0,0.05)", line_width=0)
    fig.add_hline(y=1.0, line=dict(color=_C["red"],    width=1.5, dash="dash"),
                  annotation_text="1.00x", annotation_font_color=_C["red"])
    fig.add_hline(y=dscr_target, line=dict(color=_C["orange"], width=1.5, dash="dash"),
                  annotation_text=f"{dscr_target:.2f}x target",
                  annotation_font_color=_C["orange"])
    fig.add_trace(go.Scatter(
        x=qs[cod:], y=dscr, name="DSCR",
        line=dict(color=_C["green"], width=2.5),
        mode="lines+markers",
        marker=dict(size=3),
    ))
    _apply_layout(fig,
        title=dict(text="Debt Service Coverage Ratio (DSCR)", font=dict(size=13, color="#fff")),
        yaxis=dict(title="DSCR (x)", gridcolor=_C["grid"]),
    )
    return fig


def chart_waterfall(vars_: dict, n: int, cod: int):
    import plotly.graph_objects as go
    qs = _quarters(n, cod)
    ops = slice(cod, n)

    opex_pay = vars_.get("waterfall_block.opex_payment",       np.zeros(n))
    ds_pay   = vars_.get("waterfall_block.senior_debt_service", np.zeros(n))
    dsra_pay = vars_.get("waterfall_block.dsra_funding",        np.zeros(n))
    sweep    = vars_.get("waterfall_block.cash_sweep",          np.zeros(n))
    eq_dist  = vars_.get("waterfall_block.equity_distribution", np.zeros(n))

    fig = go.Figure()
    for arr, name, color in [
        (opex_pay, "OpEx + Tax",    _C["red"]),
        (ds_pay,   "Debt Service",  _C["navy"]),
        (dsra_pay, "DSRA Funding",  _C["purple"]),
        (sweep,    "Cash Sweep",    _C["amber"]),
        (eq_dist,  "Equity Dist.",  _C["green"]),
    ]:
        fig.add_trace(go.Bar(x=qs[cod:], y=arr[ops], name=name,
                              marker_color=color, opacity=0.88))
    _apply_layout(fig,
        title=dict(text="Cash Waterfall — Priority Allocation (INR Lakhs)", font=dict(size=13, color="#fff")),
        yaxis=dict(title="INR Lakhs", gridcolor=_C["grid"]),
        barmode="stack",
    )
    return fig


def chart_equity_cashflow(vars_: dict, n: int, cod: int):
    import plotly.graph_objects as go
    qs = _quarters(n)

    eq_cf = vars_.get("returns_block.equity_cashflow", np.zeros(n))

    colors = [_C["red"] if v < 0 else _C["green"] for v in eq_cf]
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=cod - 0.5,
                  fillcolor="rgba(255,255,255,0.03)", line_width=0)
    fig.add_trace(go.Bar(x=qs, y=eq_cf, name="Equity Cash Flow",
                          marker_color=colors, opacity=0.88))
    _apply_layout(fig,
        title=dict(text="Equity Cash Flow (INR Lakhs)", font=dict(size=13, color="#fff")),
        yaxis=dict(title="INR Lakhs", gridcolor=_C["grid"]),
    )
    return fig


def chart_tornado(sens: SensitivityResults):
    import plotly.graph_objects as go

    base = sens.base_kpis.equity_irr or 0.0
    items = sorted(sens.tornado_data.items(),
                   key=lambda kv: abs(kv[1][1] - kv[1][0]))

    names  = [_ASSUMPTION_LABELS.get(k, k) for k, _ in items]
    lo_del = [(v[0] - base) * 100 for _, v in items]
    hi_del = [(v[1] - base) * 100 for _, v in items]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=lo_del, name="Low case",
                          orientation="h", marker_color=_C["red"], opacity=0.82))
    fig.add_trace(go.Bar(y=names, x=hi_del, name="High case",
                          orientation="h", marker_color=_C["green"], opacity=0.82))
    fig.add_vline(x=0, line=dict(color=_C["text"], width=1.5, dash="dash"))
    _apply_layout(fig,
        title=dict(text=f"Sensitivity — Equity IRR Impact (Base: {_pct(base)})",
                   font=dict(size=13, color="#fff")),
        xaxis=dict(title="Equity IRR Change (pp)", gridcolor=_C["grid"]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        barmode="overlay",
        height=380,
    )
    return fig


def chart_scenario_comparison(results_map: dict):
    import plotly.graph_objects as go

    names = list(results_map.keys())
    eq_irr     = [r.kpis.equity_irr    * 100 if r.kpis.equity_irr    is not None else 0 for r in results_map.values()]
    proj_irr   = [r.kpis.project_irr   * 100 if r.kpis.project_irr   is not None else 0 for r in results_map.values()]
    min_dscr   = [r.kpis.min_dscr              if r.kpis.min_dscr     is not None else 0 for r in results_map.values()]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=eq_irr, name="Equity IRR (%)",
                          marker_color=_C["green"], opacity=0.85))
    fig.add_trace(go.Bar(x=names, y=proj_irr, name="Project IRR (%)",
                          marker_color=_C["teal"], opacity=0.85))
    fig.add_trace(go.Scatter(x=names, y=min_dscr, name="Min DSCR (x)",
                              yaxis="y2", mode="lines+markers",
                              line=dict(color=_C["orange"], width=2.5),
                              marker=dict(size=8)))
    fig.add_hline(y=1.20, line=dict(color=_C["amber"], width=1, dash="dash"),
                  annotation_text="1.20x DSCR target", annotation_font_color=_C["amber"],
                  yref="y2")
    _apply_layout(fig,
        title=dict(text="Scenario Comparison", font=dict(size=13, color="#fff")),
        yaxis=dict(title="IRR (%)", gridcolor=_C["grid"]),
        yaxis2=dict(title="DSCR (x)", overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)"),
        barmode="group",
    )
    return fig


# ---------------------------------------------------------------------------
# KPI card renderer — uses st.columns so HTML renders reliably
# ---------------------------------------------------------------------------

def _render_kpi_cards(kpis) -> None:
    def badge(v, lo, hi):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        cls  = "badge-pass" if v >= hi else ("badge-warn" if v >= lo else "badge-fail")
        text = "PASS" if v >= hi else ("CAUTION" if v >= lo else "FAIL")
        return f'<span class="kpi-badge {cls}">{text}</span>'

    cards = [
        ("green",  "Equity IRR",   _pct(kpis.equity_irr),
         "Annualised (XIRR)",      badge(kpis.equity_irr, 0.12, 0.14)),
        ("blue",   "Project IRR",  _pct(kpis.project_irr),
         "Ungeared return",        ""),
        ("orange", "Min DSCR",     _x(kpis.min_dscr),
         "Minimum across tenor",   badge(kpis.min_dscr, 1.0, 1.10)),
        ("blue",   "Avg DSCR",     _x(kpis.avg_dscr),
         "Average during repayment",""),
        ("purple", "LLCR",         _x(kpis.llcr),
         "Life loan cycle ratio",  badge(kpis.llcr, 1.20, 1.40)),
        ("purple", "PLCR",         _x(kpis.plcr),
         "Project loan cycle ratio",""),
        ("green",  "NPV (Equity)", f"₹{_f(kpis.npv_equity, 0)} L",
         "@ 12% discount rate",    ""),
        ("orange", "Debt Payback", f"{_f(kpis.debt_payback_period)} yrs",
         f"Peak: ₹{_f(kpis.peak_debt_outstanding, 0)} L", ""),
    ]

    cols = st.columns(4)
    for i, (color, label, value, sub, bdg) in enumerate(cards):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="kpi-card {color}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
                {bdg}
            </div>
            """, unsafe_allow_html=True)
        # After 4 cards, open a new row of columns
        if i == 3:
            cols = st.columns(4)


# ---------------------------------------------------------------------------
# Sidebar — input assumptions
# ---------------------------------------------------------------------------

def _sidebar() -> dict:
    st.sidebar.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">☀️</div>
        <h2>Solar IPP Model</h2>
        <p>Karnataka Project Finance</p>
    </div>
    """, unsafe_allow_html=True)

    overrides = {}

    with st.sidebar.expander("⚡ Generation", expanded=True):
        overrides["capacity_mw"] = st.number_input(
            "Capacity (MW)", value=100.0, min_value=1.0, max_value=2000.0, step=5.0,
            help="Installed AC capacity in MW")
        overrides["cuf"] = st.slider(
            "CUF (%)", min_value=14.0, max_value=32.0, value=22.0, step=0.1,
            format="%.1f%%",
            help="Capacity Utilisation Factor") / 100.0
        overrides["degradation_rate"] = st.slider(
            "Degradation (% p.a.)", min_value=0.2, max_value=1.0, value=0.5, step=0.05,
            format="%.2f%%") / 100.0
        overrides["auxiliary_consumption"] = st.number_input(
            "Auxiliary Consumption (%)", value=0.5, min_value=0.0, max_value=5.0, step=0.05,
            format="%.2f") / 100.0

    with st.sidebar.expander("💰 Revenue"):
        overrides["tariff"] = st.number_input(
            "PPA Tariff (₹/kWh)", value=2.65, min_value=1.0, max_value=6.0, step=0.01,
            format="%.2f")
        overrides["tariff_escalation"] = st.number_input(
            "Tariff Escalation (% p.a.)", value=0.0, min_value=0.0, max_value=10.0, step=0.1,
            format="%.1f") / 100.0
        overrides["ppa_tenor_years"] = st.number_input(
            "PPA Tenor (years)", value=25, min_value=10, max_value=30, step=1)

    with st.sidebar.expander("🏗️ Capital Expenditure"):
        overrides["capex_per_mw"] = st.number_input(
            "CapEx / MW (₹ Lakhs)", value=450.0, min_value=200.0, max_value=800.0, step=5.0,
            format="%.1f", help="All-in project capital cost per MW AC")
        overrides["maintenance_capex_pct"] = st.number_input(
            "Maintenance CapEx (% of total p.a.)", value=0.0, min_value=0.0, max_value=5.0,
            step=0.1, format="%.1f") / 100.0
        st.markdown('<div style="font-size:0.72rem;color:#7aa8cc;margin:10px 0 4px 0;letter-spacing:0.06em">CONSTRUCTION DRAWDOWN SCHEDULE</div>', unsafe_allow_html=True)
        q1 = st.slider("Q1 (%)", 0, 100, 25, 5, format="%d%%", key="cs_q1")
        q2 = st.slider("Q2 (%)", 0, 100, 25, 5, format="%d%%", key="cs_q2")
        q3 = st.slider("Q3 (%)", 0, 100, 25, 5, format="%d%%", key="cs_q3")
        q4 = st.slider("Q4 (%)", 0, 100, 25, 5, format="%d%%", key="cs_q4")
        total_pct = q1 + q2 + q3 + q4
        if total_pct != 100:
            st.markdown(f'<div style="font-size:0.7rem;color:#ff5555">⚠ Sums to {total_pct}% — adjust to 100%</div>', unsafe_allow_html=True)
        overrides["capex_schedule"] = [q1/100, q2/100, q3/100, q4/100]

    with st.sidebar.expander("🔧 Operations & Maintenance"):
        overrides["opex_per_mw_pa"] = st.number_input(
            "O&M / MW / year (₹ Lakhs)", value=8.0, min_value=2.0, max_value=25.0, step=0.5,
            format="%.1f")
        overrides["opex_escalation"] = st.slider(
            "O&M Escalation (% p.a.)", min_value=0.0, max_value=8.0, value=3.0, step=0.1,
            format="%.1f%%") / 100.0
        overrides["insurance_percent_of_capex"] = st.number_input(
            "Insurance (% of CapEx p.a.)", value=0.5, min_value=0.0, max_value=2.0,
            step=0.05, format="%.2f") / 100.0
        overrides["land_lease_lakhs_pa"] = st.number_input(
            "Land Lease (₹ Lakhs p.a.)", value=0.0, min_value=0.0, max_value=500.0, step=5.0,
            format="%.1f")

    with st.sidebar.expander("🏦 Debt & Financing"):
        # --- Debt sizing mode ---
        st.markdown('<div style="font-size:0.72rem;color:#7aa8cc;margin:2px 0 4px 0;letter-spacing:0.06em">DEBT SIZING MODE</div>', unsafe_allow_html=True)
        dsm = st.radio(
            "Debt Sizing",
            ["Cost-based (fixed %)", "CFADS sculpted (DSCR target)"],
            index=0, horizontal=False,
            help="Cost-based: debt = capex × debt%. Sculpted: debt sized so DSCR = target throughout tenor.",
            label_visibility="collapsed")
        overrides["debt_sizing_mode"] = 0.0 if dsm.startswith("Cost") else 1.0

        st.markdown('<div style="font-size:0.72rem;color:#7aa8cc;margin:10px 0 4px 0;letter-spacing:0.06em">DEBT PARAMETERS</div>', unsafe_allow_html=True)
        overrides["debt_pct"] = st.slider(
            "Debt / Total Cost (%)", min_value=40.0, max_value=85.0, value=70.0, step=1.0,
            format="%.0f%%") / 100.0
        overrides["interest_rate"] = st.slider(
            "Interest Rate (% p.a.)", min_value=6.0, max_value=15.0, value=9.75, step=0.25,
            format="%.2f%%") / 100.0
        overrides["dscr_target"] = st.number_input(
            "DSCR Target (x)", value=1.20, min_value=1.0, max_value=2.0, step=0.05,
            format="%.2f")
        overrides["moratorium_periods"] = st.number_input(
            "Moratorium (quarters post-COD)", value=0, min_value=0, max_value=8, step=1)
        overrides["dsra_months"] = st.number_input(
            "DSRA Coverage (months)", value=6, min_value=0, max_value=12, step=1)
        overrides["cash_sweep_rate"] = st.slider(
            "Cash Sweep Rate (%)", min_value=0.0, max_value=100.0, value=0.0, step=5.0,
            format="%.0f%%") / 100.0

        # --- IDC switch ---
        st.markdown('<div style="font-size:0.72rem;color:#7aa8cc;margin:10px 0 4px 0;letter-spacing:0.06em">INTEREST DURING CONSTRUCTION (IDC)</div>', unsafe_allow_html=True)
        idc_mode = st.radio(
            "IDC Treatment",
            ["Equity funded (default)", "Capitalised into debt base"],
            index=0, horizontal=False,
            help="Equity funded: IDC paid by sponsors during construction. Capitalised: IDC added to debt principal (requires goal-seek solver).",
            label_visibility="collapsed")
        overrides["idc_capitalised"] = 1.0 if idc_mode.startswith("Capitalised") else 0.0

    with st.sidebar.expander("📊 Tax & Depreciation"):
        overrides["tax_rate"] = st.slider(
            "Corporate Tax Rate (%)", min_value=15.0, max_value=35.0, value=25.0, step=0.5,
            format="%.1f%%") / 100.0
        dep_method = st.radio("Depreciation Method", ["SLM", "WDV"], horizontal=True)
        overrides["depreciation_method"] = dep_method.lower()
        overrides["use_wdv"] = 1.0 if dep_method == "WDV" else 0.0
        if dep_method == "SLM":
            overrides["depreciation_rate"] = st.slider(
                "SLM Rate (% p.a.)", min_value=2.5, max_value=10.0, value=5.0, step=0.5,
                format="%.1f%%") / 100.0
        else:
            overrides["wdv_rate"] = st.slider(
                "WDV Rate (% p.a.)", min_value=20.0, max_value=60.0, value=40.0, step=5.0,
                format="%.0f%%") / 100.0

    with st.sidebar.expander("📐 Returns & Hurdle"):
        overrides["equity_irr_target"] = st.slider(
            "Equity IRR Hurdle (% p.a.)", min_value=8.0, max_value=25.0, value=14.0, step=0.5,
            format="%.1f%%",
            help="Target equity IRR used for NPV computation and bankability benchmarking.") / 100.0

    st.sidebar.markdown("---")
    run_clicked    = st.sidebar.button("▶  Run Model", use_container_width=True)
    export_clicked = st.sidebar.button("📥  Export to Excel", use_container_width=True)
    include_audit  = st.sidebar.checkbox(
        "Include audit trail",
        value=False,
        help="Adds a full audit sheet to the workbook — every variable, expression and computed array. Makes the file significantly larger.")

    return overrides, run_clicked, export_clicked, include_audit


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    import plotly.graph_objects as go

    # ---- Header ----
    st.markdown("""
    <div class="page-header">
        <div class="sun-icon">☀️</div>
        <div>
            <h1>Solar IPP &mdash; Project Finance Model</h1>
            <p>Karnataka &nbsp;·&nbsp; 100 MW AC &nbsp;·&nbsp; Quarterly Model &nbsp;·&nbsp; 26-Year Horizon &nbsp;·&nbsp; INR Lakhs</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Sidebar inputs ----
    overrides, run_clicked, export_clicked, include_audit = _sidebar()

    # ---- Load & compile (cached) ----
    with st.spinner("Loading model…"):
        executor, compiled, validation = _load_and_compile()

    if not validation.valid:
        st.error("Model template parse errors: " + "; ".join(validation.errors))
        return

    # ---- Session state — persist last run ----
    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "sens" not in st.session_state:
        st.session_state["sens"] = None
    if "scenario_map" not in st.session_state:
        st.session_state["scenario_map"] = None

    # Auto-run on first load
    if st.session_state["results"] is None:
        run_clicked = True

    if run_clicked:
        with st.spinner("Running model…"):
            try:
                results = executor.run(compiled, overrides)
                st.session_state["results"] = results
                st.session_state["overrides_used"] = overrides.copy()
            except Exception as exc:
                st.error(f"Model run failed: {exc}")
                return

    results: ModelResults = st.session_state["results"]
    if results is None:
        st.info("Configure assumptions in the sidebar and click **Run Model**.")
        return

    k = results.kpis

    # ---- Convergence status (compact, inline) ----
    pills_html = ""
    for meta in results.convergence:
        cls  = "conv-ok" if meta.converged else "conv-fail"
        icon = "✓" if meta.converged else "✗"
        pills_html += (f'<span class="conv-pill {cls}">{icon} {meta.loop_id} '
                       f'· {meta.iterations} iters · res {meta.final_residual:.1e}</span>')
    if pills_html:
        st.markdown(pills_html, unsafe_allow_html=True)

    # ---- Warnings — collapsed by default so they don't dominate ----
    if results.warnings:
        n_warn = len(results.warnings)
        has_bankability = any("BANKABILITY" in w or "DSCR<1" in w for w in results.warnings)
        label = f"⚠ {n_warn} warning{'s' if n_warn > 1 else ''}" + (" — Bankability issue detected" if has_bankability else "")
        with st.expander(label, expanded=has_bankability):
            for w in results.warnings:
                st.markdown(f'<div class="warn-banner" style="margin-bottom:6px">⚠ {w}</div>',
                            unsafe_allow_html=True)

    # ---- KPI dashboard ----
    st.markdown('<div class="section-label">Key Performance Indicators</div>', unsafe_allow_html=True)
    _render_kpi_cards(k)

    # ---- Derived quick stats ----
    vars_ = results.variables
    n     = compiled.model_def.project_skeleton.total_periods
    cod   = compiled.model_def.project_skeleton.milestones.cod
    dscr_target = st.session_state["overrides_used"].get("dscr_target", 1.20)

    # ---- Tabs ----
    tab_fin, tab_debt, tab_scen, tab_sens = st.tabs([
        "📈 Financial Overview",
        "🏦 Debt & Coverage",
        "🔄 Scenarios",
        "🌪️ Sensitivity",
    ])

    # ========== Tab 1 — Financial Overview ==========
    with tab_fin:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_generation_revenue(vars_, n, cod),
                            use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.plotly_chart(chart_cashflow(vars_, n, cod),
                            use_container_width=True, config={"displayModeBar": False})

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(chart_waterfall(vars_, n, cod),
                            use_container_width=True, config={"displayModeBar": False})
        with c4:
            st.plotly_chart(chart_equity_cashflow(vars_, n, cod),
                            use_container_width=True, config={"displayModeBar": False})

    # ========== Tab 2 — Debt & Coverage ==========
    with tab_debt:
        st.plotly_chart(chart_debt_schedule(vars_, n, cod),
                        use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(chart_dscr(vars_, n, cod, dscr_target),
                        use_container_width=True, config={"displayModeBar": False})

    # ========== Tab 3 — Scenarios ==========
    with tab_scen:
        if st.button("▶  Run All Scenarios", key="run_scen"):
            with st.spinner("Running 7 scenarios…"):
                smap = {}
                for s in _SCENARIOS:
                    try:
                        merged = {**st.session_state["overrides_used"], **s["overrides"]}
                        smap[s["name"]] = executor.run(compiled, merged)
                    except Exception:
                        pass
                st.session_state["scenario_map"] = smap

        smap = st.session_state.get("scenario_map")
        if smap:
            st.plotly_chart(chart_scenario_comparison(smap),
                            use_container_width=True, config={"displayModeBar": False})

            # Scenario table
            rows_html = ""
            for i, (name, r) in enumerate(smap.items()):
                kk = r.kpis
                flag = (" <span class='flag-dscr'>⚠ DSCR&lt;1</span>"
                        if kk.min_dscr is not None and kk.min_dscr < 1.0 else "")
                row_cls = "base-row" if i == 0 else ""
                rows_html += f"""
                <tr class="{row_cls}">
                    <td>{name}{flag}</td>
                    <td>{_pct(kk.equity_irr)}</td>
                    <td>{_pct(kk.project_irr)}</td>
                    <td>{_x(kk.min_dscr)}</td>
                    <td>{_x(kk.avg_dscr)}</td>
                    <td>{_x(kk.llcr)}</td>
                    <td>₹{_f(kk.npv_equity, 0)} L</td>
                    <td>{_f(kk.debt_payback_period)} yrs</td>
                </tr>"""

            table_html = f"""
            <table class="scenario-table">
              <thead>
                <tr>
                  <th>Scenario</th>
                  <th>Equity IRR</th>
                  <th>Project IRR</th>
                  <th>Min DSCR</th>
                  <th>Avg DSCR</th>
                  <th>LLCR</th>
                  <th>NPV (Equity)</th>
                  <th>Debt Payback</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table>"""
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.info("Click **Run All Scenarios** to compare Base, Management, Downside, Stress and Upside cases.")

    # ========== Tab 4 — Sensitivity ==========
    with tab_sens:
        if st.button("▶  Run Sensitivity Analysis", key="run_sens"):
            with st.spinner("Running sensitivity sweep…"):
                try:
                    sens = executor.run_sensitivity(
                        compiled, st.session_state["overrides_used"], _SWEEP)
                    st.session_state["sens"] = sens
                except Exception as exc:
                    st.error(f"Sensitivity failed: {exc}")

        sens = st.session_state.get("sens")
        if sens:
            st.plotly_chart(chart_tornado(sens),
                            use_container_width=True, config={"displayModeBar": False})

            # Sensitivity table
            rows_html = ""
            base_irr = sens.base_kpis.equity_irr or 0.0
            ranked = sorted(sens.tornado_data.items(),
                            key=lambda kv: abs(kv[1][1] - kv[1][0]), reverse=True)
            for name, (lo, hi) in ranked:
                swing = hi - lo
                color = _C["green"] if swing >= 0 else _C["red"]
                rows_html += f"""<tr>
                    <td>{_ASSUMPTION_LABELS.get(name, name)}</td>
                    <td>{_pct(lo)}</td>
                    <td>{_pct(hi)}</td>
                    <td style="color:{color}; font-weight:700">{_pct(swing)}</td>
                </tr>"""

            table_html = f"""
            <p style="font-size:0.78rem; color:{_C['text']}; margin-bottom:10px">
                Base-case Equity IRR: <strong style="color:#fff">{_pct(base_irr)}</strong>
            </p>
            <table class="scenario-table">
              <thead>
                <tr>
                  <th>Assumption</th>
                  <th>Low Case IRR</th>
                  <th>High Case IRR</th>
                  <th>Swing</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table>"""
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.info("Click **Run Sensitivity Analysis** to generate the tornado chart.")

    # ========== Excel Export ==========
    if export_clicked:
        import io
        from engine.excel_exporter import export_to_excel

        fname    = "solar_ipp_audit.xlsx" if include_audit else "solar_ipp_model.xlsx"
        spinner_msg = "Generating Excel workbook with audit trail — this may take a moment…" if include_audit else "Generating Excel workbook…"
        with st.spinner(spinner_msg):
            out_path = _ROOT / "output" / fname
            out_path.parent.mkdir(exist_ok=True)
            smap = st.session_state.get("scenario_map") or {}
            sens = st.session_state.get("sens")
            export_to_excel(
                results=results,
                compiled=compiled,
                path=str(out_path),
                sensitivity_results=sens,
                mc_results=None,
                scenario_results=smap,
                include_audit_trail=include_audit,
            )
            with open(out_path, "rb") as f:
                xlsx_bytes = f.read()

        label = "⬇ Download (with Audit Trail)" if include_audit else "⬇ Download Excel"
        st.sidebar.download_button(
            label=label,
            data=xlsx_bytes,
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.sidebar.success("Ready — click Download above.")


if __name__ == "__main__":
    main()
