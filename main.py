"""
main.py — Programmatic runner for the Solar IPP financial model.

Loads the Karnataka 100 MW baseline template, compiles it, and executes:
  1. Base case  — schema defaults, prints KPI table
  2. Scenarios  — directional overrides, prints comparison table
  3. Sensitivity — tornado chart ranked by equity IRR impact
  4. Monte Carlo — distributional KPI statistics (optional, slow)

Usage
-----
    cd solar_ipp_model
    python main.py                  # base + scenarios + sensitivity
    python main.py --monte-carlo    # also run Monte Carlo (adds ~30 s)
    python main.py --mc-iter 500    # Monte Carlo with custom iteration count
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from the repo root or from solar_ipp_model/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dsl.parser import DSLParser
from engine.executor import ModelExecutor, ModelResults, MonteCarloResults, SensitivityResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEP  = "=" * 70
_SEP2 = "-" * 70

def _fmt_pct(v: float | None, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  n/a  "
    return f"{v * 100:.{decimals}f}%"

def _fmt_x(v: float | None, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  n/a  "
    return f"{v:.{decimals}f}x"

def _fmt_f(v: float | None, decimals: int = 1) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  n/a  "
    return f"{v:,.{decimals}f}"

def _kpi_row(label: str, value: str, width: int = 28) -> str:
    return f"  {label:<{width}} {value}"


# ---------------------------------------------------------------------------
# Section 1 — Base case
# ---------------------------------------------------------------------------

def run_base_case(executor: ModelExecutor, compiled) -> ModelResults:
    print(_SEP)
    print("  BASE CASE  —  Karnataka 100 MW Solar IPP")
    print(_SEP)

    results = executor.run(compiled, {})
    k = results.kpis

    print(_kpi_row("Equity IRR",           _fmt_pct(k.equity_irr)))
    print(_kpi_row("Project IRR",          _fmt_pct(k.project_irr)))
    print(_kpi_row("Min DSCR",             _fmt_x(k.min_dscr)))
    print(_kpi_row("Avg DSCR",             _fmt_x(k.avg_dscr)))
    print(_kpi_row("LLCR",                 _fmt_x(k.llcr)))
    print(_kpi_row("PLCR",                 _fmt_x(k.plcr)))
    print(_kpi_row("NPV (equity, INR L)",  _fmt_f(k.npv_equity)))
    print(_kpi_row("Peak debt (INR L)",    _fmt_f(k.peak_debt_outstanding)))
    print(_kpi_row("Debt payback",         f"{_fmt_f(k.debt_payback_period)} yrs"))

    # Convergence
    print()
    print("  Solve-loop convergence:")
    for meta in results.convergence:
        status = "converged" if meta.converged else "DID NOT CONVERGE"
        print(f"    [{meta.loop_id}]  {status}  "
              f"iters={meta.iterations}  residual={meta.final_residual:.2e}")

    # Validation warnings
    if results.warnings:
        print()
        print("  Warnings:")
        for w in results.warnings:
            print(f"    ! {w}")

    print()
    return results


# ---------------------------------------------------------------------------
# Section 2 — Scenarios
# ---------------------------------------------------------------------------

_SCENARIOS: list[dict] = [
    {
        "name": "Base Case",
        "overrides": {},
    },
    {
        "name": "Management Case",
        "overrides": {"cuf": 0.235, "capex_per_mw": 430.0},
    },
    {
        "name": "Downside Case",
        "overrides": {"cuf": 0.209, "opex_escalation": 0.033, "tariff": 2.57},
    },
    {
        "name": "Construction Stress",
        "overrides": {"capex_per_mw": 504.0, "moratorium_periods": 4},
    },
    {
        "name": "Interest Rate Stress",
        "overrides": {"interest_rate": 0.1125},   # +150 bps
    },
    {
        "name": "Combined Stress",
        "overrides": {
            "cuf": 0.209,
            "capex_per_mw": 504.0,
            "interest_rate": 0.1125,
            "moratorium_periods": 4,
        },
    },
    {
        "name": "Upside Case",
        "overrides": {"cuf": 0.245, "capex_per_mw": 420.0, "degradation_rate": 0.004},
    },
]


def run_scenarios(executor: ModelExecutor, compiled) -> None:
    print(_SEP)
    print("  SCENARIO COMPARISON")
    print(_SEP)

    col_w = 22
    hdr   = f"  {'Scenario':<{col_w}}  {'Eq IRR':>8}  {'Proj IRR':>9}  {'MinDSCR':>8}  {'LLCR':>7}  {'NPV Eq (L)':>12}"
    print(hdr)
    print("  " + _SEP2[2:])

    for s in _SCENARIOS:
        try:
            r = executor.run(compiled, s["overrides"])
            k = r.kpis
            row = (
                f"  {s['name']:<{col_w}}"
                f"  {_fmt_pct(k.equity_irr):>8}"
                f"  {_fmt_pct(k.project_irr):>9}"
                f"  {_fmt_x(k.min_dscr):>8}"
                f"  {_fmt_x(k.llcr, 2):>7}"
                f"  {_fmt_f(k.npv_equity, 0):>12}"
            )
            # Flag scenarios where DSCR < 1.0
            flag = " *** DSCR<1" if (k.min_dscr is not None and k.min_dscr < 1.0) else ""
            print(row + flag)
        except Exception as exc:
            print(f"  {s['name']:<{col_w}}  ERROR: {exc}")

    print()


# ---------------------------------------------------------------------------
# Section 3 — Sensitivity (tornado chart)
# ---------------------------------------------------------------------------

_SWEEP: list[dict] = [
    {"assumption": "cuf",           "low": 0.18,   "high": 0.26},
    {"assumption": "tariff",        "low": 2.20,   "high": 3.10},
    {"assumption": "capex_per_mw",  "low": 380.0,  "high": 520.0},
    {"assumption": "interest_rate", "low": 0.085,  "high": 0.115},
    {"assumption": "opex_per_mw_pa","low": 6.0,    "high": 12.0},
    {"assumption": "degradation_rate", "low": 0.003, "high": 0.008},
    {"assumption": "debt_pct",      "low": 0.60,   "high": 0.80},
]


def run_sensitivity(executor: ModelExecutor, compiled) -> SensitivityResults:
    print(_SEP)
    print("  SENSITIVITY ANALYSIS  —  Impact on Equity IRR (tornado chart)")
    print(_SEP)

    sens = executor.run_sensitivity(compiled, {}, _SWEEP)

    # Sort by absolute swing (high_impact - low_impact), descending
    ranked = sorted(
        sens.tornado_data.items(),
        key=lambda kv: abs(kv[1][1] - kv[1][0]),
        reverse=True,
    )

    bar_scale = 40  # characters per 10% IRR swing
    base_irr  = sens.base_kpis.equity_irr or 0.0

    print(f"  Base-case Equity IRR: {_fmt_pct(base_irr)}\n")
    print(f"  {'Assumption':<22}  {'Low':>8}  {'High':>8}  {'Swing':>8}  Chart")
    print("  " + _SEP2[2:])

    for name, (lo, hi) in ranked:
        swing     = hi - lo
        bar_units = int(abs(swing) / 0.10 * bar_scale)
        bar       = ("+" if swing >= 0 else "-") * min(bar_units, 50)
        print(f"  {name:<22}  {_fmt_pct(lo):>8}  {_fmt_pct(hi):>8}  "
              f"{_fmt_pct(swing):>8}  {bar}")

    print()
    return sens


# ---------------------------------------------------------------------------
# Section 4 — Monte Carlo
# ---------------------------------------------------------------------------

_MC_DIST_CONFIG: list[dict] = [
    {"assumption": "cuf",           "distribution": "triangular",
     "low": 0.18, "base": 0.22, "high": 0.26},
    {"assumption": "tariff",        "distribution": "uniform",
     "low": 2.40, "base": 2.65, "high": 2.90},
    {"assumption": "capex_per_mw",  "distribution": "triangular",
     "low": 400.0, "base": 450.0, "high": 520.0},
    {"assumption": "interest_rate", "distribution": "triangular",
     "low": 0.085, "base": 0.0975, "high": 0.115},
    {"assumption": "opex_per_mw_pa","distribution": "pert",
     "low": 6.0, "base": 8.0, "high": 12.0},
]


def run_monte_carlo(executor: ModelExecutor, compiled, n_iterations: int = 1000) -> MonteCarloResults:
    print(_SEP)
    print(f"  MONTE CARLO  —  {n_iterations} iterations")
    print(_SEP)
    print("  Running... (this may take a minute)")

    mc = executor.run_monte_carlo(
        compiled, {}, _MC_DIST_CONFIG, n_iterations=n_iterations, seed=42
    )

    def _row(kpi: str, label: str) -> None:
        p10 = mc.p10.get(kpi)
        p50 = mc.p50.get(kpi)
        p90 = mc.p90.get(kpi)
        fmt = _fmt_pct if kpi in ("equity_irr",) else _fmt_x if kpi in ("min_dscr", "llcr") else _fmt_f
        print(f"  {label:<22}  P10={fmt(p10):>8}  P50={fmt(p50):>8}  P90={fmt(p90):>8}")

    print(f"  {'KPI':<22}  {'P10':>12}  {'P50':>12}  {'P90':>12}")
    print("  " + _SEP2[2:])
    _row("equity_irr", "Equity IRR")
    _row("min_dscr",   "Min DSCR")
    _row("llcr",       "LLCR")
    _row("npv_equity", "NPV Equity (L)")

    print()
    prob_d = mc.prob_dscr_below_1
    prob_i = mc.prob_irr_below_hurdle
    if not math.isnan(prob_d):
        print(f"  Prob(DSCR < 1.0x)          : {prob_d * 100:.1f}%")
    if not math.isnan(prob_i):
        print(f"  Prob(Equity IRR < hurdle)  : {prob_i * 100:.1f}%")

    # Distribution stats
    valid_irr  = mc.equity_irr_dist[~np.isnan(mc.equity_irr_dist)]
    valid_dscr = mc.min_dscr_dist[~np.isnan(mc.min_dscr_dist)]
    if len(valid_irr) > 0:
        print(f"\n  Equity IRR  mean={_fmt_pct(float(np.mean(valid_irr)))}  "
              f"std={_fmt_pct(float(np.std(valid_irr)))}")
    if len(valid_dscr) > 0:
        print(f"  Min DSCR    mean={_fmt_x(float(np.mean(valid_dscr)))}  "
              f"std={float(np.std(valid_dscr)):.4f}x")

    print()
    return mc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solar IPP financial model runner (Karnataka 100 MW baseline)"
    )
    parser.add_argument(
        "--monte-carlo", action="store_true",
        help="Run Monte Carlo simulation in addition to base case / scenarios / sensitivity"
    )
    parser.add_argument(
        "--mc-iter", type=int, default=1000,
        help="Number of Monte Carlo iterations (default: 1000)"
    )
    parser.add_argument(
        "--template", type=str,
        default=str(_HERE / "dsl" / "templates" / "solar_ipp_base.yaml"),
        help="Path to model definition YAML (default: dsl/templates/solar_ipp_base.yaml)"
    )
    parser.add_argument(
        "--export", type=str, default=None,
        metavar="FILE",
        help="Export results to Excel workbook at the given path (e.g. output/model.xlsx)"
    )
    parser.add_argument(
        "--audit-trail", action="store_true",
        help="Include full audit trail sheet in Excel export (large file)"
    )
    args = parser.parse_args()

    template_path = Path(args.template)
    if not template_path.exists():
        print(f"ERROR: template not found: {template_path}")
        sys.exit(1)

    # --- Load & compile ---
    print()
    print(_SEP)
    print("  SOLAR IPP PROJECT FINANCE MODEL")
    print(f"  Template : {template_path.name}")
    print(_SEP)
    print("  Loading model definition...")

    model_def, validation = DSLParser().load_file(template_path)

    if not validation.valid:
        print("  PARSE ERRORS:")
        for e in validation.errors:
            print(f"    ! {e}")
        sys.exit(1)

    if validation.warnings:
        print("  Parse warnings:")
        for w in validation.warnings:
            print(f"    ~ {w}")

    skel = model_def.project_skeleton
    n_blocks = len(model_def.calculation_blocks.blocks)
    print(f"  Model     : {skel.model_id}")
    print(f"  Periods   : {skel.total_periods}  ({skel.construction_periods} construction "
          f"+ {skel.operations_periods} operations, quarterly)")
    print(f"  Blocks    : {n_blocks}")
    print("  Compiling...")

    executor = ModelExecutor()
    compiled = executor.compile(model_def)
    print("  Done.\n")

    # --- Run sections ---
    base_results = run_base_case(executor, compiled)
    run_scenarios(executor, compiled)
    sens = run_sensitivity(executor, compiled)

    mc = None
    if args.monte_carlo:
        mc = run_monte_carlo(executor, compiled, n_iterations=args.mc_iter)

    if args.export:
        from engine.excel_exporter import export_to_excel

        # Build scenario results dict for the cover sheet
        scenario_map = {}
        for s in _SCENARIOS:
            try:
                r = executor.run(compiled, s["overrides"])
                scenario_map[s["name"]] = r
            except Exception:
                pass

        out_path = export_to_excel(
            results=base_results,
            compiled=compiled,
            path=args.export,
            sensitivity_results=sens,
            mc_results=mc,
            scenario_results=scenario_map,
            include_audit_trail=args.audit_trail,
        )
        print(_SEP)
        print(f"  Excel export written to: {out_path}")

    print(_SEP)
    print("  Run complete.")
    print(_SEP)
    print()


if __name__ == "__main__":
    main()
