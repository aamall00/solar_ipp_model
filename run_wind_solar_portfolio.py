"""
run_wind_solar_portfolio.py
===========================
Programmatic runner for a two-asset portfolio:
  1. Wind IPP   — standard assumptions, DSRA = 6 months
  2. Solar IPP  — no DSRA, revenue support of ₹1/kWh

Usage
-----
    cd solar_ipp_model
    python run_wind_solar_portfolio.py
    # Output: output/wind_solar_portfolio.xlsx
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — run from repo root or from solar_ipp_model/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from env_loader import load_local_env
from agents.assumption_agent import IngestionResult, InferredAssumption
from agents.blueprint_agent import BlueprintAgent
from agents.portfolio_agent import AssetIngestion, AssetSpec
from dsl.types import ValidationResult
from engine.excel_exporter import export_portfolio_to_excel
from engine.executor import ModelExecutor
from engine.portfolio_runner import AssetResult, PortfolioRunner

load_local_env()


# ---------------------------------------------------------------------------
# Asset assumption dictionaries
# ---------------------------------------------------------------------------

# Standard wind IPP assumptions
_WIND_ASSUMPTIONS: dict = {
    # Generation
    "capacity_mw":          50.0,
    "capacity_factor":      0.30,
    "availability_factor":  0.95,
    "auxiliary_consumption": 0.005,
    # Revenue
    "tariff":               3.20,       # INR/kWh
    "tariff_escalation":    0.0,
    "ppa_tenor_years":      25.0,
    # Capex
    "capex_per_mw":         700.0,      # INR Lakhs/MW
    "capex_schedule":       [0.25, 0.25, 0.25, 0.25],
    # O&M
    "opex_per_mw_pa":       20.0,
    "opex_escalation":      0.03,
    # Debt
    "debt_pct":             0.70,
    "interest_rate":        0.0975,
    "debt_tenor_years":     18.0,
    "moratorium_periods":   0.0,
    "dscr_target":          1.20,
    "debt_sizing_mode":     0.0,
    "dsra_months":          6.0,        # standard 6-month DSRA
    "cash_sweep_rate":      0.0,
    # Tax
    "tax_rate":             0.25,
    "depreciation_rate":    0.05,
    "depreciation_method":  "slm",
    "wdv_rate":             0.40,
    "use_wdv":              0.0,
    # Equity
    "equity_irr_target":    0.14,
    # Optional features — not applicable for wind
    "has_revenue_subsidy":  0.0,
    "subsidy_per_kwh":      0.0,
    "subsidy_escalation":   0.0,
    # O&M sub-components
    "insurance_percent_of_capex": 0.005,
    "land_lease_lakhs_pa":  0.0,
    "maintenance_capex_pct": 0.0,
}

# Solar IPP assumptions — NO DSRA, revenue support ₹1/kWh
_SOLAR_ASSUMPTIONS: dict = {
    # Generation
    "capacity_mw":          100.0,
    "cuf":                  0.22,
    "degradation_rate":     0.005,
    "auxiliary_consumption": 0.005,
    # Revenue
    "tariff":               2.65,       # INR/kWh
    "tariff_escalation":    0.0,
    "ppa_tenor_years":      25.0,
    # Capex
    "capex_per_mw":         450.0,      # INR Lakhs/MW
    "capex_schedule":       [0.25, 0.25, 0.25, 0.25],
    # O&M
    "opex_per_mw_pa":       8.0,
    "opex_escalation":      0.03,
    # Debt
    "debt_pct":             0.70,
    "interest_rate":        0.0975,
    "debt_tenor_years":     18.0,
    "moratorium_periods":   0.0,
    "dscr_target":          1.20,
    "debt_sizing_mode":     0.0,
    "dsra_months":          0.0,        # NO DSRA — waived for this solar project
    "cash_sweep_rate":      0.0,
    # Tax
    "tax_rate":             0.25,
    "depreciation_rate":    0.05,
    "depreciation_method":  "slm",
    "wdv_rate":             0.40,
    "use_wdv":              0.0,
    # Equity
    "equity_irr_target":    0.14,
    # Revenue support ₹1/kWh
    "has_revenue_subsidy":  1.0,
    "subsidy_per_kwh":      1.0,        # ₹1/kWh generation-linked revenue support
    "subsidy_escalation":   0.0,
    # O&M sub-components
    "insurance_percent_of_capex": 0.005,
    "land_lease_lakhs_pa":  0.0,
    "maintenance_capex_pct": 0.0,
}


# ---------------------------------------------------------------------------
# Build IngestionResult directly (no LLM extraction needed)
# ---------------------------------------------------------------------------

def _make_ingestion(assumptions: dict, asset_type: str) -> IngestionResult:
    """Wrap a pre-defined assumption dict into an IngestionResult."""
    return IngestionResult(
        filled_assumptions=dict(assumptions),
        missing_required=[],
        inferred_assumptions={},
        validation=ValidationResult(valid=True, errors=[], warnings=[]),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_SEP = "=" * 70

def _fmt_pct(v):
    if v is None:
        return "n/a"
    return f"{v * 100:.2f}%"

def _fmt_x(v):
    if v is None:
        return "n/a"
    return f"{v:.3f}×"


def main() -> None:
    print()
    print(_SEP)
    print("  WIND + SOLAR PORTFOLIO — PROJECT FINANCE MODEL")
    print(_SEP)
    print("  Assets:")
    print("    1. Wind IPP  - 50 MW, Rs.3.20/kWh, 30% CF, DSRA = 6 months")
    print("    2. Solar IPP - 100 MW, Rs.2.65/kWh, 22% CUF, NO DSRA, Rs.1/kWh revenue support")
    print()

    # --- Build AssetSpec and IngestionResult for each asset ---
    wind_spec  = AssetSpec(
        name="SPV-Wind Wind",
        asset_type="wind",
        spv_name="SPV-Wind",
        description="50 MW wind IPP, Rs.3.20/kWh tariff, 30% capacity factor, 70% debt at 9.75%",
    )
    solar_spec = AssetSpec(
        name="SPV-Solar Solar",
        asset_type="solar",
        spv_name="SPV-Solar",
        description="100 MW solar IPP, Rs.2.65/kWh tariff, 22% CUF, no DSRA, Rs.1/kWh revenue support",
    )

    ingestions = [
        AssetIngestion(
            spec=wind_spec,
            ingestion_result=_make_ingestion(_WIND_ASSUMPTIONS, "wind"),
        ),
        AssetIngestion(
            spec=solar_spec,
            ingestion_result=_make_ingestion(_SOLAR_ASSUMPTIONS, "solar"),
        ),
    ]

    # --- Run portfolio pipeline ---
    print("  Building and running models...")
    runner  = PortfolioRunner(explain=False, review=False)
    results = runner.run(ingestions)

    # --- Print KPI summary ---
    print()
    print(_SEP)
    print(f"  {'Asset':<28}  {'Eq IRR':>8}  {'MinDSCR':>8}  {'LLCR':>7}  {'Proj IRR':>9}")
    print("  " + "-" * 66)
    for ar in results:
        k = ar.model_results.kpis
        print(
            f"  {ar.spec.name:<28}"
            f"  {_fmt_pct(k.equity_irr):>8}"
            f"  {_fmt_x(k.min_dscr):>8}"
            f"  {_fmt_x(k.llcr):>7}"
            f"  {_fmt_pct(k.project_irr):>9}"
        )
        if ar.validation.warnings:
            for w in ar.validation.warnings:
                print(f"    ~ {w.message}")
    print()

    # --- Export to Excel ---
    output_dir = _HERE / "output"
    output_dir.mkdir(exist_ok=True)
    export_path = str(output_dir / "wind_solar_portfolio.xlsx")

    print("  Exporting to Excel...")
    out = export_portfolio_to_excel(results, path=export_path)
    print()
    print(_SEP)
    print(f"  Portfolio Excel written to: {out}")
    print(_SEP)
    print()


if __name__ == "__main__":
    main()
