"""
engine/excel_exporter.py — Export Solar IPP model results to a formatted Excel workbook.

Sheets produced
---------------
  Cover             Project summary card: KPIs, assumptions, convergence status
  Income Statement  Revenue → EBITDA → EBIT → PBT → Tax → PAT, quarterly + annual
  Cash Flow         CFADS, debt service, free cashflow, equity cashflow, project cashflow
  Debt Schedule     Drawdowns, outstanding balance, principal, interest, total DS, DSRA
  Generation        Capacity, CUF, gross/net generation, effective tariff, revenue
  Waterfall         5-bucket priority allocation each period
  Returns           Equity invested, distributions, IRR cashflows
  Sensitivity       Tornado chart (if SensitivityResults provided)
  Monte Carlo       Percentile table + distribution stats (if MonteCarloResults provided)
  Assumptions       Full dump of effective assumptions used in the run
  Audit Trail       Every intermediate variable, every period (optional)

Usage
-----
    from engine.excel_exporter import export_to_excel

    export_to_excel(
        results=base_results,
        compiled=compiled,
        path="output/karnataka_100mw.xlsx",
        sensitivity_results=sens,     # optional
        mc_results=mc,                # optional
        scenario_results=scenarios,   # optional dict[name → ModelResults]
        include_audit_trail=False,    # True adds a large raw-data sheet
    )
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openpyxl
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.chart.series import DataPoint
from openpyxl.styles import (
    Alignment,
    Border,
    Font,
    GradientFill,
    PatternFill,
    Side,
)
from openpyxl.utils import get_column_letter

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_HEADER_DARK  = "1F3864"   # dark navy — sheet section headers
C_HEADER_MID   = "2E74B5"   # blue — block / subsection headers
C_HEADER_LIGHT = "D6E4F0"   # light blue — column header row
C_SECTION_ALT  = "EBF3FB"   # very light blue — alternate row fill
C_POSITIVE     = "E2EFDA"   # light green — positive KPI highlight
C_WARNING      = "FFF2CC"   # amber — borderline KPI
C_DANGER       = "FFE0E0"   # red-tint — breached KPI
C_WHITE        = "FFFFFF"
C_LABEL_COL    = "F2F2F2"   # light grey — row label column


# ---------------------------------------------------------------------------
# Fonts / fills / borders (reusable)
# ---------------------------------------------------------------------------

def _font(bold=False, size=10, color="000000", italic=False) -> Font:
    return Font(bold=bold, size=size, color=color, italic=italic)

def _fill(hex_color: str) -> PatternFill:
    return PatternFill("solid", fgColor=hex_color)

def _thin_border(bottom=False, top=False, left=False, right=False) -> Border:
    thin = Side(style="thin")
    return Border(
        bottom=thin if bottom else None,
        top=thin if top else None,
        left=thin if left else None,
        right=thin if right else None,
    )

def _thick_bottom() -> Border:
    return Border(bottom=Side(style="medium"))


# ---------------------------------------------------------------------------
# Number formats
# ---------------------------------------------------------------------------
FMT_LAKHS     = '#,##0.0'           # INR Lakhs
FMT_LAKHS_0   = '#,##0'
FMT_KWH       = '#,##0'
FMT_PCT       = '0.00%'
FMT_PCT1      = '0.0%'
FMT_RATIO     = '0.000'
FMT_RATIO2    = '0.00'
FMT_INT       = '#,##0'
FMT_DATE      = 'DD-MMM-YY'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col_label(period: int, cod: int, ppy: int = 4) -> str:
    """Return a human-readable period label like 'Q2 CON' or 'Q7 OPS'."""
    q = (period % ppy) + 1
    if period < cod:
        return f"Q{period + 1} CON"
    ops_q = period - cod + 1
    return f"Q{ops_q} OPS"


def _annual_col_labels(n_periods: int, cod: int, ppy: int) -> List[str]:
    """One label per period."""
    return [_col_label(t, cod, ppy) for t in range(n_periods)]


def _annual_sum(arr: np.ndarray, ppy: int = 4) -> np.ndarray:
    """Sum quarters into annual figures; last incomplete year is summed as-is."""
    n = len(arr)
    n_years = math.ceil(n / ppy)
    result = np.zeros(n_years)
    for y in range(n_years):
        result[y] = arr[y * ppy : (y + 1) * ppy].sum()
    return result


def _apply_header_row(ws, row: int, labels: List[str],
                      start_col: int = 1,
                      fill_hex: str = C_HEADER_LIGHT,
                      font_bold: bool = True,
                      font_size: int = 9) -> None:
    for i, lbl in enumerate(labels):
        c = ws.cell(row=row, column=start_col + i, value=lbl)
        c.font = _font(bold=font_bold, size=font_size, color="000000")
        c.fill = _fill(fill_hex)
        c.alignment = Alignment(horizontal="center", wrap_text=True)
        c.border = _thick_bottom()


def _write_series(ws, row: int, label: str, arr: np.ndarray,
                  data_col_start: int = 3,
                  num_fmt: str = FMT_LAKHS,
                  label_fill: str = C_LABEL_COL,
                  data_fill: str = C_WHITE,
                  bold_label: bool = False,
                  indent: int = 0) -> None:
    """Write a label in col 1 + unit col 2 + series data starting at data_col_start."""
    lbl_cell = ws.cell(row=row, column=1, value=(" " * indent * 2) + label)
    lbl_cell.font = _font(bold=bold_label, size=9)
    lbl_cell.fill = _fill(label_fill)
    lbl_cell.alignment = Alignment(horizontal="left")

    for t, v in enumerate(arr):
        cell = ws.cell(row=row, column=data_col_start + t, value=float(v) if not math.isnan(float(v)) else None)
        cell.number_format = num_fmt
        cell.font = _font(size=9)
        cell.fill = _fill(data_fill)
        cell.alignment = Alignment(horizontal="right")


def _write_blank_row(ws, row: int) -> None:
    pass  # just leave it empty


def _section_header(ws, row: int, title: str, n_cols: int,
                    fill_hex: str = C_HEADER_MID) -> None:
    ws.merge_cells(start_row=row, start_column=1,
                   end_row=row, end_column=min(n_cols, 200))
    c = ws.cell(row=row, column=1, value=title.upper())
    c.font = _font(bold=True, size=10, color="FFFFFF")
    c.fill = _fill(fill_hex)
    c.alignment = Alignment(horizontal="left", vertical="center")


def _kv(ws, row: int, key: str, value: Any,
        key_col: int = 1, val_col: int = 2,
        num_fmt: str | None = None,
        bold_val: bool = False,
        fill_hex: str | None = None) -> None:
    k = ws.cell(row=row, column=key_col, value=key)
    k.font = _font(bold=False, size=10)
    if fill_hex:
        k.fill = _fill(fill_hex)

    v = ws.cell(row=row, column=val_col, value=value)
    v.font = _font(bold=bold_val, size=10)
    if num_fmt:
        v.number_format = num_fmt
    if fill_hex:
        v.fill = _fill(fill_hex)


def _autofit(ws, min_width: int = 8, max_width: int = 18) -> None:
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            try:
                max_len = max(max_len, len(str(cell.value or "")))
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(max(max_len + 1, min_width), max_width)


# ---------------------------------------------------------------------------
# Period header builder
# ---------------------------------------------------------------------------

def _write_period_headers(ws, header_row: int, n_periods: int,
                          cod: int, ppy: int,
                          data_col_start: int = 3) -> None:
    """Write period labels as column headers."""
    # Phase label row above period labels
    for t in range(n_periods):
        c = ws.cell(row=header_row, column=data_col_start + t,
                    value=_col_label(t, cod, ppy))
        c.font = _font(bold=True, size=8,
                       color="FFFFFF" if t < cod else "1F3864")
        c.fill = _fill(C_HEADER_DARK if t < cod else C_HEADER_LIGHT)
        c.alignment = Alignment(horizontal="center")
        c.border = _thick_bottom()


# ---------------------------------------------------------------------------
# Sheet: Cover
# ---------------------------------------------------------------------------

def _sheet_cover(wb: Workbook, results, compiled, scenario_results, sheet_prefix: str = "") -> None:
    ws = wb.create_sheet(f"{sheet_prefix}Cover")
    ws.sheet_view.showGridLines = False

    k   = results.kpis
    skel = compiled.model_def.project_skeleton
    asmp = results.assumptions_used

    # Title banner
    ws.merge_cells("A1:H1")
    c = ws["A1"]
    c.value = "PROJECT FINANCE MODEL  —  SOLAR IPP"
    c.font = Font(bold=True, size=18, color="FFFFFF")
    c.fill = _fill(C_HEADER_DARK)
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 40

    ws.merge_cells("A2:H2")
    c = ws["A2"]
    c.value = skel.model_id.replace("_", " ").title()
    c.font = Font(bold=False, size=12, color="FFFFFF", italic=True)
    c.fill = _fill(C_HEADER_MID)
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[2].height = 22

    row = 4

    # ---- Key Metrics ----
    ws.merge_cells(f"A{row}:H{row}")
    ws[f"A{row}"].value = "KEY PERFORMANCE INDICATORS"
    ws[f"A{row}"].font  = _font(bold=True, size=11, color="FFFFFF")
    ws[f"A{row}"].fill  = _fill(C_HEADER_MID)
    ws[f"A{row}"].alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[row].height = 20
    row += 1

    def _kpi_row(label, value, fmt, threshold_ok=None):
        nonlocal row
        cell_k = ws.cell(row=row, column=1, value=label)
        cell_k.font = _font(size=10)
        cell_k.fill = _fill(C_LABEL_COL)

        cell_v = ws.cell(row=row, column=2, value=value)
        cell_v.font = _font(bold=True, size=10)
        cell_v.number_format = fmt
        if threshold_ok is True:
            cell_v.fill = _fill(C_POSITIVE)
        elif threshold_ok is False:
            cell_v.fill = _fill(C_DANGER)
        else:
            cell_v.fill = _fill(C_WARNING)
        row += 1

    def _ok(v, lo, hi=None):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if hi is not None:
            return lo <= v <= hi
        return v >= lo

    _kpi_row("Equity IRR",             k.equity_irr,             FMT_PCT,   _ok(k.equity_irr, 0.10, 0.35))
    _kpi_row("Project IRR",            k.project_irr,            FMT_PCT,   _ok(k.project_irr, 0.06))
    _kpi_row("Min DSCR",               k.min_dscr,               FMT_RATIO, _ok(k.min_dscr, 1.10))
    _kpi_row("Avg DSCR",               k.avg_dscr,               FMT_RATIO, _ok(k.avg_dscr, 1.20))
    _kpi_row("LLCR",                   k.llcr,                   FMT_RATIO, _ok(k.llcr, 1.10))
    _kpi_row("PLCR",                   k.plcr,                   FMT_RATIO, _ok(k.plcr, 1.10))
    _kpi_row("NPV Equity (INR Lakhs)", k.npv_equity,             FMT_LAKHS, _ok(k.npv_equity, 0))
    _kpi_row("Peak Debt (INR Lakhs)",  k.peak_debt_outstanding,  FMT_LAKHS, None)
    _kpi_row("Debt Payback (years)",   k.debt_payback_period,    FMT_RATIO2,_ok(k.debt_payback_period, 1, 18))
    row += 1

    # ---- Project Parameters ----
    ws.merge_cells(f"A{row}:H{row}")
    ws[f"A{row}"].value = "PROJECT PARAMETERS"
    ws[f"A{row}"].font  = _font(bold=True, size=11, color="FFFFFF")
    ws[f"A{row}"].fill  = _fill(C_HEADER_MID)
    ws[f"A{row}"].alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[row].height = 20
    row += 1

    params = [
        ("Model ID",                  skel.model_id,                                       "@"),
        ("Project Type",              skel.project_type.value,                              "@"),
        ("Currency",                  f"{skel.currency} {skel.currency_unit}",              "@"),
        ("Periods per Year",          skel.periods_per_year,                                FMT_INT),
        ("Construction Periods",      skel.construction_periods,                            FMT_INT),
        ("Operations Periods",        skel.operations_periods,                              FMT_INT),
        ("Total Periods",             skel.total_periods,                                   FMT_INT),
        ("COD Period",                compiled.cod_period,                                  FMT_INT),
        ("Debt Maturity Period",      compiled.debt_maturity_period,                        FMT_INT),
        ("Report Date",               date.today().strftime("%d-%b-%Y"),                    "@"),
    ]
    for label, value, fmt in params:
        cell_k = ws.cell(row=row, column=1, value=label)
        cell_k.font = _font(size=10)
        cell_k.fill = _fill(C_LABEL_COL)
        cell_v = ws.cell(row=row, column=2, value=value)
        cell_v.font = _font(size=10)
        cell_v.number_format = fmt
        row += 1
    row += 1

    # ---- Key Assumptions ----
    ws.merge_cells(f"A{row}:H{row}")
    ws[f"A{row}"].value = "KEY ASSUMPTIONS"
    ws[f"A{row}"].font  = _font(bold=True, size=11, color="FFFFFF")
    ws[f"A{row}"].fill  = _fill(C_HEADER_MID)
    ws[f"A{row}"].alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[row].height = 20
    row += 1

    key_asmp = [
        ("Installed Capacity",        asmp.get("capacity_mw"),         FMT_INT,   "MW"),
        ("CUF (P50)",                 asmp.get("cuf"),                 FMT_PCT,   "ratio"),
        ("Degradation Rate (p.a.)",   asmp.get("degradation_rate"),    FMT_PCT,   "per year"),
        ("PPA Tariff",                asmp.get("tariff"),              "0.00",    "INR/kWh"),
        ("Tariff Escalation",         asmp.get("tariff_escalation"),   FMT_PCT,   "p.a."),
        ("Capex per MW",              asmp.get("capex_per_mw"),        FMT_LAKHS, "INR Lakh/MW"),
        ("Total Capex",               (asmp.get("capex_per_mw") or 0) * (asmp.get("capacity_mw") or 0),
                                                                        FMT_LAKHS, "INR Lakhs"),
        ("Debt %",                    asmp.get("debt_pct"),            FMT_PCT,   "ratio"),
        ("Interest Rate",             asmp.get("interest_rate"),       FMT_PCT,   "p.a."),
        ("Debt Tenor",                asmp.get("debt_tenor_periods", (compiled.debt_maturity_period - compiled.cod_period)) / compiled.periods_per_year,
                                                                        FMT_RATIO2,"years"),
        ("Moratorium Periods",        asmp.get("moratorium_periods"),  FMT_INT,   "quarters"),
        ("DSCR Target",               asmp.get("dscr_target"),         FMT_RATIO, "x"),
        ("DSRA Cover",                asmp.get("dsra_months"),         FMT_INT,   "months"),
        ("Tax Rate",                  asmp.get("tax_rate"),            FMT_PCT,   "effective"),
        ("Depreciation Method",       asmp.get("depreciation_method", "slm"),
                                                                        "@",       ""),
        ("Equity IRR Target",         asmp.get("equity_irr_target"),   FMT_PCT,   "hurdle"),
    ]
    for label, value, fmt, unit in key_asmp:
        ws.cell(row=row, column=1, value=label).font = _font(size=10)
        ws.cell(row=row, column=1).fill = _fill(C_LABEL_COL)
        v_cell = ws.cell(row=row, column=2, value=value)
        v_cell.number_format = fmt
        v_cell.font = _font(size=10)
        ws.cell(row=row, column=3, value=unit).font = _font(size=9, italic=True, color="666666")
        row += 1

    # ---- Scenario comparison (if provided) ----
    if scenario_results:
        row += 1
        ws.merge_cells(f"A{row}:H{row}")
        ws[f"A{row}"].value = "SCENARIO SUMMARY"
        ws[f"A{row}"].font  = _font(bold=True, size=11, color="FFFFFF")
        ws[f"A{row}"].fill  = _fill(C_HEADER_MID)
        ws[f"A{row}"].alignment = Alignment(horizontal="left", vertical="center")
        ws.row_dimensions[row].height = 20
        row += 1

        hdrs = ["Scenario", "Equity IRR", "Project IRR", "Min DSCR", "LLCR", "NPV Equity (L)"]
        for ci, h in enumerate(hdrs):
            c = ws.cell(row=row, column=ci + 1, value=h)
            c.font = _font(bold=True, size=9, color="FFFFFF")
            c.fill = _fill(C_HEADER_DARK)
            c.alignment = Alignment(horizontal="center")
        row += 1

        for sname, sres in scenario_results.items():
            sk = sres.kpis
            vals = [sname,
                    sk.equity_irr, sk.project_irr, sk.min_dscr, sk.llcr, sk.npv_equity]
            fmts = ["@", FMT_PCT, FMT_PCT, FMT_RATIO, FMT_RATIO, FMT_LAKHS_0]
            for ci, (v, f) in enumerate(zip(vals, fmts)):
                c = ws.cell(row=row, column=ci + 1, value=v)
                c.number_format = f
                c.font = _font(size=9)
                if ci == 3 and isinstance(v, float) and not math.isnan(v):
                    c.fill = _fill(C_POSITIVE if v >= 1.10 else (C_WARNING if v >= 1.0 else C_DANGER))
            row += 1

    # Column widths
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 16

    # Freeze top rows
    ws.freeze_panes = "A3"


# ---------------------------------------------------------------------------
# Generic financial schedule sheet
# ---------------------------------------------------------------------------

def _write_schedule_sheet(
    wb: Workbook,
    sheet_name: str,
    sections: List[Dict],
    compiled,
    n_periods: int,
    label_col_width: int = 32,
    unit_col_width: int = 12,
) -> None:
    """
    Generic writer for Income Statement / Cash Flow / Debt Schedule sheets.

    sections is a list of dicts:
      {"type": "header",  "title": str}
      {"type": "row",     "label": str, "array": np.ndarray, "fmt": str,
                          "unit": str, "indent": int, "bold": bool, "fill": str}
      {"type": "blank"}
      {"type": "dscr",    "label": str, "cfads": ndarray, "ds": ndarray,
                          "mask": ndarray}
    """
    ws = wb.create_sheet(sheet_name)
    ws.sheet_view.showGridLines = False

    cod  = compiled.cod_period
    ppy  = compiled.periods_per_year

    DATA_COL = 3  # columns 1=label, 2=unit, data starts at 3

    # ---- Title row ----
    total_cols = DATA_COL + n_periods - 1
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=min(total_cols, 200))
    c = ws.cell(row=1, column=1, value=sheet_name.upper())
    c.font  = _font(bold=True, size=13, color="FFFFFF")
    c.fill  = _fill(C_HEADER_DARK)
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 28

    # ---- Period header row ----
    ws.cell(row=2, column=1, value="Line Item").font = _font(bold=True, size=9)
    ws.cell(row=2, column=1).fill = _fill(C_HEADER_LIGHT)
    ws.cell(row=2, column=2, value="Unit").font = _font(bold=True, size=9)
    ws.cell(row=2, column=2).fill = _fill(C_HEADER_LIGHT)
    _write_period_headers(ws, header_row=2, n_periods=n_periods,
                          cod=cod, ppy=ppy, data_col_start=DATA_COL)
    ws.freeze_panes = ws.cell(row=3, column=DATA_COL)

    row = 3
    for sec in sections:
        t = sec.get("type", "row")

        if t == "blank":
            row += 1
            continue

        if t == "header":
            _section_header(ws, row, sec["title"], total_cols,
                            fill_hex=sec.get("fill", C_HEADER_MID))
            row += 1
            continue

        if t == "row":
            arr     = sec["array"]
            label   = sec["label"]
            unit    = sec.get("unit", "INR Lakhs")
            fmt     = sec.get("fmt", FMT_LAKHS)
            indent  = sec.get("indent", 0)
            bold    = sec.get("bold", False)
            r_fill  = sec.get("data_fill", C_WHITE)
            l_fill  = sec.get("label_fill", C_LABEL_COL if not bold else C_SECTION_ALT)

            # Label cell
            lc = ws.cell(row=row, column=1,
                         value=("    " * indent) + label)
            lc.font  = _font(bold=bold, size=9)
            lc.fill  = _fill(l_fill)
            lc.alignment = Alignment(horizontal="left")

            # Unit cell
            uc = ws.cell(row=row, column=2, value=unit)
            uc.font  = _font(size=8, italic=True, color="666666")
            uc.fill  = _fill(l_fill)

            # Data cells
            alt = (row % 2 == 0)
            cell_fill = r_fill if r_fill != C_WHITE else (C_SECTION_ALT if alt else C_WHITE)
            for t_idx, v in enumerate(arr):
                fv = float(v)
                dc = ws.cell(row=row, column=DATA_COL + t_idx,
                             value=None if math.isnan(fv) else fv)
                dc.number_format = fmt
                dc.font  = _font(bold=bold, size=9)
                dc.fill  = _fill(cell_fill)
                dc.alignment = Alignment(horizontal="right")

            row += 1
            continue

        if t == "dscr":
            cfads = sec["cfads"]
            ds    = sec["ds"]
            mask  = sec.get("mask")
            label = sec.get("label", "DSCR")

            lc = ws.cell(row=row, column=1, value=label)
            lc.font = _font(bold=True, size=9)
            lc.fill = _fill(C_SECTION_ALT)
            ws.cell(row=row, column=2, value="x").font = _font(size=8, italic=True)
            ws.cell(row=row, column=2).fill = _fill(C_SECTION_ALT)

            for t_idx in range(n_periods):
                d = float(ds[t_idx])
                c_val = float(cfads[t_idx])
                if d > 0 and (mask is None or float(mask[t_idx]) > 0):
                    dscr_v = c_val / d
                    dc = ws.cell(row=row, column=DATA_COL + t_idx, value=dscr_v)
                    dc.number_format = FMT_RATIO
                    dc.font  = _font(bold=True, size=9)
                    if dscr_v >= 1.25:
                        dc.fill = _fill(C_POSITIVE)
                    elif dscr_v >= 1.0:
                        dc.fill = _fill(C_WARNING)
                    else:
                        dc.fill = _fill(C_DANGER)
                else:
                    dc = ws.cell(row=row, column=DATA_COL + t_idx, value=None)
                    dc.fill = _fill(C_WHITE)
            row += 1

    # Column widths
    ws.column_dimensions["A"].width = label_col_width
    ws.column_dimensions["B"].width = unit_col_width
    for t_idx in range(n_periods):
        ws.column_dimensions[get_column_letter(DATA_COL + t_idx)].width = 11


# ---------------------------------------------------------------------------
# Sheet builders
# ---------------------------------------------------------------------------

def _sheet_income_statement(wb, results, compiled, sheet_prefix: str = "") -> None:
    v   = results.variables
    n   = compiled.n_periods
    cod = compiled.cod_period

    def _get(key):
        arr = v.get(key, np.zeros(n))
        return arr

    sections = [
        {"type": "header", "title": "Revenue"},
        {"type": "row", "label": "Net Generation", "array": _get("generation_block.net_generation_kwh"),
         "unit": "kWh", "fmt": FMT_KWH, "indent": 1},
        {"type": "row", "label": "PPA Revenue", "array": _get("revenue_block.revenue"),
         "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1, "bold": True},
        {"type": "blank"},
        {"type": "header", "title": "Operating Costs"},
        {"type": "row", "label": "Base O&M",        "array": _get("opex_block.base_opex"),     "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Insurance",        "array": _get("opex_block.insurance"),     "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Land Lease",       "array": _get("opex_block.land_lease"),    "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Total OPEX",       "array": _get("opex_block.total_opex"),    "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True,
         "label_fill": C_SECTION_ALT},
        {"type": "blank"},
        {"type": "row", "label": "EBITDA",           "array": _get("cashflow_block.ebitda"),    "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True,
         "label_fill": C_SECTION_ALT},
        {"type": "blank"},
        {"type": "header", "title": "Below-the-Line (P&L)"},
        {"type": "row", "label": "Depreciation",     "array": _get("depreciation_block.depreciation"), "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "EBIT",             "array": _get("income_statement_block.ebit"),           "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True,
         "label_fill": C_SECTION_ALT},
        {"type": "row", "label": "Interest Expense", "array": _get("debt_service_block.interest_payment"), "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "PBT",              "array": _get("income_statement_block.pbt"),            "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True,
         "label_fill": C_SECTION_ALT},
        {"type": "row", "label": "Tax",              "array": _get("income_statement_block.tax"),            "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "PAT  (PBT − Tax)", "array": _get("income_statement_block.pbt") - _get("income_statement_block.tax"),
         "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True,
         "label_fill": C_SECTION_ALT},
        {"type": "blank"},
        {"type": "header", "title": "EBITDA Margin"},
        {"type": "row", "label": "EBITDA Margin %",
         "array": np.where(_get("revenue_block.revenue") > 0,
                           _get("cashflow_block.ebitda") / _get("revenue_block.revenue"),
                           np.nan),
         "unit": "%", "fmt": FMT_PCT1, "indent": 1},
    ]
    _write_schedule_sheet(wb, f"{sheet_prefix}Income Statement", sections, compiled, n)


def _sheet_cashflow(wb, results, compiled, sheet_prefix: str = "") -> None:
    v   = results.variables
    n   = compiled.n_periods

    def _get(key):
        return v.get(key, np.zeros(n))

    ds   = _get("debt_service_block.total_debt_service")
    mask = _get("debt_service_block.outstanding_debt_balance") > 0

    sections = [
        {"type": "header", "title": "Operating Cash Flow"},
        {"type": "row", "label": "EBITDA",                 "array": _get("cashflow_block.ebitda"),         "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True},
        {"type": "row", "label": "Tax Paid",               "array": -_get("income_statement_block.tax"),                "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Maintenance Capex",      "array": -_get("cashflow_block.capex_during_ops"), "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "CFADS  (pre-debt-svc)",  "array": _get("cashflow_block.cfads"),          "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True,
         "label_fill": C_SECTION_ALT},
        {"type": "blank"},
        {"type": "header", "title": "Debt Service"},
        {"type": "row", "label": "Interest",               "array": _get("debt_service_block.interest_payment"),    "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Principal Repayment",    "array": _get("debt_service_block.principal_repayment"), "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Total Debt Service",     "array": ds,                                    "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True},
        {"type": "dscr", "label": "DSCR",
         "cfads": _get("cashflow_block.cfads"), "ds": ds, "mask": mask},
        {"type": "blank"},
        {"type": "header", "title": "Free & Equity Cashflows"},
        {"type": "row", "label": "Free Cashflow  (post-DS)", "array": _get("cashflow_block.free_cashflow"), "unit": "INR Lakhs", "fmt": FMT_LAKHS},
        {"type": "row", "label": "DSRA Funding",            "array": _get("waterfall_block.dsra_funding"),  "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Cash Sweep",              "array": _get("waterfall_block.cash_sweep"),    "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Equity Distribution",     "array": _get("waterfall_block.equity_distribution"), "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "blank"},
        {"type": "header", "title": "IRR Cashflows"},
        {"type": "row", "label": "Equity Invested  (−ve)",  "array": _get("cashflow_block.equity_invested"), "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Equity Cashflow",         "array": _get("cashflow_block.equity_cashflow"), "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True},
        {"type": "row", "label": "Project Cashflow",        "array": _get("cashflow_block.project_cashflow"),"unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True},
        {"type": "blank"},
        {"type": "header", "title": "Construction Financing"},
        {"type": "row", "label": "Capex Drawdown",          "array": _get("construction_block.capex_drawdown"),     "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Debt Drawdown",           "array": _get("debt_drawdown_block.drawdown"),           "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "IDC  (interest during construction)", "array": _get("idc_block.idc_per_period"),  "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Cumulative Capex",        "array": _get("construction_block.cumulative_capex"),   "unit": "INR Lakhs", "fmt": FMT_LAKHS},
    ]
    _write_schedule_sheet(wb, f"{sheet_prefix}Cash Flow", sections, compiled, n)


def _sheet_debt_schedule(wb, results, compiled, sheet_prefix: str = "") -> None:
    v   = results.variables
    n   = compiled.n_periods

    def _get(key):
        return v.get(key, np.zeros(n))

    ds   = _get("debt_service_block.total_debt_service")
    bal  = _get("debt_service_block.outstanding_debt_balance")
    mask = bal > 0

    sections = [
        {"type": "header", "title": "Debt Sizing"},
        {"type": "row", "label": "Total Project Cost",  "array": _get("debt_drawdown_block.total_project_cost"), "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True},
        {"type": "row", "label": "Debt Amount",         "array": _get("debt_drawdown_block.debt_amount"),        "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Equity Amount",       "array": _get("debt_drawdown_block.equity_amount"),      "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "blank"},
        {"type": "header", "title": "Construction Drawdown"},
        {"type": "row", "label": "Debt Drawdown",       "array": _get("debt_drawdown_block.drawdown"),          "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Cumulative Drawdown", "array": _get("debt_drawdown_block.cumulative_drawdown"), "unit": "INR Lakhs", "fmt": FMT_LAKHS},
        {"type": "blank"},
        {"type": "header", "title": "Debt Service Schedule"},
        {"type": "row", "label": "Opening Balance",     "array": _get("debt_service_block.opening"),          "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Interest Charge",     "array": _get("debt_service_block.interest_payment"),  "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Principal Repayment", "array": _get("debt_service_block.principal_repayment"),"unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Total Debt Service",  "array": ds,                                           "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True},
        {"type": "row", "label": "Closing Balance",     "array": bal,                                          "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True,
         "label_fill": C_SECTION_ALT},
        {"type": "blank"},
        {"type": "dscr", "label": "DSCR",
         "cfads": _get("cashflow_block.cfads"), "ds": ds, "mask": mask},
        {"type": "blank"},
        {"type": "header", "title": "Debt Service Reserve Account"},
        {"type": "row", "label": "DSRA Required",       "array": _get("dsra_block.dsra_required"),            "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "DSRA Funding",        "array": _get("waterfall_block.dsra_funding"),        "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
    ]
    _write_schedule_sheet(wb, f"{sheet_prefix}Debt Schedule", sections, compiled, n)


def _sheet_project_model(wb, results, compiled, sheet_prefix: str = "") -> None:
    """
    Combined 'Project Model' sheet: Income Statement → Cash Flow → Debt Schedule
    (including full DSRA ledger) in a single scrollable schedule.
    Replaces the three separate sheets.
    """
    v   = results.variables
    n   = compiled.n_periods

    def _get(key):
        return v.get(key, np.zeros(n))

    def _safe_divide(numerator, denominator):
        num = np.asarray(numerator, dtype=np.float64)
        den = np.asarray(denominator, dtype=np.float64)
        out = np.full_like(num, np.nan, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(num, den, out=out, where=den != 0)
        return out

    ds   = _get("debt_service_block.total_debt_service")
    bal  = _get("debt_service_block.outstanding_debt_balance")
    mask = bal > 0

    sections = [
        # ── INCOME STATEMENT ────────────────────────────────────────────────
        {"type": "header", "title": "INCOME STATEMENT", "fill": C_HEADER_DARK},
        {"type": "blank"},
        {"type": "header", "title": "Revenue"},
        {"type": "row", "label": "Net Generation",      "array": _get("generation_block.net_generation_kwh"),
         "unit": "kWh", "fmt": FMT_KWH, "indent": 1},
        {"type": "row", "label": "PPA Revenue",          "array": _get("revenue_block.revenue"),
         "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True},
        {"type": "blank"},
        {"type": "header", "title": "Operating Costs"},
        {"type": "row", "label": "Base O&M",             "array": _get("opex_block.base_opex"),      "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Insurance",            "array": _get("opex_block.insurance"),      "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Land Lease",           "array": _get("opex_block.land_lease"),     "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Total OPEX",           "array": _get("opex_block.total_opex"),     "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True,
         "label_fill": C_SECTION_ALT},
        {"type": "blank"},
        {"type": "row", "label": "EBITDA",               "array": _get("cashflow_block.ebitda"),     "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True,
         "label_fill": C_SECTION_ALT},
        {"type": "blank"},
        {"type": "header", "title": "Below-the-Line (P&L)"},
        {"type": "row", "label": "Depreciation",         "array": _get("depreciation_block.depreciation"),       "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "EBIT",                 "array": _get("income_statement_block.ebit"),            "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True,
         "label_fill": C_SECTION_ALT},
        {"type": "row", "label": "Interest Expense",     "array": _get("debt_service_block.interest_payment"),   "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "PBT",                  "array": _get("income_statement_block.pbt"),             "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True,
         "label_fill": C_SECTION_ALT},
        {"type": "row", "label": "Tax",                  "array": _get("income_statement_block.tax"),             "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "PAT  (PBT − Tax)",     "array": _get("income_statement_block.pbt") - _get("income_statement_block.tax"),
         "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True, "label_fill": C_SECTION_ALT},
        {"type": "blank"},
        {"type": "header", "title": "EBITDA Margin"},
        {"type": "row", "label": "EBITDA Margin %",
         "array": _safe_divide(_get("cashflow_block.ebitda"), _get("revenue_block.revenue")),
         "unit": "%", "fmt": FMT_PCT1, "indent": 1},

        # ── CASH FLOW STATEMENT ──────────────────────────────────────────────
        {"type": "blank"},
        {"type": "header", "title": "CASH FLOW STATEMENT", "fill": C_HEADER_DARK},
        {"type": "blank"},
        {"type": "header", "title": "Operating Cash Flow"},
        {"type": "row", "label": "EBITDA",                   "array": _get("cashflow_block.ebitda"),          "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True},
        {"type": "row", "label": "Tax Paid",                 "array": -_get("income_statement_block.tax"),                 "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Maintenance Capex",        "array": -_get("cashflow_block.capex_during_ops"), "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "CFADS  (pre-debt-svc)",    "array": _get("cashflow_block.cfads"),           "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True,
         "label_fill": C_SECTION_ALT},
        {"type": "blank"},
        {"type": "header", "title": "Debt Service"},
        {"type": "row", "label": "Interest",                 "array": _get("debt_service_block.interest_payment"),    "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Principal Repayment",      "array": _get("debt_service_block.principal_repayment"), "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Total Debt Service",       "array": ds,                                     "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True},
        {"type": "dscr", "label": "DSCR",
         "cfads": _get("cashflow_block.cfads"), "ds": ds, "mask": mask},
        {"type": "blank"},
        {"type": "header", "title": "Free & Equity Cashflows"},
        {"type": "row", "label": "Free Cashflow  (post-DS)", "array": _get("cashflow_block.free_cashflow"),   "unit": "INR Lakhs", "fmt": FMT_LAKHS},
        {"type": "row", "label": "DSRA Funding",             "array": _get("waterfall_block.dsra_funding"),   "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Cash Sweep",               "array": _get("waterfall_block.cash_sweep"),     "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Equity Distribution",      "array": _get("waterfall_block.equity_distribution"), "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "blank"},
        {"type": "header", "title": "IRR Cashflows"},
        {"type": "row", "label": "Equity Invested  (−ve)",   "array": _get("cashflow_block.equity_invested"), "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Equity Cashflow",          "array": _get("cashflow_block.equity_cashflow"), "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True},
        {"type": "row", "label": "Project Cashflow",         "array": _get("cashflow_block.project_cashflow"),"unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True},
        {"type": "blank"},
        {"type": "header", "title": "Construction Financing"},
        {"type": "row", "label": "Capex Drawdown",           "array": _get("construction_block.capex_drawdown"),     "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Debt Drawdown",            "array": _get("debt_drawdown_block.drawdown"),           "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "IDC  (interest during construction)", "array": _get("idc_block.idc_per_period"),   "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Cumulative Capex",         "array": _get("construction_block.cumulative_capex"),   "unit": "INR Lakhs", "fmt": FMT_LAKHS},

        # ── DEBT SCHEDULE ────────────────────────────────────────────────────
        {"type": "blank"},
        {"type": "header", "title": "DEBT SCHEDULE", "fill": C_HEADER_DARK},
        {"type": "blank"},
        {"type": "header", "title": "Debt Sizing"},
        {"type": "row", "label": "Total Project Cost",       "array": _get("debt_drawdown_block.total_project_cost"), "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True},
        {"type": "row", "label": "Debt Amount",              "array": _get("debt_drawdown_block.debt_amount"),        "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Equity Amount",            "array": _get("debt_drawdown_block.equity_amount"),      "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "blank"},
        {"type": "header", "title": "Construction Drawdown"},
        {"type": "row", "label": "Debt Drawdown",            "array": _get("debt_drawdown_block.drawdown"),           "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Cumulative Drawdown",      "array": _get("debt_drawdown_block.cumulative_drawdown"), "unit": "INR Lakhs", "fmt": FMT_LAKHS},
        {"type": "blank"},
        {"type": "header", "title": "Debt Service Schedule"},
        {"type": "row", "label": "Opening Balance",          "array": _get("debt_service_block.opening"),            "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Interest Charge",          "array": _get("debt_service_block.interest_payment"),   "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Principal Repayment",      "array": _get("debt_service_block.principal_repayment"), "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Total Debt Service",       "array": ds,                                            "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True},
        {"type": "row", "label": "Closing Balance",          "array": bal,                                           "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True,
         "label_fill": C_SECTION_ALT},
        {"type": "blank"},
        {"type": "dscr", "label": "DSCR",
         "cfads": _get("cashflow_block.cfads"), "ds": ds, "mask": mask},
        {"type": "blank"},
        {"type": "header", "title": "Debt Service Reserve Account"},
        {"type": "row", "label": "DSRA Required",            "array": _get("dsra_block.dsra_required"),              "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True},
        {"type": "row", "label": "Opening Balance",          "array": _get("dsra_block.opening_balance"),            "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Top-Up  (Funding)",        "array": _get("dsra_block.top_up"),                     "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Release",                  "array": _get("dsra_block.release"),                    "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Closing Balance",          "array": _get("dsra_block.closing_balance"),            "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True,
         "label_fill": C_SECTION_ALT},
    ]
    _write_schedule_sheet(wb, f"{sheet_prefix}Project Model", sections, compiled, n)


def _sheet_generation(wb, results, compiled, sheet_prefix: str = "") -> None:
    v   = results.variables
    n   = compiled.n_periods
    asmp = results.assumptions_used

    def _get(key):
        return v.get(key, np.zeros(n))

    # Effective tariff series (revenue / generation, where gen > 0)
    gen = _get("generation_block.net_generation_kwh")
    rev = _get("revenue_block.revenue")
    eff_tariff = np.where(gen > 0, rev * 100000.0 / gen, np.nan)

    sections = [
        {"type": "header", "title": "Solar Generation"},
        {"type": "row", "label": "Gross Generation",   "array": _get("generation_block.gross_generation_kwh"), "unit": "kWh", "fmt": FMT_KWH, "indent": 1},
        {"type": "row", "label": "Net Generation",     "array": gen,                                           "unit": "kWh", "fmt": FMT_KWH, "bold": True},
        {"type": "blank"},
        {"type": "header", "title": "Revenue"},
        {"type": "row", "label": "Effective Tariff",   "array": eff_tariff,  "unit": "INR/kWh", "fmt": "0.0000", "indent": 1},
        {"type": "row", "label": "PPA Revenue",        "array": rev,         "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True},
        {"type": "blank"},
        {"type": "header", "title": "Depreciation Schedule"},
        {"type": "row", "label": "Depreciable Asset Base",  "array": _get("depreciation_block.depreciable_asset_base"), "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "Depreciation Charge",     "array": _get("depreciation_block.depreciation"),           "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True},
    ]
    _write_schedule_sheet(wb, f"{sheet_prefix}Generation & Revenue", sections, compiled, n)


def _sheet_waterfall(wb, results, compiled, sheet_prefix: str = "") -> None:
    v   = results.variables
    n   = compiled.n_periods

    def _get(key):
        return v.get(key, np.zeros(n))

    rev  = _get("revenue_block.revenue")
    opex = _get("waterfall_block.opex_payment")
    ds   = _get("waterfall_block.senior_debt_service")
    dsra = _get("waterfall_block.dsra_funding")
    swp  = _get("waterfall_block.cash_sweep")
    eq   = _get("waterfall_block.equity_distribution")
    total_out = opex + ds + dsra + swp + eq

    sections = [
        {"type": "header", "title": "Cash Waterfall  (5-Bucket Priority Allocation)"},
        {"type": "row", "label": "Available Cash  (Revenue)",      "array": rev,       "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True},
        {"type": "blank"},
        {"type": "row", "label": "1. OPEX Payment",                "array": opex,      "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "   Remaining after OPEX",        "array": rev - opex,"unit": "INR Lakhs", "fmt": FMT_LAKHS},
        {"type": "blank"},
        {"type": "row", "label": "2. Senior Debt Service",         "array": ds,        "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "   Remaining after DS",          "array": rev - opex - ds, "unit": "INR Lakhs", "fmt": FMT_LAKHS},
        {"type": "blank"},
        {"type": "row", "label": "3. DSRA Funding",                "array": dsra,      "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "row", "label": "   Remaining after DSRA",        "array": rev - opex - ds - dsra, "unit": "INR Lakhs", "fmt": FMT_LAKHS},
        {"type": "blank"},
        {"type": "row", "label": "4. Cash Sweep",                  "array": swp,       "unit": "INR Lakhs", "fmt": FMT_LAKHS, "indent": 1},
        {"type": "blank"},
        {"type": "row", "label": "5. Equity Distribution",         "array": eq,        "unit": "INR Lakhs", "fmt": FMT_LAKHS, "bold": True,
         "label_fill": C_SECTION_ALT},
        {"type": "blank"},
        {"type": "row", "label": "Total Allocated",                "array": total_out, "unit": "INR Lakhs", "fmt": FMT_LAKHS},
        {"type": "row", "label": "Check  (Revenue − Allocated)",   "array": rev - total_out, "unit": "INR Lakhs", "fmt": FMT_LAKHS},
    ]
    _write_schedule_sheet(wb, f"{sheet_prefix}Waterfall", sections, compiled, n)


def _sheet_sensitivity(wb, sensitivity_results) -> None:
    if sensitivity_results is None:
        return

    ws = wb.create_sheet("Sensitivity")
    ws.sheet_view.showGridLines = False

    # Title
    ws.merge_cells("A1:G1")
    ws["A1"].value = "SENSITIVITY ANALYSIS  —  TORNADO CHART  (Equity IRR)"
    ws["A1"].font  = _font(bold=True, size=13, color="FFFFFF")
    ws["A1"].fill  = _fill(C_HEADER_DARK)
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 28

    base_irr = sensitivity_results.base_kpis.equity_irr or 0.0

    # Base KPIs summary
    row = 3
    ws.cell(row=row, column=1, value="Base-Case Equity IRR").font = _font(bold=True)
    ws.cell(row=row, column=2, value=base_irr).number_format = FMT_PCT
    ws.cell(row=row, column=2).font = _font(bold=True)
    row += 2

    # Table headers
    hdrs = ["Assumption", "Low Value  (IRR)", "Base  (IRR)", "High Value  (IRR)", "Swing  (High−Low)", "Direction"]
    for ci, h in enumerate(hdrs):
        c = ws.cell(row=row, column=ci + 1, value=h)
        c.font  = _font(bold=True, size=9, color="FFFFFF")
        c.fill  = _fill(C_HEADER_DARK)
        c.alignment = Alignment(horizontal="center")
    row += 1

    ranked = sorted(
        sensitivity_results.tornado_data.items(),
        key=lambda kv: abs(kv[1][1] - kv[1][0]),
        reverse=True,
    )

    for name, (lo_delta, hi_delta) in ranked:
        lo_irr = base_irr + lo_delta
        hi_irr = base_irr + hi_delta
        swing  = hi_delta - lo_delta
        direction = "Positive" if swing >= 0 else "Negative"

        cells = [name, lo_irr, base_irr, hi_irr, swing, direction]
        fmts  = ["@", FMT_PCT, FMT_PCT, FMT_PCT, FMT_PCT, "@"]
        for ci, (val, fmt) in enumerate(zip(cells, fmts)):
            c = ws.cell(row=row, column=ci + 1, value=val)
            c.number_format = fmt
            c.font = _font(size=9)
            c.fill = _fill(C_SECTION_ALT if row % 2 == 0 else C_WHITE)
            if ci == 4:
                c.fill = _fill(C_POSITIVE if swing > 0 else C_DANGER)
        row += 1

    # Bar chart
    chart_row = row + 2
    if len(ranked) > 0:
        chart = BarChart()
        chart.type = "bar"
        chart.grouping = "clustered"
        chart.title = "Equity IRR Sensitivity — Tornado"
        chart.y_axis.title = "Assumption"
        chart.x_axis.title = "Equity IRR"
        chart.width  = 22
        chart.height = max(8, len(ranked) * 0.9)

        # Data: low IRR and high IRR columns (cols 2 and 4)
        data_start = 6  # header row
        n_rows = len(ranked)
        lo_ref  = Reference(ws, min_col=2, min_row=data_start, max_row=data_start + n_rows - 1)
        hi_ref  = Reference(ws, min_col=4, min_row=data_start, max_row=data_start + n_rows - 1)
        cats    = Reference(ws, min_col=1, min_row=data_start, max_row=data_start + n_rows - 1)

        from openpyxl.chart import Series
        s1 = Series(lo_ref, title="Low scenario IRR")
        s2 = Series(hi_ref, title="High scenario IRR")
        chart.series.append(s1)
        chart.series.append(s2)
        chart.set_categories(cats)
        ws.add_chart(chart, f"A{chart_row}")

    ws.column_dimensions["A"].width = 28
    for col in ["B", "C", "D", "E", "F"]:
        ws.column_dimensions[col].width = 18


def _sheet_monte_carlo(wb, mc_results) -> None:
    if mc_results is None:
        return

    ws = wb.create_sheet("Monte Carlo")
    ws.sheet_view.showGridLines = False

    ws.merge_cells("A1:F1")
    ws["A1"].value = f"MONTE CARLO SIMULATION  —  {mc_results.n_iterations} ITERATIONS"
    ws["A1"].font  = _font(bold=True, size=13, color="FFFFFF")
    ws["A1"].fill  = _fill(C_HEADER_DARK)
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 28

    row = 3

    # Percentile table
    ws.merge_cells(f"A{row}:F{row}")
    ws.cell(row=row, column=1, value="KPI PERCENTILE TABLE").font = _font(bold=True, size=10, color="FFFFFF")
    ws.cell(row=row, column=1).fill = _fill(C_HEADER_MID)
    row += 1

    hdrs = ["KPI", "P10", "P25", "P50  (Median)", "P75", "P90"]
    for ci, h in enumerate(hdrs):
        c = ws.cell(row=row, column=ci + 1, value=h)
        c.font  = _font(bold=True, size=9, color="FFFFFF")
        c.fill  = _fill(C_HEADER_DARK)
        c.alignment = Alignment(horizontal="center")
    row += 1

    kpi_rows = [
        ("Equity IRR",       "equity_irr",  FMT_PCT),
        ("Min DSCR",         "min_dscr",    FMT_RATIO),
        ("LLCR",             "llcr",        FMT_RATIO),
        ("NPV Equity (L)",   "npv_equity",  FMT_LAKHS),
    ]

    dist_map = {
        "equity_irr": mc_results.equity_irr_dist,
        "min_dscr":   mc_results.min_dscr_dist,
        "llcr":       mc_results.llcr_dist,
        "npv_equity": mc_results.npv_dist,
    }

    for label, key, fmt in kpi_rows:
        arr   = dist_map[key]
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            continue
        pcts = [np.percentile(valid, q) for q in [10, 25, 50, 75, 90]]
        ws.cell(row=row, column=1, value=label).font = _font(bold=True, size=9)
        ws.cell(row=row, column=1).fill = _fill(C_LABEL_COL)
        for ci, pv in enumerate(pcts):
            c = ws.cell(row=row, column=ci + 2, value=float(pv))
            c.number_format = fmt
            c.font = _font(size=9)
            c.fill = _fill(C_SECTION_ALT if row % 2 == 0 else C_WHITE)
        row += 1

    row += 1

    # Risk metrics
    ws.merge_cells(f"A{row}:F{row}")
    ws.cell(row=row, column=1, value="RISK METRICS").font = _font(bold=True, size=10, color="FFFFFF")
    ws.cell(row=row, column=1).fill = _fill(C_HEADER_MID)
    row += 1

    risk_items = [
        ("Prob(DSCR < 1.0x)",         mc_results.prob_dscr_below_1,    FMT_PCT1),
        ("Prob(Equity IRR < Hurdle)",  mc_results.prob_irr_below_hurdle, FMT_PCT1),
        ("Iterations Run",             mc_results.n_iterations,          FMT_INT),
    ]
    for label, val, fmt in risk_items:
        ws.cell(row=row, column=1, value=label).font = _font(size=10)
        ws.cell(row=row, column=1).fill = _fill(C_LABEL_COL)
        c = ws.cell(row=row, column=2, value=val)
        c.number_format = fmt
        c.font = _font(bold=True, size=10)
        if isinstance(val, float) and not math.isnan(val) and "Prob" in label:
            c.fill = _fill(C_DANGER if val > 0.20 else (C_WARNING if val > 0.05 else C_POSITIVE))
        row += 1

    row += 1

    # Distribution stats
    ws.merge_cells(f"A{row}:F{row}")
    ws.cell(row=row, column=1, value="DISTRIBUTION STATISTICS").font = _font(bold=True, size=10, color="FFFFFF")
    ws.cell(row=row, column=1).fill = _fill(C_HEADER_MID)
    row += 1

    stat_hdrs = ["KPI", "Min", "Mean", "Max", "Std Dev", "Count"]
    for ci, h in enumerate(stat_hdrs):
        c = ws.cell(row=row, column=ci + 1, value=h)
        c.font  = _font(bold=True, size=9, color="FFFFFF")
        c.fill  = _fill(C_HEADER_DARK)
        c.alignment = Alignment(horizontal="center")
    row += 1

    for label, key, fmt in kpi_rows:
        arr   = dist_map[key]
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            continue
        stats = [float(np.min(valid)), float(np.mean(valid)),
                 float(np.max(valid)), float(np.std(valid)), len(valid)]
        ws.cell(row=row, column=1, value=label).font = _font(size=9)
        ws.cell(row=row, column=1).fill = _fill(C_LABEL_COL)
        for ci, sv in enumerate(stats):
            c = ws.cell(row=row, column=ci + 2, value=sv)
            c.number_format = fmt if ci < 4 else FMT_INT
            c.font = _font(size=9)
            c.fill = _fill(C_SECTION_ALT if row % 2 == 0 else C_WHITE)
        row += 1

    for col in ["A", "B", "C", "D", "E", "F"]:
        ws.column_dimensions[col].width = 20


def _sheet_assumptions(wb, results, compiled, sheet_prefix: str = "") -> None:
    ws = wb.create_sheet(f"{sheet_prefix}Assumptions")
    ws.sheet_view.showGridLines = False

    ws.merge_cells("A1:D1")
    ws["A1"].value = "EFFECTIVE ASSUMPTIONS  —  BASE CASE RUN"
    ws["A1"].font  = _font(bold=True, size=13, color="FFFFFF")
    ws["A1"].fill  = _fill(C_HEADER_DARK)
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 28

    row = 3
    hdrs = ["Assumption", "Value", "Unit", "Schema Default"]
    for ci, h in enumerate(hdrs):
        c = ws.cell(row=row, column=ci + 1, value=h)
        c.font  = _font(bold=True, size=9, color="FFFFFF")
        c.fill  = _fill(C_HEADER_DARK)
    row += 1

    schema_defaults = {
        a.name: a.value
        for a in compiled.model_def.assumption_schema.assumptions
    }

    unit_map = {
        a.name: a.unit
        for a in compiled.model_def.assumption_schema.assumptions
    }

    for name, value in sorted(results.assumptions_used.items()):
        default = schema_defaults.get(name, "—")
        unit    = unit_map.get(name, "")

        # Scalar-safe display value
        if isinstance(value, np.ndarray):
            display_value = str(list(np.round(value, 4)))
        elif isinstance(value, (list, tuple)):
            display_value = str([round(float(x), 4) if isinstance(x, (int, float)) else x for x in value])
        else:
            display_value = value

        if isinstance(default, np.ndarray):
            display_default = str(list(np.round(default, 4)))
        elif isinstance(default, (list, tuple)):
            display_default = str(default)
        else:
            display_default = default

        is_override = (default is not None and str(display_value) != str(display_default))

        ws.cell(row=row, column=1, value=name).font = _font(size=9, bold=is_override)
        ws.cell(row=row, column=1).fill = _fill(C_LABEL_COL)

        v_cell = ws.cell(row=row, column=2, value=display_value)
        v_cell.font = _font(size=9, bold=is_override)
        if is_override:
            v_cell.fill = _fill(C_WARNING)

        ws.cell(row=row, column=3, value=str(unit)).font = _font(size=9, italic=True, color="666666")
        ws.cell(row=row, column=4, value=display_default).font = _font(size=9, color="999999")

        row += 1

    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 18
    ws.column_dimensions["D"].width = 18


def _sheet_audit_trail(wb, results, compiled, sheet_prefix: str = "") -> None:
    ws = wb.create_sheet(f"{sheet_prefix}Audit Trail")
    ws.sheet_view.showGridLines = False

    ws.merge_cells("A1:D1")
    ws["A1"].value = "AUDIT TRAIL  —  ALL INTERMEDIATE VARIABLES"
    ws["A1"].font  = _font(bold=True, size=11, color="FFFFFF")
    ws["A1"].fill  = _fill(C_HEADER_DARK)
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 24

    n   = compiled.n_periods
    cod = compiled.cod_period
    ppy = compiled.periods_per_year
    DATA_COL = 3

    # Period header
    ws.cell(row=2, column=1, value="Variable").font = _font(bold=True, size=9)
    ws.cell(row=2, column=1).fill = _fill(C_HEADER_LIGHT)
    ws.cell(row=2, column=2, value="Expression").font = _font(bold=True, size=9)
    ws.cell(row=2, column=2).fill = _fill(C_HEADER_LIGHT)
    _write_period_headers(ws, header_row=2, n_periods=n, cod=cod, ppy=ppy,
                          data_col_start=DATA_COL)
    ws.freeze_panes = ws.cell(row=3, column=DATA_COL)

    row = 3
    for entry in results.audit_trail:
        ws.cell(row=row, column=1, value=entry.variable).font = _font(size=8)
        ws.cell(row=row, column=1).fill = _fill(C_LABEL_COL)
        ws.cell(row=row, column=2, value=entry.expression).font = _font(size=7, italic=True, color="555555")

        for t_idx, v in enumerate(entry.values):
            fv = float(v)
            c  = ws.cell(row=row, column=DATA_COL + t_idx,
                         value=None if math.isnan(fv) else fv)
            c.number_format = FMT_LAKHS
            c.font = _font(size=8)
            c.fill = _fill(C_SECTION_ALT if row % 2 == 0 else C_WHITE)
        row += 1

    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["B"].width = 50
    for t_idx in range(n):
        ws.column_dimensions[get_column_letter(DATA_COL + t_idx)].width = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_to_excel(
    results,
    compiled,
    path: str | Path = "solar_ipp_output.xlsx",
    sensitivity_results=None,
    mc_results=None,
    scenario_results: Optional[Dict[str, Any]] = None,
    include_audit_trail: bool = False,
) -> Path:
    """
    Export a ModelResults object to a formatted Excel workbook.

    Parameters
    ----------
    results             : ModelResults from executor.run()
    compiled            : CompiledModel from executor.compile()
    path                : Output file path (default: solar_ipp_output.xlsx)
    sensitivity_results : Optional SensitivityResults for tornado sheet
    mc_results          : Optional MonteCarloResults for Monte Carlo sheet
    scenario_results    : Optional dict {scenario_name → ModelResults}
    include_audit_trail : If True, adds a raw Audit Trail sheet (large file)

    Returns
    -------
    Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    n_periods = compiled.n_periods

    _sheet_cover(wb, results, compiled, scenario_results)
    _sheet_project_model(wb, results, compiled)

    if sensitivity_results is not None:
        _sheet_sensitivity(wb, sensitivity_results)

    if mc_results is not None:
        _sheet_monte_carlo(wb, mc_results)

    _sheet_assumptions(wb, results, compiled)

    if include_audit_trail:
        _sheet_audit_trail(wb, results, compiled)

    wb.save(path)
    return path


def export_portfolio_to_excel(
    asset_results,
    path: str | Path = "portfolio_output.xlsx",
    include_audit_trail: bool = True,
) -> Path:
    """
    Export a list of AssetResult objects to a single Excel workbook.

    Sheet layout
    ------------
    Portfolio Summary  — KPIs for all assets side by side
    [SPV-1] Cover          — project summary for asset 1
    [SPV-1] Project Model  — combined Income Statement / Cash Flow / Debt Schedule
    [SPV-1] Assumptions
    [SPV-1] Audit Trail    — all intermediate variables (if include_audit_trail=True)
    [SPV-2] Cover          — project summary for asset 2
    ... (repeated for each asset)

    Parameters
    ----------
    asset_results : List[AssetResult] from PortfolioRunner.run()
    path          : Output file path.
    include_audit_trail : If True, adds an Audit Trail sheet per asset (default: True)

    Returns
    -------
    Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    wb.remove(wb.active)

    # ---- Portfolio summary sheet ----
    _sheet_portfolio_summary(wb, asset_results)

    # ---- Per-asset sheets ----
    for ar in asset_results:
        # Excel sheet names cannot contain [ ] * ? : / \  — use parentheses instead
        safe_spv = ar.spec.spv_name.replace("[", "(").replace("]", ")")
        prefix = f"({safe_spv}) "
        _sheet_cover(wb, ar.model_results, ar.compiled, None, sheet_prefix=prefix)
        _sheet_project_model(wb, ar.model_results, ar.compiled, sheet_prefix=prefix)
        _sheet_assumptions(wb, ar.model_results, ar.compiled, sheet_prefix=prefix)
        if include_audit_trail:
            _sheet_audit_trail(wb, ar.model_results, ar.compiled, sheet_prefix=prefix)

    wb.save(path)
    return path


def _sheet_portfolio_summary(wb: Workbook, asset_results) -> None:
    """
    Write a side-by-side KPI summary for all assets in the portfolio.
    One column per asset.
    """
    ws = wb.create_sheet("Portfolio Summary")
    ws.sheet_view.showGridLines = False

    # Title
    n_assets = len(asset_results)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=2 + n_assets)
    c = ws.cell(row=1, column=1, value="PORTFOLIO SUMMARY  —  ALL ASSETS")
    c.font  = _font(bold=True, size=14, color="FFFFFF")
    c.fill  = _fill(C_HEADER_DARK)
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 32

    # Asset headers (row 2)
    ws.cell(row=2, column=1, value="KPI").font = _font(bold=True, size=10)
    ws.cell(row=2, column=2, value="Unit").font = _font(bold=True, size=10)
    for i, ar in enumerate(asset_results, start=3):
        c = ws.cell(row=2, column=i, value=ar.spec.name)
        c.font      = _font(bold=True, size=10, color="FFFFFF")
        c.fill      = _fill(C_HEADER_MID)
        c.alignment = Alignment(horizontal="center")

    # KPI rows
    kpi_rows = [
        ("Asset Type",         "",     lambda ar: ar.spec.asset_type.title()),
        ("SPV Name",           "",     lambda ar: ar.spec.spv_name),
        ("Capacity",           "MW",   lambda ar: ar.model_results.kpis.capacity_mw if hasattr(ar.model_results.kpis, "capacity_mw") else ar.model_results.assumptions_used.get("capacity_mw", "—")),
        ("Equity IRR",         "%",    lambda ar: f"{ar.model_results.kpis.equity_irr * 100:.2f}%" if ar.model_results.kpis.equity_irr else "n/a"),
        ("Project IRR",        "%",    lambda ar: f"{ar.model_results.kpis.project_irr * 100:.2f}%" if ar.model_results.kpis.project_irr else "n/a"),
        ("Min DSCR",           "×",    lambda ar: f"{ar.model_results.kpis.min_dscr:.3f}×" if ar.model_results.kpis.min_dscr else "n/a"),
        ("Avg DSCR",           "×",    lambda ar: f"{ar.model_results.kpis.avg_dscr:.3f}×" if ar.model_results.kpis.avg_dscr else "n/a"),
        ("LLCR",               "×",    lambda ar: f"{ar.model_results.kpis.llcr:.3f}×" if ar.model_results.kpis.llcr else "n/a"),
        ("Debt Amount",        "₹ Lk", lambda ar: f"{ar.model_results.kpis.total_debt_amount:,.0f}" if ar.model_results.kpis.total_debt_amount else "n/a"),
        ("Total Capex",        "₹ Lk", lambda ar: f"{ar.model_results.assumptions_used.get('capex_per_mw', 0) * ar.model_results.assumptions_used.get('capacity_mw', 0):,.0f}"),
    ]

    for row_i, (label, unit, getter) in enumerate(kpi_rows, start=3):
        ws.cell(row=row_i, column=1, value=label).font = _font(size=10)
        ws.cell(row=row_i, column=2, value=unit).font  = _font(size=10, italic=True)
        if row_i % 2 == 0:
            ws.cell(row=row_i, column=1).fill = _fill(C_SECTION_ALT)
            ws.cell(row=row_i, column=2).fill = _fill(C_SECTION_ALT)
        for col_i, ar in enumerate(asset_results, start=3):
            try:
                val = getter(ar)
            except Exception:
                val = "n/a"
            cell = ws.cell(row=row_i, column=col_i, value=val)
            cell.font      = _font(size=10)
            cell.alignment = Alignment(horizontal="center")
            if row_i % 2 == 0:
                cell.fill = _fill(C_SECTION_ALT)

    # Column widths
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 10
    for i in range(n_assets):
        ws.column_dimensions[get_column_letter(3 + i)].width = 20
