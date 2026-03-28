"""
engine/kpi.py — Key Performance Indicator computation.

All functions are pure NumPy / SciPy — no LLM arithmetic.

Conventions
-----------
  Monetary values  : INR Lakhs
  Rates            : decimal per annum  (0.0975 = 9.75% p.a.)
  Period index     : 0 = financial close, period-end cash-flow convention
                     CF[t] occurs at the END of period t, i.e. (t+1) periods
                     from the origin for discounting purposes
  DSCR             : NEVER computed where total_debt_service == 0
                     (enforced by is_debt_outstanding phase mask)

Public API
----------
  xirr(cashflows, periods_per_year)           → float (annual rate)
  npv_at_rate(cashflows, rate_pa, ppy)        → float (scalar INR Lakhs)
  dscr_series(cfads, debt_svc, mask)          → np.ndarray (NaN where mask=False)
  min_dscr(cfads, debt_svc, mask)             → float
  avg_dscr(cfads, debt_svc, mask)             → float
  llcr(cfads, balance, rate_pa, cod, mat, ppy)→ float
  plcr(cfads, balance, rate_pa, cod, ppy)     → float
  debt_payback_period(balance, cod, ppy)      → float  (years from COD)
  peak_debt_outstanding(balance)              → float  (INR Lakhs)
  compute_all_kpis(...)                       → KPIResult
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from dsl.types import KPIResult

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KPIComputationError(Exception):
    """Raised when a KPI cannot be computed (e.g. IRR does not converge)."""


class DSCRPhaseError(KPIComputationError):
    """Raised if DSCR is requested but the debt-outstanding mask is all-False."""


# ---------------------------------------------------------------------------
# Validation range for base-case KPIs (Karnataka solar IPP)
# Used by compute_all_kpis to emit warnings — not hard errors.
# ---------------------------------------------------------------------------

_KPI_VALIDATION_RANGES = {
    "equity_irr":         (0.10, 0.25),   # 10 – 25%
    "project_irr":        (0.08, 0.20),   # 8 – 20%
    "min_dscr":           (1.00, 3.00),   # should always exceed 1.0×
    "llcr":               (1.00, 5.00),
    "debt_payback_period": (10.0, 20.0),  # years
}


# ---------------------------------------------------------------------------
# 1. XIRR — annualised internal rate of return
# ---------------------------------------------------------------------------


def xirr(
    cashflows: np.ndarray,
    periods_per_year: int = 4,
    bracket: tuple[float, float] = (-0.9999, 20.0),
) -> float:
    """
    Annualised IRR for a uniform-period cash-flow series.

    Algorithm
    ---------
    Solve NPV(r_periodic) = 0 using Brent's method, then annualise:
        annual_irr = (1 + r_periodic)^periods_per_year - 1

    Parameters
    ----------
    cashflows       : array of cash flows, period-end convention.
                      Must contain at least one sign change.
    periods_per_year: 4 = quarterly, 12 = monthly, 1 = annual
    bracket         : (low, high) search bounds for the periodic rate.
                      Default wide enough for any realistic project.

    Returns
    -------
    float : annualised IRR as a decimal (e.g. 0.155 for 15.5%).

    Raises
    ------
    KPIComputationError : if no sign change, or Brent fails to converge.
    """
    cf = np.asarray(cashflows, dtype=np.float64)
    if len(cf) == 0:
        raise KPIComputationError("xirr: cashflows array is empty")

    if not _has_sign_change(cf):
        raise KPIComputationError(
            "xirr: cash flow series has no sign change — IRR is undefined. "
            "Ensure there is at least one outflow (negative) and one inflow (positive)."
        )

    def _npv(r_periodic: float) -> float:
        t = np.arange(1, len(cf) + 1, dtype=np.float64)
        # Suppress expected overflow/underflow at Brent bracket boundaries
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            val = float(np.sum(cf / (1.0 + r_periodic) ** t))
        return val

    # Verify the bracket straddles zero
    lo, hi = bracket
    try:
        f_lo = _npv(lo)
        f_hi = _npv(hi)
    except Exception as exc:
        raise KPIComputationError(f"xirr: NPV evaluation failed: {exc}") from exc

    if f_lo * f_hi > 0:
        # Try to widen the search automatically
        for trial_hi in [50.0, 200.0]:
            if _npv(lo) * _npv(trial_hi) < 0:
                hi = trial_hi
                break
        else:
            raise KPIComputationError(
                f"xirr: NPV does not change sign over [{lo:.4f}, {hi:.4f}]. "
                f"NPV({lo:.4f}) = {f_lo:.2f}, NPV({hi:.4f}) = {f_hi:.2f}. "
                f"Check that equity cash flows include construction-period outflows."
            )

    try:
        r_periodic = brentq(_npv, lo, hi, xtol=1e-12, rtol=1e-10, maxiter=500)
    except ValueError as exc:
        raise KPIComputationError(f"xirr: Brent's method failed: {exc}") from exc

    return float((1.0 + r_periodic) ** periods_per_year - 1.0)


def _has_sign_change(cf: np.ndarray) -> bool:
    """Return True if the cash-flow array contains both positive and negative values."""
    return bool(np.any(cf < 0) and np.any(cf > 0))


# ---------------------------------------------------------------------------
# 2. NPV at a given discount rate
# ---------------------------------------------------------------------------


def npv_at_rate(
    cashflows: np.ndarray,
    rate_pa: float,
    periods_per_year: int = 4,
) -> float:
    """
    Present value of a cash-flow series discounted at rate_pa (per annum).

    CF[t] is discounted by (1 + r_periodic)^(t+1) — period-end convention.
    Returns a scalar in the same units as cashflows (INR Lakhs).
    """
    cf = np.asarray(cashflows, dtype=np.float64)
    r_periodic = (1.0 + rate_pa) ** (1.0 / periods_per_year) - 1.0
    t = np.arange(1, len(cf) + 1, dtype=np.float64)
    if r_periodic == 0.0:
        return float(np.sum(cf))
    return float(np.sum(cf / (1.0 + r_periodic) ** t))


# ---------------------------------------------------------------------------
# 3. DSCR — Debt Service Coverage Ratio
# ---------------------------------------------------------------------------


def dscr_series(
    cfads: np.ndarray,
    total_debt_service: np.ndarray,
    is_debt_outstanding: np.ndarray,
) -> np.ndarray:
    """
    Compute the per-period DSCR array.

    DSCR[t] = CFADS[t] / TotalDebtService[t]
    Only computed where is_debt_outstanding[t] == True AND debt_service[t] > 0.
    All other periods receive np.nan (never 0 or inf — callers must filter).

    Parameters
    ----------
    cfads               : Cash Flow Available for Debt Service
    total_debt_service  : Interest + Principal (must be positive in debt periods)
    is_debt_outstanding : Boolean mask; True in periods debt exists

    Returns
    -------
    np.ndarray of same length as inputs.
    """
    cfads = np.asarray(cfads, dtype=np.float64)
    tds = np.asarray(total_debt_service, dtype=np.float64)
    mask = np.asarray(is_debt_outstanding, dtype=bool)

    n = len(cfads)
    result = np.full(n, np.nan)

    # Compute only where mask is True AND debt_service > 0
    active = mask & (tds > 1e-10)  # 1e-10 guard against floating-point near-zero

    if not np.any(active):
        raise DSCRPhaseError(
            "dscr_series: no periods have both is_debt_outstanding=True and "
            "total_debt_service > 0. Cannot compute DSCR."
        )

    result[active] = cfads[active] / tds[active]
    return result


def min_dscr(
    cfads: np.ndarray,
    total_debt_service: np.ndarray,
    is_debt_outstanding: np.ndarray,
) -> float:
    """
    Minimum DSCR across all debt-service periods.
    Raises DSCRPhaseError if no debt periods exist.
    """
    series = dscr_series(cfads, total_debt_service, is_debt_outstanding)
    valid = series[~np.isnan(series)]
    if len(valid) == 0:
        raise DSCRPhaseError("min_dscr: no valid DSCR periods after masking")
    return float(np.min(valid))


def avg_dscr(
    cfads: np.ndarray,
    total_debt_service: np.ndarray,
    is_debt_outstanding: np.ndarray,
) -> float:
    """
    Average DSCR across all debt-service periods (arithmetic mean).
    """
    series = dscr_series(cfads, total_debt_service, is_debt_outstanding)
    valid = series[~np.isnan(series)]
    if len(valid) == 0:
        raise DSCRPhaseError("avg_dscr: no valid DSCR periods after masking")
    return float(np.mean(valid))


# ---------------------------------------------------------------------------
# 4. LLCR — Loan Life Coverage Ratio
# ---------------------------------------------------------------------------


def llcr(
    cfads: np.ndarray,
    outstanding_balance: np.ndarray,
    interest_rate_pa: float,
    cod_period: int,
    debt_maturity_period: int,
    periods_per_year: int = 4,
) -> float:
    """
    Loan Life Coverage Ratio computed at COD.

    LLCR = PV(CFADS over loan life) / Outstanding Debt Balance at COD

    The "loan life" runs from COD (inclusive) to debt_maturity_period (inclusive).
    Discounts at the loan interest rate (per annum).

    Returns
    -------
    float : LLCR (e.g. 1.35 means PV of CFADS is 1.35× the debt at COD).

    Raises
    ------
    KPIComputationError : if outstanding_balance at COD is zero or negative.
    """
    cfads = np.asarray(cfads, dtype=np.float64)
    balance = np.asarray(outstanding_balance, dtype=np.float64)

    debt_at_cod = float(balance[cod_period])
    if debt_at_cod <= 0:
        raise KPIComputationError(
            f"llcr: outstanding_balance at COD (period {cod_period}) is "
            f"{debt_at_cod:.2f} — cannot compute LLCR with zero debt."
        )

    # CFADS slice covering the loan life
    loan_life_cfads = cfads[cod_period : debt_maturity_period + 1]

    # Per-period discount rate
    r_periodic = (1.0 + interest_rate_pa) ** (1.0 / periods_per_year) - 1.0
    t = np.arange(1, len(loan_life_cfads) + 1, dtype=np.float64)
    pv_cfads = float(np.sum(loan_life_cfads / (1.0 + r_periodic) ** t))

    return pv_cfads / debt_at_cod


# ---------------------------------------------------------------------------
# 5. PLCR — Project Life Coverage Ratio
# ---------------------------------------------------------------------------


def plcr(
    cfads: np.ndarray,
    outstanding_balance: np.ndarray,
    interest_rate_pa: float,
    cod_period: int,
    periods_per_year: int = 4,
) -> float:
    """
    Project Life Coverage Ratio computed at COD.

    PLCR = PV(CFADS over full project life from COD) / Outstanding Debt at COD

    Unlike LLCR, the CFADS window runs to the end of the model horizon,
    not just to debt maturity.
    """
    cfads = np.asarray(cfads, dtype=np.float64)
    balance = np.asarray(outstanding_balance, dtype=np.float64)

    debt_at_cod = float(balance[cod_period])
    if debt_at_cod <= 0:
        raise KPIComputationError(
            f"plcr: outstanding_balance at COD (period {cod_period}) is "
            f"{debt_at_cod:.2f} — cannot compute PLCR with zero debt."
        )

    # Full project life from COD
    project_life_cfads = cfads[cod_period:]

    r_periodic = (1.0 + interest_rate_pa) ** (1.0 / periods_per_year) - 1.0
    t = np.arange(1, len(project_life_cfads) + 1, dtype=np.float64)
    pv_cfads = float(np.sum(project_life_cfads / (1.0 + r_periodic) ** t))

    return pv_cfads / debt_at_cod


# ---------------------------------------------------------------------------
# 6. NPV of equity cash flows
# ---------------------------------------------------------------------------


def npv_equity(
    equity_cashflows: np.ndarray,
    discount_rate_pa: float,
    periods_per_year: int = 4,
) -> float:
    """
    NPV of equity cash flows at the target equity discount rate.

    Parameters
    ----------
    equity_cashflows : negative during construction, positive during operations
    discount_rate_pa : equity IRR hurdle rate (per annum, decimal)
    periods_per_year : 4 = quarterly

    Returns
    -------
    float : NPV in INR Lakhs. Positive = project exceeds hurdle rate.
    """
    return npv_at_rate(equity_cashflows, discount_rate_pa, periods_per_year)


# ---------------------------------------------------------------------------
# 7. Debt payback period
# ---------------------------------------------------------------------------


def debt_payback_period(
    outstanding_balance: np.ndarray,
    cod_period: int,
    periods_per_year: int = 4,
    zero_threshold: float = 1.0,
) -> float:
    """
    Number of years from COD until the outstanding debt balance reaches zero.

    Uses the last period where the balance exceeds zero_threshold (default 1 Lakh,
    i.e. effectively zero in context of typical project sizes).

    Returns
    -------
    float : payback period in years from COD.
            Returns 0.0 if debt is already repaid at COD.

    Notes
    -----
    zero_threshold: INR Lakhs. Rounding errors can leave tiny residuals;
    this threshold ensures we don't report spuriously long payback periods.
    """
    balance = np.asarray(outstanding_balance, dtype=np.float64)
    ops_balance = balance[cod_period:]

    if np.all(ops_balance <= zero_threshold):
        return 0.0  # Already repaid (or no debt)

    # Last operational period index where balance is still meaningful
    nonzero_indices = np.where(ops_balance > zero_threshold)[0]
    last_nonzero = int(nonzero_indices[-1])

    # last_nonzero is the 0-based index within ops_balance.
    # The balance is still outstanding at the END of that period,
    # so the payback occurs sometime in the NEXT period.
    # Payback period = (last_nonzero + 1) periods from COD → convert to years.
    return float((last_nonzero + 1) / periods_per_year)


# ---------------------------------------------------------------------------
# 8. Peak debt outstanding
# ---------------------------------------------------------------------------


def peak_debt_outstanding(outstanding_balance: np.ndarray) -> float:
    """
    Maximum outstanding debt balance over the full model horizon (INR Lakhs).
    """
    return float(np.max(np.asarray(outstanding_balance, dtype=np.float64)))


# ---------------------------------------------------------------------------
# 9. Assembled KPI computation
# ---------------------------------------------------------------------------


def compute_all_kpis(
    equity_cashflows: np.ndarray,
    project_cashflows: np.ndarray,
    cfads: np.ndarray,
    total_debt_service: np.ndarray,
    outstanding_balance: np.ndarray,
    is_debt_outstanding: np.ndarray,
    interest_rate_pa: float,
    equity_irr_target: float,
    cod_period: int,
    debt_maturity_period: int,
    periods_per_year: int = 4,
) -> tuple[KPIResult, list[str]]:
    """
    Compute all KPIs and return (KPIResult, warnings).

    Each KPI is computed independently — a failure in one does not block others.
    Failures are captured as warning strings (not exceptions), so the caller
    always receives a (partial) KPIResult.

    Parameters
    ----------
    equity_cashflows    : (-equity_invested during construction) + distributions
    project_cashflows   : (-total_project_cost) + CFADS
    cfads               : Cash Flow Available for Debt Service (per period)
    total_debt_service  : Interest + Principal (per period)
    outstanding_balance : Debt balance at end of each period
    is_debt_outstanding : Boolean mask
    interest_rate_pa    : Annual loan interest rate (decimal)
    equity_irr_target   : Equity hurdle rate for NPV computation (decimal p.a.)
    cod_period          : Period index of COD
    debt_maturity_period: Period index of final debt repayment
    periods_per_year    : 4 = quarterly

    Returns
    -------
    (KPIResult, list_of_warning_strings)
    """
    result = KPIResult()
    computation_warnings: list[str] = []

    def _try(name: str, fn):
        """Execute fn(), store result, capture any exception as a warning."""
        try:
            return fn()
        except Exception as exc:
            computation_warnings.append(f"KPI '{name}' could not be computed: {exc}")
            return None

    # --- IRR metrics ---
    result.equity_irr = _try(
        "equity_irr",
        lambda: xirr(equity_cashflows, periods_per_year),
    )
    result.project_irr = _try(
        "project_irr",
        lambda: xirr(project_cashflows, periods_per_year),
    )

    # --- DSCR metrics ---
    result.min_dscr = _try(
        "min_dscr",
        lambda: min_dscr(cfads, total_debt_service, is_debt_outstanding),
    )
    result.avg_dscr = _try(
        "avg_dscr",
        lambda: avg_dscr(cfads, total_debt_service, is_debt_outstanding),
    )

    # --- Coverage ratios ---
    result.llcr = _try(
        "llcr",
        lambda: llcr(
            cfads, outstanding_balance, interest_rate_pa,
            cod_period, debt_maturity_period, periods_per_year,
        ),
    )
    result.plcr = _try(
        "plcr",
        lambda: plcr(
            cfads, outstanding_balance, interest_rate_pa,
            cod_period, periods_per_year,
        ),
    )

    # --- NPV ---
    result.npv_equity = _try(
        "npv_equity",
        lambda: npv_equity(equity_cashflows, equity_irr_target, periods_per_year),
    )

    # --- Debt metrics ---
    result.debt_payback_period = _try(
        "debt_payback_period",
        lambda: debt_payback_period(outstanding_balance, cod_period, periods_per_year),
    )
    result.peak_debt_outstanding = _try(
        "peak_debt_outstanding",
        lambda: peak_debt_outstanding(outstanding_balance),
    )

    # --- Range validation warnings (advisory, not errors) ---
    _emit_range_warnings(result, computation_warnings)

    return result, computation_warnings


def _emit_range_warnings(result: KPIResult, warnings_out: list[str]) -> None:
    """
    Emit advisory warnings if KPI values fall outside expected ranges
    for a Karnataka solar IPP project.
    """
    checks = [
        ("equity_irr", result.equity_irr, _KPI_VALIDATION_RANGES["equity_irr"]),
        ("project_irr", result.project_irr, _KPI_VALIDATION_RANGES["project_irr"]),
        ("min_dscr", result.min_dscr, _KPI_VALIDATION_RANGES["min_dscr"]),
        ("llcr", result.llcr, _KPI_VALIDATION_RANGES["llcr"]),
        (
            "debt_payback_period",
            result.debt_payback_period,
            _KPI_VALIDATION_RANGES["debt_payback_period"],
        ),
    ]
    for name, value, (lo, hi) in checks:
        if value is None:
            continue
        if value < lo:
            warnings_out.append(
                f"KPI RANGE WARNING: {name} = {value:.4f} is below expected minimum "
                f"{lo:.4f} for Karnataka solar IPP. Verify assumptions."
            )
        elif value > hi:
            warnings_out.append(
                f"KPI RANGE WARNING: {name} = {value:.4f} is above expected maximum "
                f"{hi:.4f} for Karnataka solar IPP. Verify assumptions."
            )

    # Special check: min_dscr < 1.0 is a bankability failure
    if result.min_dscr is not None and result.min_dscr < 1.0:
        warnings_out.append(
            f"BANKABILITY FAILURE: min_dscr = {result.min_dscr:.4f}x < 1.0x. "
            f"Project cannot service its debt in at least one period. "
            f"Consider reducing debt quantum, extending tenor, or improving CFADS."
        )
