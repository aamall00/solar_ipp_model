"""
engine/solvers.py — Iterative solvers for circular model dependencies.

Three solver types, matching the solve_loop DSL declaration:

  goal_seek    → scipy.optimize.brentq on a scalar objective function.
                 Used for: debt sizing with capitalised IDC (find debt_amount
                 such that debt_amount = debt_pct × (capex + cumulative_idc(debt_amount))).

  sculpting    → Iterative relaxation with configurable damping.
                 Used for: DSCR-sculpted debt service (find principal_repayment[t]
                 such that CFADS[t] / (interest[t] + principal[t]) = dscr_target ∀t).

  fixed_point  → Simple fixed-point iteration.
                 Used for: DSRA top-up interactions where the reserve balance
                 feeds back into available cash.

All solvers raise ModelConvergenceError on failure with full diagnostics.
None of them compute arithmetic themselves — they call user-supplied objectives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Error type (spec requirement)
# ---------------------------------------------------------------------------


class ModelConvergenceError(Exception):
    """
    Raised when a solve loop fails to converge within max_iterations.

    Attributes
    ----------
    loop_id       : The solve_loop declaration that failed.
    last_value    : The free variable value at the last iteration.
    residual      : The remaining residual (actual - target) at termination.
    suggestion    : Human-readable hint about likely conflicting assumptions.
    """

    def __init__(
        self,
        loop_id: str,
        last_value: float,
        residual: float,
        suggestion: str = "",
    ) -> None:
        self.loop_id = loop_id
        self.last_value = last_value
        self.residual = residual
        self.suggestion = suggestion
        detail = (
            f"Solve loop '{loop_id}' did not converge.\n"
            f"  Last free-variable value : {last_value:.8g}\n"
            f"  Residual at termination  : {residual:.6e}\n"
        )
        if suggestion:
            detail += f"  Suggestion               : {suggestion}\n"
        super().__init__(detail)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class GoalSeekResult:
    """Result of a goal_seek solve."""
    loop_id: str
    converged: bool
    iterations: int
    solution: float          # value of the free variable at convergence
    final_residual: float    # |f(solution) - target_value|
    convergence_history: List[float] = field(default_factory=list)


@dataclass
class SculptingResult:
    """Result of a sculpting solve for DSCR-based debt service."""
    loop_id: str
    converged: bool
    iterations: int
    final_residual: float    # max |DSCR[t] - dscr_target| across repayment periods
    # Time-series arrays (length = model n_periods)
    principal_repayment: np.ndarray = field(default_factory=lambda: np.array([]))
    interest_payment: np.ndarray    = field(default_factory=lambda: np.array([]))
    outstanding_balance: np.ndarray = field(default_factory=lambda: np.array([]))
    total_debt_service: np.ndarray  = field(default_factory=lambda: np.array([]))
    dscr_series: np.ndarray         = field(default_factory=lambda: np.array([]))


@dataclass
class FixedPointResult:
    """Result of a fixed-point iteration."""
    loop_id: str
    converged: bool
    iterations: int
    solution: float
    final_residual: float
    convergence_history: List[float] = field(default_factory=list)


@dataclass
class ArrayFixedPointResult:
    """Result of an array-valued fixed-point iteration."""
    loop_id: str
    converged: bool
    iterations: int
    solution: np.ndarray          # converged array (e.g. idc_per_period)
    final_residual: float         # max(|x_new - x_old|) at termination
    convergence_history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Goal-seek solver  (Brent's method)
# ---------------------------------------------------------------------------


def goal_seek_solver(
    objective: Callable[[float], float],
    loop_id: str,
    target_value: float = 0.0,
    x_init: float = 1.0,
    bracket_low: Optional[float] = None,
    bracket_high: Optional[float] = None,
    tolerance: float = 1e-6,
    max_iterations: int = 50,
) -> GoalSeekResult:
    """
    Find x such that objective(x) == target_value using Brent's method.

    The objective should encapsulate the subgraph evaluation:
        objective(x) = compute_target_expression(with free_variable = x)

    Parameters
    ----------
    objective      : f(x) → computed value; solver finds x where f(x) = target_value
    loop_id        : identifier for error messages
    target_value   : desired value of objective (default 0 → root-finding)
    x_init         : initial guess for bracket search
    bracket_low    : lower bound of search bracket (auto-detected if None)
    bracket_high   : upper bound of search bracket (auto-detected if None)
    tolerance      : |f(x) - target_value| < tolerance → converged
    max_iterations : passed to brentq as maxiter

    Returns
    -------
    GoalSeekResult

    Raises
    ------
    ModelConvergenceError : if brentq cannot converge or bracket cannot be found
    """
    history: List[float] = []

    def residual(x: float) -> float:
        val = objective(x)
        history.append(val)
        return val - target_value

    # --- Auto-detect bracket if not provided ---
    lo, hi = _auto_bracket(residual, x_init, bracket_low, bracket_high, loop_id)

    # --- Brent's method ---
    try:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            solution = brentq(
                residual,
                lo,
                hi,
                xtol=tolerance,
                rtol=tolerance * 1e-2,
                maxiter=max_iterations,
                full_output=False,
            )
    except ValueError as exc:
        last_val = history[-1] if history else float("nan")
        raise ModelConvergenceError(
            loop_id=loop_id,
            last_value=last_val,
            residual=abs(residual(last_val)) if not np.isnan(last_val) else float("nan"),
            suggestion=(
                "Brent's method failed. Check that the objective function changes sign "
                "across the bracket, and that the free variable has a feasible range. "
                f"Bracket tried: [{lo:.6g}, {hi:.6g}]. Error: {exc}"
            ),
        ) from exc

    final_residual = abs(residual(solution))

    return GoalSeekResult(
        loop_id=loop_id,
        converged=final_residual <= tolerance,
        iterations=len(history),
        solution=solution,
        final_residual=final_residual,
        convergence_history=history,
    )


def _auto_bracket(
    residual_fn: Callable[[float], float],
    x_init: float,
    bracket_low: Optional[float],
    bracket_high: Optional[float],
    loop_id: str,
) -> Tuple[float, float]:
    """
    Find a bracket [lo, hi] such that residual_fn(lo) and residual_fn(hi)
    have opposite signs.  Uses exponential expansion from x_init if bounds
    are not provided.
    """
    if bracket_low is not None and bracket_high is not None:
        f_lo = residual_fn(bracket_low)
        f_hi = residual_fn(bracket_high)
        if f_lo * f_hi < 0:
            return bracket_low, bracket_high
        # Provided bracket doesn't straddle zero — fall through to auto-detect

    # Expand outward from x_init
    lo, hi = x_init * 0.01 if x_init > 0 else -1e6, x_init * 100 if x_init > 0 else 1e6

    # Standard expansion candidates covering common project finance ranges
    candidates = [
        (1e-6, 1e8),
        (-1e8, -1e-6),
        (x_init * 0.001, x_init * 1000) if x_init > 0 else (-1e6, 1e6),
        (-1e6, 1e6),
    ]
    for lo_c, hi_c in candidates:
        try:
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                f_lo = residual_fn(lo_c)
                f_hi = residual_fn(hi_c)
            if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi < 0:
                return lo_c, hi_c
        except Exception:
            continue

    raise ModelConvergenceError(
        loop_id=loop_id,
        last_value=x_init,
        residual=float("nan"),
        suggestion=(
            "Cannot find a bracket [lo, hi] where the objective changes sign. "
            "The goal_seek has no solution in the searched range, or the objective "
            "is flat/monotone. Check that the free variable meaningfully affects "
            "the target expression."
        ),
    )


# ---------------------------------------------------------------------------
# 2. Sculpting solver  (DSCR-based debt service sizing)
# ---------------------------------------------------------------------------


def sculpting_solver(
    cfads: np.ndarray,
    total_debt: float,
    interest_rate_per_period: float,
    dscr_target: float,
    cod_period: int,
    debt_maturity_period: int,
    moratorium_periods: int = 0,
    loop_id: str = "sculpting",
    tolerance: float = 1e-6,
    max_iterations: int = 50,
    damping: float = 0.4,
) -> SculptingResult:
    """
    Compute a DSCR-sculpted debt service schedule.

    Algorithm (PV-scaling with iterative refinement)
    -------------------------------------------------
    The standard sum-normalization approach oscillates because changing the
    principal schedule changes interest (via balance), which changes the
    target principal.  This implementation uses PV-scaling instead:

    Each iteration:
      1. Compute DS_target[t] = CFADS[t] / dscr_target  (proportional to CFADS)
      2. PV-scale DS_target so PV(DS_actual) = total_debt:
            scale = total_debt / PV(DS_target)
            DS_actual = DS_target × scale
      3. Walk forward computing balance, interest, and principal:
            balance[0] = total_debt
            interest[t] = balance[t] × r
            principal[t] = max(0, DS_actual[t] − interest[t])
            balance[t+1] = balance[t] − principal[t]
      4. Residual = remaining balance after last repayment period
         (should = 0 when the schedule exactly repays total_debt)
      5. If |residual| > tolerance, dampen DS_target and repeat.

    The DSCR series in the result reflects the actual schedule (which for
    flat CFADS equals the level annuity, and for escalating CFADS is
    proportional to the CFADS profile).

    Parameters
    ----------
    cfads                   : full model CFADS time-series (n_periods,)
    total_debt              : total principal to be repaid (INR Lakhs)
    interest_rate_per_period: r = annual_rate / periods_per_year
    dscr_target             : target DSCR (shapes the DS profile; actual DSCR
                              = CFADS / DS_actual which may differ from target)
    cod_period              : period index of COD
    debt_maturity_period    : period index of final repayment (inclusive)
    moratorium_periods      : number of periods after COD with interest-only
    tolerance               : |remaining_balance / total_debt| < tol → converged
    max_iterations          : hard cap
    damping                 : not used directly (kept for API compatibility)
    """
    cfads = np.asarray(cfads, dtype=np.float64)
    n_total = len(cfads)

    if total_debt <= 0:
        raise ValueError(f"sculpting_solver: total_debt must be > 0, got {total_debt}")
    if dscr_target <= 0:
        raise ValueError(f"sculpting_solver: dscr_target must be > 0, got {dscr_target}")
    if not (0 < damping <= 1):
        raise ValueError(f"sculpting_solver: damping must be in (0, 1], got {damping}")

    r = interest_rate_per_period
    repay_start = cod_period + moratorium_periods
    repay_end = debt_maturity_period + 1        # exclusive
    n_repay = repay_end - repay_start
    mora_start = cod_period
    mora_end = repay_start

    if n_repay <= 0:
        raise ModelConvergenceError(
            loop_id=loop_id,
            last_value=0.0,
            residual=float("nan"),
            suggestion=(
                f"No repayment periods: repay_start={repay_start}, "
                f"repay_end={repay_end}. Check moratorium_periods and debt_maturity_period."
            ),
        )

    cfads_repay = cfads[repay_start:repay_end]

    # PV discount factors for repayment periods (period-end, relative to repay_start)
    t_vec = np.arange(1, n_repay + 1, dtype=np.float64)
    pv_factors = 1.0 / (1.0 + r) ** t_vec

    # Initial DS profile: proportional to CFADS, shaped by dscr_target
    raw_ds = cfads_repay / dscr_target   # unconstrained shape

    final_residual = float("nan")
    converged = False
    principal = np.zeros(n_total)
    interest_arr = np.zeros(n_total)
    balance_arr = np.zeros(n_total)

    for iteration in range(max_iterations):
        # PV-scale DS so that PV(DS_actual) = total_debt
        pv_raw = float(np.dot(raw_ds, pv_factors))
        if pv_raw <= 1e-12:
            break   # degenerate — CFADS is essentially zero
        scale = total_debt / pv_raw
        actual_ds = raw_ds * scale

        # Walk forward: compute balance, interest, principal
        principal[:] = 0.0
        interest_arr[:] = 0.0
        balance_arr[:] = 0.0

        bal = total_debt

        # Moratorium: interest-only
        for t in range(mora_start, mora_end):
            balance_arr[t] = bal
            interest_arr[t] = bal * r

        # Repayment periods
        for i in range(n_repay):
            t = repay_start + i
            balance_arr[t] = bal
            interest_arr[t] = bal * r
            p = max(0.0, actual_ds[i] - interest_arr[t])
            principal[t] = p
            bal = max(0.0, bal - p)

        # Convergence: remaining balance should be zero
        final_residual = abs(bal) / total_debt

        if final_residual < tolerance:
            converged = True
            break

        # Adjust raw_ds to push more repayment into later periods
        # (the remaining balance arises when flooring clips early principal)
        # Damped proportional scaling of raw_ds
        raw_ds = raw_ds * (1.0 + damping * (bal / total_debt))

    if not converged:
        raise ModelConvergenceError(
            loop_id=loop_id,
            last_value=float(np.mean(principal[repay_start:repay_end])),
            residual=final_residual,
            suggestion=(
                f"Sculpting did not converge after {max_iterations} iterations "
                f"(remaining_balance / total_debt = {final_residual:.4e} > tol {tolerance:.2e}). "
                f"Check that CFADS[t] > interest[t] throughout repayment periods. "
                f"Current: dscr_target={dscr_target}, r_per_period={r:.6f}."
            ),
        )

    # Build final DS and DSCR series
    total_ds = interest_arr.copy()
    total_ds[repay_start:repay_end] += principal[repay_start:repay_end]

    dscr_out = np.full(n_total, np.nan)
    for i in range(n_repay):
        t = repay_start + i
        if total_ds[t] > 1e-10:
            dscr_out[t] = cfads[t] / total_ds[t]

    valid_dscr = dscr_out[~np.isnan(dscr_out)]
    dscr_residual = (
        float(np.max(np.abs(valid_dscr - dscr_target)))
        if len(valid_dscr) > 0
        else 0.0
    )

    return SculptingResult(
        loop_id=loop_id,
        converged=True,
        iterations=iteration + 1,
        final_residual=dscr_residual,
        principal_repayment=principal,
        interest_payment=interest_arr,
        outstanding_balance=balance_arr,
        total_debt_service=total_ds,
        dscr_series=dscr_out,
    )


def _compute_balance(
    principal: np.ndarray,
    cod_period: int,
    total_debt: float,
    n_total: int,
) -> np.ndarray:
    """
    Compute the opening outstanding balance at each period given a principal schedule.

    balance[cod_period]     = total_debt  (full draw at COD)
    balance[t]              = balance[t-1] - principal[t-1]   for t > cod_period
    balance[t < cod_period] = 0  (pre-COD, handled separately by debt_drawdown)
    """
    balance = np.zeros(n_total)
    balance[cod_period] = total_debt
    for t in range(cod_period + 1, n_total):
        balance[t] = max(0.0, balance[t - 1] - principal[t - 1])
    return balance


# ---------------------------------------------------------------------------
# 3. Fixed-point solver
# ---------------------------------------------------------------------------


def fixed_point_solver(
    f: Callable[[float], float],
    x_init: float,
    loop_id: str,
    tolerance: float = 1e-6,
    max_iterations: int = 25,
) -> FixedPointResult:
    """
    Simple fixed-point iteration: x_{n+1} = f(x_n) until |x_{n+1} - x_n| < tolerance.

    Suitable for mild circular dependencies where the iteration map is
    a contraction (|f'(x)| < 1 near the solution). Used for DSRA interactions
    where the reserve balance has a small feedback effect on available cash.

    Parameters
    ----------
    f              : the iteration function; x_new = f(x_current)
    x_init         : starting value
    loop_id        : identifier for error messages
    tolerance      : |x_new - x_old| < tolerance → converged
    max_iterations : hard cap (spec default: 25)

    Returns
    -------
    FixedPointResult

    Raises
    ------
    ModelConvergenceError : if |residual| > tolerance after max_iterations
    """
    history: List[float] = []
    x = float(x_init)

    for iteration in range(max_iterations):
        x_new = float(f(x))
        history.append(x_new)
        residual = abs(x_new - x)

        if residual < tolerance:
            return FixedPointResult(
                loop_id=loop_id,
                converged=True,
                iterations=iteration + 1,
                solution=x_new,
                final_residual=residual,
                convergence_history=history,
            )
        x = x_new

    # Exceeded max_iterations
    final_residual = abs(history[-1] - history[-2]) if len(history) >= 2 else float("nan")
    raise ModelConvergenceError(
        loop_id=loop_id,
        last_value=history[-1] if history else x_init,
        residual=final_residual,
        suggestion=(
            f"Fixed-point iteration did not converge after {max_iterations} iterations "
            f"(|x_new - x_old| = {final_residual:.4e} > tolerance {tolerance:.4e}). "
            f"The iteration map may not be a contraction near the solution. "
            f"Consider: (1) Check for DSRA over-specification, "
            f"(2) Increase max_iterations, "
            f"(3) Verify that circular dependency is genuinely mild."
        ),
    )


# ---------------------------------------------------------------------------
# 4. Array fixed-point solver  (time-series circular dependencies)
# ---------------------------------------------------------------------------


def array_fixed_point_solver(
    f: "Callable[[np.ndarray], np.ndarray]",
    x_init: np.ndarray,
    loop_id: str,
    tolerance: float = 1e-6,
    max_iterations: int = 25,
) -> ArrayFixedPointResult:
    """
    Fixed-point iteration for array-valued state: x_{n+1} = f(x_n).

    Converges when max(|x_{n+1} - x_n|) < tolerance.  Suitable for
    within-period circular dependencies where the iteration map is a
    contraction — e.g. IDC ↔ debt drawdown where the contraction ratio
    is debt_pct × r_q ≈ 0.017 (converges in ~3 iterations for solar IPP).

    Parameters
    ----------
    f              : array iteration function; x_new = f(x_current)
    x_init         : starting array (typically zeros)
    loop_id        : identifier for error messages
    tolerance      : max(|x_new - x_old|) < tolerance → converged
    max_iterations : hard cap

    Returns
    -------
    ArrayFixedPointResult

    Raises
    ------
    ModelConvergenceError : if residual > tolerance after max_iterations
    """
    history: List[float] = []
    x = np.asarray(x_init, dtype=np.float64).copy()

    for iteration in range(max_iterations):
        x_new = np.asarray(f(x), dtype=np.float64)
        residual = float(np.max(np.abs(x_new - x)))
        history.append(residual)

        if residual < tolerance:
            return ArrayFixedPointResult(
                loop_id=loop_id,
                converged=True,
                iterations=iteration + 1,
                solution=x_new,
                final_residual=residual,
                convergence_history=history,
            )
        x = x_new

    final_residual = history[-1] if history else float("nan")
    raise ModelConvergenceError(
        loop_id=loop_id,
        last_value=float("nan"),
        residual=final_residual,
        suggestion=(
            f"Array fixed-point iteration did not converge after {max_iterations} "
            f"iterations (max|x_new - x_old| = {final_residual:.4e} > tol {tolerance:.4e}). "
            f"Check that the circular dependency is a contraction mapping."
        ),
    )


# ---------------------------------------------------------------------------
# Batch wrappers — used by the executor for vectorised scenario runs
# ---------------------------------------------------------------------------


def goal_seek_batch(
    objectives: List[Callable[[float], float]],
    loop_id: str,
    **kwargs,
) -> List[GoalSeekResult]:
    """
    Run goal_seek independently for each objective in objectives.
    Returns a list of GoalSeekResult (one per scenario).
    Failed scenarios have their error captured in the result (converged=False).
    """
    results = []
    for obj in objectives:
        try:
            results.append(goal_seek_solver(obj, loop_id=loop_id, **kwargs))
        except ModelConvergenceError as exc:
            results.append(
                GoalSeekResult(
                    loop_id=loop_id,
                    converged=False,
                    iterations=0,
                    solution=exc.last_value,
                    final_residual=exc.residual,
                )
            )
    return results
