"""
engine/sensitivity.py — Sensitivity analysis and Monte Carlo simulation.

Two analysis modes
------------------
run_sensitivity(executor, compiled, base_assumptions, sweep_config)
    One-at-a-time (OAT) sensitivity: tornado chart.
    For each assumption in sweep_config, runs the model at its low and high
    values while holding everything else at the base case.
    Returns SensitivityResults with:
      tornado_data = {assumption_name → (low_Δ, high_Δ)}
      where Δ is the change in the chosen KPI metric from the base case.

run_monte_carlo(executor, compiled, base_assumptions, distribution_config, n_iterations)
    Samples each uncertain assumption from its declared distribution
    (triangular, pert, lognormal, uniform) and runs the model n_iterations
    times.  Returns MonteCarloResults with percentile distributions of key KPIs.

All sampling and statistics are handled by NumPy / SciPy — no LLM arithmetic.

Distribution parameterisation
------------------------------
  triangular : triang(c, loc, scale)  c = (mode-low)/(high-low)
  pert       : approximated via Beta(α, β) scaled to [low, high]
               α = 1 + 4*(mode-low)/(high-low)
               β = 1 + 4*(high-mode)/(high-low)
  lognormal  : median = base, σ ≈ range_pct / 2
  uniform    : Uniform[low, high]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import beta as _beta_dist
from scipy.stats import lognorm as _lognorm
from scipy.stats import triang as _triang
from scipy.stats import uniform as _uniform

from dsl.types import DistributionType


# ---------------------------------------------------------------------------
# Private: distribution samplers
# ---------------------------------------------------------------------------


def _sample_triangular(
    low: float,
    mode: float,
    high: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if high <= low:
        return np.full(n, mode)
    c = np.clip((mode - low) / (high - low), 0.0, 1.0)
    return _triang.rvs(c=c, loc=low, scale=high - low, size=n, random_state=rng)


def _sample_pert(
    low: float,
    mode: float,
    high: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Modified PERT via Beta distribution.
    α = 1 + 4*(mode-low)/(high-low),  β = 1 + 4*(high-mode)/(high-low)
    """
    if high <= low:
        return np.full(n, mode)
    spread = high - low
    c = np.clip((mode - low) / spread, 1e-9, 1 - 1e-9)
    alpha = 1.0 + 4.0 * c
    beta_param = 1.0 + 4.0 * (1.0 - c)
    samples = _beta_dist.rvs(alpha, beta_param, size=n, random_state=rng)
    return low + samples * spread


def _sample_lognormal(
    base: float,
    range_pct: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Lognormal with median = |base|, σ ≈ range_pct / 2.
    Sign of base is preserved.
    """
    if base == 0.0:
        return np.zeros(n)
    sigma = max(range_pct / 2.0, 1e-9)
    mu = np.log(abs(base)) - 0.5 * sigma ** 2
    samples = _lognorm.rvs(s=sigma, scale=np.exp(mu), size=n, random_state=rng)
    return samples if base > 0 else -samples


def _sample_uniform(
    low: float,
    high: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if high <= low:
        return np.full(n, (low + high) / 2.0)
    return _uniform.rvs(loc=low, scale=high - low, size=n, random_state=rng)


def _draw_samples(
    dist_type: str,
    low: Optional[float],
    base: float,
    high: Optional[float],
    range_pct: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Dispatch to the correct sampler, filling in missing bounds from range_pct."""
    lo = low  if low  is not None else base * (1.0 - range_pct)
    hi = high if high is not None else base * (1.0 + range_pct)

    if dist_type == DistributionType.triangular.value:
        return _sample_triangular(lo, base, hi, n, rng)
    elif dist_type == DistributionType.pert.value:
        return _sample_pert(lo, base, hi, n, rng)
    elif dist_type == DistributionType.lognormal.value:
        return _sample_lognormal(base, range_pct, n, rng)
    elif dist_type == DistributionType.uniform.value:
        return _sample_uniform(lo, hi, n, rng)
    else:
        # Default: triangular
        return _sample_triangular(lo, base, hi, n, rng)


# ---------------------------------------------------------------------------
# Public: run_sensitivity — one-at-a-time tornado chart
# ---------------------------------------------------------------------------


def run_sensitivity(
    executor: "ModelExecutor",
    compiled: "CompiledModel",
    base_assumptions: Dict[str, Any],
    sweep_config: List[Dict[str, Any]],
    kpi_metric: str = "equity_irr",
) -> "SensitivityResults":
    """
    One-at-a-time (OAT) sensitivity analysis.

    Parameters
    ----------
    executor         : ModelExecutor instance.
    compiled         : CompiledModel from executor.compile().
    base_assumptions : Base-case assumption overrides (may be empty).
    sweep_config     : List of sweep specifications.  Each entry must contain:
                         assumption (str)  — assumption name
                         low       (float) — low bound value
                         high      (float) — high bound value
    kpi_metric       : KPI field name to use as the sensitivity metric.
                       Must be an attribute of KPIResult (default: "equity_irr").

    Returns
    -------
    SensitivityResults
      base_kpis    : KPIResult for the base case.
      tornado_data : {assumption_name → (low_impact, high_impact)}
                     Impact = kpi(scenario) − kpi(base_case).
    """
    from engine.executor import SensitivityResults

    base_results = executor.run(compiled, base_assumptions)
    base_val = _get_kpi_value(base_results.kpis, kpi_metric)

    tornado_data: Dict[str, Tuple[float, float]] = {}

    for entry in sweep_config:
        name = str(entry["assumption"])
        lo   = float(entry["low"])
        hi   = float(entry["high"])

        low_val  = _run_scenario_kpi(executor, compiled, base_assumptions, name, lo,  kpi_metric, base_val)
        high_val = _run_scenario_kpi(executor, compiled, base_assumptions, name, hi, kpi_metric, base_val)

        tornado_data[name] = (low_val - base_val, high_val - base_val)

    return SensitivityResults(tornado_data=tornado_data, base_kpis=base_results.kpis)


def _run_scenario_kpi(
    executor: "ModelExecutor",
    compiled: "CompiledModel",
    base_assumptions: Dict[str, Any],
    assumption_name: str,
    value: float,
    kpi_metric: str,
    fallback: float,
) -> float:
    scenario = dict(base_assumptions)
    scenario[assumption_name] = value
    try:
        result = executor.run(compiled, scenario)
        return _get_kpi_value(result.kpis, kpi_metric)
    except Exception:
        return fallback


def _get_kpi_value(kpis: "KPIResult", metric: str) -> float:
    val = getattr(kpis, metric, None)
    return float(val) if val is not None else 0.0


# ---------------------------------------------------------------------------
# Public: run_monte_carlo — distributional simulation
# ---------------------------------------------------------------------------


def run_monte_carlo(
    executor: "ModelExecutor",
    compiled: "CompiledModel",
    base_assumptions: Dict[str, Any],
    distribution_config: Optional[List[Dict[str, Any]]] = None,
    n_iterations: int = 3000,
    seed: Optional[int] = None,
) -> "MonteCarloResults":
    """
    Monte Carlo simulation over uncertain assumptions.

    Parameters
    ----------
    executor             : ModelExecutor instance.
    compiled             : CompiledModel from executor.compile().
    base_assumptions     : Base-case assumption overrides.
    distribution_config  : List of distribution specifications.  If None, the
                           assumption schema's vary=True entries are used.
                           Each entry may contain:
                             assumption  (str)   — assumption name
                             distribution (str)  — triangular | pert | lognormal | uniform
                             base        (float) — central value (optional; falls back to schema)
                             low         (float) — lower bound (optional)
                             high        (float) — upper bound (optional)
                             range_pct   (float) — fractional half-width if low/high absent
    n_iterations         : Number of Monte Carlo draws.
    seed                 : RNG seed for reproducibility.

    Returns
    -------
    MonteCarloResults with KPI distributions and summary statistics.
    """
    from engine.executor import MonteCarloResults

    rng = np.random.default_rng(seed)

    if distribution_config is None:
        distribution_config = _config_from_schema(compiled, base_assumptions)

    if not distribution_config:
        raise ValueError(
            "No distribution_config provided and no vary=True assumptions in schema. "
            "Either pass distribution_config explicitly or set sensitivity.vary=True "
            "on one or more assumptions in the schema."
        )

    # --- Draw samples for each uncertain assumption ---
    assumption_names = [str(e["assumption"]) for e in distribution_config]
    samples: Dict[str, np.ndarray] = {}

    for entry in distribution_config:
        name      = str(entry["assumption"])
        base_val  = float(
            entry.get("base")
            if entry.get("base") is not None
            else (base_assumptions.get(name) or _schema_default(compiled, name) or 0.0)
        )
        dist_type = str(entry.get("distribution", DistributionType.triangular.value))
        range_pct = float(entry.get("range_pct", 0.10))
        low       = float(entry["low"])  if entry.get("low")  is not None else None
        high      = float(entry["high"]) if entry.get("high") is not None else None

        samples[name] = _draw_samples(dist_type, low, base_val, high, range_pct, n_iterations, rng)

    # --- Build batch assumption dicts ---
    assumptions_batch: List[Dict[str, Any]] = []
    for i in range(n_iterations):
        a = dict(base_assumptions)
        for name in assumption_names:
            a[name] = float(samples[name][i])
        assumptions_batch.append(a)

    # --- Run batch ---
    all_results = executor.run_batch(compiled, assumptions_batch)

    # --- Collect KPI distributions ---
    def _extract(attr: str) -> np.ndarray:
        return np.array([
            float(getattr(r.kpis, attr)) if getattr(r.kpis, attr) is not None else np.nan
            for r in all_results
        ])

    equity_irr_dist = _extract("equity_irr")
    min_dscr_dist   = _extract("min_dscr")
    llcr_dist       = _extract("llcr")
    npv_dist        = _extract("npv_equity")

    hurdle = float(base_assumptions.get("equity_irr_target", 0.14))

    kpi_names  = ["equity_irr", "min_dscr", "llcr", "npv_equity"]
    kpi_arrays = [equity_irr_dist, min_dscr_dist, llcr_dist, npv_dist]

    p10 = {n: _percentile(a, 10)  for n, a in zip(kpi_names, kpi_arrays)}
    p50 = {n: _percentile(a, 50)  for n, a in zip(kpi_names, kpi_arrays)}
    p90 = {n: _percentile(a, 90)  for n, a in zip(kpi_names, kpi_arrays)}

    valid_dscr = min_dscr_dist[~np.isnan(min_dscr_dist)]
    prob_dscr_below_1 = (
        float(np.mean(valid_dscr < 1.0)) if len(valid_dscr) > 0 else float("nan")
    )

    valid_irr = equity_irr_dist[~np.isnan(equity_irr_dist)]
    prob_irr_below_hurdle = (
        float(np.mean(valid_irr < hurdle)) if len(valid_irr) > 0 else float("nan")
    )

    return MonteCarloResults(
        equity_irr_dist=equity_irr_dist,
        min_dscr_dist=min_dscr_dist,
        llcr_dist=llcr_dist,
        npv_dist=npv_dist,
        n_iterations=n_iterations,
        p10=p10,
        p50=p50,
        p90=p90,
        prob_dscr_below_1=prob_dscr_below_1,
        prob_irr_below_hurdle=prob_irr_below_hurdle,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _percentile(arr: np.ndarray, q: float) -> float:
    valid = arr[~np.isnan(arr)]
    return float(np.percentile(valid, q)) if len(valid) > 0 else float("nan")


def _config_from_schema(
    compiled: "CompiledModel",
    base_assumptions: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build a distribution_config list from schema assumptions that have vary=True."""
    config: List[Dict[str, Any]] = []
    for a_def in compiled.model_def.assumption_schema.assumptions:
        if not a_def.sensitivity.vary:
            continue
        base_val = float(
            base_assumptions.get(a_def.name)
            or (a_def.value if isinstance(a_def.value, (int, float)) else 0.0)
        )
        config.append({
            "assumption":   a_def.name,
            "distribution": a_def.sensitivity.distribution.value,
            "base":         base_val,
            "range_pct":    a_def.sensitivity.range_pct,
        })
    return config


def _schema_default(compiled: "CompiledModel", name: str) -> Optional[float]:
    a_def = compiled.model_def.assumption_schema.by_name(name)
    if a_def is None:
        return None
    if isinstance(a_def.value, (int, float)):
        return float(a_def.value)
    return None
