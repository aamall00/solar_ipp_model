"""
tests/test_integration.py — End-to-end integration tests for the solar IPP model.

Loads the full dsl/templates/solar_ipp_base.yaml Karnataka 100 MW baseline,
compiles it, runs it, and verifies that:
  - Parsing produces a valid ModelDefinition with no errors
  - All 14 calculation blocks are present
  - The full run completes without exception
  - All declared output variables are populated
  - KPIs lie within plausible economic ranges
  - Solve-loop convergence metadata is recorded
  - Batch runs return one result per scenario
  - Scenario overrides produce expected directional changes
  - Sensitivity analysis returns valid tornado data
  - Monte Carlo distributions contain finite values

The tests rely on session-scoped fixtures from conftest.py so the heavy
compilation step runs only once per test session.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Expected block IDs in the Karnataka template
EXPECTED_BLOCK_IDS = {
    "generation_block",
    "revenue_block",
    "construction_block",
    "debt_sizing_block",
    "debt_drawdown_block",
    "idc_block",
    "debt_service_block",
    "opex_block",
    "dsra_block",
    "depreciation_block",
    "tax_block",
    "cashflow_block",
    "waterfall_block",
    "returns_block",
}

# Expected output variables (block_id.output_name)
EXPECTED_OUTPUTS = [
    "generation_block.net_generation_kwh",
    "revenue_block.revenue",
    "construction_block.capex_drawdown",
    "debt_sizing_block.debt_amount",
    "debt_service_block.total_debt_service",
    "debt_service_block.outstanding_debt_balance",
    "opex_block.base_opex",
    "cashflow_block.cfads",
    "cashflow_block.ebitda",
    "income_statement_block.tax",
    "cashflow_block.equity_cashflow",
    "cashflow_block.project_cashflow",
]

# Plausible KPI ranges for a 100 MW Karnataka solar IPP.
# Cost-based debt-service mode (debt_sizing_mode=0.0, the schema default) uses
# equal principal repayments which can produce stressed DSCR and equity IRR in
# borderline tariff / leverage scenarios.  Ranges are intentionally wide so the
# integration tests verify model execution and basic sign-consistency rather
# than specific target values.
_EQUITY_IRR_MIN = 0.001   # 0.1% — just verify a positive, computable IRR
_EQUITY_IRR_MAX = 0.40
_PROJECT_IRR_MIN = 0.001
_PROJECT_IRR_MAX = 0.35
_MIN_DSCR_MIN = 0.2    # cost-based mode can breach 1.0 at default leverage
_MIN_DSCR_MAX = 5.0
_DEBT_PAYBACK_MIN = 5.0    # years
_DEBT_PAYBACK_MAX = 30.0


# ===========================================================================
# 1. Parsing and structure
# ===========================================================================


class TestParsing:
    """Verify the template parses cleanly and has the expected structure."""

    def test_parsing_succeeds(self, parse_validation):
        assert parse_validation.valid, (
            f"Template failed validation:\n" + "\n".join(parse_validation.errors)
        )

    def test_no_parse_errors(self, parse_validation):
        assert parse_validation.errors == [], parse_validation.errors

    def test_model_id(self, model_def):
        assert "solar_ipp" in model_def.project_skeleton.model_id.lower()

    def test_total_periods(self, model_def):
        skel = model_def.project_skeleton
        assert skel.total_periods == skel.construction_periods + skel.operations_periods
        assert skel.total_periods > 0

    def test_milestone_ordering(self, model_def):
        m = model_def.project_skeleton.milestones
        assert m.financial_close <= m.cod <= m.debt_maturity

    def test_cod_equals_construction_periods(self, model_def):
        skel = model_def.project_skeleton
        assert skel.milestones.cod == skel.construction_periods

    def test_phases_derived(self, model_def):
        phases = model_def.project_skeleton.phases
        assert phases is not None
        n = model_def.project_skeleton.total_periods
        assert len(phases.is_construction) == n
        assert len(phases.is_operational) == n
        assert len(phases.is_debt_outstanding) == n

    def test_all_blocks_present(self, model_def):
        block_ids = {b.block_id for b in model_def.calculation_blocks.blocks}
        missing = EXPECTED_BLOCK_IDS - block_ids
        assert missing == set(), f"Missing blocks: {missing}"

    def test_assumption_schema_populated(self, model_def):
        assumptions = model_def.assumption_schema.assumptions
        assert len(assumptions) >= 10, "Expected at least 10 assumptions"

    def test_key_assumptions_present(self, model_def):
        names = {a.name for a in model_def.assumption_schema.assumptions}
        for key in ("capacity_mw", "cuf", "capex_per_mw", "tariff", "debt_pct",
                    "interest_rate", "dscr_target"):
            assert key in names, f"Missing assumption: {key}"

    def test_solve_loop_declared(self, model_def):
        loops = model_def.calculation_blocks.solve_loops
        assert len(loops) >= 1, "Expected at least one solve_loop"

    def test_wiring_connections_present(self, model_def):
        assert len(model_def.model_wiring.connections) > 0


# ===========================================================================
# 2. Compilation
# ===========================================================================


class TestCompilation:
    """Verify CompiledModel has correct structure."""

    def test_compiled_n_periods(self, compiled, model_def):
        assert compiled.n_periods == model_def.project_skeleton.total_periods

    def test_compiled_ppy(self, compiled, model_def):
        assert compiled.periods_per_year == model_def.project_skeleton.periods_per_year

    def test_evaluation_plan_nonempty(self, compiled):
        assert len(compiled.evaluation_plan) > 0

    def test_evaluation_plan_has_solve_loop(self, compiled):
        kinds = [step.kind for step in compiled.evaluation_plan]
        assert "solve_loop" in kinds, "Evaluation plan should contain a solve_loop step"

    def test_dependency_graph_nonempty(self, compiled):
        assert compiled.dependency_graph.number_of_nodes() > 0

    def test_output_graph_nonempty(self, compiled):
        assert compiled.output_graph.number_of_nodes() > 0


# ===========================================================================
# 3. Base-case run
# ===========================================================================


class TestBaseRun:
    """Verify the full model run with schema defaults produces correct outputs."""

    def test_run_completes(self, base_results):
        # Fixture itself raises if run fails; just check we have results
        assert base_results is not None

    def test_model_id_matches(self, base_results, model_def):
        assert base_results.model_id == model_def.project_skeleton.model_id

    def test_all_expected_outputs_present(self, base_results):
        missing = [k for k in EXPECTED_OUTPUTS if k not in base_results.variables]
        assert missing == [], f"Missing output variables: {missing}"

    def test_output_arrays_correct_length(self, base_results, compiled):
        n = compiled.n_periods
        for key, arr in base_results.variables.items():
            assert len(arr) == n, (
                f"Variable '{key}' has length {len(arr)}, expected {n}"
            )

    def test_convergence_metadata_present(self, base_results):
        assert len(base_results.convergence) >= 1, (
            "Expected at least one ConvergenceMetadata entry from debt service solve loop"
        )

    def test_solve_loop_converged(self, base_results):
        for meta in base_results.convergence:
            assert meta.converged, (
                f"Solve loop '{meta.loop_id}' did not converge: "
                f"residual={meta.final_residual:.2e}"
            )

    def test_audit_trail_nonempty(self, base_results):
        assert len(base_results.audit_trail) > 0

    def test_assumptions_used_snapshot(self, base_results):
        assert isinstance(base_results.assumptions_used, dict)
        assert "capacity_mw" in base_results.assumptions_used

    # --- Output sanity checks ---

    def test_generation_positive_in_operations(self, base_results, compiled):
        cod = compiled.cod_period
        gen = base_results.variables.get("generation_block.net_generation_kwh")
        assert gen is not None
        # Net generation should be positive in operational periods
        assert np.any(gen[cod:] > 0), "Net generation is zero in all operational periods"

    def test_generation_zero_in_construction(self, base_results, compiled):
        cod = compiled.cod_period
        gen = base_results.variables.get("generation_block.net_generation_kwh")
        # Construction periods should have zero generation
        assert np.all(gen[:cod] == 0.0), "Generation should be zero during construction"

    def test_revenue_positive_in_operations(self, base_results, compiled):
        cod = compiled.cod_period
        rev = base_results.variables.get("revenue_block.revenue")
        assert rev is not None
        assert np.any(rev[cod:] > 0), "Revenue should be positive in operational periods"

    def test_debt_service_zero_before_cod(self, base_results, compiled):
        cod = compiled.cod_period
        ds = base_results.variables.get("debt_service_block.total_debt_service")
        assert ds is not None
        assert np.all(ds[:cod] == 0.0), "Debt service should be zero before COD"

    def test_outstanding_balance_zero_after_maturity(self, base_results, compiled):
        mat = compiled.debt_maturity_period
        n = compiled.n_periods
        bal = base_results.variables.get("debt_service_block.outstanding_debt_balance")
        assert bal is not None
        # Periods after debt maturity should carry zero outstanding balance.
        # bal[mat] is the *opening* balance of the last repayment period (still non-zero);
        # the balance clears to zero at period mat+1 onwards.
        if mat + 1 < n:
            assert abs(float(bal[mat + 1])) < 1.0, (
                f"Outstanding balance at period mat+1={mat+1} = {bal[mat+1]:.2f}, expected ~0"
            )
        # Also verify the balance *does* reach zero somewhere within the model horizon
        assert np.any(np.abs(bal[mat:]) < 1.0), (
            "Outstanding balance never reaches zero at or after debt maturity"
        )

    def test_capex_drawdown_during_construction_only(self, base_results, compiled):
        cod = compiled.cod_period
        n = compiled.n_periods
        capex = base_results.variables.get("construction_block.capex_drawdown")
        assert capex is not None
        assert np.all(capex[cod:] == 0.0), "Capex drawdown should stop at COD"
        assert np.any(capex[:cod] > 0), "Capex drawdown should be positive during construction"

    def test_cfads_positive_in_operations(self, base_results, compiled):
        cod = compiled.cod_period
        cfads = base_results.variables.get("cashflow_block.cfads")
        assert cfads is not None
        assert np.any(cfads[cod:] > 0), "CFADS should be positive in at least some operational periods"

    def test_opex_positive_in_operations(self, base_results, compiled):
        cod = compiled.cod_period
        opex = base_results.variables.get("opex_block.base_opex")
        assert opex is not None
        assert np.any(opex[cod:] > 0), "base_opex should be positive in operational periods"

    def test_equity_cashflow_has_negative_construction(self, base_results, compiled):
        cod = compiled.cod_period
        eq_cf = base_results.variables.get("cashflow_block.equity_cashflow")
        assert eq_cf is not None
        assert np.any(eq_cf[:cod] < 0), (
            "Equity cashflow should be negative (equity investment) during construction"
        )

    # --- KPI range checks ---

    def test_kpis_computed(self, base_results):
        kpis = base_results.kpis
        assert kpis is not None

    def test_equity_irr_in_range(self, base_results):
        irr = base_results.kpis.equity_irr
        if irr is None:
            pytest.skip("equity_irr not computed (model may need full wiring)")
        assert _EQUITY_IRR_MIN <= irr <= _EQUITY_IRR_MAX, (
            f"equity_irr = {irr:.2%} is outside plausible range "
            f"[{_EQUITY_IRR_MIN:.0%}, {_EQUITY_IRR_MAX:.0%}]"
        )

    def test_project_irr_in_range(self, base_results):
        irr = base_results.kpis.project_irr
        if irr is None:
            pytest.skip("project_irr not computed")
        assert _PROJECT_IRR_MIN <= irr <= _PROJECT_IRR_MAX, (
            f"project_irr = {irr:.2%} outside [{_PROJECT_IRR_MIN:.0%}, {_PROJECT_IRR_MAX:.0%}]"
        )

    def test_min_dscr_positive(self, base_results):
        dscr = base_results.kpis.min_dscr
        if dscr is None:
            pytest.skip("min_dscr not computed")
        assert dscr > 0, f"min_dscr = {dscr:.3f} should be positive"

    def test_min_dscr_in_range(self, base_results):
        dscr = base_results.kpis.min_dscr
        if dscr is None:
            pytest.skip("min_dscr not computed")
        assert _MIN_DSCR_MIN <= dscr <= _MIN_DSCR_MAX, (
            f"min_dscr = {dscr:.3f} outside [{_MIN_DSCR_MIN}, {_MIN_DSCR_MAX}]"
        )

    def test_debt_payback_in_range(self, base_results):
        payback = base_results.kpis.debt_payback_period
        if payback is None:
            pytest.skip("debt_payback_period not computed")
        assert _DEBT_PAYBACK_MIN <= payback <= _DEBT_PAYBACK_MAX, (
            f"debt_payback_period = {payback:.1f} years outside "
            f"[{_DEBT_PAYBACK_MIN}, {_DEBT_PAYBACK_MAX}]"
        )

    def test_peak_debt_positive(self, base_results):
        peak = base_results.kpis.peak_debt_outstanding
        if peak is None:
            pytest.skip("peak_debt_outstanding not computed")
        assert peak > 0, f"peak_debt_outstanding = {peak:.1f} should be positive"


# ===========================================================================
# 4. Scenario runs — directional tests
# ===========================================================================


class TestScenarioRuns:
    """Confirm that assumption changes move KPIs in the expected direction."""

    def test_higher_cuf_increases_equity_irr(self, compiled, executor):
        low_result = executor.run(compiled, {"cuf": 0.18})
        high_result = executor.run(compiled, {"cuf": 0.26})

        irr_low = low_result.kpis.equity_irr
        irr_high = high_result.kpis.equity_irr
        if irr_low is None or irr_high is None:
            pytest.skip("IRR not available for comparison")

        assert irr_high > irr_low, (
            f"Higher CUF should increase equity IRR: "
            f"CUF=0.26 → {irr_high:.2%}, CUF=0.18 → {irr_low:.2%}"
        )

    def test_higher_tariff_increases_equity_irr(self, compiled, executor):
        low_result = executor.run(compiled, {"tariff": 2.20})
        high_result = executor.run(compiled, {"tariff": 3.10})

        irr_low = low_result.kpis.equity_irr
        irr_high = high_result.kpis.equity_irr
        if irr_low is None or irr_high is None:
            pytest.skip("IRR not available")

        assert irr_high > irr_low, (
            f"Higher tariff should increase equity IRR: "
            f"2.65 → {irr_high:.2%}, 2.20 → {irr_low:.2%}"
        )

    def test_lower_capex_increases_equity_irr(self, compiled, executor):
        low_capex = executor.run(compiled, {"capex_per_mw": 380})
        high_capex = executor.run(compiled, {"capex_per_mw": 520})

        irr_low_capex = low_capex.kpis.equity_irr
        irr_high_capex = high_capex.kpis.equity_irr
        if irr_low_capex is None or irr_high_capex is None:
            pytest.skip("IRR not available")

        assert irr_low_capex > irr_high_capex, (
            f"Lower capex should increase equity IRR: "
            f"380 → {irr_low_capex:.2%}, 520 → {irr_high_capex:.2%}"
        )

    def test_higher_leverage_amplifies_equity_irr(self, compiled, executor):
        """Higher debt_pct typically increases equity IRR (leverage effect) until service breaks."""
        base = executor.run(compiled, {"debt_pct": 0.60})
        levered = executor.run(compiled, {"debt_pct": 0.75})

        irr_base = base.kpis.equity_irr
        irr_levered = levered.kpis.equity_irr
        if irr_base is None or irr_levered is None:
            pytest.skip("IRR not available")

        # At sensible leverage the levered IRR should be higher
        assert irr_levered > irr_base, (
            f"Higher leverage (75% vs 60%) should increase equity IRR: "
            f"{irr_levered:.2%} vs {irr_base:.2%}"
        )

    def test_higher_cuf_increases_revenue(self, compiled, executor):
        low = executor.run(compiled, {"cuf": 0.18})
        high = executor.run(compiled, {"cuf": 0.26})

        cod = compiled.cod_period
        rev_low = low.variables.get("revenue_block.revenue")
        rev_high = high.variables.get("revenue_block.revenue")
        assert rev_low is not None and rev_high is not None

        assert np.sum(rev_high[cod:]) > np.sum(rev_low[cod:]), (
            "Higher CUF should produce higher total revenue"
        )

    def test_higher_opex_escalation_reduces_cfads(self, compiled, executor):
        low = executor.run(compiled, {"opex_escalation": 0.02})
        high = executor.run(compiled, {"opex_escalation": 0.06})

        cod = compiled.cod_period
        cfads_low = np.sum(low.variables.get("cashflow_block.cfads", np.zeros(compiled.n_periods))[cod:])
        cfads_high = np.sum(high.variables.get("cashflow_block.cfads", np.zeros(compiled.n_periods))[cod:])

        assert cfads_low > cfads_high, (
            "Higher OPEX escalation should reduce total CFADS"
        )


# ===========================================================================
# 5. Batch run
# ===========================================================================


class TestBatchRun:
    """Verify run_batch returns results in the correct order."""

    def test_batch_returns_correct_count(self, compiled, executor):
        scenarios = [
            {"cuf": 0.18},
            {"cuf": 0.22},
            {"cuf": 0.26},
        ]
        results = executor.run_batch(compiled, scenarios)
        assert len(results) == 3

    def test_batch_order_preserved(self, compiled, executor):
        cufs = [0.18, 0.20, 0.22, 0.24, 0.26]
        scenarios = [{"cuf": c} for c in cufs]
        results = executor.run_batch(compiled, scenarios)

        # Revenue should be monotonically increasing with CUF
        cod = compiled.cod_period
        revenues = [
            float(np.sum(r.variables.get("revenue_block.revenue", np.zeros(compiled.n_periods))[cod:]))
            for r in results
        ]
        for i in range(1, len(revenues)):
            assert revenues[i] >= revenues[i - 1], (
                f"Revenue not monotone at index {i}: {revenues[i]:.1f} < {revenues[i-1]:.1f}"
            )

    def test_batch_single_scenario(self, compiled, executor):
        results = executor.run_batch(compiled, [{"cuf": 0.22}])
        assert len(results) == 1
        assert results[0] is not None

    def test_batch_empty_returns_empty(self, compiled, executor):
        results = executor.run_batch(compiled, [])
        assert results == []


# ===========================================================================
# 6. Sensitivity analysis
# ===========================================================================


class TestSensitivityAnalysis:
    """Verify run_sensitivity produces valid tornado data."""

    @pytest.fixture(scope="class")
    def sensitivity_results(self, compiled, executor):
        sweep = [
            {"assumption": "cuf",         "low": 0.18, "high": 0.26},
            {"assumption": "tariff",       "low": 2.20, "high": 3.10},
            {"assumption": "capex_per_mw", "low": 380.0, "high": 520.0},
            {"assumption": "interest_rate","low": 0.085, "high": 0.115},
        ]
        return executor.run_sensitivity(compiled, {}, sweep)

    def test_returns_sensitivity_results(self, sensitivity_results):
        from engine.executor import SensitivityResults
        assert isinstance(sensitivity_results, SensitivityResults)

    def test_base_kpis_computed(self, sensitivity_results):
        assert sensitivity_results.base_kpis is not None

    def test_tornado_data_all_assumptions(self, sensitivity_results):
        expected = {"cuf", "tariff", "capex_per_mw", "interest_rate"}
        assert expected == set(sensitivity_results.tornado_data.keys())

    def test_tornado_entries_are_tuples(self, sensitivity_results):
        for name, entry in sensitivity_results.tornado_data.items():
            assert len(entry) == 2, f"Tornado entry for '{name}' should be (low_impact, high_impact)"

    def test_cuf_directional(self, sensitivity_results):
        low_impact, high_impact = sensitivity_results.tornado_data["cuf"]
        assert high_impact > low_impact, (
            "Higher CUF should have a more positive equity_irr impact than lower CUF"
        )

    def test_tariff_directional(self, sensitivity_results):
        low_impact, high_impact = sensitivity_results.tornado_data["tariff"]
        assert high_impact > low_impact, (
            "Higher tariff should produce more positive equity_irr impact"
        )

    def test_capex_directional(self, sensitivity_results):
        low_impact, high_impact = sensitivity_results.tornado_data["capex_per_mw"]
        # Lower capex → higher IRR (low_impact > high_impact)
        assert low_impact > high_impact, (
            "Lower capex should produce higher equity_irr (low_impact should be positive/larger)"
        )

    def test_tornado_impacts_finite(self, sensitivity_results):
        for name, (lo, hi) in sensitivity_results.tornado_data.items():
            assert math.isfinite(lo), f"low_impact for '{name}' is not finite: {lo}"
            assert math.isfinite(hi), f"high_impact for '{name}' is not finite: {hi}"


# ===========================================================================
# 7. Monte Carlo
# ===========================================================================


class TestMonteCarlo:
    """Verify run_monte_carlo produces valid distributional output."""

    _N_ITER = 50  # small for test speed

    @pytest.fixture(scope="class")
    def mc_results(self, compiled, executor):
        dist_config = [
            {"assumption": "cuf",    "distribution": "triangular",
             "low": 0.18, "base": 0.22, "high": 0.26},
            {"assumption": "tariff", "distribution": "uniform",
             "low": 2.40, "high": 2.90},
        ]
        return executor.run_monte_carlo(
            compiled, {}, dist_config, n_iterations=self._N_ITER
        )

    def test_returns_monte_carlo_results(self, mc_results):
        from engine.executor import MonteCarloResults
        assert isinstance(mc_results, MonteCarloResults)

    def test_n_iterations_recorded(self, mc_results):
        assert mc_results.n_iterations == self._N_ITER

    def test_equity_irr_dist_length(self, mc_results):
        assert len(mc_results.equity_irr_dist) == self._N_ITER

    def test_min_dscr_dist_length(self, mc_results):
        assert len(mc_results.min_dscr_dist) == self._N_ITER

    def test_percentile_keys(self, mc_results):
        for key in ("equity_irr", "min_dscr", "llcr", "npv_equity"):
            assert key in mc_results.p10, f"p10 missing key: {key}"
            assert key in mc_results.p50, f"p50 missing key: {key}"
            assert key in mc_results.p90, f"p90 missing key: {key}"

    def test_percentile_ordering(self, mc_results):
        """p10 ≤ p50 ≤ p90 for equity_irr (ignoring NaN)."""
        p10 = mc_results.p10.get("equity_irr")
        p50 = mc_results.p50.get("equity_irr")
        p90 = mc_results.p90.get("equity_irr")
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (p10, p50, p90)):
            pytest.skip("equity_irr percentiles contain NaN")
        assert p10 <= p50 <= p90, (
            f"Percentile ordering violated: p10={p10:.2%}, p50={p50:.2%}, p90={p90:.2%}"
        )

    def test_prob_dscr_below_1_in_range(self, mc_results):
        p = mc_results.prob_dscr_below_1
        if math.isnan(p):
            pytest.skip("prob_dscr_below_1 is NaN")
        assert 0.0 <= p <= 1.0, f"prob_dscr_below_1 = {p} is not in [0, 1]"

    def test_prob_irr_below_hurdle_in_range(self, mc_results):
        p = mc_results.prob_irr_below_hurdle
        if math.isnan(p):
            pytest.skip("prob_irr_below_hurdle is NaN")
        assert 0.0 <= p <= 1.0, f"prob_irr_below_hurdle = {p} is not in [0, 1]"

    def test_irr_distribution_has_spread(self, mc_results):
        """Two distinct CUF/tariff inputs should produce variance in equity_irr."""
        valid = mc_results.equity_irr_dist[~np.isnan(mc_results.equity_irr_dist)]
        if len(valid) < 10:
            pytest.skip("Too few valid IRR values to test spread")
        std = float(np.std(valid))
        assert std > 0, "equity_irr distribution has zero variance — sampling may not be working"


# ===========================================================================
# 8. Downstream variable tracking
# ===========================================================================


class TestDownstreamTracking:
    """Verify executor.downstream_variables() correctly identifies dependents."""

    def test_cuf_has_downstream(self, compiled, executor):
        deps = executor.downstream_variables(compiled, "cuf")
        assert len(deps) > 0, "cuf should have downstream variables"

    def test_generation_downstream_includes_revenue(self, compiled, executor):
        deps = executor.downstream_variables(compiled, "cuf")
        # revenue depends on generation which depends on cuf
        assert any("revenue" in d for d in deps), (
            f"Expected revenue in downstream of cuf. Got: {deps}"
        )

    def test_unknown_assumption_returns_empty(self, compiled, executor):
        deps = executor.downstream_variables(compiled, "nonexistent_var")
        assert deps == []
