"""
Unit tests for dsl/expression.py

Run with: pytest tests/test_expression.py -v
"""

import math

import numpy as np
import pytest

from dsl.expression import (
    ExpressionEvaluator,
    ExpressionError,
    UndefinedVariableError,
    UnsupportedNodeError,
    evaluate,
)

# ---------------------------------------------------------------------------
# Fixture: standard evaluator (12 quarterly periods = 3 years)
# ---------------------------------------------------------------------------

N = 12  # total periods
PPY = 4  # periods per year


@pytest.fixture
def ev() -> ExpressionEvaluator:
    return ExpressionEvaluator(n_periods=N, periods_per_year=PPY)


def ctx(**kwargs) -> dict:
    """Helper: build context with numpy arrays from keyword args."""
    return {k: np.asarray(v, dtype=float) if isinstance(v, (list, np.ndarray)) else v
            for k, v in kwargs.items()}


# ===========================================================================
# 1. Arithmetic
# ===========================================================================


class TestArithmetic:
    def test_scalar_add(self, ev):
        result = ev.evaluate("2 + 3", {})
        assert result == 5

    def test_scalar_mul(self, ev):
        assert ev.evaluate("4 * 5", {}) == 20

    def test_scalar_div(self, ev):
        assert ev.evaluate("10 / 4", {}) == pytest.approx(2.5)

    def test_scalar_pow(self, ev):
        assert ev.evaluate("2 ** 10", {}) == pytest.approx(1024.0)

    def test_unary_neg(self, ev):
        assert ev.evaluate("-7", {}) == -7

    def test_array_add(self, ev):
        a = np.ones(N) * 3
        b = np.ones(N) * 2
        result = ev.evaluate("a + b", ctx(a=a, b=b))
        np.testing.assert_array_almost_equal(result, np.full(N, 5.0))

    def test_array_mul_by_scalar(self, ev):
        a = np.arange(N, dtype=float)
        result = ev.evaluate("a * 2", ctx(a=a))
        np.testing.assert_array_almost_equal(result, np.arange(N) * 2)

    def test_element_wise_pow(self, ev):
        a = np.full(N, 2.0)
        result = ev.evaluate("a ** 3", ctx(a=a))
        np.testing.assert_array_almost_equal(result, np.full(N, 8.0))

    def test_div_by_zero_scalar_returns_zero(self, ev):
        result = ev.evaluate("5 / 0", {})
        assert result == 0.0

    def test_div_by_zero_array_returns_zero_element(self, ev):
        a = np.ones(N)
        b = np.zeros(N)
        b[3] = 2.0  # only period 3 is non-zero
        result = ev.evaluate("a / b", ctx(a=a, b=b))
        assert result[3] == pytest.approx(0.5)
        assert result[0] == 0.0  # divide by zero → 0

    def test_nested_arithmetic(self, ev):
        result = ev.evaluate("(3 + 4) * (2 - 1)", {})
        assert result == 7


# ===========================================================================
# 2. Variable resolution
# ===========================================================================


class TestVariableResolution:
    def test_scalar_variable(self, ev):
        result = ev.evaluate("x", ctx(x=3.14))
        assert result == pytest.approx(3.14)

    def test_array_variable(self, ev):
        arr = np.linspace(0, 1, N)
        result = ev.evaluate("arr", ctx(arr=arr))
        np.testing.assert_array_almost_equal(result, arr)

    def test_period_index(self, ev):
        result = ev.evaluate("period_index", {})
        np.testing.assert_array_almost_equal(result, np.arange(N))

    def test_dotted_phase_ref(self, ev):
        mask = np.array([0] * 3 + [1] * (N - 3), dtype=float)
        result = ev.evaluate(
            "revenue * phase.is_operational",
            ctx(revenue=np.ones(N), **{"phase.is_operational": mask}),
        )
        np.testing.assert_array_almost_equal(result, mask)

    def test_undefined_variable_raises(self, ev):
        with pytest.raises(UndefinedVariableError, match="unknown_var"):
            ev.evaluate("unknown_var", {})

    def test_unsupported_node_raises(self, ev):
        # List comprehension is not allowed
        with pytest.raises(UnsupportedNodeError):
            ev.evaluate("[x for x in range(5)]", {})

    def test_import_attempt_blocked(self, ev):
        # Should fail — ast.parse('import os', mode='eval') raises SyntaxError
        with pytest.raises(ExpressionError):
            ev.evaluate("import os", {})


# ===========================================================================
# 3. Comparisons and boolean ops
# ===========================================================================


class TestComparisons:
    def test_scalar_gt(self, ev):
        assert ev.evaluate("5 > 3", {}) is True

    def test_array_gt_produces_bool_array(self, ev):
        a = np.array([1.0, 2.0, 3.0] + [0.0] * (N - 3))
        result = ev.evaluate("a > 1.5", ctx(a=a))
        assert result[0] == False
        assert result[1] == True

    def test_combined_bool_and(self, ev):
        a = np.ones(N)
        b = np.ones(N) * 2
        result = ev.evaluate("a > 0 and b > 1", ctx(a=a, b=b))
        # Both True → result is True-ish array or scalar
        assert np.all(result)


# ===========================================================================
# 4. cumsum
# ===========================================================================


class TestCumsum:
    def test_basic_cumsum(self, ev):
        draws = np.zeros(N)
        draws[0] = 10.0
        draws[1] = 20.0
        draws[2] = 30.0
        result = ev.evaluate("cumsum(draws)", ctx(draws=draws))
        assert result[0] == pytest.approx(10.0)
        assert result[1] == pytest.approx(30.0)
        assert result[2] == pytest.approx(60.0)
        # Remaining periods stay at 60
        assert result[-1] == pytest.approx(60.0)

    def test_cumsum_all_ones(self, ev):
        ones = np.ones(N)
        result = ev.evaluate("cumsum(ones)", ctx(ones=ones))
        np.testing.assert_array_almost_equal(result, np.arange(1, N + 1, dtype=float))


# ===========================================================================
# 5. lag
# ===========================================================================


class TestLag:
    def test_lag_1(self, ev):
        series = np.arange(1, N + 1, dtype=float)
        result = ev.evaluate("lag(series, 1)", ctx(series=series))
        assert result[0] == 0.0        # padded
        assert result[1] == pytest.approx(1.0)
        assert result[-1] == pytest.approx(N - 1.0)

    def test_lag_0_is_identity(self, ev):
        series = np.arange(N, dtype=float)
        result = ev.evaluate("lag(series, 0)", ctx(series=series))
        np.testing.assert_array_almost_equal(result, series)

    def test_lag_exceeds_periods(self, ev):
        series = np.ones(N)
        result = ev.evaluate(f"lag(series, {N + 5})", ctx(series=series))
        np.testing.assert_array_almost_equal(result, np.zeros(N))

    def test_lag_negative_raises(self, ev):
        series = np.ones(N)
        with pytest.raises(ExpressionError, match=">="):
            ev.evaluate("lag(series, -1)", ctx(series=series))


# ===========================================================================
# 6. pv
# ===========================================================================


class TestPV:
    def test_pv_zero_rate_sums_series(self, ev):
        series = np.full(N, 10.0)
        result = ev.evaluate("pv(series, 0)", ctx(series=series))
        assert result == pytest.approx(N * 10.0)

    def test_pv_known_value(self, ev):
        # Single cash flow of 100 in period 0, discounted at 10% per period
        # PV = 100 / (1.10)^1 = 90.909...
        series = np.zeros(N)
        series[0] = 100.0
        result = ev.evaluate("pv(series, 0.10)", ctx(series=series))
        assert result == pytest.approx(100.0 / 1.10, rel=1e-6)

    def test_pv_is_scalar(self, ev):
        series = np.ones(N) * 5.0
        result = ev.evaluate("pv(series, 0.02)", ctx(series=series))
        assert isinstance(result, float)

    def test_pv_decreasing_with_rate(self, ev):
        series = np.ones(N) * 100.0
        pv_low = ev.evaluate("pv(series, 0.01)", ctx(series=series))
        pv_high = ev.evaluate("pv(series, 0.15)", ctx(series=series))
        assert pv_low > pv_high  # higher rate → lower PV


# ===========================================================================
# 7. annuity
# ===========================================================================


class TestAnnuity:
    def test_annuity_length(self, ev):
        result = ev.evaluate("annuity(1000, 0.025, 8)", {})
        assert len(result) == N  # always full n_periods length

    def test_annuity_zero_rate(self, ev):
        # 1000 / 5 = 200 per period for 5 periods
        result = ev.evaluate("annuity(1000, 0, 5)", {})
        for i in range(5):
            assert result[i] == pytest.approx(200.0)
        for i in range(5, N):
            assert result[i] == 0.0

    def test_annuity_positive_rate_correct_payment(self, ev):
        # Analytical check: PMT = PV * r(1+r)^n / ((1+r)^n - 1)
        pv, r, n = 10000.0, 0.025, 8  # 8 periods, 2.5% per period
        expected_pmt = pv * r * (1 + r) ** n / ((1 + r) ** n - 1)
        result = ev.evaluate("annuity(10000, 0.025, 8)", {})
        for i in range(8):
            assert result[i] == pytest.approx(expected_pmt, rel=1e-6)
        for i in range(8, N):
            assert result[i] == 0.0

    def test_annuity_payments_recover_principal(self, ev):
        # Sum of annuity payments discounted at the same rate should = principal
        pv_amount, r, n = 5000.0, 0.03, 10
        result = ev.evaluate("annuity(5000, 0.03, 10)", {})
        payments = result[:n]
        t = np.arange(1, n + 1, dtype=float)
        pv_of_payments = float(np.sum(payments / (1 + r) ** t))
        assert pv_of_payments == pytest.approx(pv_amount, rel=1e-5)

    def test_annuity_zero_principal(self, ev):
        result = ev.evaluate("annuity(0, 0.05, 8)", {})
        np.testing.assert_array_almost_equal(result, np.zeros(N))

    def test_annuity_n_exceeds_model_periods(self, ev):
        # n_periods larger than self.n_periods — should fill entire array
        result = ev.evaluate(f"annuity(1000, 0.02, {N + 50})", {})
        assert len(result) == N
        # All periods should be filled with the payment
        assert all(result[i] > 0 for i in range(N))


# ===========================================================================
# 8. max_series / min_series
# ===========================================================================


class TestMaxMinSeries:
    def test_max_series(self, ev):
        a = np.array([1.0, 5.0, 3.0] + [0.0] * (N - 3))
        b = np.array([4.0, 2.0, 3.0] + [0.0] * (N - 3))
        result = ev.evaluate("max_series(a, b)", ctx(a=a, b=b))
        assert result[0] == pytest.approx(4.0)
        assert result[1] == pytest.approx(5.0)
        assert result[2] == pytest.approx(3.0)

    def test_min_series(self, ev):
        a = np.array([1.0, 5.0, 3.0] + [0.0] * (N - 3))
        b = np.array([4.0, 2.0, 3.0] + [0.0] * (N - 3))
        result = ev.evaluate("min_series(a, b)", ctx(a=a, b=b))
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(3.0)

    def test_max_with_zero_useful_for_floor(self, ev):
        series = np.array([-3.0, 0.0, 5.0] + [2.0] * (N - 3))
        result = ev.evaluate("max_series(series, 0)", ctx(series=series))
        assert result[0] == pytest.approx(0.0)   # floored at zero
        assert result[2] == pytest.approx(5.0)


# ===========================================================================
# 9. scalar_to_series
# ===========================================================================


class TestScalarToSeries:
    def test_broadcasts_float(self, ev):
        result = ev.evaluate("scalar_to_series(3.14)", {})
        assert len(result) == N
        np.testing.assert_array_almost_equal(result, np.full(N, 3.14))

    def test_broadcasts_zero(self, ev):
        result = ev.evaluate("scalar_to_series(0)", {})
        np.testing.assert_array_almost_equal(result, np.zeros(N))


# ===========================================================================
# 10. escalate
# ===========================================================================


class TestEscalate:
    def test_period_zero_equals_base(self, ev):
        result = ev.evaluate("escalate(100, 0.05)", {})
        # period 0: 100 * (1.05)^(0/4) = 100 * 1.0 = 100
        assert result[0] == pytest.approx(100.0)

    def test_after_one_year_correctly_escalated(self, ev):
        result = ev.evaluate("escalate(100, 0.08)", {})
        # After 4 quarters (period 4): 100 * (1.08)^(4/4) = 100 * 1.08
        assert result[4] == pytest.approx(100.0 * 1.08, rel=1e-6)

    def test_zero_rate_flat(self, ev):
        result = ev.evaluate("escalate(50, 0)", {})
        np.testing.assert_array_almost_equal(result, np.full(N, 50.0))

    def test_escalation_is_monotone(self, ev):
        result = ev.evaluate("escalate(10, 0.06)", {})
        assert all(result[i] <= result[i + 1] for i in range(N - 1))


# ===========================================================================
# 11. referenced_variables
# ===========================================================================


class TestReferencedVariables:
    def test_simple_names(self, ev):
        refs = ev.referenced_variables("a + b * c")
        assert set(refs) == {"a", "b", "c"}

    def test_dotted_phase(self, ev):
        refs = ev.referenced_variables("revenue * phase.is_operational")
        assert "revenue" in refs
        assert "phase.is_operational" in refs

    def test_function_args(self, ev):
        refs = ev.referenced_variables("lag(debt_drawn, 1)")
        assert "debt_drawn" in refs

    def test_period_index_excluded(self, ev):
        refs = ev.referenced_variables("escalate(base_opex, 0.05) * period_index")
        assert "period_index" not in refs
        assert "base_opex" in refs


# ===========================================================================
# 12. Compound / realistic expressions
# ===========================================================================


class TestRealisticExpressions:
    def test_generation_formula(self, ev):
        """
        p50_gen = installed_mw * cuf * hours_per_period * phase.is_operational
        Mirrors the generation block logic.
        """
        installed_mw = 100.0
        cuf = 0.22
        # 3 construction, 9 operational
        is_operational = np.array([0] * 3 + [1] * (N - 3), dtype=float)
        hours_per_period = (365 / 4) * 24  # quarterly

        ctx_data = {
            "installed_mw": installed_mw,
            "cuf": cuf,
            "hours_per_period": hours_per_period,
            "phase.is_operational": is_operational,
        }
        result = ev.evaluate(
            "installed_mw * cuf * hours_per_period * phase.is_operational",
            ctx_data,
        )
        # Construction periods → 0
        for t in range(3):
            assert result[t] == pytest.approx(0.0)
        # Operational periods → positive
        expected_quarterly_gen = installed_mw * cuf * hours_per_period * 1000  # kWh
        for t in range(3, N):
            assert result[t] == pytest.approx(installed_mw * cuf * hours_per_period)

    def test_capex_draw(self, ev):
        """capex_draw = total_capex * capex_schedule_t * phase.is_construction"""
        total_capex = 45000.0  # INR Lakhs
        capex_schedule = np.array([0.30, 0.40, 0.30] + [0.0] * (N - 3))
        is_construction = np.array([1] * 3 + [0] * (N - 3), dtype=float)

        result = ev.evaluate(
            "total_capex * capex_schedule * phase.is_construction",
            {
                "total_capex": total_capex,
                "capex_schedule": capex_schedule,
                "phase.is_construction": is_construction,
            },
        )
        assert result[0] == pytest.approx(45000.0 * 0.30)
        assert result[1] == pytest.approx(45000.0 * 0.40)
        assert result[2] == pytest.approx(45000.0 * 0.30)
        for t in range(3, N):
            assert result[t] == pytest.approx(0.0)

    def test_idc_uses_lag_and_cumsum(self, ev):
        """
        idc = lag(cumsum(debt_drawn), 1) * (interest_rate / periods_per_year)
        IDC in any period = opening cumulative debt × quarterly rate
        """
        debt_drawn = np.array([3000.0, 4000.0, 3000.0] + [0.0] * (N - 3))
        interest_rate = 0.0975
        rate_per_period = interest_rate / PPY  # 0.024375

        result = ev.evaluate(
            "lag(cumsum(debt_drawn), 1) * (interest_rate / periods_per_year)",
            {
                "debt_drawn": debt_drawn,
                "interest_rate": interest_rate,
                "periods_per_year": PPY,
            },
        )
        # Period 0: lag of cumsum at t=-1 = 0 → idc = 0
        assert result[0] == pytest.approx(0.0)
        # Period 1: cumsum[0] = 3000 → idc = 3000 * 0.024375
        assert result[1] == pytest.approx(3000.0 * rate_per_period, rel=1e-6)
        # Period 2: cumsum[1] = 7000 → idc = 7000 * 0.024375
        assert result[2] == pytest.approx(7000.0 * rate_per_period, rel=1e-6)

    def test_module_level_evaluate(self):
        """Test the convenience module-level function."""
        result = evaluate("a * 2 + 1", ctx(a=np.ones(N)), n_periods=N, periods_per_year=PPY)
        np.testing.assert_array_almost_equal(result, np.full(N, 3.0))


# ===========================================================================
# 13. Edge cases and error paths
# ===========================================================================


class TestEdgeCases:
    def test_empty_expression_raises(self, ev):
        with pytest.raises(ExpressionError):
            ev.evaluate("", {})

    def test_unknown_function_raises(self, ev):
        with pytest.raises(UnsupportedNodeError, match="Unknown function"):
            ev.evaluate("os.system('ls')", {})

    def test_invalid_n_periods(self):
        with pytest.raises(ValueError, match="n_periods"):
            ExpressionEvaluator(n_periods=0, periods_per_year=4)

    def test_invalid_periods_per_year(self):
        with pytest.raises(ValueError, match="periods_per_year"):
            ExpressionEvaluator(n_periods=10, periods_per_year=0)

    def test_deeply_nested_expression(self, ev):
        # max_series of escalated values masked by phase
        mask = np.array([0] * 3 + [1] * (N - 3), dtype=float)
        result = ev.evaluate(
            "max_series(escalate(100, 0.05) * phase.is_operational, 0)",
            {"phase.is_operational": mask},
        )
        # Construction periods: max(0, 0) = 0
        for t in range(3):
            assert result[t] == pytest.approx(0.0)
        # Operational periods: positive escalated value
        for t in range(3, N):
            assert result[t] > 0.0
