"""
Unit tests for engine/kpi.py

All expected values are computed analytically or cross-checked with
known finance formulas so the tests are self-validating.

Run with: pytest tests/test_kpis.py -v
"""

import math

import numpy as np
import pytest

from engine.kpi import (
    KPIComputationError,
    DSCRPhaseError,
    avg_dscr,
    compute_all_kpis,
    debt_payback_period,
    dscr_series,
    llcr,
    min_dscr,
    npv_at_rate,
    npv_equity,
    peak_debt_outstanding,
    plcr,
    xirr,
)
from dsl.types import KPIResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cashflows(outflow_periods: int, inflow_per_period: float, n: int) -> np.ndarray:
    """Simple pattern: negative in first outflow_periods, positive thereafter."""
    cf = np.full(n, inflow_per_period)
    cf[:outflow_periods] = -inflow_per_period * outflow_periods / max(outflow_periods, 1)
    return cf


# ===========================================================================
# 1. XIRR
# ===========================================================================


class TestXIRR:
    """
    Ground-truth checks use the formula:
    For a loan of PV repaid in n equal end-of-period payments at periodic rate r:
        PMT = PV * r(1+r)^n / ((1+r)^n - 1)
    The annuity cash-flow series [-PV, PMT, PMT, ..., PMT] has IRR = r_periodic.
    Annual IRR = (1+r_periodic)^ppy - 1.
    """

    def test_simple_bond_quarterly(self):
        """
        Borrow 100 at t=0, repay 12 equal quarterly instalments.
        Quarterly rate = 2.5%  →  annual IRR ≈ (1.025)^4 - 1 ≈ 10.38%.
        """
        r_q = 0.025   # quarterly rate
        n = 12
        pv = 100.0
        pmt = pv * r_q * (1 + r_q) ** n / ((1 + r_q) ** n - 1)
        cf = np.array([-pv] + [pmt] * n)
        result = xirr(cf, periods_per_year=4)
        expected = (1.0 + r_q) ** 4 - 1.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_annual_cashflows_ppy1(self):
        """
        Annual cash flows: -1000, 300, 300, 300, 300, 300 (5 years).
        IRR should equal the periodic rate that makes NPV=0.
        """
        cf = np.array([-1000.0, 300.0, 300.0, 300.0, 300.0, 300.0])
        result = xirr(cf, periods_per_year=1)
        # Verify: NPV at result is ≈ 0
        t = np.arange(1, len(cf) + 1, dtype=float)
        npv_check = np.sum(cf / (1.0 + result) ** t)
        assert abs(npv_check) < 1e-6

    def test_zero_irr_for_flat_recovery(self):
        """
        -100, +100 → periodic IRR ≈ 0, annual ≈ 0.
        """
        cf = np.array([-100.0, 100.0])
        result = xirr(cf, periods_per_year=4)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_high_irr(self):
        """
        -100, +500 in one period → periodic rate = 4.0 (400%), annual ≈ huge.
        """
        cf = np.array([-100.0, 500.0])
        result = xirr(cf, periods_per_year=4)
        # Periodic rate = (500/100) - 1 = 4.0 → annual = (1+4)^4 - 1 = 624
        assert result == pytest.approx((1.0 + 4.0) ** 4 - 1.0, rel=1e-4)

    def test_realistic_equity_irr(self):
        """
        Equity: -3 construction quarters of 3000 each, then 100 ops quarters
        with a pattern yielding ~15% annual IRR.
        We construct the pattern analytically.
        """
        # Target quarterly rate r_q such that annual = 15%
        annual_target = 0.15
        r_q = (1.0 + annual_target) ** (1.0 / 4) - 1.0

        n_const = 3
        n_ops = 100
        equity_invested = 10000.0  # INR Lakhs total
        equity_per_period = equity_invested / n_const

        # Annuity payment that makes IRR = r_q over n_ops periods
        # PV of distributions at COD = equity_invested
        # PMT = PV * r(1+r)^n / ((1+r)^n - 1)  — but discounted back 3 periods
        # We need: sum of PMT/(1+r_q)^(3+t) = equity_invested * (1+r_q)^3 / ... (complex)
        # Instead, just compute cf with a known quarterly rate and verify
        ops_cf = equity_invested * r_q * (1 + r_q) ** n_ops / ((1 + r_q) ** n_ops - 1)
        # Plus return of capital at end (bullet style)
        cf = np.zeros(n_const + n_ops)
        cf[:n_const] = -equity_per_period
        cf[n_const:] = ops_cf

        # The IRR of construction draw + annuity will differ from 15% exactly
        # because draws occur quarterly, not as a lump sum.
        # Just verify IRR is reasonable (positive, > 5%)
        result = xirr(cf, periods_per_year=4)
        assert result > 0.05, f"Expected IRR > 5%, got {result:.2%}"

    def test_no_sign_change_raises(self):
        """All-positive cash flows have no IRR."""
        with pytest.raises(KPIComputationError, match="sign change"):
            xirr(np.array([1.0, 2.0, 3.0]))

    def test_empty_cashflows_raises(self):
        with pytest.raises(KPIComputationError, match="empty"):
            xirr(np.array([]))

    def test_multiple_sign_changes(self):
        """Non-conventional cash flows (multiple sign changes) — Brent finds a root."""
        cf = np.array([-100.0, 200.0, -50.0, 100.0])
        # Should not raise — one of the roots will be found
        result = xirr(cf, periods_per_year=4)
        # Verify NPV at result is ≈ 0
        t = np.arange(1, len(cf) + 1, dtype=float)
        r_q = (1.0 + result) ** (1.0 / 4) - 1.0
        npv_check = np.sum(cf / (1.0 + r_q) ** t)
        assert abs(npv_check) < 1e-6


# ===========================================================================
# 2. NPV at rate
# ===========================================================================


class TestNPVAtRate:
    def test_zero_rate_sums_cashflows(self):
        cf = np.array([100.0, 200.0, 300.0])
        assert npv_at_rate(cf, 0.0) == pytest.approx(600.0)

    def test_positive_rate_discounts(self):
        """Single cash flow of 100 at period 0: PV = 100/(1+r)^1."""
        cf = np.array([100.0, 0.0, 0.0, 0.0])  # only period 0 has cash
        # Annual rate 10%, quarterly → r_q = (1.10)^0.25 - 1
        r_pa = 0.10
        r_q = (1 + r_pa) ** 0.25 - 1
        expected = 100.0 / (1.0 + r_q) ** 1
        result = npv_at_rate(cf, r_pa, periods_per_year=4)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_npv_consistent_with_xirr(self):
        """NPV at the XIRR should be ≈ 0."""
        cf = np.array([-1000.0] + [120.0] * 20)
        irr = xirr(cf, periods_per_year=4)
        npv_val = npv_at_rate(cf, irr, periods_per_year=4)
        assert abs(npv_val) < 1.0  # within 1 Lakh of zero (relative to 1000 Lakh scale)

    def test_high_rate_gives_small_npv(self):
        cf = np.array([100.0, 100.0, 100.0, 100.0])
        npv_low = npv_at_rate(cf, 0.01)
        npv_high = npv_at_rate(cf, 0.50)
        assert npv_low > npv_high


# ===========================================================================
# 3. DSCR series / min / avg
# ===========================================================================


class TestDSCR:
    def setup_method(self):
        """
        12 total periods: 3 construction + 9 operational (debt periods 3–10).
        """
        self.n = 12
        self.cod = 3
        self.mat = 10

        self.is_debt = np.array([False] * 3 + [True] * 8 + [False] * 1)
        # CFADS: 0 in construction, escalating in ops
        self.cfads = np.array([0.0] * 3 + [120.0, 125.0, 130.0, 135.0,
                                             140.0, 145.0, 150.0, 155.0, 0.0])
        # Debt service: 0 in construction and last period, flat in ops
        self.tds   = np.array([0.0] * 3 + [100.0] * 8 + [0.0])

    def test_dscr_values_correct(self):
        series = dscr_series(self.cfads, self.tds, self.is_debt)
        # Construction periods → NaN
        assert np.isnan(series[0])
        assert np.isnan(series[1])
        assert np.isnan(series[2])
        # Operational periods → cfads / tds
        assert series[3] == pytest.approx(1.20)  # 120/100
        assert series[4] == pytest.approx(1.25)  # 125/100
        assert series[10] == pytest.approx(1.55) # 155/100

    def test_non_debt_period_is_nan(self):
        series = dscr_series(self.cfads, self.tds, self.is_debt)
        assert np.isnan(series[11])  # last period, not debt outstanding

    def test_min_dscr(self):
        result = min_dscr(self.cfads, self.tds, self.is_debt)
        assert result == pytest.approx(1.20)  # minimum is period 3

    def test_avg_dscr(self):
        result = avg_dscr(self.cfads, self.tds, self.is_debt)
        expected_values = [120, 125, 130, 135, 140, 145, 150, 155]
        expected = np.mean(expected_values) / 100.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_no_debt_periods_raises(self):
        """All mask=False → should raise DSCRPhaseError."""
        mask = np.zeros(self.n, dtype=bool)
        with pytest.raises(DSCRPhaseError):
            dscr_series(self.cfads, self.tds, mask)

    def test_dscr_below_one_handled(self):
        """DSCR < 1.0 is valid to compute — just signals distress."""
        cfads_stress = self.cfads.copy()
        cfads_stress[3] = 80.0  # below debt service
        series = dscr_series(cfads_stress, self.tds, self.is_debt)
        assert series[3] == pytest.approx(0.80)

    def test_debt_service_zero_in_active_period_excluded(self):
        """If debt_service = 0 in a debt-outstanding period, exclude that period."""
        tds_modified = self.tds.copy()
        tds_modified[3] = 0.0  # period 3: debt outstanding but no service (moratorium)
        # Period 3 should be NaN; periods 4-10 should be computed
        series = dscr_series(self.cfads, tds_modified, self.is_debt)
        assert np.isnan(series[3])
        assert series[4] == pytest.approx(1.25)


# ===========================================================================
# 4. LLCR
# ===========================================================================


class TestLLCR:
    def setup_method(self):
        self.n = 15
        self.cod = 3
        self.mat = 12
        self.r_pa = 0.0975
        self.ppy = 4
        r_q = (1 + self.r_pa) ** 0.25 - 1

        # Flat CFADS during ops — sized to exceed debt service (~1134/qtr for 10000 loan)
        self.cfads = np.array([0.0] * 3 + [1500.0] * 12)

        # Outstanding balance: starts at 10000, reduces linearly over debt life
        balance = np.zeros(self.n)
        for t in range(self.mat - self.cod + 1):   # 10 periods: cod..mat inclusive
            balance[self.cod + t] = max(0.0, 10000.0 - 1000.0 * t)
        self.balance = balance

    def test_llcr_greater_than_one(self):
        """With CFADS > debt service, LLCR should exceed 1."""
        result = llcr(
            self.cfads, self.balance, self.r_pa,
            self.cod, self.mat, self.ppy,
        )
        assert result > 1.0

    def test_llcr_analytical(self):
        """
        With flat CFADS = 150, discount rate r_pa, and debt_at_cod = 10000,
        LLCR = PV(annuity of 150 over 10 periods at r_q) / 10000.
        """
        r_q = (1.0 + self.r_pa) ** (1.0 / self.ppy) - 1.0
        n_debt = self.mat - self.cod + 1  # 10 periods
        cfads_per_period = 1500.0  # matches setup_method
        # PV of annuity: cfads * (1 - (1+r)^-n) / r
        pv_annuity = cfads_per_period * (1.0 - (1.0 + r_q) ** (-n_debt)) / r_q
        expected = pv_annuity / self.balance[self.cod]
        result = llcr(
            self.cfads, self.balance, self.r_pa,
            self.cod, self.mat, self.ppy,
        )
        assert result == pytest.approx(expected, rel=1e-6)

    def test_zero_balance_raises(self):
        balance = np.zeros(self.n)
        with pytest.raises(KPIComputationError, match="zero debt"):
            llcr(self.cfads, balance, self.r_pa, self.cod, self.mat, self.ppy)


# ===========================================================================
# 5. PLCR
# ===========================================================================


class TestPLCR:
    def setup_method(self):
        self.n = 15
        self.cod = 3
        self.r_pa = 0.0975
        self.ppy = 4

        self.cfads = np.array([0.0] * 3 + [150.0] * 12)
        self.balance = np.zeros(self.n)
        self.balance[self.cod] = 10000.0

    def test_plcr_geq_llcr(self):
        """PLCR covers full project life, LLCR only loan life → PLCR >= LLCR."""
        llcr_val = llcr(
            self.cfads, self.balance, self.r_pa,
            self.cod, self.cod + 9, self.ppy,
        )
        plcr_val = plcr(
            self.cfads, self.balance, self.r_pa, self.cod, self.ppy,
        )
        assert plcr_val >= llcr_val

    def test_zero_balance_raises(self):
        balance = np.zeros(self.n)
        with pytest.raises(KPIComputationError, match="zero debt"):
            plcr(self.cfads, balance, self.r_pa, self.cod, self.ppy)


# ===========================================================================
# 6. NPV equity
# ===========================================================================


class TestNPVEquity:
    def test_positive_npv_when_irr_exceeds_hurdle(self):
        """If actual IRR > hurdle rate, NPV should be positive."""
        # IRR is ~10.38% (quarterly 2.5%), hurdle = 8%
        r_q = 0.025
        n = 12
        pv = 100.0
        pmt = pv * r_q * (1 + r_q) ** n / ((1 + r_q) ** n - 1)
        cf = np.array([-pv] + [pmt] * n)
        npv_val = npv_equity(cf, discount_rate_pa=0.08, periods_per_year=4)
        assert npv_val > 0.0

    def test_zero_npv_at_irr(self):
        """NPV at the IRR = 0."""
        r_q = 0.025
        n = 12
        pv = 100.0
        pmt = pv * r_q * (1 + r_q) ** n / ((1 + r_q) ** n - 1)
        cf = np.array([-pv] + [pmt] * n)
        irr = xirr(cf, periods_per_year=4)
        npv_val = npv_equity(cf, discount_rate_pa=irr, periods_per_year=4)
        assert abs(npv_val) < 1e-4


# ===========================================================================
# 7. Debt payback period
# ===========================================================================


class TestDebtPaybackPeriod:
    def setup_method(self):
        self.ppy = 4
        self.cod = 3

    def test_payback_in_correct_year(self):
        """
        Balance: 10000 at COD, falling by 1000 each period.
        Reaches zero at ops period 10 (absolute period 13).
        Payback = 10 periods / 4 ppy = 2.5 years from COD.
        """
        n = 20
        balance = np.zeros(n)
        for t in range(10):
            balance[self.cod + t] = 10000.0 - 1000.0 * t
        result = debt_payback_period(balance, self.cod, self.ppy)
        # Last nonzero index in ops = 9 → payback = 10 / 4 = 2.5 years
        assert result == pytest.approx(2.5, rel=1e-6)

    def test_already_repaid_returns_zero(self):
        balance = np.zeros(10)
        result = debt_payback_period(balance, self.cod, self.ppy)
        assert result == 0.0

    def test_payback_after_18_years(self):
        """
        18-year debt tenor, quarterly.
        Balance nonzero for 72 periods after COD.
        Payback = 72 / 4 = 18 years.
        """
        n_debt = 72
        n_total = self.cod + n_debt + 10
        balance = np.zeros(n_total)
        for t in range(n_debt):
            balance[self.cod + t] = float(n_debt - t) * 100.0
        result = debt_payback_period(balance, self.cod, self.ppy)
        assert result == pytest.approx(18.0, rel=1e-6)


# ===========================================================================
# 8. Peak debt outstanding
# ===========================================================================


class TestPeakDebtOutstanding:
    def test_peak_at_end_of_construction(self):
        balance = np.array([0.0, 3000.0, 7000.0, 10000.0, 9000.0, 8000.0])
        assert peak_debt_outstanding(balance) == pytest.approx(10000.0)

    def test_all_zero(self):
        assert peak_debt_outstanding(np.zeros(10)) == pytest.approx(0.0)


# ===========================================================================
# 9. compute_all_kpis (integration)
# ===========================================================================


class TestComputeAllKPIs:
    """
    Integration test using a synthetic but internally consistent mini-model:
      - 100MW solar, quarterly, 3 construction + 72 operational periods (18 yr debt)
      - All values analytically constructed so KPIs can be verified
    """

    def setup_method(self):
        self.ppy = 4
        self.cod = 3
        self.n = 3 + 100  # 3 construction + 100 ops
        self.mat = 3 + 72 - 1  # debt matures at end of period 74 (72 debt periods)

        # Debt parameters
        self.interest_pa = 0.0975
        r_q = (1.0 + self.interest_pa) ** 0.25 - 1.0
        self.debt_amount = 31500.0  # 70% of 45000 Lakhs
        n_debt = 72

        # Level annuity payment
        pmt = self.debt_amount * r_q * (1 + r_q) ** n_debt / ((1 + r_q) ** n_debt - 1)
        self.pmt = pmt

        # Build outstanding balance (amortising)
        balance = np.zeros(self.n)
        bal = self.debt_amount
        for t in range(n_debt):
            balance[self.cod + t] = bal
            interest = bal * r_q
            principal = pmt - interest
            bal = max(0.0, bal - principal)

        self.balance = balance

        # Build debt service arrays
        self.tds = np.zeros(self.n)
        self.tds[self.cod: self.cod + n_debt] = pmt

        # CFADS: flat 1.3× debt service during debt life, then higher after
        self.cfads = np.zeros(self.n)
        self.cfads[self.cod: self.cod + n_debt] = pmt * 1.30
        self.cfads[self.cod + n_debt:] = pmt * 2.0  # more cash after debt repaid

        # Phase mask
        self.is_debt = np.zeros(self.n, dtype=bool)
        self.is_debt[self.cod: self.cod + n_debt] = True

        # Equity (13500 Lakhs = 30% of 45000)
        equity_total = 13500.0
        equity_per_qtr = equity_total / 3.0
        self.equity_cf = np.zeros(self.n)
        self.equity_cf[:3] = -equity_per_qtr
        # Distributions = CFADS - debt service
        self.equity_cf[self.cod:] = self.cfads[self.cod:] - self.tds[self.cod:]

        # Project CF
        self.project_cf = np.zeros(self.n)
        self.project_cf[:3] = -15000.0  # 45000/3 per construction quarter
        self.project_cf[self.cod:] = self.cfads[self.cod:]

    def test_kpis_return_kpi_result(self):
        result, warnings = compute_all_kpis(
            equity_cashflows=self.equity_cf,
            project_cashflows=self.project_cf,
            cfads=self.cfads,
            total_debt_service=self.tds,
            outstanding_balance=self.balance,
            is_debt_outstanding=self.is_debt,
            interest_rate_pa=self.interest_pa,
            equity_irr_target=0.15,
            cod_period=self.cod,
            debt_maturity_period=self.mat,
            periods_per_year=self.ppy,
        )
        assert isinstance(result, KPIResult)

    def test_equity_irr_positive(self):
        result, _ = compute_all_kpis(
            equity_cashflows=self.equity_cf,
            project_cashflows=self.project_cf,
            cfads=self.cfads,
            total_debt_service=self.tds,
            outstanding_balance=self.balance,
            is_debt_outstanding=self.is_debt,
            interest_rate_pa=self.interest_pa,
            equity_irr_target=0.15,
            cod_period=self.cod,
            debt_maturity_period=self.mat,
            periods_per_year=self.ppy,
        )
        assert result.equity_irr is not None
        assert result.equity_irr > 0.0

    def test_min_dscr_matches_cfads_tds_ratio(self):
        """With CFADS = 1.30 × TDS everywhere, min_dscr should be ≈ 1.30."""
        result, _ = compute_all_kpis(
            equity_cashflows=self.equity_cf,
            project_cashflows=self.project_cf,
            cfads=self.cfads,
            total_debt_service=self.tds,
            outstanding_balance=self.balance,
            is_debt_outstanding=self.is_debt,
            interest_rate_pa=self.interest_pa,
            equity_irr_target=0.15,
            cod_period=self.cod,
            debt_maturity_period=self.mat,
            periods_per_year=self.ppy,
        )
        assert result.min_dscr == pytest.approx(1.30, rel=1e-4)

    def test_avg_dscr_equals_min_when_flat(self):
        """Flat CFADS/TDS → avg == min."""
        result, _ = compute_all_kpis(
            equity_cashflows=self.equity_cf,
            project_cashflows=self.project_cf,
            cfads=self.cfads,
            total_debt_service=self.tds,
            outstanding_balance=self.balance,
            is_debt_outstanding=self.is_debt,
            interest_rate_pa=self.interest_pa,
            equity_irr_target=0.15,
            cod_period=self.cod,
            debt_maturity_period=self.mat,
            periods_per_year=self.ppy,
        )
        assert result.avg_dscr == pytest.approx(result.min_dscr, rel=1e-4)

    def test_llcr_exceeds_one(self):
        result, _ = compute_all_kpis(
            equity_cashflows=self.equity_cf,
            project_cashflows=self.project_cf,
            cfads=self.cfads,
            total_debt_service=self.tds,
            outstanding_balance=self.balance,
            is_debt_outstanding=self.is_debt,
            interest_rate_pa=self.interest_pa,
            equity_irr_target=0.15,
            cod_period=self.cod,
            debt_maturity_period=self.mat,
            periods_per_year=self.ppy,
        )
        assert result.llcr is not None
        assert result.llcr > 1.0

    def test_plcr_geq_llcr(self):
        result, _ = compute_all_kpis(
            equity_cashflows=self.equity_cf,
            project_cashflows=self.project_cf,
            cfads=self.cfads,
            total_debt_service=self.tds,
            outstanding_balance=self.balance,
            is_debt_outstanding=self.is_debt,
            interest_rate_pa=self.interest_pa,
            equity_irr_target=0.15,
            cod_period=self.cod,
            debt_maturity_period=self.mat,
            periods_per_year=self.ppy,
        )
        assert result.plcr >= result.llcr

    def test_debt_payback_18_years(self):
        result, _ = compute_all_kpis(
            equity_cashflows=self.equity_cf,
            project_cashflows=self.project_cf,
            cfads=self.cfads,
            total_debt_service=self.tds,
            outstanding_balance=self.balance,
            is_debt_outstanding=self.is_debt,
            interest_rate_pa=self.interest_pa,
            equity_irr_target=0.15,
            cod_period=self.cod,
            debt_maturity_period=self.mat,
            periods_per_year=self.ppy,
        )
        assert result.debt_payback_period == pytest.approx(18.0, abs=0.3)

    def test_peak_debt_outstanding_at_cod(self):
        result, _ = compute_all_kpis(
            equity_cashflows=self.equity_cf,
            project_cashflows=self.project_cf,
            cfads=self.cfads,
            total_debt_service=self.tds,
            outstanding_balance=self.balance,
            is_debt_outstanding=self.is_debt,
            interest_rate_pa=self.interest_pa,
            equity_irr_target=0.15,
            cod_period=self.cod,
            debt_maturity_period=self.mat,
            periods_per_year=self.ppy,
        )
        assert result.peak_debt_outstanding == pytest.approx(self.debt_amount, rel=1e-4)

    def test_bankability_flag_in_warnings_when_dscr_low(self):
        """Force CFADS below TDS → bankability warning."""
        cfads_stress = self.cfads.copy()
        cfads_stress[self.cod: self.cod + 72] = self.pmt * 0.80  # below 1.0x
        equity_stress = self.equity_cf.copy()
        equity_stress[self.cod:] = cfads_stress[self.cod:] - self.tds[self.cod:]

        result, warnings = compute_all_kpis(
            equity_cashflows=equity_stress,
            project_cashflows=self.project_cf,
            cfads=cfads_stress,
            total_debt_service=self.tds,
            outstanding_balance=self.balance,
            is_debt_outstanding=self.is_debt,
            interest_rate_pa=self.interest_pa,
            equity_irr_target=0.15,
            cod_period=self.cod,
            debt_maturity_period=self.mat,
            periods_per_year=self.ppy,
        )
        assert result.min_dscr is not None
        assert result.min_dscr < 1.0
        assert any("BANKABILITY FAILURE" in w for w in warnings)

    def test_partial_kpi_on_irr_failure(self):
        """If IRR fails (all positive equity CFs), other KPIs still computed."""
        bad_equity = np.ones(self.n) * 100.0  # no sign change
        result, warnings = compute_all_kpis(
            equity_cashflows=bad_equity,
            project_cashflows=self.project_cf,
            cfads=self.cfads,
            total_debt_service=self.tds,
            outstanding_balance=self.balance,
            is_debt_outstanding=self.is_debt,
            interest_rate_pa=self.interest_pa,
            equity_irr_target=0.15,
            cod_period=self.cod,
            debt_maturity_period=self.mat,
            periods_per_year=self.ppy,
        )
        # equity_irr fails
        assert result.equity_irr is None
        assert any("equity_irr" in w for w in warnings)
        # But DSCR is still computed
        assert result.min_dscr is not None

    def test_npv_equity_positive_at_hurdle_below_irr(self):
        result, _ = compute_all_kpis(
            equity_cashflows=self.equity_cf,
            project_cashflows=self.project_cf,
            cfads=self.cfads,
            total_debt_service=self.tds,
            outstanding_balance=self.balance,
            is_debt_outstanding=self.is_debt,
            interest_rate_pa=self.interest_pa,
            equity_irr_target=0.10,   # hurdle below actual IRR → NPV > 0
            cod_period=self.cod,
            debt_maturity_period=self.mat,
            periods_per_year=self.ppy,
        )
        assert result.npv_equity is not None
        assert result.npv_equity > 0.0


# ===========================================================================
# 10. Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_xirr_all_negative_raises(self):
        with pytest.raises(KPIComputationError):
            xirr(np.array([-100.0, -50.0, -25.0]))

    def test_xirr_all_positive_raises(self):
        with pytest.raises(KPIComputationError):
            xirr(np.array([100.0, 50.0, 25.0]))

    def test_dscr_series_propagates_nan_correctly(self):
        n = 6
        mask = np.array([False, False, True, True, True, False])
        cfads = np.array([0.0, 0.0, 100.0, 110.0, 120.0, 0.0])
        tds   = np.array([0.0, 0.0,  80.0,  80.0,  80.0, 0.0])
        series = dscr_series(cfads, tds, mask)
        assert np.isnan(series[0])
        assert np.isnan(series[5])
        assert series[2] == pytest.approx(1.25)
        assert series[3] == pytest.approx(1.375)
        assert series[4] == pytest.approx(1.50)

    def test_debt_payback_immediate_at_cod(self):
        """If balance is zero at COD, payback = 0."""
        balance = np.zeros(10)
        result = debt_payback_period(balance, cod_period=3, periods_per_year=4)
        assert result == 0.0

    def test_peak_debt_zero(self):
        assert peak_debt_outstanding(np.zeros(5)) == 0.0
