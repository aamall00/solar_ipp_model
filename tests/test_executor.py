"""
Unit and integration tests for engine/executor.py

Test hierarchy
--------------
1.  TestCompile            — compile() builds a valid CompiledModel
2.  TestNamespaceInit      — _init_namespace injects assumptions & phases correctly
3.  TestStandardBlock      — _execute_standard computes body steps in order
4.  TestWaterfallBlock     — _execute_waterfall allocates buckets by priority
5.  TestLedgerBlock        — _execute_ledger walks sequential balance
6.  TestGoalSeekLoop       — _run_goal_seek converges to target
7.  TestFixedPointLoop     — _run_fixed_point converges to fixed-point
8.  TestRunEnd2End         — executor.run() on a 3-block model produces correct KPIs
9.  TestRunBatch           — run_batch() returns correct list of results
10. TestDownstreamVariables— downstream_variables() traces dependency graph
11. TestAuditTrail         — audit trail records every output
12. TestConvergenceMetadata— convergence list is populated after solve loops
13. TestStubs              — run_sensitivity / run_monte_carlo raise NotImplementedError

Run with: pytest tests/test_executor.py -v
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from dsl.types import (
    AssumptionDefinition,
    AssumptionSchema,
    AssumptionType,
    BlockCategory,
    BlockInput,
    BlockOutput,
    BlockType,
    CalculationBlock,
    CalculationBlocks,
    CalculationStep,
    LedgerFlow,
    Milestones,
    ModelDefinition,
    ModelWiring,
    Phases,
    ProjectSkeleton,
    ProjectType,
    SolveLoop,
    SolveLoopType,
    WaterfallBucket,
)
from engine.executor import (
    AuditEntry,
    CompiledModel,
    ModelExecutor,
    ModelResults,
)
from engine.solvers import ModelConvergenceError


# ===========================================================================
# Helpers — minimal model factories
# ===========================================================================

N_PERIODS = 12     # 3 years at quarterly
PPY = 4
COD = 4            # 4 construction quarters → cod at period 4
MAT = 11           # debt matures at period 11 (last period index for n=12 model)


def _phases(n: int, cod: int, mat: int) -> Phases:
    return Phases(
        is_construction=[t < cod for t in range(n)],
        is_operational=[t >= cod for t in range(n)],
        is_debt_outstanding=[cod <= t <= mat for t in range(n)],
    )


def _skeleton(n_construction: int = COD, n_ops: int = N_PERIODS - COD) -> ProjectSkeleton:
    total = n_construction + n_ops
    mat = total - 1
    return ProjectSkeleton(
        model_id="test_model",
        project_type=ProjectType.solar_ipp,
        periods_per_year=PPY,
        construction_periods=n_construction,
        operations_periods=n_ops,
        milestones=Milestones(
            financial_close=0,
            cod=n_construction,
            debt_maturity=mat,
        ),
        phases=_phases(total, n_construction, mat),
    )


def _assumption(name: str, value: float) -> AssumptionDefinition:
    return AssumptionDefinition(
        name=name,
        type=AssumptionType.scalar,
        unit="ratio",
        value=value,
    )


def _simple_model() -> ModelDefinition:
    """
    Minimal 2-block model (no solve loops):
      - block_a: computes 'doubled = x * 2'  (x from assumption.x_val)
      - block_b: computes 'tripled = doubled * 3'  (doubled from block_a)

    All outputs are scalars broadcast to n_periods arrays by the evaluator.
    """
    block_a = CalculationBlock(
        block_id="block_a",
        category=BlockCategory.revenue,
        block_type=BlockType.standard,
        inputs=[BlockInput(name="x", source="assumption.x_val")],
        outputs=[BlockOutput(name="doubled", unit="")],
        body=[CalculationStep(target="doubled", expr="x * 2")],
    )
    block_b = CalculationBlock(
        block_id="block_b",
        category=BlockCategory.cashflow,
        block_type=BlockType.standard,
        inputs=[BlockInput(name="doubled", source="block_a.doubled")],
        outputs=[BlockOutput(name="tripled", unit="")],
        body=[CalculationStep(target="tripled", expr="doubled * 3")],
    )
    return ModelDefinition(
        project_skeleton=_skeleton(),
        assumption_schema=AssumptionSchema(
            assumptions=[_assumption("x_val", 5.0)]
        ),
        calculation_blocks=CalculationBlocks(blocks=[block_a, block_b]),
        model_wiring=ModelWiring(),
    )


def _waterfall_model() -> ModelDefinition:
    """
    Waterfall model: 'cash' input split into 'ops_cost' (priority 1) and 'reserve' (priority 2),
    with any residual going to equity.
    """
    wf_block = CalculationBlock(
        block_id="wf_block",
        category=BlockCategory.waterfall,
        block_type=BlockType.waterfall,
        inputs=[BlockInput(name="cash_in", source="assumption.cash_in")],
        outputs=[
            BlockOutput(name="ops_cost", unit=""),
            BlockOutput(name="reserve", unit=""),
        ],
        available_cash_input="assumption.cash_in",
        buckets=[
            WaterfallBucket(
                name="ops_cost",
                priority=1,
                target_expr="100.0",
                recipient="ops_block",
            ),
            WaterfallBucket(
                name="reserve",
                priority=2,
                target_expr="50.0",
                recipient="reserve_block",
            ),
        ],
    )
    return ModelDefinition(
        project_skeleton=_skeleton(),
        assumption_schema=AssumptionSchema(
            assumptions=[_assumption("cash_in", 200.0)]
        ),
        calculation_blocks=CalculationBlocks(blocks=[wf_block]),
        model_wiring=ModelWiring(),
    )


def _ledger_model() -> ModelDefinition:
    """
    Ledger model: reserve account starting at 0, additions from 'assumption.add',
    deductions from 'assumption.ded'.
    """
    # We need a dummy source block to provide the addition/deduction arrays
    add_block = CalculationBlock(
        block_id="add_block",
        category=BlockCategory.dsra,
        block_type=BlockType.standard,
        inputs=[BlockInput(name="add_val", source="assumption.add_val")],
        outputs=[BlockOutput(name="addition", unit="")],
        body=[CalculationStep(target="addition", expr="add_val")],
    )
    ded_block = CalculationBlock(
        block_id="ded_block",
        category=BlockCategory.dsra,
        block_type=BlockType.standard,
        inputs=[BlockInput(name="ded_val", source="assumption.ded_val")],
        outputs=[BlockOutput(name="deduction", unit="")],
        body=[CalculationStep(target="deduction", expr="ded_val")],
    )
    ledger_block = CalculationBlock(
        block_id="reserve_ledger",
        category=BlockCategory.dsra,
        block_type=BlockType.ledger,
        inputs=[],
        outputs=[BlockOutput(name="closing_balance", unit="")],
        opening_balance=1000.0,
        additions=[LedgerFlow(source="add_block.addition")],
        deductions=[LedgerFlow(source="ded_block.deduction")],
    )
    return ModelDefinition(
        project_skeleton=_skeleton(),
        assumption_schema=AssumptionSchema(
            assumptions=[
                _assumption("add_val", 100.0),
                _assumption("ded_val", 50.0),
            ]
        ),
        calculation_blocks=CalculationBlocks(
            blocks=[add_block, ded_block, ledger_block]
        ),
        model_wiring=ModelWiring(),
    )


def _goal_seek_model() -> ModelDefinition:
    """
    Goal-seek model: find scalar 'target_x' such that 'target_x * multiplier == 100'.
    free_variable = 'assumption.target_x'
    """
    block = CalculationBlock(
        block_id="gs_block",
        category=BlockCategory.revenue,
        block_type=BlockType.standard,
        inputs=[
            BlockInput(name="x", source="assumption.target_x"),
            BlockInput(name="m", source="assumption.multiplier"),
        ],
        outputs=[BlockOutput(name="result", unit="")],
        body=[CalculationStep(target="result", expr="x * m")],
    )
    loop = SolveLoop(
        loop_id="gs_loop",
        type=SolveLoopType.goal_seek,
        free_variable="assumption.target_x",
        target_expression="gs_block.result",
        target_value=100.0,
        tolerance=1e-8,
        max_iterations=50,
    )
    return ModelDefinition(
        project_skeleton=_skeleton(),
        assumption_schema=AssumptionSchema(
            assumptions=[
                _assumption("target_x", 1.0),    # initial guess
                _assumption("multiplier", 4.0),
            ]
        ),
        calculation_blocks=CalculationBlocks(blocks=[block], solve_loops=[loop]),
        model_wiring=ModelWiring(),
    )


def _fixed_point_model() -> ModelDefinition:
    """
    Fixed-point model: x = 0.5 * x + 30.
    Fixed point is x* = 60.
    free_variable = 'assumption.fp_x', target_expression = '0.5 * fp_x + 30.0'
    """
    block = CalculationBlock(
        block_id="fp_block",
        category=BlockCategory.revenue,
        block_type=BlockType.standard,
        inputs=[BlockInput(name="fp_x", source="assumption.fp_x")],
        outputs=[BlockOutput(name="next_x", unit="")],
        body=[CalculationStep(target="next_x", expr="0.5 * fp_x + 30.0")],
    )
    loop = SolveLoop(
        loop_id="fp_loop",
        type=SolveLoopType.fixed_point,
        free_variable="assumption.fp_x",
        target_expression="fp_block.next_x",
        target_value=0.0,    # not used for fixed_point
        tolerance=1e-8,
        max_iterations=50,
    )
    return ModelDefinition(
        project_skeleton=_skeleton(),
        assumption_schema=AssumptionSchema(
            assumptions=[_assumption("fp_x", 10.0)]
        ),
        calculation_blocks=CalculationBlocks(blocks=[block], solve_loops=[loop]),
        model_wiring=ModelWiring(),
    )


# ===========================================================================
# 1. TestCompile
# ===========================================================================


class TestCompile:
    def setup_method(self):
        self.executor = ModelExecutor()
        self.model = _simple_model()

    def test_returns_compiled_model(self):
        cm = self.executor.compile(self.model)
        assert isinstance(cm, CompiledModel)

    def test_n_periods_correct(self):
        cm = self.executor.compile(self.model)
        assert cm.n_periods == N_PERIODS

    def test_cod_period_correct(self):
        cm = self.executor.compile(self.model)
        assert cm.cod_period == COD

    def test_evaluation_plan_nonempty(self):
        cm = self.executor.compile(self.model)
        assert len(cm.evaluation_plan) > 0

    def test_evaluation_plan_block_kinds(self):
        cm = self.executor.compile(self.model)
        kinds = {step.kind for step in cm.evaluation_plan}
        assert "block" in kinds

    def test_dependency_graph_has_both_blocks(self):
        cm = self.executor.compile(self.model)
        assert "block_a" in cm.dependency_graph.nodes
        assert "block_b" in cm.dependency_graph.nodes

    def test_dependency_graph_edge_direction(self):
        cm = self.executor.compile(self.model)
        # block_a → block_b (block_b depends on block_a)
        assert cm.dependency_graph.has_edge("block_a", "block_b")

    def test_output_graph_contains_outputs(self):
        cm = self.executor.compile(self.model)
        assert "block_a.doubled" in cm.output_graph.nodes
        assert "block_b.tripled" in cm.output_graph.nodes

    def test_compile_from_dict_raises_on_invalid(self):
        """compile() called with an invalid dict raises ValueError."""
        with pytest.raises(ValueError, match="invalid"):
            self.executor.compile({"project_skeleton": {}})

    def test_topo_order_deps_before_dependents(self):
        """block_a must appear before block_b in the evaluation plan."""
        cm = self.executor.compile(self.model)
        block_steps = [s for s in cm.evaluation_plan if s.kind == "block"]
        block_ids = [s.block.block_id for s in block_steps]
        assert block_ids.index("block_a") < block_ids.index("block_b")


# ===========================================================================
# 2. TestNamespaceInit
# ===========================================================================


class TestNamespaceInit:
    def setup_method(self):
        self.executor = ModelExecutor()
        self.model = _simple_model()
        self.compiled = self.executor.compile(self.model)

    def test_phase_masks_injected(self):
        ns, _ = self.executor._init_namespace(self.compiled, {})
        assert "phase.is_construction" in ns
        assert "phase.is_operational" in ns
        assert "phase.is_debt_outstanding" in ns

    def test_phase_construction_mask_length(self):
        ns, _ = self.executor._init_namespace(self.compiled, {})
        assert len(ns["phase.is_construction"]) == N_PERIODS

    def test_phase_construction_true_before_cod(self):
        ns, _ = self.executor._init_namespace(self.compiled, {})
        assert all(ns["phase.is_construction"][:COD] == 1.0)
        assert all(ns["phase.is_construction"][COD:] == 0.0)

    def test_assumption_default_value_used(self):
        ns, eff = self.executor._init_namespace(self.compiled, {})
        assert eff["x_val"] == pytest.approx(5.0)
        assert ns["assumption.x_val"] == pytest.approx(5.0)

    def test_assumption_override_takes_priority(self):
        ns, eff = self.executor._init_namespace(self.compiled, {"x_val": 99.0})
        assert eff["x_val"] == pytest.approx(99.0)
        assert ns["assumption.x_val"] == pytest.approx(99.0)

    def test_missing_assumption_defaults_to_zero(self):
        """An assumption with no value and no default_rule resolves to 0."""
        model2 = copy.deepcopy(self.model)
        model2.assumption_schema.assumptions[0].value = None
        compiled2 = self.executor.compile(model2)
        ns, eff = self.executor._init_namespace(compiled2, {})
        assert eff["x_val"] == pytest.approx(0.0)


# ===========================================================================
# 3. TestStandardBlock
# ===========================================================================


class TestStandardBlock:
    def setup_method(self):
        self.executor = ModelExecutor()
        self.model = _simple_model()
        self.compiled = self.executor.compile(self.model)

    def test_outputs_in_namespace_after_run(self):
        ns, _ = self.executor._init_namespace(self.compiled, {})
        audit: list = []
        block_a = self.compiled.model_def.calculation_blocks.blocks[0]
        self.executor._execute_standard(block_a, self.compiled, ns, audit)
        assert "block_a.doubled" in ns

    def test_doubled_value_correct(self):
        ns, _ = self.executor._init_namespace(self.compiled, {})
        audit: list = []
        block_a = self.compiled.model_def.calculation_blocks.blocks[0]
        self.executor._execute_standard(block_a, self.compiled, ns, audit)
        expected = np.full(N_PERIODS, 10.0)   # x_val=5, doubled=10
        np.testing.assert_allclose(ns["block_a.doubled"], expected)

    def test_chained_computation_correct(self):
        """block_b.tripled = block_a.doubled * 3 = 5*2*3 = 30 for all periods."""
        results = self.executor.run(self.compiled)
        expected = np.full(N_PERIODS, 30.0)
        np.testing.assert_allclose(results.variables["block_b.tripled"], expected)

    def test_standard_block_audit_entry(self):
        ns, _ = self.executor._init_namespace(self.compiled, {})
        audit: list = []
        block_a = self.compiled.model_def.calculation_blocks.blocks[0]
        self.executor._execute_standard(block_a, self.compiled, ns, audit)
        assert any(e.variable == "block_a.doubled" for e in audit)

    def test_assumption_override_propagates(self):
        """Override x_val=7 → doubled=14 → tripled=42."""
        results = self.executor.run(self.compiled, {"x_val": 7.0})
        expected = np.full(N_PERIODS, 42.0)
        np.testing.assert_allclose(results.variables["block_b.tripled"], expected)


# ===========================================================================
# 4. TestWaterfallBlock
# ===========================================================================


class TestWaterfallBlock:
    def setup_method(self):
        self.executor = ModelExecutor()
        self.model = _waterfall_model()
        self.compiled = self.executor.compile(self.model)

    def _run(self, cash_in: float = 200.0):
        return self.executor.run(self.compiled, {"cash_in": cash_in})

    def test_ops_cost_allocated_first(self):
        results = self._run(200.0)
        np.testing.assert_allclose(results.variables["wf_block.ops_cost"],
                                   np.full(N_PERIODS, 100.0))

    def test_reserve_allocated_second(self):
        results = self._run(200.0)
        np.testing.assert_allclose(results.variables["wf_block.reserve"],
                                   np.full(N_PERIODS, 50.0))

    def test_residual_is_remainder(self):
        results = self._run(200.0)
        # 200 - 100 - 50 = 50 residual
        np.testing.assert_allclose(results.variables["wf_block.residual"],
                                   np.full(N_PERIODS, 50.0))

    def test_insufficient_cash_ops_gets_available(self):
        """When only 80 cash, ops_cost (target 100) gets 80, reserve gets 0."""
        results = self._run(80.0)
        np.testing.assert_allclose(results.variables["wf_block.ops_cost"],
                                   np.full(N_PERIODS, 80.0))
        np.testing.assert_allclose(results.variables["wf_block.reserve"],
                                   np.zeros(N_PERIODS))

    def test_residual_never_negative(self):
        results = self._run(10.0)
        assert np.all(results.variables["wf_block.residual"] >= 0.0)

    def test_equity_distribution_equals_residual(self):
        results = self._run(200.0)
        np.testing.assert_allclose(
            results.variables["wf_block.equity_distribution"],
            results.variables["wf_block.residual"],
        )


# ===========================================================================
# 5. TestLedgerBlock
# ===========================================================================


class TestLedgerBlock:
    def setup_method(self):
        self.executor = ModelExecutor()
        self.model = _ledger_model()
        self.compiled = self.executor.compile(self.model)

    def _run(self, add: float = 100.0, ded: float = 50.0):
        return self.executor.run(self.compiled, {"add_val": add, "ded_val": ded})

    def test_opening_balance_period0_is_initial(self):
        results = self._run()
        assert results.variables["reserve_ledger.opening_balance"][0] == pytest.approx(1000.0)

    def test_closing_balance_grows_by_net(self):
        """With add=100, ded=50: each period net=+50 → balance increases."""
        results = self._run(100.0, 50.0)
        cb = results.variables["reserve_ledger.closing_balance"]
        # period 0: 1000 + 100 - 50 = 1050
        assert cb[0] == pytest.approx(1050.0)
        # period 1: 1050 + 100 - 50 = 1100
        assert cb[1] == pytest.approx(1100.0)

    def test_closing_balance_never_negative(self):
        """Even with deductions > additions + opening, balance floors at 0."""
        results = self._run(add=0.0, ded=9999.0)
        assert np.all(results.variables["reserve_ledger.closing_balance"] >= 0.0)

    def test_opening_equals_prior_closing(self):
        results = self._run()
        opening = results.variables["reserve_ledger.opening_balance"]
        closing = results.variables["reserve_ledger.closing_balance"]
        np.testing.assert_allclose(opening[1:], closing[:-1])

    def test_net_movement_sign_correct(self):
        """With add=100, ded=50, net_movement should be +50 always."""
        results = self._run(100.0, 50.0)
        nm = results.variables["reserve_ledger.net_movement"]
        np.testing.assert_allclose(nm, np.full(N_PERIODS, 50.0))

    def test_ledger_audit_recorded(self):
        results = self.executor.run(self.compiled)
        audit_vars = {e.variable for e in results.audit_trail}
        assert "reserve_ledger.opening_balance" in audit_vars
        assert "reserve_ledger.closing_balance" in audit_vars


# ===========================================================================
# 6. TestGoalSeekLoop
# ===========================================================================


class TestGoalSeekLoop:
    def setup_method(self):
        self.executor = ModelExecutor()
        self.model = _goal_seek_model()
        self.compiled = self.executor.compile(self.model)

    def test_goal_seek_converges_to_correct_value(self):
        """target_x * 4 == 100 → target_x = 25."""
        results = self.executor.run(self.compiled)
        # After solve, assumption.target_x should be stored in assumptions_used
        assert results.assumptions_used.get("target_x") == pytest.approx(1.0)
        # The block result should equal 100
        block_result = results.variables["gs_block.result"]
        np.testing.assert_allclose(block_result, np.full(N_PERIODS, 100.0), atol=1e-4)

    def test_convergence_metadata_recorded(self):
        results = self.executor.run(self.compiled)
        assert len(results.convergence) == 1
        conv = results.convergence[0]
        assert conv.loop_id == "gs_loop"
        assert conv.converged is True

    def test_convergence_residual_small(self):
        results = self.executor.run(self.compiled)
        conv = results.convergence[0]
        assert abs(conv.final_residual) < 1e-4

    def test_solve_result_in_variables(self):
        results = self.executor.run(self.compiled)
        assert "gs_block.result" in results.variables

    def test_different_multiplier_same_target(self):
        """multiplier=5 → target_x should be 20."""
        results = self.executor.run(self.compiled, {"multiplier": 5.0, "target_x": 1.0})
        block_result = results.variables["gs_block.result"]
        np.testing.assert_allclose(block_result, np.full(N_PERIODS, 100.0), atol=1e-4)


# ===========================================================================
# 7. TestFixedPointLoop
# ===========================================================================


class TestFixedPointLoop:
    def setup_method(self):
        self.executor = ModelExecutor()
        self.model = _fixed_point_model()
        self.compiled = self.executor.compile(self.model)

    def test_fixed_point_converges(self):
        """x = 0.5*x + 30 → x* = 60."""
        results = self.executor.run(self.compiled)
        assert len(results.convergence) == 1
        conv = results.convergence[0]
        assert conv.converged is True

    def test_fixed_point_block_output_near_60(self):
        """After convergence, next_x should ≈ 60 for all periods."""
        results = self.executor.run(self.compiled)
        np.testing.assert_allclose(
            results.variables["fp_block.next_x"],
            np.full(N_PERIODS, 60.0),
            atol=1e-4,
        )

    def test_fixed_point_loop_id_in_convergence(self):
        results = self.executor.run(self.compiled)
        assert results.convergence[0].loop_id == "fp_loop"

    def test_fixed_point_residual_small(self):
        results = self.executor.run(self.compiled)
        assert abs(results.convergence[0].final_residual) < 1e-4


# ===========================================================================
# 8. TestIDCForwardMarch
# ===========================================================================


class TestIDCForwardMarch:
    """
    Validates _run_array_fixed_point (idc_capitalised=1) using average-balance
    convention: idc[t] = (opening + closing) / 2 × r_q.

    Setup:
      capex = 45,000 INR Lakhs, capex_schedule = [30%, 40%, 30%] over 3 quarters
      debt_pct = 0.70, interest_rate = 9.75% pa → r_q = 0.024375

    Period-0 closed-form (D_prev=0, average-balance):
      divisor      = 1 − debt_pct × r_q / 2 ≈ 0.991469
      total_capex  = capex[0] / divisor ≈ 13616.3
      IDC[0]       = total_capex − capex[0] ≈ 116.3
      drawdown[0]  = debt_pct × total_capex ≈ 9531.4
    """

    def setup_method(self):
        self.executor = ModelExecutor()
        self.ppy = 4
        self.n_construction = 3
        self.n_ops = 20
        self.n = self.n_construction + self.n_ops
        self.capex_total = 45000.0
        self.debt_pct = 0.70
        self.r_q = 0.0975 / 4
        self.capex_schedule = [0.30, 0.40, 0.30] + [0.0] * self.n_ops
        self.capex_draws = [self.capex_total * f for f in self.capex_schedule[:self.n_construction]]

        cod = self.n_construction
        mat = self.n - 1

        skeleton = ProjectSkeleton(
            model_id="idc_fm_test",
            project_type=ProjectType.solar_ipp,
            periods_per_year=self.ppy,
            construction_periods=self.n_construction,
            operations_periods=self.n_ops,
            milestones=Milestones(financial_close=0, cod=cod, debt_maturity=mat),
            phases=_phases(self.n, cod, mat),
        )

        construction_block = CalculationBlock(
            block_id="construction_block",
            category=BlockCategory.construction,
            inputs=[
                BlockInput(name="capex_per_mw",    source="assumption.capex_per_mw"),
                BlockInput(name="capacity_mw",     source="assumption.capacity_mw"),
                BlockInput(name="capex_schedule",  source="assumption.capex_schedule"),
                BlockInput(name="is_construction", source="phase.is_construction"),
            ],
            outputs=[
                BlockOutput(name="capex_drawdown",  unit="INR_Lakhs"),
                BlockOutput(name="cumulative_capex", unit="INR_Lakhs"),
            ],
            body=[
                CalculationStep(target="capex_drawdown",
                                expr="capex_per_mw * capacity_mw * capex_schedule * is_construction"),
                CalculationStep(target="cumulative_capex",
                                expr="cumsum(capex_drawdown)"),
            ],
        )

        # idc_block: body IS the executable formula, re-evaluated each iteration
        idc_block = CalculationBlock(
            block_id="idc_block",
            category=BlockCategory.idc,
            inputs=[
                BlockInput(name="cumulative_drawdown",
                           source="debt_drawdown_block.cumulative_drawdown"),
                BlockInput(name="interest_rate",    source="assumption.interest_rate"),
                BlockInput(name="is_construction",  source="phase.is_construction"),
            ],
            outputs=[
                BlockOutput(name="idc_per_period", unit="INR_Lakhs"),
                BlockOutput(name="cumulative_idc",  unit="INR_Lakhs"),
                BlockOutput(name="idc_total",       unit="INR_Lakhs"),
            ],
            body=[
                CalculationStep(target="r_simple",
                                expr="scalar_to_series(interest_rate / 4.0)"),
                CalculationStep(target="idc_per_period",
                                expr="(lag(cumulative_drawdown, 1) + cumulative_drawdown)"
                                     " / 2.0 * r_simple * is_construction"),
                CalculationStep(target="cumulative_idc", expr="cumsum(idc_per_period)"),
                CalculationStep(target="idc_total",
                                expr="scalar_to_series(max(cumulative_idc))"),
            ],
        )

        debt_drawdown_block = CalculationBlock(
            block_id="debt_drawdown_block",
            category=BlockCategory.debt_drawdown,
            inputs=[
                BlockInput(name="debt_amount",     source="debt_sizing_block.debt_amount"),
                BlockInput(name="capex_schedule",  source="assumption.capex_schedule"),
                BlockInput(name="is_construction", source="phase.is_construction"),
            ],
            outputs=[
                BlockOutput(name="drawdown",            unit="INR_Lakhs"),
                BlockOutput(name="cumulative_drawdown", unit="INR_Lakhs"),
            ],
            body=[
                CalculationStep(target="drawdown",
                                expr="debt_amount * capex_schedule * is_construction"),
                CalculationStep(target="cumulative_drawdown", expr="cumsum(drawdown)"),
            ],
        )

        debt_sizing_block = CalculationBlock(
            block_id="debt_sizing_block",
            category=BlockCategory.debt_sizing,
            inputs=[
                BlockInput(name="capex_per_mw", source="assumption.capex_per_mw"),
                BlockInput(name="capacity_mw",  source="assumption.capacity_mw"),
                BlockInput(name="debt_pct",     source="assumption.debt_pct"),
                BlockInput(name="idc_total",    source="idc_block.idc_total"),
            ],
            outputs=[
                BlockOutput(name="total_project_cost", unit="INR_Lakhs"),
                BlockOutput(name="debt_amount",        unit="INR_Lakhs"),
                BlockOutput(name="equity_amount",      unit="INR_Lakhs"),
            ],
            body=[
                CalculationStep(
                    target="base_capex",
                    expr="scalar_to_series(capex_per_mw * capacity_mw)"),
                CalculationStep(
                    target="total_project_cost",
                    expr="scalar_to_series(max(base_capex) + max(idc_total))"),
                CalculationStep(
                    target="debt_amount",
                    expr="total_project_cost * debt_pct"),
                CalculationStep(
                    target="equity_amount",
                    expr="total_project_cost * (1.0 - debt_pct)"),
            ],
        )

        idc_loop = SolveLoop(
            loop_id="idc_capitalisation",
            type=SolveLoopType.array_fixed_point,
            free_variable="idc_block.idc_per_period",
            target_expression="idc_block.idc_per_period",
            target_value=0.0,
            owned_blocks=["debt_sizing_block", "debt_drawdown_block", "idc_block"],
        )

        schema = AssumptionSchema(assumptions=[
            AssumptionDefinition(name="capex_per_mw",   type=AssumptionType.scalar,
                                 unit="INR_Lakhs_per_MW", value=450.0),
            AssumptionDefinition(name="capacity_mw",    type=AssumptionType.scalar,
                                 unit="MW", value=100.0),
            AssumptionDefinition(name="debt_pct",       type=AssumptionType.scalar,
                                 unit="ratio", value=self.debt_pct),
            AssumptionDefinition(name="interest_rate",  type=AssumptionType.scalar,
                                 unit="ratio", value=0.0975),
            AssumptionDefinition(name="capex_schedule", type=AssumptionType.schedule,
                                 unit="fraction", value=self.capex_schedule),
        ])

        model = ModelDefinition(
            project_skeleton=skeleton,
            assumption_schema=schema,
            calculation_blocks=CalculationBlocks(
                blocks=[construction_block, debt_sizing_block,
                        debt_drawdown_block, idc_block],
                solve_loops=[idc_loop],
            ),
            model_wiring=ModelWiring(),
        )
        self.compiled = self.executor.compile(model)

    def test_converges_iteratively(self):
        """array_fixed_point converges in a small number of iterations (> 0)."""
        results = self.executor.run(self.compiled)
        assert results.convergence[0].converged
        assert results.convergence[0].iterations > 0
        assert results.convergence[0].final_residual < 1e-6

    def test_idc_period0_spot_check(self):
        """
        New model: debt_amount = (capex_total + idc_total) × debt_pct,
        drawdown[t] = debt_amount × capex_schedule[t].

        Closed-form for idc[0] = debt_amount × s[0]/2 × r_q:
          total_idc_factor = Σ (cum_s[t-1] + cum_s[t]) / 2  over construction
          debt_amount = capex_total × debt_pct / (1 − debt_pct × r_q × total_idc_factor)
          idc[0] = debt_amount × s[0] / 2 × r_q
        """
        results = self.executor.run(self.compiled)
        idc = results.variables["idc_block.idc_per_period"]
        s = self.capex_schedule[:self.n_construction]
        cum_s = [sum(s[:t + 1]) for t in range(self.n_construction)]
        prev_cum_s = [0.0] + cum_s[:-1]
        total_factor = sum((p + c) / 2.0 for p, c in zip(prev_cum_s, cum_s))
        debt_amount = (self.capex_total * self.debt_pct
                       / (1.0 - self.debt_pct * self.r_q * total_factor))
        expected_idc0 = debt_amount * s[0] / 2.0 * self.r_q
        assert float(idc[0]) == pytest.approx(expected_idc0, rel=1e-5)

    def test_average_balance_formula_holds(self):
        """
        For every construction period:
          IDC(t) == (opening_debt(t) + closing_debt(t)) / 2 × r_q
        """
        results = self.executor.run(self.compiled)
        idc = results.variables["idc_block.idc_per_period"]
        cum_debt = results.variables["debt_drawdown_block.cumulative_drawdown"]

        for t in range(self.n_construction):
            opening = float(cum_debt[t - 1]) if t > 0 else 0.0
            closing = float(cum_debt[t])
            expected = self.r_q * (opening + closing) / 2.0
            assert float(idc[t]) == pytest.approx(expected, rel=1e-5), \
                f"Period {t}: IDC={idc[t]:.4f}, expected={expected:.4f}"

    def test_drawdown_equals_debt_amount_times_schedule(self):
        """drawdown[t] = debt_amount × capex_schedule[t] for every construction period."""
        results = self.executor.run(self.compiled)
        drawdown = results.variables["debt_drawdown_block.drawdown"]
        debt_amount = float(results.variables["debt_sizing_block.debt_amount"][0])

        for t in range(self.n_construction):
            expected = debt_amount * self.capex_schedule[t]
            assert float(drawdown[t]) == pytest.approx(expected, rel=1e-6), \
                f"Period {t}: drawdown={drawdown[t]:.4f}, expected={expected:.4f}"

    def test_total_debt_equals_debt_pct_times_total_project_cost(self):
        """Aggregate: total_debt = debt_pct × (capex + total_IDC)."""
        results = self.executor.run(self.compiled)
        debt_amount = float(results.variables["debt_sizing_block.debt_amount"][0])
        idc_total   = float(results.variables["idc_block.idc_total"][0])
        assert debt_amount == pytest.approx(self.debt_pct * (self.capex_total + idc_total), rel=1e-6)

    def test_no_idc_after_construction(self):
        """IDC is zero outside construction."""
        results = self.executor.run(self.compiled)
        idc = results.variables["idc_block.idc_per_period"]
        for t in range(self.n_construction, self.n):
            assert float(idc[t]) == pytest.approx(0.0, abs=1e-8)


# ===========================================================================
# 9. TestRunEnd2End
# ===========================================================================


class TestRunEnd2End:
    """End-to-end run() on the 2-block simple model."""

    def setup_method(self):
        self.executor = ModelExecutor()
        self.model = _simple_model()
        self.compiled = self.executor.compile(self.model)

    def test_returns_model_results(self):
        results = self.executor.run(self.compiled)
        assert isinstance(results, ModelResults)

    def test_model_id_matches(self):
        results = self.executor.run(self.compiled)
        assert results.model_id == "test_model"

    def test_variables_contains_both_outputs(self):
        results = self.executor.run(self.compiled)
        assert "block_a.doubled" in results.variables
        assert "block_b.tripled" in results.variables

    def test_variables_do_not_contain_assumptions(self):
        results = self.executor.run(self.compiled)
        for k in results.variables:
            assert not k.startswith("assumption.")

    def test_variables_do_not_contain_phases(self):
        results = self.executor.run(self.compiled)
        for k in results.variables:
            assert not k.startswith("phase.")

    def test_assumptions_used_snapshot(self):
        results = self.executor.run(self.compiled, {"x_val": 3.0})
        assert results.assumptions_used["x_val"] == pytest.approx(3.0)

    def test_audit_trail_nonempty(self):
        results = self.executor.run(self.compiled)
        assert len(results.audit_trail) > 0

    def test_warnings_is_list(self):
        results = self.executor.run(self.compiled)
        assert isinstance(results.warnings, list)

    def test_kpis_is_kpiresult_type(self):
        from dsl.types import KPIResult
        results = self.executor.run(self.compiled)
        assert isinstance(results.kpis, KPIResult)

    def test_run_with_none_assumptions_uses_defaults(self):
        results = self.executor.run(self.compiled, None)
        expected = np.full(N_PERIODS, 10.0)
        np.testing.assert_allclose(results.variables["block_a.doubled"], expected)


# ===========================================================================
# 9. TestRunBatch
# ===========================================================================


class TestRunBatch:
    def setup_method(self):
        self.executor = ModelExecutor()
        self.model = _simple_model()
        self.compiled = self.executor.compile(self.model)

    def test_batch_returns_correct_count(self):
        batch = [{"x_val": v} for v in [1.0, 2.0, 3.0]]
        results = self.executor.run_batch(self.compiled, batch)
        assert len(results) == 3

    def test_batch_order_preserved(self):
        """Each scenario's tripled value matches 6 * x_val."""
        x_vals = [1.0, 2.0, 5.0, 10.0]
        batch = [{"x_val": v} for v in x_vals]
        results = self.executor.run_batch(self.compiled, batch)
        for i, x in enumerate(x_vals):
            expected = np.full(N_PERIODS, x * 6.0)
            np.testing.assert_allclose(
                results[i].variables["block_b.tripled"], expected,
                err_msg=f"Scenario {i} (x_val={x}) failed"
            )

    def test_batch_empty_list(self):
        results = self.executor.run_batch(self.compiled, [])
        assert results == []

    def test_batch_single_item(self):
        results = self.executor.run_batch(self.compiled, [{"x_val": 4.0}])
        assert len(results) == 1
        np.testing.assert_allclose(
            results[0].variables["block_b.tripled"],
            np.full(N_PERIODS, 24.0),
        )

    def test_batch_results_are_independent(self):
        """Mutating one result's variables should not affect another."""
        batch = [{"x_val": 1.0}, {"x_val": 2.0}]
        results = self.executor.run_batch(self.compiled, batch)
        results[0].variables["block_a.doubled"][:] = 999.0
        # Result 1 must still have the original values
        np.testing.assert_allclose(
            results[1].variables["block_a.doubled"],
            np.full(N_PERIODS, 4.0),
        )


# ===========================================================================
# 10. TestDownstreamVariables
# ===========================================================================


class TestDownstreamVariables:
    def setup_method(self):
        self.executor = ModelExecutor()
        self.model = _simple_model()
        self.compiled = self.executor.compile(self.model)

    def test_x_val_downstream_includes_block_a(self):
        downstream = self.executor.downstream_variables(self.compiled, "x_val")
        assert "block_a.doubled" in downstream

    def test_x_val_downstream_includes_block_b(self):
        """block_b.tripled depends transitively on x_val via block_a."""
        downstream = self.executor.downstream_variables(self.compiled, "x_val")
        assert "block_b.tripled" in downstream

    def test_unknown_assumption_returns_empty(self):
        downstream = self.executor.downstream_variables(self.compiled, "nonexistent")
        assert downstream == []

    def test_returns_list(self):
        downstream = self.executor.downstream_variables(self.compiled, "x_val")
        assert isinstance(downstream, list)


# ===========================================================================
# 11. TestAuditTrail
# ===========================================================================


class TestAuditTrail:
    def setup_method(self):
        self.executor = ModelExecutor()
        self.model = _simple_model()
        self.compiled = self.executor.compile(self.model)
        self.results = self.executor.run(self.compiled)

    def test_audit_entries_are_audit_entry_type(self):
        for entry in self.results.audit_trail:
            assert isinstance(entry, AuditEntry)

    def test_audit_entry_has_expression(self):
        for entry in self.results.audit_trail:
            assert isinstance(entry.expression, str)
            assert len(entry.expression) > 0

    def test_audit_entry_values_correct_length(self):
        for entry in self.results.audit_trail:
            assert len(entry.values) == N_PERIODS

    def test_audit_entry_block_id_nonempty(self):
        for entry in self.results.audit_trail:
            assert entry.block_id

    def test_audit_block_a_doubled_present(self):
        variables = {e.variable for e in self.results.audit_trail}
        assert "block_a.doubled" in variables

    def test_audit_values_match_variables(self):
        """Audit trail values for block_a.doubled should equal variables."""
        entry = next(
            e for e in self.results.audit_trail if e.variable == "block_a.doubled"
        )
        np.testing.assert_allclose(entry.values, self.results.variables["block_a.doubled"])


# ===========================================================================
# 12. TestConvergenceMetadata
# ===========================================================================


class TestConvergenceMetadata:
    def test_no_loops_empty_convergence(self):
        executor = ModelExecutor()
        model = _simple_model()
        compiled = executor.compile(model)
        results = executor.run(compiled)
        assert results.convergence == []

    def test_goal_seek_convergence_list_length(self):
        executor = ModelExecutor()
        model = _goal_seek_model()
        compiled = executor.compile(model)
        results = executor.run(compiled)
        assert len(results.convergence) == 1

    def test_goal_seek_convergence_attributes(self):
        executor = ModelExecutor()
        model = _goal_seek_model()
        compiled = executor.compile(model)
        results = executor.run(compiled)
        conv = results.convergence[0]
        assert conv.loop_id == "gs_loop"
        assert conv.converged is True
        assert conv.iterations > 0
        assert isinstance(conv.final_residual, float)

    def test_fixed_point_convergence_attributes(self):
        executor = ModelExecutor()
        model = _fixed_point_model()
        compiled = executor.compile(model)
        results = executor.run(compiled)
        conv = results.convergence[0]
        assert conv.loop_id == "fp_loop"
        assert conv.converged is True


# ===========================================================================
# 13. TestSensitivityAndMonteCarlo (formerly TestStubs — now implemented)
# ===========================================================================


class TestSensitivityAndMonteCarlo:
    """
    Verify that run_sensitivity and run_monte_carlo are implemented and
    return the correct result types.  Full behavioural coverage lives in
    tests/test_integration.py::TestSensitivityAnalysis and ::TestMonteCarlo.
    """

    def setup_method(self):
        self.executor = ModelExecutor()
        self.compiled = self.executor.compile(_simple_model())

    def test_run_sensitivity_returns_sensitivity_results(self):
        from engine.executor import SensitivityResults

        # _simple_model has one assumption: x_val (default 5.0)
        sweep = [{"assumption": "x_val", "low": 3.0, "high": 8.0}]
        result = self.executor.run_sensitivity(self.compiled, {}, sweep)
        assert isinstance(result, SensitivityResults)
        assert "x_val" in result.tornado_data

    def test_run_monte_carlo_returns_monte_carlo_results(self):
        from engine.executor import MonteCarloResults

        dist_config = [
            {
                "assumption": "x_val",
                "distribution": "triangular",
                "low": 3.0,
                "base": 5.0,
                "high": 8.0,
            }
        ]
        result = self.executor.run_monte_carlo(
            self.compiled, {}, dist_config, n_iterations=5, seed=42
        )
        assert isinstance(result, MonteCarloResults)
        assert result.n_iterations == 5
        assert len(result.equity_irr_dist) == 5


# ===========================================================================
# 14. TestWaterfallEdgeCases
# ===========================================================================


class TestWaterfallEdgeCases:
    def test_zero_cash_all_zeros(self):
        executor = ModelExecutor()
        model = _waterfall_model()
        compiled = executor.compile(model)
        results = executor.run(compiled, {"cash_in": 0.0})
        np.testing.assert_allclose(results.variables["wf_block.ops_cost"], np.zeros(N_PERIODS))
        np.testing.assert_allclose(results.variables["wf_block.reserve"], np.zeros(N_PERIODS))
        np.testing.assert_allclose(results.variables["wf_block.residual"], np.zeros(N_PERIODS))

    def test_exact_fill_no_residual(self):
        """cash_in = 150 exactly fills ops_cost(100) + reserve(50), residual = 0."""
        executor = ModelExecutor()
        model = _waterfall_model()
        compiled = executor.compile(model)
        results = executor.run(compiled, {"cash_in": 150.0})
        np.testing.assert_allclose(results.variables["wf_block.residual"], np.zeros(N_PERIODS))


# ===========================================================================
# 15. TestMultiStepBody
# ===========================================================================


class TestMultiStepBody:
    """Test a standard block with multiple sequential body steps."""

    def test_intermediate_variable_available_in_next_step(self):
        """
        block: a = x + 1;  b = a * 2
        x=5 → a=6, b=12
        """
        block = CalculationBlock(
            block_id="multi_block",
            category=BlockCategory.revenue,
            block_type=BlockType.standard,
            inputs=[BlockInput(name="x", source="assumption.x_val")],
            outputs=[BlockOutput(name="b", unit="")],
            body=[
                CalculationStep(target="a", expr="x + 1"),
                CalculationStep(target="b", expr="a * 2"),
            ],
        )
        model = ModelDefinition(
            project_skeleton=_skeleton(),
            assumption_schema=AssumptionSchema(
                assumptions=[_assumption("x_val", 5.0)]
            ),
            calculation_blocks=CalculationBlocks(blocks=[block]),
            model_wiring=ModelWiring(),
        )
        executor = ModelExecutor()
        compiled = executor.compile(model)
        results = executor.run(compiled)
        expected = np.full(N_PERIODS, 12.0)
        np.testing.assert_allclose(results.variables["multi_block.b"], expected)
