"""
engine/executor.py — ModelExecutor: compile, run, and batch-run model definitions.

Pipeline
--------
  compile(model_def)  → CompiledModel
    Build dependency graph → topological sort → create evaluation plan
    Pre-allocate numpy arrays per output variable

  run(compiled, assumptions)  → ModelResults
    Inject assumptions → execute plan (standard/waterfall/ledger blocks)
    For each declared solve_loop invoke the matching solver
    Compute KPIs → build audit trail

  run_batch(compiled, assumptions_batch)  → List[ModelResults]
    Sequential run per scenario (vectorised optimisation deferred to step 10)

  run_sensitivity / run_monte_carlo → stubs (implemented in step 10)

Critical constraints (from spec)
---------------------------------
  LLM arithmetic is NEVER used — all computation goes through NumPy.
  DSCR is NEVER computed in non-debt periods (enforced in kpi.py).
  Every run produces an audit_trail mapping variable → array + expression.
  Failed solve loops raise ModelConvergenceError with full diagnostics.
"""

from __future__ import annotations

import copy
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

from dsl.expression import ExpressionEvaluator
from dsl.parser import DSLParser, load_model_from_dict
from dsl.types import (
    BlockType,
    CalculationBlock,
    CalculationBlocks,
    ConvergenceMetadata,
    KPIResult,
    ModelDefinition,
    SolveLoop,
    SolveLoopType,
    ValidationResult,
)
from engine.kpi import compute_all_kpis
from engine.solvers import (
    ModelConvergenceError,
    ArrayFixedPointResult,
    FixedPointResult,
    GoalSeekResult,
    SculptingResult,
    array_fixed_point_solver,
    fixed_point_solver,
    goal_seek_solver,
    sculpting_solver,
)


# ---------------------------------------------------------------------------
# Compiled-model data structures
# ---------------------------------------------------------------------------


@dataclass
class AuditEntry:
    """One computed value in the audit trail."""
    variable: str          # "block_id.output_name"
    expression: str        # DSL expression that produced it
    block_id: str
    values: np.ndarray     # (n_periods,) array


@dataclass
class EvaluationStep:
    """One step in the ordered execution plan."""
    kind: str                                        # "block" | "solve_loop"
    block: Optional[CalculationBlock] = None
    solve_loop: Optional[SolveLoop] = None
    loop_subgraph: Optional[List["EvaluationStep"]] = None   # blocks inside loop


@dataclass
class CompiledModel:
    model_def: ModelDefinition
    evaluation_plan: List[EvaluationStep]
    dependency_graph: nx.DiGraph          # block-level, for downstream tracking
    output_graph: nx.DiGraph              # output-variable-level, for dirty tracking
    n_periods: int
    periods_per_year: int
    cod_period: int
    debt_maturity_period: int
    evaluator: ExpressionEvaluator


# ---------------------------------------------------------------------------
# Model results
# ---------------------------------------------------------------------------


@dataclass
class ModelResults:
    model_id: str
    variables: Dict[str, np.ndarray]          # all outputs: "block.output" → array
    kpis: KPIResult
    convergence: List[ConvergenceMetadata]
    audit_trail: List[AuditEntry]
    warnings: List[str]
    assumptions_used: Dict[str, Any]          # snapshot of effective assumptions


# Stubs used in step 10
@dataclass
class SensitivityResults:
    tornado_data: Dict[str, Tuple[float, float]]   # assumption → (low_impact, high_impact)
    base_kpis: KPIResult


@dataclass
class MonteCarloResults:
    equity_irr_dist: np.ndarray
    min_dscr_dist: np.ndarray
    llcr_dist: np.ndarray
    npv_dist: np.ndarray
    n_iterations: int
    p10: Dict[str, float]
    p50: Dict[str, float]
    p90: Dict[str, float]
    prob_dscr_below_1: float
    prob_irr_below_hurdle: float


# ---------------------------------------------------------------------------
# ModelExecutor
# ---------------------------------------------------------------------------


class ModelExecutor:
    """
    Compile and execute a model definition.

    Usage
    -----
    executor = ModelExecutor()
    compiled = executor.compile(model_definition)
    results  = executor.run(compiled, assumptions_dict)
    """

    # ------------------------------------------------------------------
    # 1. compile()
    # ------------------------------------------------------------------

    def compile(
        self,
        model_definition: Union[ModelDefinition, Dict[str, Any]],
    ) -> CompiledModel:
        """
        Compile a ModelDefinition into an executable plan.

        Parameters
        ----------
        model_definition : ModelDefinition object or raw dict (auto-parsed).

        Returns
        -------
        CompiledModel ready for run().

        Raises
        ------
        ValueError : if the model definition is invalid or has unexpected cycles.
        """
        if isinstance(model_definition, dict):
            model_def, validation = load_model_from_dict(model_definition)
            if not validation.valid:
                raise ValueError(
                    "Model definition is invalid:\n"
                    + "\n".join(validation.errors)
                )
        else:
            model_def = model_definition

        skeleton = model_def.project_skeleton
        n_periods = skeleton.total_periods
        ppy = skeleton.periods_per_year
        cod = skeleton.milestones.cod
        mat = skeleton.milestones.debt_maturity

        evaluator = ExpressionEvaluator(n_periods=n_periods, periods_per_year=ppy)

        # Build block-level dependency graph
        dep_graph = self._build_block_dependency_graph(model_def)

        # Topological sort (ignoring declared solve loop back-edges)
        sorted_blocks = self._topological_sort(dep_graph, model_def)

        # Build output-variable-level graph (for dirty-node tracking in sensitivity)
        out_graph = self._build_output_dependency_graph(model_def)

        # Create evaluation plan with solve loops spliced in
        plan = self._build_evaluation_plan(
            sorted_blocks, model_def.calculation_blocks.solve_loops
        )

        return CompiledModel(
            model_def=model_def,
            evaluation_plan=plan,
            dependency_graph=dep_graph,
            output_graph=out_graph,
            n_periods=n_periods,
            periods_per_year=ppy,
            cod_period=cod,
            debt_maturity_period=mat,
            evaluator=evaluator,
        )

    # ------------------------------------------------------------------
    # 2. run()
    # ------------------------------------------------------------------

    def run(
        self,
        compiled: CompiledModel,
        assumptions: Optional[Dict[str, Any]] = None,
    ) -> ModelResults:
        """
        Execute a compiled model with the given assumption overrides.

        Parameters
        ----------
        compiled    : output of compile()
        assumptions : dict of {assumption_name → value} overrides.
                      If None, schema defaults are used.

        Returns
        -------
        ModelResults containing time-series variables, KPIs, convergence
        metadata, and the full audit trail.
        """
        assumptions = assumptions or {}

        # --- Initialize namespace ---
        namespace, effective_assumptions = self._init_namespace(compiled, assumptions)

        # --- Execute the evaluation plan ---
        audit: List[AuditEntry] = []
        convergence: List[ConvergenceMetadata] = []
        exec_warnings: List[str] = []

        self._execute_plan(
            compiled.evaluation_plan,
            compiled,
            namespace,
            audit,
            convergence,
            exec_warnings,
        )

        # --- Extract output arrays ---
        variables = {k: v for k, v in namespace.items()
                     if "." in k and not k.startswith("assumption.")
                     and not k.startswith("phase.")}

        # --- Compute KPIs ---
        kpis, kpi_warnings = self._compute_kpis(compiled, namespace)
        exec_warnings.extend(kpi_warnings)

        return ModelResults(
            model_id=compiled.model_def.project_skeleton.model_id,
            variables=variables,
            kpis=kpis,
            convergence=convergence,
            audit_trail=audit,
            warnings=exec_warnings,
            assumptions_used=effective_assumptions,
        )

    # ------------------------------------------------------------------
    # 3. run_batch()
    # ------------------------------------------------------------------

    def run_batch(
        self,
        compiled: CompiledModel,
        assumptions_batch: List[Dict[str, Any]],
        n_workers: int = 1,
    ) -> List[ModelResults]:
        """
        Run the model for multiple assumption sets.

        Parameters
        ----------
        compiled          : output of compile()
        assumptions_batch : list of assumption dicts, one per scenario
        n_workers         : number of parallel workers (1 = sequential)

        Returns
        -------
        List of ModelResults, one per scenario, in the same order.

        Notes
        -----
        Vectorised numpy broadcasting across scenarios is deferred to step 10.
        This implementation is sequential (correct, not optimised).
        """
        if n_workers == 1:
            return [self.run(compiled, a) for a in assumptions_batch]

        results: List[Optional[ModelResults]] = [None] * len(assumptions_batch)

        def _run_one(idx_and_assumptions):
            idx, assumptions = idx_and_assumptions
            try:
                return idx, self.run(compiled, assumptions)
            except Exception as exc:
                return idx, exc

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_run_one, (i, a)): i
                for i, a in enumerate(assumptions_batch)
            }
            for future in as_completed(futures):
                idx, result = future.result()
                if isinstance(result, Exception):
                    raise result
                results[idx] = result

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Stubs for step 10
    # ------------------------------------------------------------------

    def run_sensitivity(
        self,
        compiled: CompiledModel,
        base_assumptions: Dict[str, Any],
        sweep_config: List[Dict[str, Any]],
        kpi_metric: str = "equity_irr",
    ) -> SensitivityResults:
        """
        One-at-a-time sensitivity analysis (tornado chart).

        For each entry in sweep_config ({assumption, low, high}), runs the
        model at the low and high values while holding all other assumptions at
        their base-case values.  Returns the change in kpi_metric relative to
        the base case.

        Delegated to engine.sensitivity.run_sensitivity.
        """
        from engine.sensitivity import run_sensitivity as _run_sensitivity

        return _run_sensitivity(
            executor=self,
            compiled=compiled,
            base_assumptions=base_assumptions,
            sweep_config=sweep_config,
            kpi_metric=kpi_metric,
        )

    def run_monte_carlo(
        self,
        compiled: CompiledModel,
        base_assumptions: Dict[str, Any],
        distribution_config: Optional[List[Dict[str, Any]]] = None,
        n_iterations: int = 3000,
        seed: Optional[int] = None,
    ) -> MonteCarloResults:
        """
        Monte Carlo simulation over uncertain assumptions.

        Samples each assumption from its declared distribution (triangular,
        pert, lognormal, uniform) and runs the model n_iterations times.
        Returns distributional statistics for key KPIs.

        If distribution_config is None, uses assumption schema entries with
        sensitivity.vary=True.

        Delegated to engine.sensitivity.run_monte_carlo.
        """
        from engine.sensitivity import run_monte_carlo as _run_monte_carlo

        return _run_monte_carlo(
            executor=self,
            compiled=compiled,
            base_assumptions=base_assumptions,
            distribution_config=distribution_config,
            n_iterations=n_iterations,
            seed=seed,
        )

    # ==================================================================
    # Internal: namespace initialisation
    # ==================================================================

    def _init_namespace(
        self,
        compiled: CompiledModel,
        assumption_overrides: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Build the initial namespace from:
          - Phase masks (auto-derived, stored as "phase.X")
          - Assumption schema defaults, overridden by assumption_overrides
          - Assumption values stored as both "assumption.X" and bare "X"
        """
        model_def = compiled.model_def
        skeleton = model_def.project_skeleton
        n = compiled.n_periods

        namespace: Dict[str, Any] = {}

        # Phase masks
        phases = skeleton.phases
        namespace["phase.is_construction"] = np.asarray(
            phases.is_construction, dtype=np.float64
        )
        namespace["phase.is_operational"] = np.asarray(
            phases.is_operational, dtype=np.float64
        )
        namespace["phase.is_debt_outstanding"] = np.asarray(
            phases.is_debt_outstanding, dtype=np.float64
        )

        # Assumption values
        effective: Dict[str, Any] = {}
        for a_def in model_def.assumption_schema.assumptions:
            # Priority: override > schema value > default_rule evaluation
            if a_def.name in assumption_overrides:
                raw_value = assumption_overrides[a_def.name]
            elif a_def.value is not None:
                raw_value = a_def.value
            elif a_def.default_rule:
                # Simple constant default rules ("= 0.025")
                rule = a_def.default_rule.strip().lstrip("=").strip()
                try:
                    raw_value = float(rule)
                except ValueError:
                    raw_value = 0.0  # unresolvable rule defaults to 0
            else:
                raw_value = 0.0  # absent assumption → 0

            # Convert schedule/time_series to arrays
            if isinstance(raw_value, list):
                arr = np.asarray(raw_value, dtype=np.float64)
                # Pad/trim to n_periods if needed (only for full-length series)
                if len(arr) != n and a_def.type.value in ("time_series",):
                    if len(arr) < n:
                        arr = np.pad(arr, (0, n - len(arr)), constant_values=arr[-1])
                    else:
                        arr = arr[:n]
                effective[a_def.name] = arr
            else:
                effective[a_def.name] = raw_value

            namespace[f"assumption.{a_def.name}"] = effective[a_def.name]

        return namespace, effective

    # ==================================================================
    # Internal: plan execution
    # ==================================================================

    def _execute_plan(
        self,
        plan: List[EvaluationStep],
        compiled: CompiledModel,
        namespace: Dict[str, Any],
        audit: List[AuditEntry],
        convergence: List[ConvergenceMetadata],
        warnings: List[str],
    ) -> None:
        """Walk the evaluation plan in order, dispatching to block/loop executors."""
        for step in plan:
            if step.kind == "block":
                self._execute_block(step.block, compiled, namespace, audit)
            elif step.kind == "solve_loop":
                self._execute_solve_loop(
                    step.solve_loop,
                    step.loop_subgraph or [],
                    compiled,
                    namespace,
                    audit,
                    convergence,
                    warnings,
                )

    # ------------------------------------------------------------------
    # Block execution
    # ------------------------------------------------------------------

    def _execute_block(
        self,
        block: CalculationBlock,
        compiled: CompiledModel,
        namespace: Dict[str, Any],
        audit: List[AuditEntry],
    ) -> None:
        """Dispatch to the correct block executor based on block_type."""
        if block.block_type == BlockType.standard:
            self._execute_standard(block, compiled, namespace, audit)
        elif block.block_type == BlockType.waterfall:
            self._execute_waterfall(block, compiled, namespace, audit)
        elif block.block_type == BlockType.ledger:
            self._execute_ledger(block, compiled, namespace, audit)

    def _execute_standard(
        self,
        block: CalculationBlock,
        compiled: CompiledModel,
        namespace: Dict[str, Any],
        audit: List[AuditEntry],
    ) -> None:
        """
        Execute a standard block:
          1. Map block input names to namespace values.
          2. Run each body step in order (results accumulate in local ctx).
          3. Store declared outputs into the global namespace.
          4. Record audit entries.
        """
        ev = compiled.evaluator

        # Build local context from declared inputs
        ctx: Dict[str, Any] = {}
        for inp in block.inputs:
            value = namespace.get(inp.source)
            if value is None:
                # Phase mask shorthand: allow "is_operational" as bare name
                bare = inp.source.split(".")[-1]
                value = namespace.get(f"phase.{bare}", 0.0)
            ctx[inp.name] = value

        # Execute body steps
        n = compiled.n_periods
        if block.body:
            # Detect forward references: a body step references a variable that
            # is the target of a *later* step (recurrence pattern, e.g. opening = lag(closing,1)).
            # When detected, switch to period-by-period sequential evaluation so that
            # lag() always reads from already-correct array positions.
            body_targets = [step.target for step in block.body]
            needs_sequential = False
            for i, step in enumerate(block.body):
                future_set = set(body_targets[i + 1:])
                # Word-boundary scan: is any future target name referenced in this expression?
                if any(re.search(r'\b' + re.escape(ft) + r'\b', step.expr) for ft in future_set):
                    needs_sequential = True
                    break

            if needs_sequential:
                # Pre-initialise every body target to zeros so forward references via lag()
                # see a valid (zero-padded) array from the very first period.
                for tgt in body_targets:
                    ctx[tgt] = np.zeros(n)

                # Step through each period; evaluate full expressions and take index t.
                # Because ctx arrays are updated in-place after each period, lag() reads
                # correct history for all t' < t.
                for t in range(n):
                    for step in block.body:
                        result = ev.evaluate(step.expr, ctx)
                        result_arr = np.asarray(result, dtype=np.float64)
                        val_t = float(result_arr) if result_arr.ndim == 0 else float(result_arr[t])
                        ctx[step.target][t] = val_t
            else:
                # Standard array evaluation — all body steps evaluate on the full period axis.
                for step in block.body:
                    result = ev.evaluate(step.expr, ctx)
                    result_arr = np.asarray(result, dtype=np.float64)
                    if result_arr.ndim == 0:
                        result_arr = np.full(n, float(result_arr))
                    ctx[step.target] = result_arr

            # Audit declared outputs (works for both array and sequential paths)
            for step in block.body:
                for out in block.outputs:
                    if out.name == step.target:
                        arr = np.asarray(ctx[step.target], dtype=np.float64)
                        if arr.ndim == 0:
                            arr = np.full(n, float(arr))
                        audit.append(AuditEntry(
                            variable=f"{block.block_id}.{out.name}",
                            expression=step.expr,
                            block_id=block.block_id,
                            values=arr.copy(),
                        ))

        # Store declared outputs into namespace — always as (n_periods,) arrays
        for out in block.outputs:
            val = ctx.get(out.name)
            if val is not None:
                arr = np.asarray(val, dtype=np.float64)
                if arr.ndim == 0:
                    arr = np.full(n, float(arr))
                namespace[f"{block.block_id}.{out.name}"] = arr

    def _execute_waterfall(
        self,
        block: CalculationBlock,
        compiled: CompiledModel,
        namespace: Dict[str, Any],
        audit: List[AuditEntry],
    ) -> None:
        """
        Waterfall block: allocate cash to priority buckets in sequence.

        Each bucket is always fully met regardless of available cash.  Any
        period where cumulative bucket targets exceed revenue is funded by an
        equity infusion (an additional outflow from the equity perspective).
        The infusion amount is stored as '{block_id}.equity_infusion' and
        must be deducted from equity_cashflow in the returns block.

        Remaining cash after all buckets (>= 0) is the equity distribution.
        """
        ev = compiled.evaluator
        n = compiled.n_periods

        # Available cash at the top of the waterfall
        available = namespace.get(block.available_cash_input, np.zeros(n))
        available = np.asarray(available, dtype=np.float64)

        # Build context for expression evaluation
        ctx: Dict[str, Any] = {}
        for inp in block.inputs:
            ctx[inp.name] = namespace.get(inp.source, np.zeros(n))
        ctx["available_cash"] = available

        remaining = available.copy()

        # Process buckets in priority order — always allocate the full target.
        # remaining is allowed to go negative; that deficit is the equity infusion.
        for bucket in sorted(block.buckets, key=lambda b: b.priority):
            target = ev.evaluate(bucket.target_expr, ctx)
            target_arr = np.asarray(target, dtype=np.float64)
            if target_arr.ndim == 0:
                target_arr = np.full(n, float(target_arr))

            allocated = target_arr.copy()
            remaining = remaining - allocated

            namespace[f"{block.block_id}.{bucket.name}"] = allocated
            ctx[bucket.name] = allocated

            audit.append(AuditEntry(
                variable=f"{block.block_id}.{bucket.name}",
                expression=bucket.target_expr,
                block_id=block.block_id,
                values=allocated.copy(),
            ))

        # Shortfall (remaining < 0): equity must inject cash to cover obligations.
        equity_infusion = np.maximum(-remaining, 0.0)
        # Surplus (remaining > 0): distributed to equity.
        residual = np.maximum(remaining, 0.0)

        # equity_distribution is always written — it is the fundamental waterfall output.
        namespace[f"{block.block_id}.equity_distribution"] = residual

        # equity_infusion and residual are written only if declared in block outputs,
        # keeping the namespace clean when these are not needed downstream.
        declared_output_names = {out.name for out in block.outputs}
        if "equity_infusion" in declared_output_names:
            namespace[f"{block.block_id}.equity_infusion"] = equity_infusion
            audit.append(AuditEntry(
                variable=f"{block.block_id}.equity_infusion",
                expression="max(-(available_cash - sum_of_bucket_targets), 0)",
                block_id=block.block_id,
                values=equity_infusion.copy(),
            ))
        if "residual" in declared_output_names:
            namespace[f"{block.block_id}.residual"] = residual

        # Also store declared outputs that were computed as bucket allocations
        for out in block.outputs:
            key = f"{block.block_id}.{out.name}"
            if key not in namespace:
                val = ctx.get(out.name, np.zeros(n))
                namespace[key] = np.asarray(val, dtype=np.float64)

    def _execute_ledger(
        self,
        block: CalculationBlock,
        compiled: CompiledModel,
        namespace: Dict[str, Any],
        audit: List[AuditEntry],
    ) -> None:
        """
        Ledger block (running balance):
          closing[t] = opening[t] + additions[t] - deductions[t]
          opening[t] = closing[t-1]    (opening[0] = block.opening_balance)

        Outputs: {block_id}.opening_balance, .closing_balance, .net_movement
        """
        n = compiled.n_periods
        opening_init = float(block.opening_balance or 0.0)

        closing = np.zeros(n)
        opening = np.zeros(n)

        for t in range(n):
            opening[t] = closing[t - 1] if t > 0 else opening_init

            additions = sum(
                float(np.asarray(namespace.get(a.source, 0.0)).flat[t])
                for a in (block.additions or [])
            )
            deductions = sum(
                float(np.asarray(namespace.get(d.source, 0.0)).flat[t])
                for d in (block.deductions or [])
            )
            closing[t] = max(0.0, opening[t] + additions - deductions)

        namespace[f"{block.block_id}.opening_balance"] = opening
        namespace[f"{block.block_id}.closing_balance"] = closing
        namespace[f"{block.block_id}.net_movement"] = closing - opening

        # Also store any declared outputs (may alias the above)
        for out in block.outputs:
            key = f"{block.block_id}.{out.name}"
            if key not in namespace:
                if "opening" in out.name:
                    namespace[key] = opening
                elif "closing" in out.name or "balance" in out.name:
                    namespace[key] = closing
                else:
                    namespace[key] = closing  # default to closing balance

        for arr_name, arr_val in [("opening_balance", opening), ("closing_balance", closing)]:
            audit.append(AuditEntry(
                variable=f"{block.block_id}.{arr_name}",
                expression="ledger",
                block_id=block.block_id,
                values=arr_val.copy(),
            ))

    # ==================================================================
    # Solve loop execution
    # ==================================================================

    def _execute_solve_loop(
        self,
        loop: SolveLoop,
        subgraph: List[EvaluationStep],
        compiled: CompiledModel,
        namespace: Dict[str, Any],
        audit: List[AuditEntry],
        convergence: List[ConvergenceMetadata],
        warnings: List[str],
    ) -> None:
        """Dispatch to the correct solver strategy based on loop.type."""
        if loop.type == SolveLoopType.goal_seek:
            self._run_goal_seek(loop, subgraph, compiled, namespace, audit, convergence)
        elif loop.type == SolveLoopType.sculpting:
            self._run_sculpting(loop, subgraph, compiled, namespace, audit, convergence, warnings)
        elif loop.type == SolveLoopType.fixed_point:
            self._run_fixed_point(loop, subgraph, compiled, namespace, audit, convergence)
        elif loop.type == SolveLoopType.array_fixed_point:
            self._run_array_fixed_point(loop, subgraph, compiled, namespace, audit, convergence)

    def _run_array_fixed_point(
        self,
        loop: SolveLoop,
        subgraph: List[EvaluationStep],
        compiled: CompiledModel,
        namespace: Dict[str, Any],
        audit: List[AuditEntry],
        convergence: List[ConvergenceMetadata],
    ) -> None:
        """
        Generic array fixed-point solver driven entirely by YAML declarations.

        The solver re-evaluates loop.owned_blocks (in declared order) each
        iteration, updating namespace in-place.  Convergence is measured on
        loop.free_variable: max(|x_new - x_old|) < loop.tolerance.

        No block-specific logic lives here — the circular dependency structure
        emerges from YAML block inputs and is resolved by repeating the YAML
        body expressions until they stop changing.

        Execution order within owned_blocks (example — IDC loop):
          1. debt_sizing_block  — reads idc_block.idc_total (prev iter / 0 initially)
          2. debt_drawdown_block — reads debt_sizing_block.debt_amount (fresh)
          3. idc_block          — reads debt_drawdown_block.cumulative_drawdown (fresh)
                                   writes idc_block.idc_per_period (→ x_new)
        """
        n = compiled.n_periods
        block_map = {b.block_id: b for b in compiled.model_def.calculation_blocks.blocks}
        owned_blocks = [block_map[bid] for bid in loop.owned_blocks if bid in block_map]

        converged = False
        iterations = 0
        final_residual = float("inf")

        for i in range(loop.max_iterations):
            x_old = np.asarray(
                namespace.get(loop.free_variable, np.zeros(n)), dtype=np.float64
            )
            if x_old.ndim == 0:
                x_old = np.full(n, float(x_old))

            # Re-evaluate owned blocks in declared order, updating namespace in-place.
            # Intermediate audit entries are discarded; the final pass below records them.
            iter_audit: List[AuditEntry] = []
            for block in owned_blocks:
                self._execute_block(block, compiled, namespace, iter_audit)

            x_new = np.asarray(
                namespace.get(loop.free_variable, np.zeros(n)), dtype=np.float64
            )
            if x_new.ndim == 0:
                x_new = np.full(n, float(x_new))

            final_residual = float(np.max(np.abs(x_new - x_old)))
            iterations = i + 1

            if final_residual < loop.tolerance:
                converged = True
                break

        # Final authoritative pass — results are identical (namespace already converged)
        # but this writes to the real audit trail.
        for block in owned_blocks:
            self._execute_block(block, compiled, namespace, audit)

        fv_arr = np.asarray(namespace.get(loop.free_variable, np.zeros(n)), dtype=np.float64)
        convergence.append(ConvergenceMetadata(
            loop_id=loop.loop_id,
            converged=converged,
            iterations=iterations,
            final_residual=final_residual,
            final_free_variable_value=float(np.max(np.abs(fv_arr))),
        ))

    def _run_goal_seek(
        self,
        loop: SolveLoop,
        subgraph: List[EvaluationStep],
        compiled: CompiledModel,
        namespace: Dict[str, Any],
        audit: List[AuditEntry],
        convergence: List[ConvergenceMetadata],
    ) -> None:
        """
        Goal-seek: find scalar free_variable value such that
            target_expression(namespace) == target_value.

        For each trial value of the free variable:
          1. Write it into namespace[loop.free_variable]
          2. Re-execute the loop subgraph
          3. Evaluate loop.target_expression
        """
        ev = compiled.evaluator
        x_init = self._get_scalar(namespace, loop.free_variable, 0.0)

        def objective(x: float) -> float:
            ns_trial = dict(namespace)  # shallow copy — arrays are shared (read-only inside loop)
            ns_trial[loop.free_variable] = float(x)
            trial_audit: List[AuditEntry] = []
            for step in subgraph:
                if step.kind == "block":
                    self._execute_block(step.block, compiled, ns_trial, trial_audit)
            val = ev.evaluate(loop.target_expression, ns_trial)
            return float(np.asarray(val, dtype=np.float64).flat[0])

        result = goal_seek_solver(
            objective=objective,
            loop_id=loop.loop_id,
            target_value=loop.target_value,
            x_init=x_init,
            tolerance=loop.tolerance,
            max_iterations=loop.max_iterations,
        )

        # Apply converged value and do a final authoritative evaluation
        namespace[loop.free_variable] = result.solution
        for step in subgraph:
            if step.kind == "block":
                self._execute_block(step.block, compiled, namespace, audit)

        convergence.append(ConvergenceMetadata(
            loop_id=loop.loop_id,
            converged=result.converged,
            iterations=result.iterations,
            final_residual=result.final_residual,
            final_free_variable_value=result.solution,
        ))

    def _run_sculpting(
        self,
        loop: SolveLoop,
        subgraph: List[EvaluationStep],
        compiled: CompiledModel,
        namespace: Dict[str, Any],
        audit: List[AuditEntry],
        convergence: List[ConvergenceMetadata],
        warnings: List[str],
    ) -> None:
        """
        Debt service sizing — two modes controlled by assumption.debt_sizing_mode:

        0.0  →  cost_based  (default)
             total_debt = capex × debt_pct  (fixed input)
             Repayment   = equal principal instalments over tenor
             Interest    = balance × r_per_period (declining)
             No CFADS feedback into repayment schedule.
             Warnings raised when DSCR < 1.0 or free cashflow < 0.

        1.0  →  cfads_based
             total_debt = PV(CFADS / dscr_target)  — computed, not assumed
             Repayment   = sculpted proportional to CFADS (DSCR = target by design)
             Iterates: subgraph → CFADS → sculpt → write DS → subgraph …
             until relative CFADS change < 1e-4 (self-consistent with interest tax shield).
        """
        n = compiled.n_periods
        ppy = compiled.periods_per_year
        cod = compiled.cod_period
        mat = compiled.debt_maturity_period

        # All assumption names and output variable names are read from loop.parameters
        # (declared in YAML) with hardcoded strings only as fallbacks for backward compat.
        p = loop.parameters
        interest_rate_var    = p.get("interest_rate_var",    "assumption.interest_rate")
        moratorium_var       = p.get("moratorium_var",        "assumption.moratorium_periods")
        debt_sizing_mode_var = p.get("debt_sizing_mode_var",  "assumption.debt_sizing_mode")
        out_principal        = p.get("output_principal",      "principal_repayment")
        out_interest         = p.get("output_interest",       "interest_payment")
        out_balance          = p.get("output_balance",        "outstanding_debt_balance")
        out_ds               = p.get("output_ds",             "total_debt_service")

        target_block_id = loop.free_variable.split(".")[0]
        debt_amount_key = p.get("debt_amount_var", f"{target_block_id}.debt_amount")

        rate_pa = self._get_scalar(namespace, interest_rate_var, 0.0)
        r_per_period = (1.0 + rate_pa) ** (1.0 / ppy) - 1.0
        moratorium = int(self._get_scalar(namespace, moratorium_var, 0))
        dscr_target = loop.target_value  # YAML target_value is authoritative

        # Mode switch: 0.0 = cost_based, 1.0 = cfads_based
        cfads_based = self._get_scalar(namespace, debt_sizing_mode_var, 0.0) >= 0.5

        repay_start = cod + moratorium
        repay_end = mat + 1   # exclusive
        n_repay = repay_end - repay_start

        if not cfads_based:
            # ----------------------------------------------------------
            # COST-BASED MODE
            # total_debt is fixed.  Repayment = equal principal each period.
            # One pass only — DS does not depend on CFADS.
            # Subgraph is run once afterwards to get final CFADS/tax values.
            # DSCR and cashflow are checked; warnings emitted on any breach.
            # ----------------------------------------------------------
            total_debt = self._find_scalar(
                namespace,
                [debt_amount_key, "assumption.total_debt"],
                0.0,
            )

            principal = np.zeros(n)
            interest_arr = np.zeros(n)
            balance_arr = np.zeros(n)

            equal_principal = total_debt / n_repay if n_repay > 0 else 0.0
            bal = total_debt

            # Moratorium periods: interest-only, no principal
            # balance_arr records the CLOSING balance (after any repayment)
            for t in range(cod, repay_start):
                interest_arr[t] = bal * r_per_period
                balance_arr[t] = bal  # no principal → closing = opening

            # Repayment periods: equal principal + declining interest
            for t in range(repay_start, repay_end):
                interest_arr[t] = bal * r_per_period
                principal[t] = equal_principal
                bal = max(0.0, bal - equal_principal)
                balance_arr[t] = bal  # closing balance = after repayment

            total_ds = interest_arr + principal

            # Build opening balance array (balance before repayment each period)
            opening_arr = np.zeros(n)
            ob = total_debt
            for t in range(cod, repay_end):
                opening_arr[t] = ob
                ob = max(0.0, ob - principal[t])

            # Write schedule into namespace using YAML-declared output names
            namespace[f"{target_block_id}.opening"]         = opening_arr
            namespace[f"{target_block_id}.{out_principal}"] = principal
            namespace[f"{target_block_id}.{out_interest}"]  = interest_arr
            namespace[f"{target_block_id}.{out_balance}"]   = balance_arr
            namespace[f"{target_block_id}.{out_ds}"]        = total_ds

            # Run subgraph once with actual DS → computes CFADS, tax, waterfall
            _tmp_audit: List[AuditEntry] = []
            for step in subgraph:
                if step.kind == "block":
                    self._execute_block(step.block, compiled, namespace, _tmp_audit)
            cfads = np.asarray(namespace.get(loop.target_expression, np.zeros(n)), dtype=np.float64)

            # Check for DSCR breaches and negative free cashflow
            breach_periods = []
            for t in range(repay_start, repay_end):
                ds_t = float(total_ds[t])
                cfads_t = float(cfads[t])
                if ds_t > 1e-6:
                    dscr_t = cfads_t / ds_t
                    if dscr_t < 1.0:
                        breach_periods.append((t, dscr_t, cfads_t, ds_t))
                if cfads_t < 0.0:
                    warnings.append(
                        f"Cost-based mode: negative CFADS at period {t}: "
                        f"{cfads_t:.1f} Lakhs"
                    )

            if breach_periods:
                n_breach = len(breach_periods)
                worst_t, worst_dscr, worst_cfads, worst_ds = min(
                    breach_periods, key=lambda x: x[1]
                )
                warnings.append(
                    f"Cost-based mode: DSCR < 1.0 in {n_breach} period(s). "
                    f"Worst: period {worst_t}, DSCR={worst_dscr:.3f} "
                    f"(CFADS={worst_cfads:.1f}, DS={worst_ds:.1f} Lakhs). "
                    "Consider reducing debt_pct or extending tenor."
                )

            convergence.append(ConvergenceMetadata(
                loop_id=loop.loop_id,
                converged=True,
                iterations=1,
                final_residual=0.0,
            ))

            for name, arr in [
                (out_principal, principal),
                (out_interest,  interest_arr),
                (out_balance,   balance_arr),
                (out_ds,        total_ds),
            ]:
                audit.append(AuditEntry(
                    variable=f"{target_block_id}.{name}",
                    expression="cost_based_equal_repayment",
                    block_id=target_block_id,
                    values=arr.copy(),
                ))

        else:
            # ----------------------------------------------------------
            # CFADS-BASED MODE
            # total_debt is NOT fixed — it is derived as PV(CFADS/dscr_target).
            # DS is shaped proportional to CFADS so DSCR = dscr_target throughout.
            # Iterates until CFADS is self-consistent with the interest tax shield.
            #
            # Outer loop:
            #   Pass 0 — run subgraph with DS=0 (placeholder) → CFADS₀
            #   Each iteration:
            #     1. total_debt = PV(CFADS[repay] / dscr_target)   (recomputed)
            #     2. sculpt_solver(CFADS, total_debt) → DS  (scale = 1 by construction)
            #     3. Write DS → namespace
            #     4. Re-run subgraph → CFADS_new
            #     5. If rel_change(CFADS) < 1e-4 → converged
            # ----------------------------------------------------------
            _MAX_OUTER = 8
            _CFADS_TOL = 1e-4

            # Pass 0: subgraph with DS=0 to seed CFADS
            _tmp_audit = []
            for step in subgraph:
                if step.kind == "block":
                    self._execute_block(step.block, compiled, namespace, _tmp_audit)
            cfads = np.asarray(namespace.get(loop.target_expression, np.zeros(n)), dtype=np.float64)

            result: SculptingResult
            outer_converged = False
            rel_change = float("inf")

            for _outer in range(_MAX_OUTER):
                # Recompute total_debt as PV(CFADS/dscr_target) over repayment window
                cfads_repay = cfads[repay_start:repay_end]
                t_vec = np.arange(1, n_repay + 1, dtype=np.float64)
                pv_factors = 1.0 / (1.0 + r_per_period) ** t_vec
                total_debt = float(np.dot(cfads_repay / dscr_target, pv_factors))

                # Update namespace so downstream blocks see the derived debt amount
                namespace[debt_amount_key] = np.full(n, total_debt)

                result = sculpting_solver(
                    cfads=cfads,
                    total_debt=total_debt,
                    interest_rate_per_period=r_per_period,
                    dscr_target=dscr_target,
                    cod_period=cod,
                    debt_maturity_period=mat,
                    moratorium_periods=moratorium,
                    loop_id=loop.loop_id,
                    tolerance=loop.tolerance,
                    max_iterations=loop.max_iterations,
                )

                # Write sculpted DS → namespace using YAML-declared output names
                # Also write opening balance (before repayment) for consistent reporting
                _cfads_opening = np.zeros(n)
                _ob = total_debt
                for _t in range(cod, mat + 1):
                    _cfads_opening[_t] = _ob
                    _ob = max(0.0, _ob - float(result.principal_repayment[_t]))
                namespace[f"{target_block_id}.opening"]         = _cfads_opening
                namespace[f"{target_block_id}.{out_principal}"] = result.principal_repayment
                namespace[f"{target_block_id}.{out_interest}"]  = result.interest_payment
                namespace[f"{target_block_id}.{out_balance}"]   = result.outstanding_balance
                namespace[f"{target_block_id}.{out_ds}"]        = result.total_debt_service

                # Re-run subgraph → updated tax (with actual interest) → updated CFADS
                _tmp_audit = []
                for step in subgraph:
                    if step.kind == "block":
                        self._execute_block(step.block, compiled, namespace, _tmp_audit)
                cfads_new = np.asarray(namespace.get(loop.target_expression, np.zeros(n)), dtype=np.float64)

                # Convergence: relative L2 change in CFADS over repayment window
                cfads_norm = float(np.linalg.norm(cfads[repay_start:repay_end]))
                cfads_change = float(
                    np.linalg.norm(cfads_new[repay_start:repay_end] - cfads[repay_start:repay_end])
                )
                rel_change = cfads_change / max(cfads_norm, 1.0)
                cfads = cfads_new

                if rel_change < _CFADS_TOL:
                    outer_converged = True
                    break

            if not outer_converged:
                warnings.append(
                    f"CFADS-based sculpting: outer loop did not converge after "
                    f"{_MAX_OUTER} iterations (rel_change={rel_change:.4e}). "
                    "DSCR may deviate slightly from target."
                )

            convergence.append(ConvergenceMetadata(
                loop_id=loop.loop_id,
                converged=result.converged,
                iterations=result.iterations,
                final_residual=result.final_residual,
            ))

            for name, arr in [
                (out_principal, result.principal_repayment),
                (out_interest,  result.interest_payment),
                (out_balance,   result.outstanding_balance),
                (out_ds,        result.total_debt_service),
            ]:
                audit.append(AuditEntry(
                    variable=f"{target_block_id}.{name}",
                    expression="cfads_based_sculpting_solver",
                    block_id=target_block_id,
                    values=arr.copy(),
                ))

    def _run_fixed_point(
        self,
        loop: SolveLoop,
        subgraph: List[EvaluationStep],
        compiled: CompiledModel,
        namespace: Dict[str, Any],
        audit: List[AuditEntry],
        convergence: List[ConvergenceMetadata],
    ) -> None:
        """
        Fixed-point iteration: x_{n+1} = evaluate_subgraph(x_n).

        The free_variable is a scalar state that feeds into the subgraph and
        is updated by re-evaluating loop.target_expression after each pass.
        """
        ev = compiled.evaluator
        x_init = self._get_scalar(namespace, loop.free_variable, 0.0)

        def f(x: float) -> float:
            ns_trial = dict(namespace)
            ns_trial[loop.free_variable] = float(x)
            trial_audit: List[AuditEntry] = []
            for step in subgraph:
                if step.kind == "block":
                    self._execute_block(step.block, compiled, ns_trial, trial_audit)
            val = ev.evaluate(loop.target_expression, ns_trial)
            return float(np.asarray(val, dtype=np.float64).flat[0])

        result = fixed_point_solver(
            f=f,
            x_init=x_init,
            loop_id=loop.loop_id,
            tolerance=loop.tolerance,
            max_iterations=loop.max_iterations,
        )

        namespace[loop.free_variable] = result.solution
        for step in subgraph:
            if step.kind == "block":
                self._execute_block(step.block, compiled, namespace, audit)

        convergence.append(ConvergenceMetadata(
            loop_id=loop.loop_id,
            converged=result.converged,
            iterations=result.iterations,
            final_residual=result.final_residual,
            final_free_variable_value=result.solution,
        ))

    # ==================================================================
    # Internal: graph construction
    # ==================================================================

    def _build_block_dependency_graph(self, model_def: ModelDefinition) -> nx.DiGraph:
        """Block-level directed graph: edge A → B means B depends on A."""
        g = nx.DiGraph()
        blocks = model_def.calculation_blocks.blocks

        for block in blocks:
            g.add_node(block.block_id)

        for block in blocks:
            for inp in block.inputs:
                if inp.source.startswith("assumption.") or inp.source.startswith("phase."):
                    continue
                source_block = inp.source.split(".")[0]
                if source_block != block.block_id and g.has_node(source_block):
                    g.add_edge(source_block, block.block_id)

        return g

    def _build_output_dependency_graph(self, model_def: ModelDefinition) -> nx.DiGraph:
        """
        Output-variable-level graph for dirty-node tracking.
        Nodes: "block_id.output_name" and "assumption.name" and "phase.X"
        Edges: from source output → target block's outputs (transitively)
        """
        g = nx.DiGraph()

        # Assumption and phase nodes
        for a in model_def.assumption_schema.assumptions:
            g.add_node(f"assumption.{a.name}")
        for phase in ("is_construction", "is_operational", "is_debt_outstanding"):
            g.add_node(f"phase.{phase}")

        # Block output nodes
        for block in model_def.calculation_blocks.blocks:
            for out in block.outputs:
                g.add_node(f"{block.block_id}.{out.name}")

        # Edges: each block input source → each block output
        for block in model_def.calculation_blocks.blocks:
            for inp in block.inputs:
                for out in block.outputs:
                    g.add_edge(inp.source, f"{block.block_id}.{out.name}")

        return g

    def _topological_sort(
        self,
        dep_graph: nx.DiGraph,
        model_def: ModelDefinition,
    ) -> List[CalculationBlock]:
        """
        Return blocks in topological order (dependencies before dependents).
        Raises ValueError if an undeclared cycle is found.
        """
        # Remove edges that correspond to declared solve loops (these are the back-edges)
        g_dag = dep_graph.copy()
        declared_free_blocks = {
            sl.free_variable.split(".")[0]
            for sl in model_def.calculation_blocks.solve_loops
            if not sl.free_variable.startswith("assumption.")
        }
        # Remove back-edges from the free variable's block to its upstream dependencies
        # (the solver handles these)
        edges_to_remove = []
        for sl in model_def.calculation_blocks.solve_loops:
            if not sl.free_variable.startswith("assumption."):
                fv_block = sl.free_variable.split(".")[0]
                # Remove all incoming edges to fv_block from loop descendants
                # (simplified: remove in-edges from blocks downstream of target)
                for edge in list(g_dag.in_edges(fv_block)):
                    edges_to_remove.append(edge)

        for edge in edges_to_remove:
            if g_dag.has_edge(*edge):
                g_dag.remove_edge(*edge)

        try:
            sorted_ids = list(nx.topological_sort(g_dag))
        except nx.NetworkXUnfeasible as exc:
            raise ValueError(
                "Unexpected cycle in dependency graph. "
                "Declare a solve_loop for each intentional cycle."
            ) from exc

        block_by_id = {b.block_id: b for b in model_def.calculation_blocks.blocks}
        return [block_by_id[bid] for bid in sorted_ids if bid in block_by_id]

    def _build_evaluation_plan(
        self,
        sorted_blocks: List[CalculationBlock],
        solve_loops: List[SolveLoop],
    ) -> List[EvaluationStep]:
        """
        Convert a topologically sorted block list into an evaluation plan,
        inserting solve_loop wrappers where declared.

        Strategy: solve loops are placed BEFORE the first block in the loop's
        "target" subgraph.  All blocks in the loop subgraph are collected
        into the loop step's loop_subgraph, and excluded from the top-level plan.
        """
        if not solve_loops:
            return [EvaluationStep(kind="block", block=b) for b in sorted_blocks]

        plan: List[EvaluationStep] = []
        handled_loop_ids: set = set()

        # Collect all blocks owned by array_fixed_point loops.  These are excluded
        # from the main plan and evaluated exclusively by their loop solver.
        # For each such loop, record the insertion index (position of its first
        # owned block in topological order) so prerequisite blocks have already run.
        fm_owned: set = set()
        fp_loops: List[tuple] = []  # list of (insert_idx, SolveLoop)

        def _loop_depends_on_owned_blocks(sl: SolveLoop, owned_blocks: set[str]) -> bool:
            refs = [sl.free_variable, sl.target_expression, *sl.parameters.values()]
            for ref in refs:
                if not isinstance(ref, str):
                    continue
                for block_id in owned_blocks:
                    if f"{block_id}." in ref:
                        return True
            return False

        for sl in solve_loops:
            if sl.type == SolveLoopType.array_fixed_point:
                owned_set = set(sl.owned_blocks)
                fm_owned |= owned_set
                insert_idx = len(sorted_blocks)
                for i, b in enumerate(sorted_blocks):
                    if b.block_id in owned_set:
                        insert_idx = i
                        break
                fp_loops.append((insert_idx, sl))

        # 1. Prepend assumption-based loops that are NOT array_fixed_point.
        #    (subgraph = all blocks; the assumption scalar feeds every block)
        for sl in solve_loops:
            if sl.type == SolveLoopType.array_fixed_point:
                continue  # handled positionally below
            if sl.free_variable.startswith("assumption."):
                subgraph = [EvaluationStep(kind="block", block=b) for b in sorted_blocks]
                plan.append(EvaluationStep(
                    kind="solve_loop",
                    solve_loop=sl,
                    loop_subgraph=subgraph,
                ))
                handled_loop_ids.add(sl.loop_id)

        # 2. Iterate sorted_blocks; splice in each array_fixed_point loop just before
        #    its first owned block, then handle block-level loops normally.
        for i, block in enumerate(sorted_blocks):
            # Insert any array_fixed_point loops whose first owned block is at index i
            for insert_idx, fp_loop in fp_loops:
                if insert_idx == i and fp_loop.loop_id not in handled_loop_ids:
                    remaining = sorted_blocks[insert_idx:]
                    subgraph = [EvaluationStep(kind="block", block=b) for b in remaining]
                    plan.append(EvaluationStep(
                        kind="solve_loop",
                        solve_loop=fp_loop,
                        loop_subgraph=subgraph,
                    ))
                    handled_loop_ids.add(fp_loop.loop_id)

                    # Block-level loops whose free_variable block is owned by this
                    # array_fixed_point loop are never reached in the normal iteration
                    # (owned blocks are skipped). Insert them immediately after so that
                    # they run once the IDC loop has finalised debt_amount.
                    fp_owned_set = set(fp_loop.owned_blocks)
                    for sl in solve_loops:
                        if sl.loop_id in handled_loop_ids:
                            continue
                        fv_block = sl.free_variable.split(".")[0]
                        if fv_block in fp_owned_set or _loop_depends_on_owned_blocks(sl, fp_owned_set):
                            post_owned = [
                                b for b in sorted_blocks if b.block_id not in fp_owned_set
                            ]
                            sl_subgraph = [EvaluationStep(kind="block", block=b) for b in post_owned]
                            plan.append(EvaluationStep(
                                kind="solve_loop",
                                solve_loop=sl,
                                loop_subgraph=sl_subgraph,
                            ))
                            handled_loop_ids.add(sl.loop_id)

            # Skip blocks owned by any array_fixed_point loop
            if block.block_id in fm_owned:
                continue

            plan.append(EvaluationStep(kind="block", block=block))

            # Block-level loops: insert after the block that owns the free variable
            for sl in solve_loops:
                if sl.loop_id in handled_loop_ids:
                    continue
                if any(
                    _loop_depends_on_owned_blocks(sl, set(fp_loop.owned_blocks))
                    for _, fp_loop in fp_loops
                    if fp_loop.loop_id not in handled_loop_ids
                ):
                    continue
                fv_block = sl.free_variable.split(".")[0]
                if fv_block == block.block_id:
                    remaining_blocks = sorted_blocks[sorted_blocks.index(block) + 1:]
                    subgraph = [EvaluationStep(kind="block", block=b) for b in remaining_blocks]
                    plan.append(EvaluationStep(
                        kind="solve_loop",
                        solve_loop=sl,
                        loop_subgraph=subgraph,
                    ))
                    handled_loop_ids.add(sl.loop_id)
                    break

        return plan

    # ==================================================================
    # Internal: KPI computation
    # ==================================================================

    def _compute_kpis(
        self,
        compiled: CompiledModel,
        namespace: Dict[str, Any],
    ) -> Tuple[KPIResult, List[str]]:
        """
        Locate required time-series in namespace and call compute_all_kpis().
        Uses naming conventions from the standard solar IPP block library.
        Silently returns empty KPIResult if required arrays are not found.
        """
        n = compiled.n_periods
        ppy = compiled.periods_per_year
        cod = compiled.cod_period
        mat = compiled.debt_maturity_period

        # Look up arrays by convention (first match wins)
        equity_cf = self._find_array(
            namespace,
            ["cashflow_block.equity_cashflow", "equity_cashflow"],
            n,
        )
        project_cf = self._find_array(
            namespace,
            ["cashflow_block.project_cashflow", "project_cashflow"],
            n,
        )
        cfads = self._find_array(
            namespace,
            ["cashflow_block.cfads", "cfads"],
            n,
        )
        total_ds = self._find_array(
            namespace,
            ["debt_service_block.total_debt_service", "total_debt_service"],
            n,
        )
        balance = self._find_array(
            namespace,
            ["debt_service_block.outstanding_debt_balance", "outstanding_balance"],
            n,
        )
        is_debt = np.asarray(
            namespace.get("phase.is_debt_outstanding", np.zeros(n)),
            dtype=bool,
        )

        rate_pa = self._get_scalar(namespace, "assumption.interest_rate", 0.10)
        equity_target = self._get_scalar(
            namespace, "assumption.equity_irr_target", 0.15
        )

        # If required arrays are missing (model not yet fully wired), return empty KPIs
        if np.all(equity_cf == 0) and np.all(cfads == 0):
            return KPIResult(), []

        return compute_all_kpis(
            equity_cashflows=equity_cf,
            project_cashflows=project_cf,
            cfads=cfads,
            total_debt_service=total_ds,
            outstanding_balance=balance,
            is_debt_outstanding=is_debt,
            interest_rate_pa=rate_pa,
            equity_irr_target=equity_target,
            cod_period=cod,
            debt_maturity_period=mat,
            periods_per_year=ppy,
        )

    # ==================================================================
    # Utilities
    # ==================================================================

    def _get_scalar(
        self,
        namespace: Dict[str, Any],
        key: str,
        default: float = 0.0,
    ) -> float:
        val = namespace.get(key, default)
        if isinstance(val, np.ndarray):
            return float(val.flat[0])
        return float(val)

    def _find_scalar(
        self,
        namespace: Dict[str, Any],
        keys: List[str],
        default: float = 0.0,
    ) -> float:
        for k in keys:
            if k in namespace:
                return self._get_scalar(namespace, k)
        return default

    def _find_array(
        self,
        namespace: Dict[str, Any],
        keys: List[str],
        n: int,
    ) -> np.ndarray:
        for k in keys:
            if k in namespace:
                val = namespace[k]
                arr = np.asarray(val, dtype=np.float64)
                if arr.ndim == 0:
                    return np.full(n, float(arr))
                return arr
        return np.zeros(n)

    def downstream_variables(
        self,
        compiled: CompiledModel,
        changed_assumption: str,
    ) -> List[str]:
        """
        Return all output variable names that are downstream of the given
        assumption in the output dependency graph.
        Used by run_sensitivity() for dirty-node tracking.
        """
        g = compiled.output_graph
        key = f"assumption.{changed_assumption}"
        if key not in g:
            return []
        return list(nx.descendants(g, key))
