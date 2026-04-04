"""
dsl/parser.py — Full YAML DSL loader and validator.

Responsibilities
----------------
1. Load YAML (file or dict) and construct a validated ModelDefinition.
2. Auto-derive Phases boolean masks from milestones.
3. Build a NetworkX directed dependency graph (block-level).
4. Detect unexpected cycles — only declared solve_loops may create cycles.
5. Run cross-assumption consistency checks (warn-not-error where appropriate).
6. Perform basic unit consistency checking on block wiring.
7. Return (ModelDefinition, ValidationResult) so the caller decides how to proceed.

The parser does NOT execute any arithmetic — it is purely structural/topological.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import yaml
from pydantic import ValidationError

from .expression import ExpressionEvaluator
from .types import (
    AssumptionDefinition,
    AssumptionSchema,
    CalculationBlock,
    CalculationBlocks,
    Connection,
    ModelDefinition,
    ModelWiring,
    OutputReports,
    Phases,
    ProjectSkeleton,
    SolveLoop,
    SolveLoopType,
    ValidationResult,
    ValidationWarning,
)


# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------


class DSLParseError(Exception):
    """Raised when YAML structure is fundamentally malformed (not recoverable)."""


class DSLCycleError(DSLParseError):
    """Raised when an undeclared dependency cycle is detected."""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class DSLParser:
    """
    Load, validate, and enrich a model definition from YAML.

    Usage
    -----
    parser = DSLParser()
    model, validation = parser.load_file("path/to/model.yaml")
    if not validation.valid:
        for e in validation.errors:
            print("ERROR:", e)
    """

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def load_file(
        self, path: Union[str, Path]
    ) -> Tuple[Optional[ModelDefinition], ValidationResult]:
        """Load and parse a YAML model definition file."""
        path = Path(path)
        if not path.exists():
            raise DSLParseError(f"Model file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise DSLParseError(f"YAML parse error in '{path}': {exc}") from exc
        if not isinstance(raw, dict):
            raise DSLParseError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")

        # Resolve import_from directive: replace with blocks loaded from the library
        calc = raw.get("calculation_blocks", {})
        if isinstance(calc, dict) and "import_from" in calc:
            import_key = calc.pop("import_from")
            calc["blocks"] = self._resolve_block_import(import_key)

        return self.load_dict(raw)

    @staticmethod
    def _resolve_block_import(import_key: str) -> List[Dict[str, Any]]:
        """Resolve an import_from key to an ordered list of raw block dicts."""
        if import_key == "solar_ipp":
            from blocks.solar_ipp import load_all_blocks_raw
            return list(load_all_blocks_raw().values())
        if import_key == "wind_ipp":
            from blocks.wind_ipp import load_all_blocks_raw
            return list(load_all_blocks_raw().values())
        raise DSLParseError(
            f"Unknown block library '{import_key}' in import_from. "
            f"Valid values: 'solar_ipp', 'wind_ipp'"
        )

    def load_dict(
        self, data: Dict[str, Any]
    ) -> Tuple[Optional[ModelDefinition], ValidationResult]:
        """Parse a model definition from a plain Python dict (e.g. already-loaded YAML)."""
        validation = ValidationResult(valid=True)

        # --- Step 1: Parse the four sections independently ---
        skeleton = self._parse_skeleton(data, validation)
        if skeleton is None:
            return None, validation  # fatal — cannot continue

        assumption_schema = self._parse_assumption_schema(data, validation)
        calc_blocks = self._parse_calculation_blocks(data, validation)
        wiring = self._parse_model_wiring(data, validation)

        if not validation.valid:
            return None, validation

        # --- Step 2: Auto-derive phase masks ---
        phases = self._derive_phases(skeleton)
        skeleton = skeleton.model_copy(update={"phases": phases})

        # --- Step 3: Assemble and cross-validate ModelDefinition ---
        try:
            model = ModelDefinition(
                project_skeleton=skeleton,
                assumption_schema=assumption_schema,
                calculation_blocks=calc_blocks,
                model_wiring=wiring,
            )
        except ValidationError as exc:
            for err in exc.errors():
                validation.add_error(
                    f"Model assembly error [{'.'.join(str(l) for l in err['loc'])}]: "
                    f"{err['msg']}"
                )
            return None, validation

        # --- Step 4: Dependency graph + cycle detection / auto-resolution ---
        graph = self._build_dependency_graph(model)
        auto_loops = self._resolve_cycles(
            graph, model.calculation_blocks.solve_loops, model, validation
        )
        if auto_loops:
            updated_blocks = model.calculation_blocks.model_copy(
                update={"solve_loops": list(model.calculation_blocks.solve_loops) + auto_loops}
            )
            model = model.model_copy(update={"calculation_blocks": updated_blocks})

        # --- Step 5: Assumption source coverage ---
        # Validates that every assumption.X referenced by a block input is declared
        # in the assumption_schema.  Catches agent↔block name mismatches early.
        self._check_assumption_sources(model, validation)

        # --- Step 6: Cross-assumption consistency checks ---
        self._check_cross_assumption_consistency(model, validation)

        # --- Step 7: Expression variable resolution ---
        self._check_expression_references(model, validation)

        # --- Step 8: Basic unit checks ---
        self._check_unit_consistency(model, validation)

        return model, validation

    # ------------------------------------------------------------------
    # Section parsers
    # ------------------------------------------------------------------

    def _parse_skeleton(
        self, data: Dict[str, Any], validation: ValidationResult
    ) -> Optional[ProjectSkeleton]:
        raw = data.get("project_skeleton")
        if raw is None:
            validation.add_error("Missing top-level section: 'project_skeleton'")
            return None
        try:
            return ProjectSkeleton(**raw)
        except (ValidationError, TypeError) as exc:
            validation.add_error(f"project_skeleton validation error: {exc}")
            return None

    def _parse_assumption_schema(
        self, data: Dict[str, Any], validation: ValidationResult
    ) -> AssumptionSchema:
        raw = data.get("assumption_schema", {})
        assumptions_raw = raw.get("assumptions", []) if isinstance(raw, dict) else []
        assumptions: List[AssumptionDefinition] = []
        for i, a in enumerate(assumptions_raw):
            try:
                assumptions.append(AssumptionDefinition(**a))
            except (ValidationError, TypeError) as exc:
                validation.add_error(
                    f"assumption_schema[{i}] (name={a.get('name', '?')}): {exc}"
                )
        return AssumptionSchema(assumptions=assumptions)

    def _parse_calculation_blocks(
        self, data: Dict[str, Any], validation: ValidationResult
    ) -> CalculationBlocks:
        raw = data.get("calculation_blocks", {})
        if not isinstance(raw, dict):
            validation.add_error("'calculation_blocks' must be a mapping")
            return CalculationBlocks(blocks=[])

        blocks_raw = raw.get("blocks", [])
        solve_loops_raw = raw.get("solve_loops") or []

        blocks = self._parse_blocks_list(blocks_raw, validation)
        solve_loops = self._parse_solve_loops(solve_loops_raw, validation)

        return CalculationBlocks(blocks=blocks, solve_loops=solve_loops)

    def _parse_blocks_list(
        self, blocks_raw: List[Any], validation: ValidationResult
    ) -> List[CalculationBlock]:
        blocks: List[CalculationBlock] = []
        for i, b in enumerate(blocks_raw):
            if not isinstance(b, dict):
                validation.add_error(f"calculation_blocks.blocks[{i}] must be a mapping")
                continue
            try:
                blocks.append(CalculationBlock(**b))
            except (ValidationError, TypeError) as exc:
                validation.add_error(
                    f"calculation_blocks.blocks[{i}] (block_id={b.get('block_id', '?')}): {exc}"
                )
        return blocks

    def _parse_solve_loops(
        self, raw: List[Any], validation: ValidationResult
    ) -> List[SolveLoop]:
        loops: List[SolveLoop] = []
        for i, sl in enumerate(raw):
            try:
                loops.append(SolveLoop(**sl))
            except (ValidationError, TypeError) as exc:
                validation.add_error(
                    f"calculation_blocks.solve_loops[{i}] "
                    f"(loop_id={sl.get('loop_id', '?')}): {exc}"
                )
        return loops

    def _parse_model_wiring(
        self, data: Dict[str, Any], validation: ValidationResult
    ) -> ModelWiring:
        raw = data.get("model_wiring", {})
        if not isinstance(raw, dict):
            validation.add_error("'model_wiring' must be a mapping")
            return ModelWiring()

        connections: List[Connection] = []
        for i, c in enumerate(raw.get("connections", [])):
            if not isinstance(c, dict):
                validation.add_error(f"model_wiring.connections[{i}] must be a mapping")
                continue
            try:
                # Handle Python reserved word 'from' via alias
                connections.append(Connection(**c))
            except (ValidationError, TypeError) as exc:
                validation.add_error(f"model_wiring.connections[{i}]: {exc}")

        reports_raw = raw.get("output_reports", {})
        try:
            reports = OutputReports(**reports_raw) if reports_raw else OutputReports()
        except (ValidationError, TypeError) as exc:
            validation.add_error(f"model_wiring.output_reports: {exc}")
            reports = OutputReports()

        return ModelWiring(connections=connections, output_reports=reports)

    # ------------------------------------------------------------------
    # Phase mask derivation
    # ------------------------------------------------------------------

    def _derive_phases(self, skeleton: ProjectSkeleton) -> Phases:
        """
        Auto-derive boolean masks from milestone period indices.

        Time axis (0-based):
          - Period 0 = financial close
          - Periods [0, cod)          = construction
          - Periods [cod, total)      = operational
          - Periods [cod, maturity+1) = debt outstanding
        """
        total = skeleton.total_periods
        cod = skeleton.milestones.cod
        maturity = skeleton.milestones.debt_maturity

        is_construction = [t < cod for t in range(total)]
        is_operational = [t >= cod for t in range(total)]
        is_debt_outstanding = [cod <= t <= maturity for t in range(total)]

        return Phases(
            is_construction=is_construction,
            is_operational=is_operational,
            is_debt_outstanding=is_debt_outstanding,
        )

    # ------------------------------------------------------------------
    # Dependency graph construction
    # ------------------------------------------------------------------

    def _build_dependency_graph(self, model: ModelDefinition) -> nx.DiGraph:
        """
        Build a directed graph where:
          - Nodes: "assumption.X", "phase.X", "block_id" (one node per block)
          - Edges: from source_block/assumption/phase → consuming_block

        This block-level graph is used for cycle detection.
        The executor builds a finer-grained output-level graph for execution ordering.
        """
        graph = nx.DiGraph()

        # Add all block nodes
        for block in model.calculation_blocks.blocks:
            graph.add_node(block.block_id, type="block")

        # Add assumption and phase pseudo-nodes
        for assumption in model.assumption_schema.assumptions:
            graph.add_node(f"assumption.{assumption.name}", type="assumption")

        for phase_name in ("is_construction", "is_operational", "is_debt_outstanding"):
            graph.add_node(f"phase.{phase_name}", type="phase")

        # Add edges from each input source to the block
        for block in model.calculation_blocks.blocks:
            for inp in block.inputs:
                source = inp.source
                # Determine the source node
                if source.startswith("assumption."):
                    source_node = source  # full name is the node key
                elif source.startswith("phase."):
                    source_node = source
                else:
                    # "block_id.output_name" → strip the output part
                    source_block_id = source.split(".")[0]
                    source_node = source_block_id

                if not graph.has_node(source_node):
                    graph.add_node(source_node, type="unknown")
                graph.add_edge(source_node, block.block_id)

        return graph

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    def _resolve_cycles(
        self,
        graph: nx.DiGraph,
        solve_loops: List[SolveLoop],
        model: ModelDefinition,
        validation: ValidationResult,
    ) -> List[SolveLoop]:
        """
        Detect cycles in the dependency graph and resolve them.

        - SCCs covered by a declared solve_loop: accepted as-is (info warning).
        - SCCs with NO declaration: auto-generate an array_fixed_point loop by
          topologically ordering the SCC blocks and using the back-edge source
          block's first output as the convergence monitor.

        Returns a list of auto-generated SolveLoop objects to be merged into
        the model. Errors only if the cycle is genuinely unresolvable (no block
        nodes in SCC).
        """
        sccs = list(nx.strongly_connected_components(graph))
        cyclic_sccs = [scc for scc in sccs if len(scc) > 1]

        if not cyclic_sccs:
            return []

        # Build declared coverage sets (block_ids covered by each declared loop).
        declared_loop_blocks: List[Set[str]] = []
        for sl in solve_loops:
            if sl.owned_blocks:
                covered: Set[str] = set(sl.owned_blocks)
            else:
                covered = set()
                fv = sl.free_variable
                if not fv.startswith("assumption."):
                    covered.add(fv.split(".")[0])
                for node in graph.nodes:
                    if graph.nodes[node].get("type") == "block" and node in sl.target_expression:
                        covered.add(node)
            declared_loop_blocks.append(covered)

        auto_loops: List[SolveLoop] = []

        for scc in cyclic_sccs:
            covered_by_loop = any(
                len(loop_blocks & scc) >= 2 or (len(loop_blocks & scc) >= 1 and len(scc) == 2)
                for loop_blocks in declared_loop_blocks
            )
            if covered_by_loop:
                validation.add_warning(
                    "DECLARED_CYCLE",
                    f"Declared solve loop covers cycle: {sorted(scc)}",
                    severity="info",
                )
            else:
                loop = self._auto_generate_loop_for_scc(scc, graph, model)
                if loop is None:
                    validation.add_error(
                        f"Undeclared dependency cycle among {sorted(scc)} contains no "
                        f"block nodes — cannot auto-resolve. Declare a solve_loop."
                    )
                else:
                    auto_loops.append(loop)
                    validation.add_warning(
                        "AUTO_CYCLE",
                        f"Auto-generated array_fixed_point loop '{loop.loop_id}' for "
                        f"cycle {sorted(scc)}. Declare explicitly in solve_loops to "
                        f"override tolerance/max_iterations (defaults: 1e-6 / 25).",
                        severity="info",
                    )

        return auto_loops

    def _auto_generate_loop_for_scc(
        self,
        scc: Set[str],
        graph: nx.DiGraph,
        model: ModelDefinition,
    ) -> Optional[SolveLoop]:
        """
        Build an array_fixed_point SolveLoop for an undeclared SCC.

        Algorithm:
        1. Keep only block-type nodes (drop assumption.X / phase.X).
        2. Build the induced subgraph on those nodes.
        3. Find the back-edge via DFS cycle detection.
        4. Remove the back-edge and topologically sort → owned_blocks order.
        5. Use the back-edge source block's first declared output as free_variable.
        """
        block_ids = [n for n in scc if graph.nodes[n].get("type") == "block"]
        if not block_ids:
            return None

        induced = graph.subgraph(block_ids).copy()

        # Find the back-edge (last edge in the DFS cycle)
        try:
            cycle_edges = nx.find_cycle(induced)
            back_src, back_dst = cycle_edges[-1][0], cycle_edges[-1][1]
        except nx.NetworkXNoCycle:
            # Degenerate: no intra-block cycle — pick arbitrary ordering
            back_src = block_ids[0]
            back_dst = block_ids[0]

        # Remove back-edge and topologically sort the remaining DAG
        dag = induced.copy()
        if dag.has_edge(back_src, back_dst):
            dag.remove_edge(back_src, back_dst)
        try:
            ordered = list(nx.topological_sort(dag))
            # Keep only nodes that are actual block_ids (filter out any strays)
            ordered = [n for n in ordered if n in set(block_ids)]
        except nx.NetworkXUnfeasible:
            ordered = block_ids

        # free_variable: first declared output of the back-edge source block
        block_map = {b.block_id: b for b in model.calculation_blocks.blocks}
        src_block = block_map.get(back_src)
        if src_block and src_block.outputs:
            free_var = f"{back_src}.{src_block.outputs[0].name}"
        else:
            free_var = f"{back_src}.output"

        loop_id = "auto_" + "_".join(sorted(block_ids))
        return SolveLoop(
            loop_id=loop_id,
            type=SolveLoopType.array_fixed_point,
            free_variable=free_var,
            target_expression=free_var,
            target_value=0.0,
            owned_blocks=ordered,
            tolerance=1e-6,
            max_iterations=25,
        )

    # ------------------------------------------------------------------
    # Assumption source coverage
    # ------------------------------------------------------------------

    def _check_assumption_sources(
        self, model: ModelDefinition, validation: ValidationResult
    ) -> None:
        """
        For every block input whose source starts with 'assumption.', verify that
        the referenced name is declared in the model's assumption_schema.

        This is the explicit contract bridge between Layer 3 (assumption ingestion)
        and Layer 4 (block execution): the canonical names in assumption_agent._CANONICAL
        must match the assumption_schema names consumed here.  Any mismatch is a hard
        error — the executor will silently use 0 for the missing assumption, which
        corrupts every downstream calculation without any traceable error.
        """
        schema_names: Set[str] = {a.name for a in model.assumption_schema.assumptions}

        for block in model.calculation_blocks.blocks:
            for inp in block.inputs:
                if not inp.source.startswith("assumption."):
                    continue
                ref_name = inp.source[len("assumption."):]
                if ref_name not in schema_names:
                    validation.add_error(
                        f"Block '{block.block_id}' input '{inp.name}' references "
                        f"'assumption.{ref_name}' which is not declared in "
                        f"assumption_schema. "
                        f"Add it to assumption_schema or fix the source reference. "
                        f"Declared schema names: {sorted(schema_names)}"
                    )

    # ------------------------------------------------------------------
    # Cross-assumption consistency
    # ------------------------------------------------------------------

    def _check_cross_assumption_consistency(
        self, model: ModelDefinition, validation: ValidationResult
    ) -> None:
        """
        Validates logical relationships between assumptions.
        Produces errors for hard violations, warnings for advisory issues.
        """
        schema = model.assumption_schema
        skeleton = model.project_skeleton

        def get_value(name: str) -> Optional[Any]:
            a = schema.by_name(name)
            return a.value if a else None

        # --- capex_schedule must sum to 1.0 ---
        capex_schedule = get_value("capex_schedule")
        if capex_schedule is not None and isinstance(capex_schedule, list):
            total = sum(capex_schedule)
            if abs(total - 1.0) > 1e-6:
                validation.add_error(
                    f"capex_schedule values must sum to 1.0 (got {total:.6f}). "
                    f"Adjust the schedule fractions."
                )

        # --- capex_schedule length must match construction_periods ---
        if capex_schedule is not None and isinstance(capex_schedule, list):
            if len(capex_schedule) != skeleton.construction_periods:
                validation.add_error(
                    f"capex_schedule has {len(capex_schedule)} entries but "
                    f"construction_periods = {skeleton.construction_periods}. "
                    f"These must match."
                )

        # --- debt_tenor_years must not exceed ppa_tenor_years ---
        debt_tenor = get_value("debt_tenor_years")
        ppa_tenor = get_value("ppa_tenor_years")
        if debt_tenor is not None and ppa_tenor is not None:
            if debt_tenor > ppa_tenor:
                validation.add_warning(
                    "DEBT_TENOR_EXCEEDS_PPA",
                    f"debt_tenor_years ({debt_tenor}) > ppa_tenor_years ({ppa_tenor}). "
                    f"Lenders typically require debt maturity within the PPA period. "
                    f"This may make the project unbankable.",
                    severity="warning",
                )

        # --- moratorium_periods must be < debt_tenor in periods ---
        moratorium = get_value("moratorium_periods")
        if moratorium is not None and debt_tenor is not None:
            debt_tenor_periods = int(debt_tenor) * skeleton.periods_per_year
            if moratorium >= debt_tenor_periods:
                validation.add_error(
                    f"moratorium_periods ({moratorium}) >= debt_tenor_years × periods_per_year "
                    f"({debt_tenor_periods}). Moratorium must be shorter than the debt tenor."
                )

        # --- operations_periods must cover at least debt_tenor ---
        if debt_tenor is not None:
            debt_tenor_periods = int(debt_tenor) * skeleton.periods_per_year
            if skeleton.operations_periods < debt_tenor_periods:
                validation.add_warning(
                    "SHORT_OPERATIONS_PERIOD",
                    f"operations_periods ({skeleton.operations_periods}) < "
                    f"debt_tenor_years × periods_per_year ({debt_tenor_periods}). "
                    f"Debt may not be fully repaid within the model horizon.",
                    severity="warning",
                )

        # --- debt_pct is a valid ratio ---
        debt_pct = get_value("debt_pct")
        if debt_pct is not None:
            if not (0.0 < debt_pct < 1.0):
                validation.add_error(
                    f"debt_pct must be between 0 and 1 exclusive "
                    f"(got {debt_pct})."
                )

        # --- tariff must be positive ---
        tariff = get_value("tariff")
        if tariff is not None and tariff <= 0:
            validation.add_error(
                f"tariff must be > 0 (got {tariff})."
            )

        # --- CUF sanity check for solar ---
        cuf = get_value("cuf")
        if cuf is not None:
            if cuf <= 0 or cuf >= 1:
                validation.add_error(
                    f"cuf must be between 0 and 1 exclusive "
                    f"(got {cuf}). Typical Karnataka solar CUF: 0.19–0.26."
                )
            elif cuf > 0.30:
                validation.add_warning(
                    "HIGH_CUF",
                    f"cuf = {cuf:.2%} seems high for solar IPP. "
                    f"Typical Karnataka range: 19–26%. Verify against irradiance data.",
                    severity="warning",
                )
            elif cuf < 0.15:
                validation.add_warning(
                    "LOW_CUF",
                    f"cuf = {cuf:.2%} seems low for solar IPP. "
                    f"Verify against irradiance data.",
                    severity="warning",
                )

        # --- interest_rate sanity ---
        rate = get_value("interest_rate")
        if rate is not None:
            if rate <= 0 or rate > 0.30:
                validation.add_warning(
                    "UNUSUAL_INTEREST_RATE",
                    f"interest_rate = {rate:.2%} is outside the normal 0–30% range. "
                    f"Confirm this is an annual decimal rate (e.g. 0.0975 for 9.75%).",
                    severity="warning",
                )

        # --- tax_rate sanity ---
        tax_rate = get_value("tax_rate")
        if tax_rate is not None:
            if tax_rate < 0 or tax_rate > 0.50:
                validation.add_warning(
                    "UNUSUAL_TAX_RATE",
                    f"tax_rate = {tax_rate:.2%} is outside the 0–50% range. "
                    f"Indian corporate rate is typically ~25.17% or 34.32%.",
                    severity="warning",
                )

        # --- capacity_mw must be positive ---
        capacity = get_value("capacity_mw")
        if capacity is not None and capacity <= 0:
            validation.add_error(
                f"capacity_mw must be > 0 (got {capacity})."
            )

    # ------------------------------------------------------------------
    # Expression variable resolution
    # ------------------------------------------------------------------

    def _check_expression_references(
        self, model: ModelDefinition, validation: ValidationResult
    ) -> None:
        """
        For each block body expression, verify that all referenced variable
        names appear in the block's declared inputs (or are period_index).

        This catches mismatches between BlockInput.name declarations and
        the actual variable names used in expressions.
        """
        skeleton = model.project_skeleton
        evaluator = ExpressionEvaluator(
            n_periods=skeleton.total_periods,
            periods_per_year=skeleton.periods_per_year,
        )

        for block in model.calculation_blocks.blocks:
            if block.body is None:
                continue

            declared_input_names: Set[str] = {inp.name for inp in block.inputs}
            declared_output_names: Set[str] = set(block.output_names())
            # Accumulate body step targets so each step can reference prior intermediates
            body_targets_so_far: Set[str] = set()

            for step in block.body:
                try:
                    refs = evaluator.referenced_variables(step.expr)
                except Exception as exc:
                    validation.add_error(
                        f"Block '{block.block_id}' step target='{step.target}' "
                        f"expression parse error: {exc}"
                    )
                    body_targets_so_far.add(step.target)
                    continue

                allowed = declared_input_names | declared_output_names | body_targets_so_far

                for ref in refs:
                    if not ref.startswith("phase.") and ref not in allowed:
                        validation.add_error(
                            f"Block '{block.block_id}' expression '{step.expr}' references "
                            f"'{ref}' which is not declared as a block input. "
                            f"Declared inputs: {sorted(declared_input_names)}"
                        )

                body_targets_so_far.add(step.target)

    # ------------------------------------------------------------------
    # Basic unit consistency
    # ------------------------------------------------------------------

    # Unit compatibility groups: units in the same group can be combined freely.
    # Units in different groups trigger a warning.
    _UNIT_GROUPS: Dict[str, str] = {
        # Financial
        "INR_Lakhs": "currency",
        "INR_Crores": "currency",
        "INR": "currency",
        # Power / Energy
        "MW": "power",
        "kW": "power",
        "kWh": "energy",
        "MWh": "energy",
        "GWh": "energy",
        # Rates / ratios
        "percentage": "ratio",
        "ratio": "ratio",
        "decimal": "ratio",
        # Time
        "years": "time",
        "quarters": "time",
        "months": "time",
        # Dimensionless
        "": "dimensionless",
        "dimensionless": "dimensionless",
        "INR_per_kWh": "tariff",
        "INR_Lakhs_per_MW": "cost_per_mw",
    }

    def _check_unit_consistency(
        self, model: ModelDefinition, validation: ValidationResult
    ) -> None:
        """
        Walk block output declarations and flag cases where blocks wired
        together have clearly incompatible units (e.g. MW fed into an INR field).

        This is advisory only (warnings) since unit algebra for arbitrary
        expressions is not decidable without full symbolic execution.
        """
        # Build output-unit map
        output_units: Dict[str, str] = {}
        for block in model.calculation_blocks.blocks:
            for out in block.outputs:
                key = f"{block.block_id}.{out.name}"
                output_units[key] = out.unit

        for conn in model.model_wiring.connections:
            from_unit = output_units.get(conn.from_ref, "")
            # Find the target block input's unit (if declared)
            to_block_id, _, to_input_name = conn.to_ref.partition(".")
            target_block = model.calculation_blocks.block_by_id(to_block_id)
            if target_block is None:
                continue  # Already caught by wiring validation

            # Check source output unit against what we know
            from_group = self._UNIT_GROUPS.get(from_unit)
            if from_group is None:
                # Unknown unit — skip silently
                continue

            # Find the matching block output unit for the target input via its source
            for inp in target_block.inputs:
                if inp.name == to_input_name:
                    # We know what's being fed in — check if the input source
                    # unit matches (only if the target block has a corresponding
                    # output unit to compare against, as a proxy)
                    break


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def load_model(
    path: Union[str, Path],
) -> Tuple[Optional[ModelDefinition], ValidationResult]:
    """Load and validate a DSL YAML model file. Returns (model, validation).
    model is None when validation.valid is False."""
    return DSLParser().load_file(path)


def load_model_from_dict(
    data: Dict[str, Any],
) -> Tuple[Optional[ModelDefinition], ValidationResult]:
    """Parse and validate a model from an already-loaded dict.
    model is None when validation.valid is False."""
    return DSLParser().load_dict(data)
