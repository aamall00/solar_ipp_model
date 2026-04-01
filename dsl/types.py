"""
DSL Type Definitions — Pydantic models for all DSL constructs.

Covers four top-level sections of a model definition file:
  1. project_skeleton
  2. assumption_schema
  3. calculation_blocks (including waterfall, ledger, solve_loops)
  4. model_wiring
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------


class ProjectType(str, Enum):
    solar_ipp = "solar_ipp"
    wind_ipp = "wind_ipp"
    toll_road = "toll_road"
    port = "port"


class AssumptionType(str, Enum):
    scalar = "scalar"
    time_series = "time_series"
    schedule = "schedule"
    curve = "curve"
    enum = "enum"


class DistributionType(str, Enum):
    triangular = "triangular"
    pert = "pert"
    lognormal = "lognormal"
    uniform = "uniform"


class BlockCategory(str, Enum):
    generation = "generation"
    revenue = "revenue"
    construction = "construction"
    idc = "idc"
    debt_sizing = "debt_sizing"
    debt_drawdown = "debt_drawdown"
    debt = "debt"
    opex = "opex"
    dsra = "dsra"
    depreciation = "depreciation"
    tax = "tax"
    cashflow = "cashflow"
    waterfall = "waterfall"
    returns = "returns"


class BlockType(str, Enum):
    standard = "standard"
    waterfall = "waterfall"
    ledger = "ledger"


class SolveLoopType(str, Enum):
    goal_seek = "goal_seek"
    sculpting = "sculpting"
    fixed_point = "fixed_point"
    array_fixed_point = "array_fixed_point"


class DebtStructure(str, Enum):
    level_annuity = "level_annuity"
    sculpted = "sculpted"
    bullet = "bullet"


class DepreciationMethod(str, Enum):
    slm = "slm"
    wdv = "wdv"


# ---------------------------------------------------------------------------
# Section 1: project_skeleton
# ---------------------------------------------------------------------------


class Milestones(BaseModel):
    """Period indices (0-based) for key project events."""
    financial_close: int = Field(ge=0, description="Period index of financial close")
    cod: int = Field(ge=0, description="Period index of Commercial Operation Date")
    debt_maturity: int = Field(ge=0, description="Period index of final debt repayment")

    @model_validator(mode="after")
    def check_ordering(self) -> "Milestones":
        if not (self.financial_close <= self.cod <= self.debt_maturity):
            raise ValueError(
                "Milestones must satisfy: financial_close <= cod <= debt_maturity. "
                f"Got financial_close={self.financial_close}, cod={self.cod}, "
                f"debt_maturity={self.debt_maturity}"
            )
        return self


class Phases(BaseModel):
    """
    Boolean masks over the full time axis.
    Auto-derived from milestones by the DSL parser; not user-specified.
    Each list has length == total_periods.
    """
    is_construction: List[bool] = Field(
        description="True during construction (financial_close to cod-1)"
    )
    is_operational: List[bool] = Field(
        description="True during operations (cod to end)"
    )
    is_debt_outstanding: List[bool] = Field(
        description="True while debt is outstanding (cod to debt_maturity)"
    )

    @model_validator(mode="after")
    def check_lengths_match(self) -> "Phases":
        lengths = {
            len(self.is_construction),
            len(self.is_operational),
            len(self.is_debt_outstanding),
        }
        if len(lengths) > 1:
            raise ValueError(
                f"All phase masks must have equal length. Got lengths: "
                f"is_construction={len(self.is_construction)}, "
                f"is_operational={len(self.is_operational)}, "
                f"is_debt_outstanding={len(self.is_debt_outstanding)}"
            )
        return self


class ProjectSkeleton(BaseModel):
    """Top-level project metadata and time axis definition."""
    model_config = {"protected_namespaces": ()}

    model_id: str = Field(description="Unique model identifier")
    project_type: ProjectType = Field(default=ProjectType.solar_ipp)
    currency: str = Field(default="INR")
    currency_unit: str = Field(default="Lakhs")
    periods_per_year: int = Field(default=4, ge=1, le=12)
    construction_periods: int = Field(ge=1)
    operations_periods: int = Field(ge=1)
    milestones: Milestones
    phases: Optional[Phases] = Field(
        default=None,
        description="Auto-derived by DSL parser; omit in source YAML"
    )

    @property
    def total_periods(self) -> int:
        return self.construction_periods + self.operations_periods

    @model_validator(mode="after")
    def check_milestones_against_periods(self) -> "ProjectSkeleton":
        total = self.construction_periods + self.operations_periods
        if self.milestones.debt_maturity >= total:
            raise ValueError(
                f"debt_maturity period ({self.milestones.debt_maturity}) must be "
                f"< total_periods ({total})"
            )
        if self.milestones.cod != self.construction_periods:
            raise ValueError(
                f"cod ({self.milestones.cod}) must equal construction_periods "
                f"({self.construction_periods}). Period 0 = financial close, "
                f"so COD = first operational period."
            )
        return self


# ---------------------------------------------------------------------------
# Section 2: assumption_schema
# ---------------------------------------------------------------------------


class AssumptionConstraints(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None

    @model_validator(mode="after")
    def min_lt_max(self) -> "AssumptionConstraints":
        if self.min is not None and self.max is not None:
            if self.min >= self.max:
                raise ValueError(
                    f"Constraint min ({self.min}) must be strictly less than max ({self.max})"
                )
        return self


class AssumptionSensitivity(BaseModel):
    vary: bool = Field(default=False)
    range_pct: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Fractional range e.g. 0.15 means ±15%"
    )
    distribution: DistributionType = Field(default=DistributionType.triangular)


class AssumptionDefinition(BaseModel):
    """
    Schema definition for a single assumption.
    Used in assumption_schema section and in the solar IPP schema YAML.
    """
    name: str
    type: AssumptionType
    unit: str = Field(description="e.g. INR_Lakhs, MW, percentage, ratio, years, GWh, kWh")
    value: Optional[Any] = Field(
        default=None,
        description="Concrete value; may be absent if defined via default_rule"
    )
    default_rule: Optional[str] = Field(
        default=None,
        description="Expression to derive a default e.g. '= cpi_assumption' or '= 0.025'"
    )
    constraints: AssumptionConstraints = Field(default_factory=AssumptionConstraints)
    sensitivity: AssumptionSensitivity = Field(default_factory=AssumptionSensitivity)

    @model_validator(mode="after")
    def must_have_value_or_rule(self) -> "AssumptionDefinition":
        if self.value is None and self.default_rule is None:
            # This is allowed at schema-definition time; flagged at ingestion time.
            pass
        return self

    @model_validator(mode="after")
    def validate_schedule_is_list(self) -> "AssumptionDefinition":
        if self.type == AssumptionType.schedule and self.value is not None:
            if not isinstance(self.value, list):
                raise ValueError(
                    f"Assumption '{self.name}' has type=schedule so value must be a list"
                )
        return self

    @model_validator(mode="after")
    def validate_constraints_against_value(self) -> "AssumptionDefinition":
        if self.value is not None and isinstance(self.value, (int, float)):
            c = self.constraints
            if c.min is not None and self.value < c.min:
                raise ValueError(
                    f"Assumption '{self.name}' value {self.value} < min constraint {c.min}"
                )
            if c.max is not None and self.value > c.max:
                raise ValueError(
                    f"Assumption '{self.name}' value {self.value} > max constraint {c.max}"
                )
        return self


class AssumptionSchema(BaseModel):
    """Collection of assumption definitions for a project type."""
    assumptions: List[AssumptionDefinition]

    def by_name(self, name: str) -> Optional[AssumptionDefinition]:
        return next((a for a in self.assumptions if a.name == name), None)

    def required_names(self) -> List[str]:
        """Assumptions with no value and no default_rule — must be user-supplied."""
        return [
            a.name for a in self.assumptions
            if a.value is None and a.default_rule is None
        ]

    def sensitivity_assumptions(self) -> List[AssumptionDefinition]:
        return [a for a in self.assumptions if a.sensitivity.vary]


# ---------------------------------------------------------------------------
# Section 3: calculation_blocks
# ---------------------------------------------------------------------------

# --- Block input / output descriptors ---


class BlockInput(BaseModel):
    name: str = Field(description="Local variable name within the block")
    source: str = Field(
        description=(
            "Dotted reference: 'assumption.{name}' or '{block_id}.{output_name}' "
            "or 'phase.{mask_name}'"
        )
    )

    @field_validator("source")
    @classmethod
    def source_format(cls, v: str) -> str:
        parts = v.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Block input source must be 'assumption.X', 'block_id.output', "
                f"or 'phase.X'. Got: '{v}'"
            )
        return v


class BlockOutput(BaseModel):
    name: str = Field(description="Output variable name (also used in wiring references)")
    unit: str = Field(default="", description="Unit of the output e.g. INR_Lakhs, kWh")


# --- Expression body ---


class CalculationStep(BaseModel):
    """A single assignment in a block body: target = expr."""
    target: str = Field(description="Variable name to assign to")
    expr: str = Field(description="Expression string in the DSL expression language")


# --- Waterfall bucket ---


class WaterfallBucket(BaseModel):
    name: str
    priority: int = Field(ge=1, description="1 = highest priority")
    target_expr: str = Field(description="Expression for the amount this bucket needs")
    recipient: str = Field(description="Destination of this bucket's funds (e.g. block output name)")


# --- Ledger additions/deductions ---


class LedgerFlow(BaseModel):
    """A single source contributing to or deducting from a ledger balance."""
    source: str = Field(description="Dotted reference to a block output or assumption")


# --- Solve loop declaration ---


class SolveLoop(BaseModel):
    loop_id: str
    type: SolveLoopType
    free_variable: str = Field(
        description="The variable to iterate: 'assumption.{name}' or '{block_id}.{output}'"
    )
    target_expression: str = Field(
        description="Expression whose value we want to equal target_value"
    )
    target_value: float
    tolerance: float = Field(default=1e-6, gt=0)
    max_iterations: int = Field(default=50, ge=1)
    owned_blocks: List[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of block_ids whose YAML body expressions are re-evaluated "
            "each iteration. Required for array_fixed_point loops. The executor "
            "re-evaluates these blocks in declared order each pass, updating namespace "
            "in-place. Blocks are excluded from the main evaluation plan."
        ),
    )
    parameters: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Solver-specific parameters declared in YAML rather than hardcoded in the "
            "executor. Keys and semantics are solver-type specific. For sculpting loops: "
            "interest_rate_var, moratorium_var, debt_sizing_mode_var, debt_amount_var, "
            "output_principal, output_interest, output_balance, output_ds."
        ),
    )

    @field_validator("free_variable")
    @classmethod
    def free_var_format(cls, v: str) -> str:
        if "." not in v:
            raise ValueError(
                f"free_variable must be 'assumption.X' or 'block_id.output'. Got: '{v}'"
            )
        return v

    @model_validator(mode="after")
    def array_fp_requires_owned_blocks(self) -> "SolveLoop":
        if self.type == SolveLoopType.array_fixed_point and not self.owned_blocks:
            raise ValueError(
                f"SolveLoop '{self.loop_id}' has type=array_fixed_point but "
                f"owned_blocks is empty. Declare the block_ids whose YAML expressions "
                f"form the circular dependency, in topological iteration order."
            )
        return self


# --- Main block model ---


class CalculationBlock(BaseModel):
    """
    A single calculation block in the model.
    block_type determines which sub-fields are required:
      - standard  → body required
      - waterfall → buckets required
      - ledger    → opening_balance, additions, deductions required
    """
    block_id: str
    category: BlockCategory
    block_type: BlockType = Field(default=BlockType.standard)
    description: Optional[str] = None

    inputs: List[BlockInput] = Field(default_factory=list)
    outputs: List[BlockOutput] = Field(default_factory=list)

    # Standard block
    body: Optional[List[CalculationStep]] = Field(
        default=None,
        description="Ordered list of assignment steps for standard blocks"
    )

    # Waterfall block
    available_cash_input: Optional[str] = Field(
        default=None,
        description="Source reference for cash available at top of waterfall"
    )
    buckets: Optional[List[WaterfallBucket]] = Field(
        default=None,
        description="Ordered priority buckets for waterfall blocks"
    )

    # Ledger block
    opening_balance: Optional[float] = Field(
        default=None,
        description="Balance at period 0 for ledger blocks"
    )
    additions: Optional[List[LedgerFlow]] = Field(
        default=None,
        description="Sources that increase the ledger balance"
    )
    deductions: Optional[List[LedgerFlow]] = Field(
        default=None,
        description="Sources that decrease the ledger balance"
    )

    @model_validator(mode="after")
    def validate_block_type_fields(self) -> "CalculationBlock":
        if self.block_type == BlockType.standard:
            if not self.body:
                raise ValueError(
                    f"Block '{self.block_id}' has block_type=standard but body is empty/absent"
                )
        elif self.block_type == BlockType.waterfall:
            if not self.buckets:
                raise ValueError(
                    f"Block '{self.block_id}' has block_type=waterfall but buckets is empty/absent"
                )
            if not self.available_cash_input:
                raise ValueError(
                    f"Block '{self.block_id}' has block_type=waterfall but "
                    f"available_cash_input is missing"
                )
            # Verify bucket priorities are unique and sequential
            priorities = [b.priority for b in self.buckets]
            if len(priorities) != len(set(priorities)):
                raise ValueError(
                    f"Block '{self.block_id}': waterfall bucket priorities must be unique"
                )
        elif self.block_type == BlockType.ledger:
            if self.opening_balance is None:
                raise ValueError(
                    f"Block '{self.block_id}' has block_type=ledger but opening_balance is absent"
                )
        return self

    @model_validator(mode="after")
    def validate_output_names_unique(self) -> "CalculationBlock":
        names = [o.name for o in self.outputs]
        if len(names) != len(set(names)):
            raise ValueError(
                f"Block '{self.block_id}': output names must be unique. Got: {names}"
            )
        return self

    def output_names(self) -> List[str]:
        return [o.name for o in self.outputs]

    def input_sources(self) -> List[str]:
        return [i.source for i in self.inputs]


class CalculationBlocks(BaseModel):
    blocks: List[CalculationBlock]
    solve_loops: List[SolveLoop] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_block_ids_unique(self) -> "CalculationBlocks":
        ids = [b.block_id for b in self.blocks]
        if len(ids) != len(set(ids)):
            dupes = [x for x in ids if ids.count(x) > 1]
            raise ValueError(f"Duplicate block_ids found: {list(set(dupes))}")
        return self

    @model_validator(mode="after")
    def validate_solve_loop_ids_unique(self) -> "CalculationBlocks":
        ids = [sl.loop_id for sl in self.solve_loops]
        if len(ids) != len(set(ids)):
            dupes = [x for x in ids if ids.count(x) > 1]
            raise ValueError(f"Duplicate solve_loop loop_ids found: {list(set(dupes))}")
        return self

    def block_by_id(self, block_id: str) -> Optional[CalculationBlock]:
        return next((b for b in self.blocks if b.block_id == block_id), None)

    def all_output_refs(self) -> Dict[str, str]:
        """Return mapping of 'block_id.output_name' -> unit."""
        refs: Dict[str, str] = {}
        for block in self.blocks:
            for out in block.outputs:
                refs[f"{block.block_id}.{out.name}"] = out.unit
        return refs


# ---------------------------------------------------------------------------
# Section 4: model_wiring
# ---------------------------------------------------------------------------


class Connection(BaseModel):
    """Explicit data-flow edge between a block output and a block input."""
    from_ref: str = Field(alias="from", description="'block_id.output_name'")
    to_ref: str = Field(alias="to", description="'block_id.input_name'")

    model_config = {"populate_by_name": True}

    @field_validator("from_ref", "to_ref")
    @classmethod
    def ref_format(cls, v: str) -> str:
        if "." not in v:
            raise ValueError(
                f"Connection references must be 'block_id.name'. Got: '{v}'"
            )
        return v


class OutputReports(BaseModel):
    """References to block outputs that populate each financial statement."""
    income_statement: List[str] = Field(default_factory=list)
    cash_flow_statement: List[str] = Field(default_factory=list)
    debt_schedule: List[str] = Field(default_factory=list)
    returns_summary: List[str] = Field(default_factory=list)


class ModelWiring(BaseModel):
    connections: List[Connection] = Field(default_factory=list)
    output_reports: OutputReports = Field(default_factory=OutputReports)


# ---------------------------------------------------------------------------
# Top-level: complete model definition
# ---------------------------------------------------------------------------


class ModelDefinition(BaseModel):
    """
    Complete model definition as loaded from a DSL YAML file.
    Corresponds to the four top-level sections.
    """
    model_config = {"protected_namespaces": ()}

    project_skeleton: ProjectSkeleton
    assumption_schema: AssumptionSchema
    calculation_blocks: CalculationBlocks
    model_wiring: ModelWiring

    @model_validator(mode="after")
    def validate_wiring_references(self) -> "ModelDefinition":
        """
        Check that every connection 'from' resolves to a known block output,
        and every connection 'to' resolves to a known block input.
        """
        all_outputs = self.calculation_blocks.all_output_refs()
        assumption_names = {
            f"assumption.{a.name}" for a in self.assumption_schema.assumptions
        }
        phase_refs = {
            "phase.is_construction",
            "phase.is_operational",
            "phase.is_debt_outstanding",
        }
        valid_sources = set(all_outputs.keys()) | assumption_names | phase_refs

        errors: List[str] = []
        for conn in self.model_wiring.connections:
            if conn.from_ref not in valid_sources:
                errors.append(
                    f"Connection from='{conn.from_ref}' does not resolve to any known "
                    f"block output or assumption"
                )
            # 'to' is a block_id.input_name — check block exists
            block_id, _, input_name = conn.to_ref.partition(".")
            block = self.calculation_blocks.block_by_id(block_id)
            if block is None:
                errors.append(
                    f"Connection to='{conn.to_ref}' references unknown block '{block_id}'"
                )
            elif input_name not in [i.name for i in block.inputs]:
                errors.append(
                    f"Connection to='{conn.to_ref}': block '{block_id}' has no input "
                    f"named '{input_name}'"
                )

        if errors:
            raise ValueError("Model wiring validation errors:\n" + "\n".join(errors))
        return self

    @model_validator(mode="after")
    def validate_report_references(self) -> "ModelDefinition":
        """Ensure all output_reports references resolve to known block outputs."""
        all_outputs = self.calculation_blocks.all_output_refs()
        all_report_refs: List[str] = (
            self.model_wiring.output_reports.income_statement
            + self.model_wiring.output_reports.cash_flow_statement
            + self.model_wiring.output_reports.debt_schedule
            + self.model_wiring.output_reports.returns_summary
        )
        errors = [
            f"output_reports reference '{ref}' does not resolve to any block output"
            for ref in all_report_refs
            if ref not in all_outputs
        ]
        if errors:
            raise ValueError("Output report reference errors:\n" + "\n".join(errors))
        return self

    @model_validator(mode="after")
    def validate_solve_loop_free_variables(self) -> "ModelDefinition":
        """Ensure solve loop free variables reference valid assumptions or block outputs."""
        assumption_names = {
            f"assumption.{a.name}" for a in self.assumption_schema.assumptions
        }
        all_outputs = self.calculation_blocks.all_output_refs()
        valid = assumption_names | set(all_outputs.keys())

        errors = [
            f"SolveLoop '{sl.loop_id}' free_variable='{sl.free_variable}' is not a "
            f"known assumption or block output"
            for sl in self.calculation_blocks.solve_loops
            if sl.free_variable not in valid
        ]
        if errors:
            raise ValueError("Solve loop variable errors:\n" + "\n".join(errors))
        return self


# ---------------------------------------------------------------------------
# Runtime result containers (used by the executor, not the DSL parser)
# ---------------------------------------------------------------------------


class KPIResult(BaseModel):
    """Key performance indicators computed for a single model run."""
    equity_irr: Optional[float] = None
    project_irr: Optional[float] = None
    min_dscr: Optional[float] = None
    avg_dscr: Optional[float] = None
    llcr: Optional[float] = None
    plcr: Optional[float] = None
    npv_equity: Optional[float] = None
    debt_payback_period: Optional[float] = None
    peak_debt_outstanding: Optional[float] = None

    def is_bankable(self, min_dscr_threshold: float = 1.10) -> bool:
        if self.min_dscr is None:
            return False
        return self.min_dscr >= min_dscr_threshold


class ConvergenceMetadata(BaseModel):
    """Convergence status for a single solve loop."""
    loop_id: str
    converged: bool
    iterations: int
    final_residual: float
    final_free_variable_value: Optional[float] = None


class ValidationWarning(BaseModel):
    """Non-fatal warning from model validation or KPI range checking."""
    code: str
    message: str
    severity: str = Field(default="warning", pattern="^(warning|info)$")


class ValidationResult(BaseModel):
    """Aggregated validation output from assumption ingestion or DSL parsing."""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[ValidationWarning] = Field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.valid = False
        self.errors.append(msg)

    def add_warning(self, code: str, msg: str, severity: str = "warning") -> None:
        self.warnings.append(ValidationWarning(code=code, message=msg, severity=severity))
