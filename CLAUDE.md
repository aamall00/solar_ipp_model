# solar_ipp_model

A DSL-driven renewable project finance engine. Transforms project descriptions or YAML templates into fully executable financial models for solar and wind power plants.

**Core principle**: All arithmetic is deterministic and auditable in Python/NumPy. LLMs are used ONLY for structured extraction and model assembly — never for calculations. No module specific behavior to be hardcoded in the business layer. The objective is that a user can simply add yaml files to bring in additional blocks in computation eg capex subsidy, equity bridge loan without having to modify code

## Architecture Overview

Multi-stage pipeline:

```
[Natural Language] → PortfolioAgent → AssumptionAgent → BlueprintAgent
                                                              ↓
[YAML Template] ──────────────────────────────────────→ DSLParser
                                                              ↓
                                                       ModelExecutor
                                                              ↓
                                               Excel / Streamlit / CLI output
```

### Key Layers

| Layer | Location | Responsibility |
|---|---|---|
| Agent Layer | `agents/` | LLM-based extraction (portfolio, assumptions, blueprint) |
| DSL Layer | `dsl/` | YAML schema, parser, expression evaluator, type definitions |
| Execution Engine | `engine/` | Block executor, KPI calculator, solvers, sensitivity analysis |
| Blocks Library | `blocks/`, `core_module/` | Reusable YAML calculation blocks |
| UI / CLI | `app.py`, `main.py` | Streamlit UI and command-line runner |

## Entry Points

### CLI (main.py)
```bash
# Base case (uses dsl/templates/solar_ipp_base.yaml)
python main.py

# With Monte Carlo
python main.py --monte-carlo --mc-iter 500

# Custom template
python main.py --template dsl/templates/solar_ipp_base.yaml

# Portfolio mode (interactive)
python main.py --portfolio

# Portfolio mode (non-interactive)
python main.py --portfolio --prompt "100 MW solar at Rs.2.65/kWh, 70% debt"

# Export to Excel
python main.py --export output/model.xlsx --audit-trail
```

### Streamlit UI
```bash
streamlit run app.py
```

### Programmatic API
```python
from dsl.parser import DSLParser
from engine.executor import ModelExecutor

model_def, validation = DSLParser().load_file("template.yaml")
executor = ModelExecutor()
compiled = executor.compile(model_def)
results = executor.run(compiled, {})                 # base case
results = executor.run(compiled, {"cuf": 0.25})      # with overrides
```

## Tech Stack

- **Python 3.10+**
- `anthropic` — Claude API for structured extraction (tool_use)
- `numpy` — all arithmetic and time-series computation
- `scipy` — numerical optimization (IRR via Brent's method)
- `networkx` — dependency graph + cycle detection
- `pyyaml` + `pydantic` — DSL parsing and validation
- `numpy-financial` — IRR/NPV helpers
- `openpyxl` — Excel export
- `streamlit` — web UI (install separately; not in requirements.txt)

## DSL Block Structure

Each calculation block is a YAML file:
```yaml
inputs:
  - assumptions.capacity_mw
  - phases.is_operational
  - generation_block.net_generation_mwh

outputs:
  - revenue_inr

body:
  revenue_inr: "net_generation_mwh * tariff * escalate(1 + tariff_escalation, year)"
```

Blocks are wired together via dependency graph; execution order is determined by topological sort. Circular dependencies (IDC, debt sculpting) must be explicitly declared in `dsl/solve_loops/`.

## Financial Model Calculation Order

1. **Generation** — quarterly AC energy from capacity, CUF, degradation
2. **Revenue** — tariff × generation × escalation
3. **Construction & Financing** — capex schedule, debt drawdown, IDC (*solve loop*)
4. **Opex** — operating costs, maintenance, insurance, land lease
5. **Depreciation** — SLM or WDV method
6. **Income Statement** — EBITDA → EBIT → PBT → tax
7. **Debt Service** — equal-principal repayment with sculpting (*solve loop*)
8. **DSRA** — reserve funding and release (optional; skip if `dsra_months == 0`)
9. **Waterfall** — priority: Opex/Tax → Debt Service → DSRA → Sweep → Equity
10. **Cashflow & KPIs** — Equity IRR, Project IRR, DSCR, LLCR, PLCR, NPV, peak debt

## Key Files

| File | Purpose |
|---|---|
| `dsl/templates/solar_ipp_base.yaml` | Ready-to-run 100 MW Karnataka solar benchmark |
| `dsl/types.py` | All Pydantic DSL types (ProjectSkeleton, CalculationBlock, etc.) |
| `dsl/parser.py` | YAML loader, validator, cycle detector, phase mask derivation |
| `dsl/expression.py` | Safe expression evaluator (no arbitrary code execution) |
| `engine/executor.py` | Core: compile, run, batch, sensitivity, Monte Carlo |
| `engine/solvers.py` | goal_seek, sculpting, fixed_point, array_fixed_point |
| `engine/kpi.py` | XIRR, NPV, DSCR, LLCR, PLCR, debt payback |
| `engine/excel_exporter.py` | Multi-sheet Excel workbook export |
| `engine/asset_registry.py` | Maps asset types to block libraries |
| `agents/assumption_agent.py` | Claude-powered canonical assumption extraction |
| `agents/blueprint_agent.py` | Claude-powered ModelDefinition assembly |
| `env_loader.py` | Discovers and loads `.env` with `ANTHROPIC_API_KEY` |

## Environment Setup

Requires `ANTHROPIC_API_KEY` for natural-language ingestion modes. Place in a `.env` file in the project root or any parent directory — `env_loader.py` will find it.

```bash
pip install -r requirements.txt
pip install streamlit  # only if using app.py
```

## Testing

```bash
pytest                          # all tests
pytest -v tests/test_executor.py
pytest tests/test_kpis.py --pdb # debug on failure
```

Test files:
- `tests/test_dsl_parser.py` — YAML parsing, phase derivation, cycle detection
- `tests/test_expression.py` — expression evaluator safety and correctness
- `tests/test_executor.py` — block compilation, dependency ordering
- `tests/test_solvers.py` — sculpting, fixed-point, goal-seek convergence
- `tests/test_kpis.py` — IRR, DSCR, LLCR, NPV correctness
- `tests/test_integration.py` — end-to-end pipeline

## Adding a New Asset Type

1. Add entry to `ASSET_REGISTRY` in `engine/asset_registry.py`
2. Create `blocks/<asset_type>/` with `generation.yaml` and `revenue.yaml`
3. Add `dsl/assumption_schemas/<asset_type>.yaml`
4. Add `dsl/project_skeletons/<asset_type>.yaml`
5. Optionally add asset-specific solve loops to `dsl/solve_loops/`

## Adding a New Calculation Block

1. Create YAML in `core_module/` (shared) or `blocks/<asset>/` (asset-specific)
2. Define `inputs`, `outputs`, and `body` expressions
3. Reference in model template or wire via conditional block selection in blueprint agent

## Important Constraints

- Time axis is **quarterly** (not monthly or annual)
- DSCR is computed only during debt-outstanding periods (NaN elsewhere)
- Tax approximation in sculpting loop is one-pass (actual DSCR within 1–2% of target)
- Defaults reflect Indian benchmark values (INR, CERC norms, Karnataka irradiance)
- Do not use LLMs for arithmetic — all computation must go through NumPy in the engine layer
- Undeclared circular dependencies raise `DSLCycleError` — declare them in `dsl/solve_loops/`
