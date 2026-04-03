# Solar IPP Model

`solar_ipp_model` is a DSL-driven renewable project finance engine for Indian IPPs. It supports:

- Single-asset template runs from YAML
- Natural-language portfolio ingestion for solar and wind assets
- A compiled execution engine with dependency graphs and solve loops
- Scenario, sensitivity, and Monte Carlo analysis
- Excel export for model outputs and audit trails
- A Streamlit UI for portfolio runs

The design goal is to keep arithmetic deterministic and auditable in Python/NumPy while using LLMs only for structured extraction and model assembly.

## What This Repository Does

At a high level, the application turns project descriptions or YAML templates into executable financial models:

1. Ingest assumptions from either a YAML template or natural-language prompt
2. Build a validated `ModelDefinition` from block YAML files
3. Compile the model into an execution plan with dependencies and solve loops
4. Execute time-series calculations in NumPy
5. Compute KPIs such as equity IRR, project IRR, DSCR, LLCR, NPV, and debt payback
6. Export the results to Excel or display them in Streamlit

The model currently focuses on quarterly project-finance style IPP evaluation with debt service, tax, DSRA, waterfall allocation, and returns metrics.

## Core Concepts

### 1. DSL-first model definition

The model is defined as YAML, not hardcoded spreadsheets:

- `project_skeleton`: time axis, milestones, currency, project type
- `assumption_schema`: typed assumptions with defaults, constraints, and conditional inclusion
- `calculation_blocks`: reusable block definitions such as generation, revenue, opex, debt service, and cashflow
- `solve_loops`: declared circular dependencies that must be iterated rather than topologically sorted
- `model_wiring`: reporting-oriented connections derived during assembly

### 2. Strict separation of responsibilities

- LLMs extract structure and assumptions
- Python validates and assembles the model
- NumPy performs all arithmetic
- The executor produces both results and an audit trail

### 3. Block library architecture

The repository mixes shared finance blocks with asset-specific blocks:

- `core_module/`: shared blocks such as construction, debt service, DSRA, depreciation, tax, waterfall, and cashflow
- `blocks/solar_ipp/`: solar-specific generation and revenue blocks
- `blocks/wind_ipp/`: wind-specific generation and revenue blocks

This makes it straightforward to add new asset classes while reusing most of the financing stack.

## Repository Layout

```text
solar_ipp_model/
|- app.py                         # Streamlit portfolio UI
|- main.py                        # CLI runner for single-asset and portfolio modes
|- run_wind_solar_portfolio.py    # Programmatic demo for a mixed portfolio
|- env_loader.py                  # Lightweight .env discovery for Anthropic credentials
|- agents/
|  |- portfolio_agent.py          # Splits a portfolio prompt into assets
|  |- assumption_agent.py         # Extracts canonical assumptions from text
|  |- blueprint_agent.py          # Builds a full model definition from assumptions
|  `- ...
|- dsl/
|  |- parser.py                   # Loads and validates model YAML
|  |- expression.py               # Safe expression evaluator for block formulas
|  |- assumption_schemas/         # Asset-specific schema templates
|  |- project_skeletons/          # Asset-specific skeleton templates
|  |- solve_loops/                # Declarative solve-loop definitions
|  `- templates/                  # Ready-to-run model templates
|- engine/
|  |- executor.py                 # Compiler and runtime executor
|  |- portfolio_runner.py         # Multi-asset orchestration
|  |- sensitivity.py              # Sensitivity and Monte Carlo helpers
|  |- kpi.py                      # KPI calculations
|  `- excel_exporter.py           # Formatted Excel output
|- core_module/                   # Shared YAML calculation blocks
|- blocks/
|  |- solar_ipp/                  # Solar-specific block YAML
|  `- wind_ipp/                   # Wind-specific block YAML
|- tests/                         # Parser, executor, KPI, and integration tests
`- output/                        # Generated workbooks
```

## Architecture Flow

### A. Portfolio / natural-language flow

This is the path used by `app.py` and `main.py --portfolio`.

1. `PortfolioAgent`
   Parses a free-form portfolio description into one or more `AssetSpec` objects.

2. `AssumptionAgent`
   Extracts canonical assumptions per asset, applies defaults, and derives cross-assumptions such as:
   - `construction_periods`
   - `operations_periods`
   - `debt_tenor_periods`
   - `use_wdv`

3. `BlueprintAgent`
   Loads the asset-specific block library and assembles:
   - project skeleton
   - assumption schema
   - included/excluded optional blocks
   - solve loop declarations
   - model wiring

4. `DSLParser`
   Validates structure, derives phase masks, checks assumption coverage, and detects undeclared cycles.

5. `ModelExecutor`
   Compiles the model into an evaluation plan, executes the blocks, resolves solve loops, and computes KPIs.

6. `ExcelExporter` / Streamlit
   Results are written to Excel and optionally shown in the UI with dependency and execution graphs.

### B. Single-template YAML flow

This is the path used by `main.py` without `--portfolio`.

1. Load `dsl/templates/solar_ipp_base.yaml`
2. Parse with `DSLParser`
3. Compile with `ModelExecutor`
4. Run base case
5. Optionally run scenario comparison, sensitivity, and Monte Carlo
6. Optionally export to Excel

## Financial Model Flow

The main statement logic follows a standard IPP sequence:

1. Generation
   Asset-specific generation blocks convert capacity and resource assumptions into quarterly energy.

2. Revenue
   Revenue blocks apply tariff and escalation to generation, plus optional subsidy support when enabled.

3. Construction and funding
   Construction, debt drawdown, and IDC blocks build the capitalized project cost during construction.

4. Opex and depreciation
   Shared blocks compute operating costs, maintenance capex, insurance, land lease, and depreciation.

5. Income statement and tax
   Revenue, opex, depreciation, and interest feed EBITDA, EBIT, PBT, and tax.

6. Debt service and DSRA
   Debt service can run in standard equal-principal mode or CFADS-based sculpting mode. DSRA is optional and excluded automatically when `dsra_months == 0`.

7. Waterfall
   Revenue is allocated by priority to opex/tax, debt service, DSRA funding, cash sweep, and finally equity.

8. Cashflow and returns
   The cashflow block computes:
   - `cfads`
   - `project_cashflow`
   - `equity_cashflow`
   - debt and DSRA cash movements

9. KPI layer
   The engine computes:
   - equity IRR
   - project IRR
   - min/avg DSCR
   - LLCR / PLCR
   - NPV of equity
   - peak debt
   - debt payback period

## Solve Loops

The engine supports declared iterative loops for circular financial logic.

### IDC capitalisation loop

Resolves the dependency between:

- construction capex
- debt drawdown
- IDC per period

### Debt service sculpting loop

When sculpting is enabled, the debt service schedule is iterated using `CFADS` and the target DSCR to shape debt service consistently.

If a cycle exists in the model graph and is not covered by a declared solve loop, parsing fails.

## Supported Asset Types

Currently supported:

- `solar`
- `wind`

Asset metadata is registered in `engine/asset_registry.py`.

## Getting Started

### Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

If you want to run the Streamlit UI, install Streamlit as well:

```bash
pip install streamlit
```

### Environment variables

Portfolio and assumption extraction use Anthropic models. Set:

```bash
$env:ANTHROPIC_API_KEY="your_key_here"
```

Optionally:

```bash
$env:CLAUDE_MODEL="claude-sonnet-4-6"
```

`env_loader.py` looks for a nearby `.env` file if `ANTHROPIC_API_KEY` is not already set.

You can also place these in a local `.env` file:

```text
ANTHROPIC_API_KEY=your_key_here
CLAUDE_MODEL=claude-sonnet-4-6
```

## Running the Project

### 1. Single-asset template run

Run the bundled solar template:

```bash
python main.py
```

This performs:

- base-case run
- scenario comparison
- sensitivity analysis

Add Monte Carlo:

```bash
python main.py --monte-carlo
```

Custom Monte Carlo iterations:

```bash
python main.py --monte-carlo --mc-iter 500
```

Export to Excel:

```bash
python main.py --export output/model.xlsx
```

Export with audit trail:

```bash
python main.py --export output/model.xlsx --audit-trail
```

Use a custom template:

```bash
python main.py --template dsl/templates/solar_ipp_base.yaml
```

### 2. Portfolio mode from natural language

Interactive portfolio run:

```bash
python main.py --portfolio
```

Non-interactive portfolio run:

```bash
python main.py --portfolio --prompt "Two assets: 100 MW solar SPV-A at Rs.2.65/kWh and 50 MW wind SPV-B at Rs.3.20/kWh"
```

This path:

- parses the portfolio prompt
- extracts assumptions per asset
- builds one model per asset
- runs the portfolio
- exports a combined workbook

### 3. Streamlit UI

```bash
streamlit run app.py
```

The UI:

- accepts a plain-English portfolio description
- runs the portfolio model
- shows per-asset KPIs
- visualizes the dependency graph and execution plan
- lets you download the generated workbook

### 4. Programmatic mixed portfolio demo

```bash
python run_wind_solar_portfolio.py
```

This creates a sample two-asset wind + solar portfolio and writes:

- `output/wind_solar_portfolio.xlsx`

## Example Prompts

Portfolio prompts can be concise:

```text
Two assets: 100 MW solar SPV-A at Rs. 2.65/kWh with 70% debt, and 50 MW wind SPV-B at Rs. 3.20/kWh with 75% debt.
```

Or more descriptive:

```text
I want to create a project finance model for two assets - one wind and one solar. The solar model will not have any DSRA requirements and will have revenue support of 1 rs/kwh.
```

## Excel Output

The exporter produces formatted workbooks with sheets such as:

- Cover
- Income Statement
- Cash Flow
- Debt Schedule
- Generation
- Waterfall
- Returns
- Sensitivity
- Monte Carlo
- Assumptions
- Audit Trail

Portfolio exports also include a `Portfolio Summary` sheet and grouped per-asset tabs.

## Tests

Run the test suite with:

```bash
pytest
```

The tests cover:

- DSL parsing and validation
- compilation and dependency planning
- base-case execution
- KPI sanity checks
- scenario directionality
- batch execution
- sensitivity analysis
- Monte Carlo output
- downstream dependency tracking

## Extending the Model

### Add a new asset type

1. Register the asset in `engine/asset_registry.py`
2. Create `blocks/<asset_type>/` with at least generation and revenue blocks
3. Add an assumption schema under `dsl/assumption_schemas/`
4. Add a project skeleton under `dsl/project_skeletons/`
5. Add or reuse solve-loop declarations under `dsl/solve_loops/`
6. Update parser import resolution if you want direct `import_from` support

### Add a new calculation block

1. Create or modify a YAML block in `core_module/` or `blocks/<asset_type>/`
2. Declare:
   - `block_id`
   - `inputs`
   - `outputs`
   - `body`
3. If optional, use `wiring.optional` with `include_when` or `exclude_when`
4. Ensure downstream blocks reference the block through declared sources
5. Add tests for both presence and behavior

### Add a new assumption

You will usually need to touch more than one place:

1. Canonical extraction schema in `agents/assumption_agent.py`
2. Asset schema template in `dsl/assumption_schemas/`
3. Any block YAML that consumes the assumption
4. Validation logic if new constraints are needed

## Implementation Notes

- Arithmetic is executed in NumPy, not by the LLM
- Time series are modeled quarterly by default
- DSCR is only meaningful in debt-outstanding periods
- Every run can produce an audit trail of intermediate variables
- Optional blocks are included or excluded declaratively based on assumptions

## Known Notes and Practical Caveats

- Anthropic credentials are required for natural-language portfolio ingestion
- The Streamlit UI imports `streamlit`, which is not listed in `requirements.txt`
- The repository is optimized for bankability-style screening and scenario analysis, not full tax/legal structuring
- Some assumptions and defaults reflect Indian utility-scale solar/wind benchmark values

## Recommended First Commands

If you just want to verify the project is working:

```bash
pip install -r requirements.txt
pytest
python main.py --export output/model.xlsx
```

If you want the LLM-assisted flow:

```bash
$env:ANTHROPIC_API_KEY="your_key_here"
python main.py --portfolio --prompt "One 100 MW solar asset at Rs.2.65/kWh with 70% debt"
```

If you want the UI:

```bash
pip install streamlit
streamlit run app.py
```

## License / Ownership

No license file is included in this repository. Add one if you plan to distribute or open-source the project.
