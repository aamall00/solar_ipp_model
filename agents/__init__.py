"""
agents/ — AI agent layer for the solar IPP financial modelling system.

Five-layer architecture
-----------------------
  Layer 1  DSL engine          dsl/           (no AI)
  Layer 2  Execution engine    engine/        (no AI)
  Layer 3  Assumption ingestion AssumptionAgent  ← this package
  Layer 4  Blueprint generation BlueprintAgent   ← this package
  Layer 5  Scenario & narrative ScenarioAgent    ← this package
           Output narrative     NarrativeAgent   ← this package
           (analyst helper)     AnalystAgent     ← this package

The AI layer NEVER computes arithmetic.  Claude is used only to:
  - extract and map assumptions from user prose  (AssumptionAgent)
  - write configuration rationales               (BlueprintAgent)
  - propose correlated stress scenarios          (ScenarioAgent)
  - generate bankability narratives              (NarrativeAgent, AnalystAgent)

All numbers come from the deterministic NumPy/SciPy execution engine.

Typical end-to-end workflow
----------------------------
    from agents import AssumptionAgent, BlueprintAgent, ScenarioAgent, NarrativeAgent
    from engine.executor import ModelExecutor

    # 1. Extract assumptions from user text
    ingestion = AssumptionAgent().extract(user_text)
    assert ingestion.validation.valid

    # 2. Build and compile the model
    model_def, validation, _ = BlueprintAgent().generate(ingestion)
    executor  = ModelExecutor()
    compiled  = executor.compile(model_def)

    # 3. Run base case
    base_results = executor.run(compiled, {})

    # 4. Run all standard scenarios
    scenario_agent = ScenarioAgent(compiled=compiled, executor=executor)
    scenarios      = scenario_agent.propose_scenarios(base_results)
    all_results    = scenario_agent.run_all_scenarios(scenarios=scenarios)

    # 5. Sensitivity + Monte Carlo
    sweep = [{"assumption": "cuf",    "low": 0.18, "high": 0.26},
             {"assumption": "tariff", "low": 2.20, "high": 3.10}]
    sensitivity = executor.run_sensitivity(compiled, {}, sweep)
    mc_results  = executor.run_monte_carlo(compiled, {}, n_iterations=500)

    # 6. Generate bankability report
    report = NarrativeAgent().report(
        base_results=base_results,
        scenario_results=all_results,
        sensitivity_results=sensitivity,
        mc_results=mc_results,
        compiled=compiled,
    )
    print(report.full_text)
"""

from env_loader import load_local_env
from agents.assumption_agent import AssumptionAgent, IngestionResult, InferredAssumption
from agents.blueprint_agent  import BlueprintAgent
from agents.scenario_agent   import ScenarioAgent, Scenario, STANDARD_SCENARIOS
from agents.narrative_agent  import NarrativeAgent, NarrativeReport
from agents.analyst_agent    import AnalystAgent

load_local_env()

__all__ = [
    # Layer 3
    "AssumptionAgent",
    "IngestionResult",
    "InferredAssumption",
    # Layer 4
    "BlueprintAgent",
    # Layer 5
    "ScenarioAgent",
    "Scenario",
    "STANDARD_SCENARIOS",
    # Output
    "NarrativeAgent",
    "NarrativeReport",
    # Analyst helper (kept from earlier implementation)
    "AnalystAgent",
]
