"""
engine/portfolio_runner.py — Orchestrates the full pipeline for a multi-asset portfolio.

Pipeline per asset
------------------
  AssetIngestion (assumptions)
    → BlueprintAgent.generate()   → ModelDefinition
    → ModelExecutor.compile()     → CompiledModel
    → ModelExecutor.run()         → ModelResults
    → AssetResult

All arithmetic goes through NumPy / ModelExecutor.  No LLM arithmetic.

Usage
-----
    from agents.portfolio_agent import PortfolioAgent
    from engine.portfolio_runner import PortfolioRunner

    agent   = PortfolioAgent()
    specs   = agent.parse("100 MW solar SPV-A and 50 MW wind SPV-B")
    ingestions = agent.collect_assumptions(specs)

    runner  = PortfolioRunner()
    results = runner.run(ingestions)

    from engine.excel_exporter import export_portfolio_to_excel
    export_portfolio_to_excel(results, "output/portfolio.xlsx")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.blueprint_agent import BlueprintAgent
from agents.portfolio_agent import AssetIngestion, AssetSpec
from dsl.types import ModelDefinition, ValidationResult
from engine.executor import CompiledModel, ModelExecutor, ModelResults


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class AssetResult:
    """Full pipeline output for one asset in the portfolio."""
    spec:          AssetSpec
    model_def:     ModelDefinition
    validation:    ValidationResult
    compiled:      CompiledModel
    model_results: ModelResults
    yaml_str:      str
    rationale:     Optional[str] = None


# ---------------------------------------------------------------------------
# PortfolioRunner
# ---------------------------------------------------------------------------


class PortfolioRunner:
    """
    Runs the full pipeline (Blueprint → Executor) for each asset in a portfolio.

    Parameters
    ----------
    claude_model : Anthropic model name passed to BlueprintAgent.
    explain      : If True, ask Claude to generate a plain-English rationale
                   for each asset's configuration.
    review       : If True, ask Claude to flag structural anomalies before
                   building each model (extra API call per asset).
    """

    def __init__(
        self,
        claude_model: str = "claude-sonnet-4-6",
        explain: bool = False,
        review: bool = False,
    ) -> None:
        self.claude_model = claude_model
        self.explain      = explain
        self.review       = review
        self._executor    = ModelExecutor()

    def run(self, ingestions: List[AssetIngestion]) -> List[AssetResult]:
        """
        Execute the full pipeline for each asset.

        Parameters
        ----------
        ingestions : List[AssetIngestion] from PortfolioAgent.collect_assumptions().

        Returns
        -------
        List[AssetResult] in the same order as ingestions.
        """
        results: List[AssetResult] = []
        for ing in ingestions:
            result = self._run_one(ing)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_one(self, ing: AssetIngestion) -> AssetResult:
        spec         = ing.spec
        ingest_result = ing.ingestion_result

        # 1. Build ModelDefinition via BlueprintAgent
        bp_agent = BlueprintAgent(
            asset_type=spec.asset_type,
            model=self.claude_model,
        )
        model_def, validation, yaml_str, rationale = bp_agent.generate(
            ingest_result,
            explain=self.explain,
            review=self.review,
        )

        if model_def is None:
            raise RuntimeError(
                f"BlueprintAgent failed for asset '{spec.name}': "
                + "; ".join(e for e in validation.errors)
            )

        # 2. Compile
        compiled = self._executor.compile(model_def)

        # 3. Run base case (no overrides — assumptions already baked in)
        model_results = self._executor.run(compiled, {})

        return AssetResult(
            spec=spec,
            model_def=model_def,
            validation=validation,
            compiled=compiled,
            model_results=model_results,
            yaml_str=yaml_str,
            rationale=rationale,
        )
