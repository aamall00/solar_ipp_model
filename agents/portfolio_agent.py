"""
agents/portfolio_agent.py — Portfolio parsing and per-asset assumption collection.

Two responsibilities
--------------------
1. parse(user_prompt)
   Uses Claude tool_use to extract a list of AssetSpec objects from a free-form
   portfolio description.  Example input:
     "Two assets in different SPVs — a 100 MW solar plant and a 50 MW wind farm"
   Example output:
     [AssetSpec(name="SPV-1 Solar", asset_type="solar", spv_name="SPV1", description="100 MW solar plant"),
      AssetSpec(name="SPV-2 Wind",  asset_type="wind",  spv_name="SPV2", description="50 MW wind farm")]

2. collect_assumptions(specs, prompt_fn)
   For each AssetSpec, calls prompt_fn(spec) to get the user's assumption text,
   then runs AssumptionAgent to produce an IngestionResult.

Claude NEVER computes arithmetic — it only identifies asset types, counts, and names.

Usage
-----
    from agents.portfolio_agent import PortfolioAgent

    agent = PortfolioAgent()
    specs = agent.parse("Two assets: 100 MW solar SPV-A and 50 MW wind SPV-B")
    results = agent.collect_assumptions(specs, prompt_fn=input)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import anthropic

from agents.assumption_agent import AssumptionAgent, IngestionResult
from engine.asset_registry import SUPPORTED_ASSET_TYPES

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class AssetSpec:
    """Describes one asset extracted from the user's portfolio prompt."""
    name:        str   # human label e.g. "SPV-1 Solar"
    asset_type:  str   # "solar" | "wind"
    spv_name:    str   # SPV / entity name e.g. "SPV-1"
    description: str   # original user text fragment for this asset


@dataclass
class AssetIngestion:
    """Pairing of an AssetSpec with its extracted IngestionResult."""
    spec:             AssetSpec
    ingestion_result: IngestionResult


# ---------------------------------------------------------------------------
# Claude tool schema
# ---------------------------------------------------------------------------

_PARSE_PORTFOLIO_TOOL: Dict[str, Any] = {
    "name": "set_portfolio",
    "description": (
        "Record each distinct asset in the user's portfolio description. "
        "For each asset, identify its type (solar or wind), the SPV/entity name "
        "if mentioned, and extract the relevant description text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "assets": {
                "type": "array",
                "description": "List of assets in the portfolio",
                "items": {
                    "type": "object",
                    "properties": {
                        "asset_type": {
                            "type": "string",
                            "enum": SUPPORTED_ASSET_TYPES,
                            "description": "Type of renewable energy asset",
                        },
                        "spv_name": {
                            "type": "string",
                            "description": (
                                "SPV or entity name if stated (e.g. 'SPV-1', 'GreenPower Ltd'). "
                                "Use 'SPV-{n}' if not explicitly named."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "The portion of the user's text describing this specific asset, "
                                "including all technical and financial details mentioned."
                            ),
                        },
                    },
                    "required": ["asset_type", "spv_name", "description"],
                },
            }
        },
        "required": ["assets"],
    },
}

_SYSTEM_PROMPT = (
    "You are a project finance analyst. "
    "The user will describe a portfolio of renewable energy assets in different SPVs. "
    "Your task is to identify each distinct asset and call set_portfolio with the list. "
    "Each asset must have: asset_type (solar or wind), spv_name, and the description text. "
    "If the user gives details for all assets together, split them by asset type and SPV. "
    "Do not compute numbers. Do not invent details not present in the user's text."
)


# ---------------------------------------------------------------------------
# PortfolioAgent
# ---------------------------------------------------------------------------


class PortfolioAgent:
    """
    Parses multi-asset portfolio prompts and orchestrates per-asset
    assumption collection.

    Parameters
    ----------
    model  : Anthropic model name.
    client : Pre-configured anthropic.Anthropic client (uses env var if None).
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        client: Optional[anthropic.Anthropic] = None,
    ) -> None:
        self.model  = model
        self.client = client or anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, user_prompt: str) -> List[AssetSpec]:
        """
        Extract a list of AssetSpec objects from a free-form portfolio description.

        Parameters
        ----------
        user_prompt : Plain-English description of the portfolio
                      (e.g. "Two assets: 100 MW solar SPV-A and 50 MW wind SPV-B").

        Returns
        -------
        List[AssetSpec] — one per distinct asset.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            tools=[_PARSE_PORTFOLIO_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_prompt}],
        )

        specs: List[AssetSpec] = []
        for block in response.content:
            if block.type != "tool_use" or block.name != "set_portfolio":
                continue
            for i, asset in enumerate(block.input.get("assets", []), start=1):
                asset_type = asset.get("asset_type", "solar").lower()
                spv_name   = asset.get("spv_name", f"SPV-{i}")
                desc       = asset.get("description", "")
                name       = f"{spv_name} {asset_type.title()}"
                specs.append(AssetSpec(
                    name=name,
                    asset_type=asset_type,
                    spv_name=spv_name,
                    description=desc,
                ))

        if not specs:
            raise ValueError(
                "Could not identify any assets from the prompt. "
                "Please describe each asset with its type (solar/wind) and key parameters."
            )

        return specs

    def collect_assumptions(
        self,
        specs: List[AssetSpec],
        prompt_fn: Optional[Callable[[AssetSpec], str]] = None,
    ) -> List[AssetIngestion]:
        """
        For each AssetSpec, collect and validate assumptions.

        Parameters
        ----------
        specs     : List of AssetSpec from parse().
        prompt_fn : Callable that takes an AssetSpec and returns a string of
                    assumption text.  Defaults to using spec.description (i.e.
                    the description extracted from the original portfolio prompt).
                    Pass prompt_fn=input for interactive CLI collection.

        Returns
        -------
        List[AssetIngestion] — one per asset, in the same order as specs.
        """
        ingestions: List[AssetIngestion] = []
        for spec in specs:
            if prompt_fn is not None:
                assumption_text = prompt_fn(spec)
            else:
                assumption_text = spec.description

            agent  = AssumptionAgent(asset_type=spec.asset_type, model=self.model, client=self.client)
            result = agent.extract(assumption_text)
            ingestions.append(AssetIngestion(spec=spec, ingestion_result=result))

        return ingestions
