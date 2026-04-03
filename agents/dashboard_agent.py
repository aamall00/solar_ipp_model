"""
agents/dashboard_agent.py — Dashboard intelligence layer.

Two responsibilities
--------------------
1. select_charts(model_context)
   Uses Claude tool_use (set_additional_charts) to analyse the asset's quarterly
   model data and propose 2–4 additional charts beyond the standard set.

2. answer_question(model_context, chat_history, question)
   Uses Claude Haiku to answer free-form analyst questions about an asset,
   grounded in the quarterly model context.

Token strategy
--------------
- Model context is pre-built as a compact dict (~5 KB) with quarterly series
  (not raw arrays) and 10 key assumptions — passed verbatim to Claude.
- Chat history is capped at the last 6 turns to bound input tokens.
- select_charts uses claude-sonnet-4-6 (called once per asset per run).
- answer_question uses claude-haiku-4-5-20251001 (fast, cheap for chat).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import anthropic

# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

_SELECT_CHARTS_TOOL: Dict[str, Any] = {
    "name": "set_additional_charts",
    "description": (
        "Record 2–4 additional chart specifications that complement the standard "
        "dashboard charts. Each chart must reference series keys available in the "
        "quarterly_series dict provided."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "charts": {
                "type": "array",
                "description": "List of additional chart specifications",
                "items": {
                    "type": "object",
                    "properties": {
                        "chart_id": {
                            "type": "string",
                            "description": "Unique snake_case identifier for this chart",
                        },
                        "title": {
                            "type": "string",
                            "description": "Human-readable chart title shown in the dashboard",
                        },
                        "chart_type": {
                            "type": "string",
                            "enum": ["line", "bar", "area", "stacked_bar"],
                            "description": "Streamlit chart type to use for rendering",
                        },
                        "series": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "One or more keys from quarterly_series to plot. "
                                "Must only use keys that exist in the provided data."
                            ),
                        },
                        "insight": {
                            "type": "string",
                            "description": (
                                "One-sentence analytical observation about this chart, "
                                "citing specific values or trends from the data."
                            ),
                        },
                    },
                    "required": ["chart_id", "title", "chart_type", "series", "insight"],
                },
            }
        },
        "required": ["charts"],
    },
}

_SELECT_CHARTS_SYSTEM = """You are a project finance analyst reviewing an IPP financial model output.

The standard dashboard already shows these charts:
- CFADS vs Debt Service (line)
- DSCR per Quarter (line)
- Revenue, OPEX & EBITDA (line)
- Outstanding Debt Balance (area)
- Principal & Interest stacked (bar)
- DSRA Balance if applicable (area)
- Equity Cashflow J-Curve cumulative (line)

Given the quarterly KPIs and time-series data below, propose 2–4 ADDITIONAL charts
that reveal further financial insight for an equity investor or lender.
Good candidates include: tax trajectory, equity distributions over time,
waterfall components, revenue degradation trend, or net free cashflow.

Rules:
- Only use series keys that exist in the quarterly_series dict.
- Do not duplicate the standard charts listed above.
- Write the insight as a concrete observation referencing numbers or trends in the data.
- Call set_additional_charts exactly once."""

_CHAT_SYSTEM = """You are a senior project finance analyst. Answer questions about this IPP asset
using the model data provided. Be concise (3–5 sentences), cite specific numbers,
and flag any bankability concerns where relevant. Do not speculate beyond the data."""

# History cap to bound token usage
_MAX_HISTORY_TURNS = 6


# ---------------------------------------------------------------------------
# DashboardAgent
# ---------------------------------------------------------------------------


class DashboardAgent:
    """
    Provides AI-driven chart selection and Q&A for the financial dashboard.

    Parameters
    ----------
    client : Pre-configured anthropic.Anthropic client (uses env var if None).
    """

    def __init__(self, client: Optional[anthropic.Anthropic] = None) -> None:
        self.client = client or anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Chart selection
    # ------------------------------------------------------------------

    def select_charts(self, model_context: dict) -> dict:
        """
        Ask Claude to propose 2–4 additional dashboard charts for this asset.

        Parameters
        ----------
        model_context : dict produced by _build_model_context() in app.py.

        Returns
        -------
        dict with key "charts" → list of chart spec dicts, or {"charts": []} on failure.
        """
        context_json = json.dumps(model_context, indent=None)

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=_SELECT_CHARTS_SYSTEM,
                tools=[_SELECT_CHARTS_TOOL],
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": context_json}],
            )
        except Exception:
            return {"charts": []}

        for block in response.content:
            if block.type == "tool_use" and block.name == "set_additional_charts":
                charts = block.input.get("charts", [])
                # Validate series keys exist in the context
                available = set(model_context.get("quarterly_series", {}).keys())
                valid = [
                    c for c in charts
                    if all(s in available for s in c.get("series", []))
                ]
                return {"charts": valid}

        return {"charts": []}

    # ------------------------------------------------------------------
    # Chat Q&A
    # ------------------------------------------------------------------

    def answer_question(
        self,
        model_context: dict,
        chat_history: List[Dict[str, str]],
        question: str,
    ) -> str:
        """
        Answer a user question about an asset using the model context.

        Parameters
        ----------
        model_context  : dict produced by _build_model_context() in app.py.
        chat_history   : List of {"role": "user"|"assistant", "content": str}.
                         Capped to last _MAX_HISTORY_TURNS turns internally.
        question       : Current user question.

        Returns
        -------
        str — plain-text analyst response.
        """
        context_json = json.dumps(model_context, indent=None)
        system = _CHAT_SYSTEM + f"\n\nModel data:\n{context_json}"

        # Cap history to avoid unbounded token growth
        history = chat_history[-_MAX_HISTORY_TURNS:]
        messages = history + [{"role": "user", "content": question}]

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                system=system,
                messages=messages,
            )
            return response.content[0].text
        except Exception as exc:
            return f"Could not generate answer: {exc}"
