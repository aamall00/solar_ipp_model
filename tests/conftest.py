"""
tests/conftest.py — Shared pytest fixtures for the solar IPP model test suite.

Session-scoped fixtures load the full Karnataka template once and reuse it
across all integration tests, avoiding repeated compilation overhead.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE_PATH = os.path.join(_ROOT, "dsl", "templates", "solar_ipp_base.yaml")


# ---------------------------------------------------------------------------
# Session-scoped: load → parse → compile
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def template_path() -> str:
    """Absolute path to the baseline Karnataka 100 MW template."""
    assert os.path.exists(TEMPLATE_PATH), (
        f"Template not found at {TEMPLATE_PATH}. "
        "Run from the project root or check the path."
    )
    return TEMPLATE_PATH


@pytest.fixture(scope="session")
def parsed_model(template_path):
    """
    (ModelDefinition, ValidationResult) for the baseline template.
    Loaded once for the full test session.
    """
    from dsl.parser import load_model

    model_def, validation = load_model(template_path)
    return model_def, validation


@pytest.fixture(scope="session")
def model_def(parsed_model):
    """The ModelDefinition extracted from the parsed template."""
    md, _ = parsed_model
    return md


@pytest.fixture(scope="session")
def parse_validation(parsed_model):
    """The ValidationResult from parsing the template."""
    _, vr = parsed_model
    return vr


@pytest.fixture(scope="session")
def executor():
    """A single ModelExecutor instance shared across the session."""
    from engine.executor import ModelExecutor

    return ModelExecutor()


@pytest.fixture(scope="session")
def compiled(model_def, executor):
    """CompiledModel built from the session-level ModelDefinition."""
    return executor.compile(model_def)


@pytest.fixture(scope="session")
def base_results(compiled, executor):
    """
    ModelResults from running the compiled model with schema defaults
    (no overrides — cost-based debt service mode).
    """
    return executor.run(compiled, {})


# ---------------------------------------------------------------------------
# Reusable assumption sets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def assumptions_high_cuf() -> Dict[str, Any]:
    """Optimistic CUF scenario (CUF = 0.26, all else at schema defaults)."""
    return {"cuf": 0.26}


@pytest.fixture(scope="session")
def assumptions_low_cuf() -> Dict[str, Any]:
    """Stressed CUF scenario (CUF = 0.18)."""
    return {"cuf": 0.18}


@pytest.fixture(scope="session")
def assumptions_high_leverage() -> Dict[str, Any]:
    """High-leverage scenario (debt_pct = 0.80)."""
    return {"debt_pct": 0.80}


@pytest.fixture(scope="session")
def assumptions_low_tariff() -> Dict[str, Any]:
    """Low-tariff stress scenario (tariff = 2.20 INR/kWh)."""
    return {"tariff": 2.20}
