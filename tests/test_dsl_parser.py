"""
Unit tests for dsl/parser.py

Run with: pytest tests/test_dsl_parser.py -v
"""

import pytest

from dsl.parser import DSLParser, DSLParseError, load_model_from_dict
from dsl.types import ModelDefinition, ValidationResult


# ---------------------------------------------------------------------------
# Minimal valid model dict (used as a base for most tests)
# ---------------------------------------------------------------------------

def minimal_model(overrides: dict = None) -> dict:
    """
    Returns a fully valid minimal solar IPP model dict.
    3 construction periods + 12 operational = 15 periods total (quarterly).
    """
    base = {
        "project_skeleton": {
            "model_id": "test_solar",
            "project_type": "solar_ipp",
            "currency": "INR",
            "currency_unit": "Lakhs",
            "periods_per_year": 4,
            "construction_periods": 3,
            "operations_periods": 12,
            "milestones": {
                "financial_close": 0,
                "cod": 3,
                "debt_maturity": 11,
            },
        },
        "assumption_schema": {
            "assumptions": [
                {
                    "name": "installed_capacity_mw",
                    "type": "scalar",
                    "unit": "MW",
                    "value": 100.0,
                    "constraints": {"min": 0.0},
                    "sensitivity": {"vary": False},
                },
                {
                    "name": "capacity_utilisation_factor_p50",
                    "type": "scalar",
                    "unit": "ratio",
                    "value": 0.22,
                    "constraints": {"min": 0.0, "max": 1.0},
                    "sensitivity": {"vary": True, "range_pct": 0.10},
                },
                {
                    "name": "ppa_tariff_per_kwh",
                    "type": "scalar",
                    "unit": "INR_per_kWh",
                    "value": 2.65,
                    "constraints": {"min": 0.0},
                    "sensitivity": {"vary": True, "range_pct": 0.10},
                },
                {
                    "name": "debt_percent_of_total_cost",
                    "type": "scalar",
                    "unit": "ratio",
                    "value": 0.70,
                    "constraints": {"min": 0.0, "max": 1.0},
                    "sensitivity": {"vary": False},
                },
                {
                    "name": "debt_tenor_years",
                    "type": "scalar",
                    "unit": "years",
                    "value": 3,      # 3 years = 12 periods, fitting model
                    "constraints": {"min": 1.0},
                    "sensitivity": {"vary": False},
                },
                {
                    "name": "ppa_tenor_years",
                    "type": "scalar",
                    "unit": "years",
                    "value": 25,
                    "constraints": {"min": 1.0},
                    "sensitivity": {"vary": False},
                },
                {
                    "name": "moratorium_periods",
                    "type": "scalar",
                    "unit": "dimensionless",
                    "value": 2,
                    "constraints": {"min": 0.0},
                    "sensitivity": {"vary": False},
                },
                {
                    "name": "capex_schedule",
                    "type": "schedule",
                    "unit": "ratio",
                    "value": [0.30, 0.40, 0.30],
                    "sensitivity": {"vary": False},
                },
                {
                    "name": "interest_rate",
                    "type": "scalar",
                    "unit": "ratio",
                    "value": 0.0975,
                    "constraints": {"min": 0.0},
                    "sensitivity": {"vary": True, "range_pct": 0.15},
                },
            ],
        },
        "calculation_blocks": {
            "blocks": [
                {
                    "block_id": "generation_block",
                    "category": "generation",
                    "block_type": "standard",
                    "inputs": [
                        {"name": "installed_mw", "source": "assumption.installed_capacity_mw"},
                        {"name": "cuf", "source": "assumption.capacity_utilisation_factor_p50"},
                        {"name": "is_operational", "source": "phase.is_operational"},
                    ],
                    "outputs": [
                        {"name": "p50_generation_kwh", "unit": "kWh"},
                    ],
                    "body": [
                        {
                            "target": "p50_generation_kwh",
                            "expr": "installed_mw * cuf * 2190 * 1000 * is_operational",
                        }
                    ],
                },
                {
                    "block_id": "revenue_block",
                    "category": "revenue",
                    "block_type": "standard",
                    "inputs": [
                        {"name": "p50_gen", "source": "generation_block.p50_generation_kwh"},
                        {"name": "tariff", "source": "assumption.ppa_tariff_per_kwh"},
                    ],
                    "outputs": [
                        {"name": "ppa_revenue", "unit": "INR_Lakhs"},
                    ],
                    "body": [
                        {
                            "target": "ppa_revenue",
                            "expr": "p50_gen * tariff / 100000",
                        }
                    ],
                },
            ],
            "solve_loops": [],
        },
        "model_wiring": {
            "connections": [
                {"from": "generation_block.p50_generation_kwh", "to": "revenue_block.p50_gen"},
            ],
            "output_reports": {
                "income_statement": ["revenue_block.ppa_revenue"],
                "cash_flow_statement": [],
                "debt_schedule": [],
                "returns_summary": [],
            },
        },
    }

    if overrides:
        _deep_update(base, overrides)
    return base


def _deep_update(base: dict, updates: dict) -> dict:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parser() -> DSLParser:
    return DSLParser()


# ===========================================================================
# 1. Basic round-trip
# ===========================================================================


class TestBasicParsing:
    def test_valid_model_returns_no_errors(self, parser):
        model, validation = parser.load_dict(minimal_model())
        assert validation.valid, f"Unexpected errors: {validation.errors}"
        assert isinstance(model, ModelDefinition)

    def test_model_id_preserved(self, parser):
        model, _ = parser.load_dict(minimal_model())
        assert model.project_skeleton.model_id == "test_solar"

    def test_project_type_parsed(self, parser):
        model, _ = parser.load_dict(minimal_model())
        assert model.project_skeleton.project_type.value == "solar_ipp"

    def test_total_periods_correct(self, parser):
        model, _ = parser.load_dict(minimal_model())
        assert model.project_skeleton.total_periods == 15

    def test_assumptions_count(self, parser):
        model, _ = parser.load_dict(minimal_model())
        assert len(model.assumption_schema.assumptions) == 9

    def test_blocks_parsed(self, parser):
        model, _ = parser.load_dict(minimal_model())
        assert len(model.calculation_blocks.blocks) == 2
        ids = [b.block_id for b in model.calculation_blocks.blocks]
        assert "generation_block" in ids
        assert "revenue_block" in ids

    def test_connection_parsed(self, parser):
        model, _ = parser.load_dict(minimal_model())
        assert len(model.model_wiring.connections) == 1
        conn = model.model_wiring.connections[0]
        assert conn.from_ref == "generation_block.p50_generation_kwh"
        assert conn.to_ref == "revenue_block.p50_gen"


# ===========================================================================
# 2. Phase mask derivation
# ===========================================================================


class TestPhaseMasks:
    def test_phases_auto_derived(self, parser):
        model, _ = parser.load_dict(minimal_model())
        assert model.project_skeleton.phases is not None

    def test_construction_mask_length(self, parser):
        model, _ = parser.load_dict(minimal_model())
        phases = model.project_skeleton.phases
        assert len(phases.is_construction) == 15

    def test_construction_first_three_periods(self, parser):
        model, _ = parser.load_dict(minimal_model())
        phases = model.project_skeleton.phases
        assert phases.is_construction[:3] == [True, True, True]
        assert phases.is_construction[3] is False

    def test_operational_starts_at_cod(self, parser):
        model, _ = parser.load_dict(minimal_model())
        phases = model.project_skeleton.phases
        assert phases.is_operational[2] is False   # last construction period
        assert phases.is_operational[3] is True    # COD

    def test_operational_and_construction_are_complementary(self, parser):
        model, _ = parser.load_dict(minimal_model())
        phases = model.project_skeleton.phases
        for t in range(15):
            assert phases.is_construction[t] != phases.is_operational[t]

    def test_debt_outstanding_mask(self, parser):
        model, _ = parser.load_dict(minimal_model())
        phases = model.project_skeleton.phases
        # COD = 3, debt_maturity = 11
        assert phases.is_debt_outstanding[2] is False   # pre-COD
        assert phases.is_debt_outstanding[3] is True    # COD
        assert phases.is_debt_outstanding[11] is True   # maturity
        assert phases.is_debt_outstanding[12] is False  # post-maturity

    def test_debt_mask_length_equals_total_periods(self, parser):
        model, _ = parser.load_dict(minimal_model())
        phases = model.project_skeleton.phases
        assert len(phases.is_debt_outstanding) == 15


# ===========================================================================
# 3. Validation errors
# ===========================================================================


class TestValidationErrors:
    def test_missing_project_skeleton(self, parser):
        data = minimal_model()
        del data["project_skeleton"]
        _, validation = parser.load_dict(data)
        assert not validation.valid
        assert any("project_skeleton" in e for e in validation.errors)

    def test_invalid_project_type(self, parser):
        data = minimal_model()
        data["project_skeleton"]["project_type"] = "nuclear_plant"
        _, validation = parser.load_dict(data)
        assert not validation.valid

    def test_milestone_ordering_violated(self, parser):
        data = minimal_model()
        data["project_skeleton"]["milestones"]["cod"] = 1  # cod < construction_periods
        _, validation = parser.load_dict(data)
        assert not validation.valid

    def test_capex_schedule_wrong_sum(self, parser):
        data = minimal_model()
        # Modify assumption value directly
        for a in data["assumption_schema"]["assumptions"]:
            if a["name"] == "capex_schedule":
                a["value"] = [0.30, 0.40, 0.25]  # sums to 0.95
        _, validation = parser.load_dict(data)
        assert not validation.valid
        assert any("capex_schedule" in e for e in validation.errors)

    def test_capex_schedule_wrong_length(self, parser):
        data = minimal_model()
        for a in data["assumption_schema"]["assumptions"]:
            if a["name"] == "capex_schedule":
                a["value"] = [0.50, 0.50]  # 2 entries, construction_periods=3
        _, validation = parser.load_dict(data)
        assert not validation.valid
        assert any("capex_schedule" in e for e in validation.errors)

    def test_moratorium_exceeds_debt_tenor(self, parser):
        data = minimal_model()
        for a in data["assumption_schema"]["assumptions"]:
            if a["name"] == "moratorium_periods":
                a["value"] = 50  # way more than debt tenor in periods
        _, validation = parser.load_dict(data)
        assert not validation.valid

    def test_debt_percent_out_of_range(self, parser):
        data = minimal_model()
        for a in data["assumption_schema"]["assumptions"]:
            if a["name"] == "debt_percent_of_total_cost":
                a["value"] = 1.20  # > 1.0
        _, validation = parser.load_dict(data)
        assert not validation.valid

    def test_negative_tariff(self, parser):
        data = minimal_model()
        for a in data["assumption_schema"]["assumptions"]:
            if a["name"] == "ppa_tariff_per_kwh":
                a["value"] = -1.0
        _, validation = parser.load_dict(data)
        assert not validation.valid

    def test_expression_references_undeclared_input(self, parser):
        data = minimal_model()
        # Introduce an undeclared variable in expression
        for b in data["calculation_blocks"]["blocks"]:
            if b["block_id"] == "generation_block":
                b["body"][0]["expr"] = "installed_mw * mystery_var * 2190 * is_operational"
        _, validation = parser.load_dict(data)
        assert not validation.valid
        assert any("mystery_var" in e for e in validation.errors)


# ===========================================================================
# 4. Validation warnings
# ===========================================================================


class TestValidationWarnings:
    def test_debt_tenor_exceeds_ppa_warns(self, parser):
        data = minimal_model()
        for a in data["assumption_schema"]["assumptions"]:
            if a["name"] == "debt_tenor_years":
                a["value"] = 30  # > ppa_tenor_years=25
        model, validation = parser.load_dict(data)
        # Should produce a warning, not necessarily an error
        warning_codes = [w.code for w in validation.warnings]
        assert "DEBT_TENOR_EXCEEDS_PPA" in warning_codes

    def test_high_cuf_warns(self, parser):
        data = minimal_model()
        for a in data["assumption_schema"]["assumptions"]:
            if a["name"] == "capacity_utilisation_factor_p50":
                a["value"] = 0.35  # > 0.30 threshold
        _, validation = parser.load_dict(data)
        warning_codes = [w.code for w in validation.warnings]
        assert "HIGH_CUF" in warning_codes

    def test_low_cuf_warns(self, parser):
        data = minimal_model()
        for a in data["assumption_schema"]["assumptions"]:
            if a["name"] == "capacity_utilisation_factor_p50":
                a["value"] = 0.12  # < 0.15 threshold
        _, validation = parser.load_dict(data)
        warning_codes = [w.code for w in validation.warnings]
        assert "LOW_CUF" in warning_codes

    def test_unusual_interest_rate_warns(self, parser):
        data = minimal_model()
        for a in data["assumption_schema"]["assumptions"]:
            if a["name"] == "interest_rate":
                a["value"] = 0.40  # 40% — unusual
        _, validation = parser.load_dict(data)
        warning_codes = [w.code for w in validation.warnings]
        assert "UNUSUAL_INTEREST_RATE" in warning_codes


# ===========================================================================
# 5. Dependency graph and cycle detection
# ===========================================================================


class TestDependencyGraph:
    def test_no_cycles_in_valid_model(self, parser):
        data = minimal_model()
        _, validation = parser.load_dict(data)
        # No cycle errors expected
        cycle_errors = [e for e in validation.errors if "cycle" in e.lower()]
        assert cycle_errors == []

    def test_graph_built_for_two_blocks(self, parser):
        data = minimal_model()
        model, validation = parser.load_dict(data)
        assert validation.valid
        # Build graph manually and check it has the right structure
        graph = parser._build_dependency_graph(model)
        assert "generation_block" in graph.nodes
        assert "revenue_block" in graph.nodes
        # revenue_block should depend on generation_block
        assert graph.has_edge("generation_block", "revenue_block")

    def test_undeclared_cycle_raises_error(self, parser):
        # Create a cycle: A → B → A with no solve_loop
        data = minimal_model()
        # Modify revenue_block to depend on itself (via generation_block which depends on it)
        for b in data["calculation_blocks"]["blocks"]:
            if b["block_id"] == "generation_block":
                # Add an input sourced from revenue_block output (creating a cycle)
                b["inputs"].append({
                    "name": "revenue_feedback",
                    "source": "revenue_block.ppa_revenue",
                })
                b["body"][0]["expr"] = (
                    "installed_mw * cuf * 2190 * 1000 * is_operational + revenue_feedback * 0"
                )
        # This should fail wiring validation (revenue_block not yet computed at generation_block time)
        # For cycle detection, we check the graph SCC
        # The wiring cross-check in ModelDefinition will catch this as an unresolved reference
        # OR the cycle detector will flag it
        _, validation = parser.load_dict(data)
        # Either a wiring error or a cycle error is acceptable
        has_issue = (not validation.valid) or any(
            "cycle" in e.lower() for e in validation.errors
        )
        assert has_issue, (
            "Expected either a validation error or cycle error for circular dependency"
        )


# ===========================================================================
# 6. Module-level convenience function
# ===========================================================================


class TestConvenienceFunction:
    def test_load_model_from_dict(self):
        model, validation = load_model_from_dict(minimal_model())
        assert validation.valid
        assert model.project_skeleton.model_id == "test_solar"

    def test_invalid_model_returns_errors(self):
        data = minimal_model()
        del data["project_skeleton"]
        _, validation = load_model_from_dict(data)
        assert not validation.valid


# ===========================================================================
# 7. Solve loop parsing
# ===========================================================================


class TestSolveLoops:
    def test_solve_loop_parsed(self, parser):
        data = minimal_model()
        data["calculation_blocks"]["solve_loops"] = [
            {
                "loop_id": "idc_goal_seek",
                "type": "goal_seek",
                "free_variable": "assumption.debt_percent_of_total_cost",
                "target_expression": "debt_sizing_block.debt_amount",
                "target_value": 31500.0,
                "tolerance": 1e-4,
                "max_iterations": 50,
            }
        ]
        model, validation = parser.load_dict(data)
        # The solve loop free_variable references a known assumption → valid
        assert len(model.calculation_blocks.solve_loops) == 1
        sl = model.calculation_blocks.solve_loops[0]
        assert sl.loop_id == "idc_goal_seek"
        assert sl.type.value == "goal_seek"

    def test_invalid_solve_loop_free_variable(self, parser):
        data = minimal_model()
        data["calculation_blocks"]["solve_loops"] = [
            {
                "loop_id": "bad_loop",
                "type": "goal_seek",
                "free_variable": "assumption.nonexistent_assumption",
                "target_expression": "some_block.output",
                "target_value": 0.0,
            }
        ]
        _, validation = parser.load_dict(data)
        # free_variable references a nonexistent assumption → error
        assert not validation.valid


# ===========================================================================
# 8. Assumption schema helpers
# ===========================================================================


class TestAssumptionSchema:
    def test_by_name_lookup(self, parser):
        model, _ = parser.load_dict(minimal_model())
        a = model.assumption_schema.by_name("installed_capacity_mw")
        assert a is not None
        assert a.value == 100.0

    def test_by_name_missing_returns_none(self, parser):
        model, _ = parser.load_dict(minimal_model())
        assert model.assumption_schema.by_name("does_not_exist") is None

    def test_sensitivity_assumptions_filtered(self, parser):
        model, _ = parser.load_dict(minimal_model())
        sensitive = model.assumption_schema.sensitivity_assumptions()
        names = [a.name for a in sensitive]
        assert "capacity_utilisation_factor_p50" in names
        assert "ppa_tariff_per_kwh" in names
        assert "interest_rate" in names
        # Non-vary assumptions should be excluded
        assert "installed_capacity_mw" not in names
