"""
DSL Expression Language — parser and evaluator.

Parses expression strings using Python's ast module (no eval/exec).
All numeric operations are delegated to NumPy — the evaluator never
produces an LLM-computed value.

Supported primitives
--------------------
Arithmetic  : +  -  *  /  **  (element-wise over arrays or scalars)
Unary       : - (negation)
Comparison  : <  >  <=  >=  ==  !=  (return bool arrays, for use as masks)
Phase refs  : phase.is_operational / phase.is_construction / phase.is_debt_outstanding
              (accessed as dotted names in the context dict)

Built-in functions
------------------
  cumsum(series)               → np.cumsum
  lag(series, n)               → shift right by n periods, zero-pad left
  pv(series, rate)             → scalar: sum of discounted values, period-end convention
  annuity(principal, rate, n)  → array length n_periods; level payment in first n slots
  max_series(a, b)             → np.maximum element-wise
  min_series(a, b)             → np.minimum element-wise
  scalar_to_series(v)          → np.full(n_periods, v)
  escalate(base, rate)         → base * (1+rate)^(period_index / periods_per_year)

Variable resolution
-------------------
Variables in expressions are local names defined by BlockInput.name.
The caller supplies a context dict:  {local_name: np.ndarray | float}
Dotted references (phase.is_operational) are stored as literal keys in the context.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ExpressionError(Exception):
    """Raised when an expression cannot be parsed or evaluated."""


class UndefinedVariableError(ExpressionError):
    """Raised when a variable in the expression has no entry in the context."""


class UnsupportedNodeError(ExpressionError):
    """Raised when the expression uses a Python construct we do not allow."""


# ---------------------------------------------------------------------------
# Safe AST node allowlist
# ---------------------------------------------------------------------------

_ALLOWED_NODES: Set[type] = {
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Attribute,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Compare,
    ast.BoolOp,
    # Arithmetic operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.FloorDiv, ast.Mod,
    ast.USub, ast.UAdd,
    # Compare operators
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    # Bool operators
    ast.And, ast.Or,
    ast.Not,
    # AST context nodes (internal to Python AST — not executable constructs)
    ast.Load, ast.Store, ast.Del,
}


def _check_ast_safety(tree: ast.AST, expr: str) -> None:
    """Walk the AST and raise UnsupportedNodeError if any disallowed node is found."""
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            raise UnsupportedNodeError(
                f"Expression '{expr}' uses disallowed construct: "
                f"{type(node).__name__}. Only arithmetic, comparisons, and "
                f"whitelisted function calls are permitted."
            )


# ---------------------------------------------------------------------------
# ExpressionEvaluator
# ---------------------------------------------------------------------------


class ExpressionEvaluator:
    """
    Stateful evaluator configured for a specific model time axis.

    Parameters
    ----------
    n_periods : int
        Total number of periods in the model (construction + operations).
    periods_per_year : int
        Periods per calendar year (4 = quarterly).
    """

    def __init__(self, n_periods: int, periods_per_year: int) -> None:
        if n_periods < 1:
            raise ValueError(f"n_periods must be >= 1, got {n_periods}")
        if periods_per_year < 1:
            raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}")

        self.n_periods = n_periods
        self.periods_per_year = periods_per_year
        # Period index array: 0-based, period-end convention
        self._period_index = np.arange(n_periods, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        expr: str,
        context: Dict[str, Any],
    ) -> Union[np.ndarray, float]:
        """
        Parse and evaluate an expression string.

        Parameters
        ----------
        expr    : DSL expression string
        context : mapping of variable names → np.ndarray or scalar

        Returns
        -------
        np.ndarray of shape (n_periods,) or a scalar float/bool.
        """
        expr = expr.strip()
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(
                f"Syntax error in expression '{expr}': {exc}"
            ) from exc

        _check_ast_safety(tree, expr)
        try:
            result = self._eval_node(tree.body, context)
        except (UndefinedVariableError, UnsupportedNodeError):
            raise
        except Exception as exc:
            raise ExpressionError(
                f"Error evaluating expression '{expr}': {exc}"
            ) from exc

        return result

    def referenced_variables(self, expr: str) -> List[str]:
        """
        Return all variable names (including dotted names) referenced in expr.
        Used by the DSL parser for dependency analysis without a context.
        """
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"Syntax error in expression '{expr}': {exc}") from exc

        _check_ast_safety(tree, expr)
        refs: List[str] = []
        self._collect_refs(tree.body, refs)
        return refs

    # ------------------------------------------------------------------
    # AST node evaluator
    # ------------------------------------------------------------------

    def _eval_node(
        self,
        node: ast.expr,
        context: Dict[str, Any],
    ) -> Union[np.ndarray, float]:

        # Literals
        if isinstance(node, ast.Constant):
            return node.value

        # Simple names: local variable or 'period_index'
        if isinstance(node, ast.Name):
            return self._resolve_name(node.id, context)

        # Dotted attribute access: phase.is_operational, etc.
        if isinstance(node, ast.Attribute):
            dotted = self._dotted_name(node)
            if dotted is not None:
                return self._resolve_name(dotted, context)
            raise UnsupportedNodeError(
                f"Complex attribute expression is not supported: {ast.dump(node)}"
            )

        # Binary operations
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, context)
            right = self._eval_node(node.right, context)
            return self._apply_binop(node.op, left, right)

        # Unary operations
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, context)
            if isinstance(node.op, ast.USub):
                return -operand  # type: ignore[operator]
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.Not):
                return ~np.asarray(operand, dtype=bool) if isinstance(operand, np.ndarray) else not operand
            raise UnsupportedNodeError(f"Unsupported unary op: {type(node.op).__name__}")

        # Comparisons: a < b, a >= b, etc.
        if isinstance(node, ast.Compare):
            return self._eval_compare(node, context)

        # Boolean ops: and, or
        if isinstance(node, ast.BoolOp):
            return self._eval_boolop(node, context)

        # Function calls
        if isinstance(node, ast.Call):
            return self._eval_call(node, context)

        raise UnsupportedNodeError(
            f"Unsupported AST node type: {type(node).__name__}"
        )

    # ------------------------------------------------------------------
    # Name resolution
    # ------------------------------------------------------------------

    def _resolve_name(self, name: str, context: Dict[str, Any]) -> Any:
        if name == "period_index":
            return self._period_index.copy()
        if name in context:
            val = context[name]
            if isinstance(val, (int, float, bool, str)):
                return val
            return np.asarray(val, dtype=np.float64)
        raise UndefinedVariableError(
            f"Variable '{name}' is not in the evaluation context. "
            f"Available: {sorted(context.keys())}"
        )

    def _dotted_name(self, node: ast.Attribute) -> Optional[str]:
        """Recursively build a dotted string from nested Attribute nodes."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        if isinstance(node.value, ast.Attribute):
            prefix = self._dotted_name(node.value)
            if prefix is not None:
                return f"{prefix}.{node.attr}"
        return None

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    def _apply_binop(
        self,
        op: ast.operator,
        left: Any,
        right: Any,
    ) -> Any:
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            # Guard against divide-by-zero: return 0 where denominator is 0
            if isinstance(right, np.ndarray):
                with np.errstate(divide="ignore", invalid="ignore"):
                    result = np.where(right == 0, 0.0, left / right)
                return result
            if right == 0:
                return 0.0
            return left / right
        if isinstance(op, ast.Pow):
            return left**right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Mod):
            return left % right
        raise UnsupportedNodeError(f"Unsupported binary operator: {type(op).__name__}")

    def _eval_compare(
        self,
        node: ast.Compare,
        context: Dict[str, Any],
    ) -> Any:
        left = self._eval_node(node.left, context)
        result = None
        for op, comparator in zip(node.ops, node.comparators):
            right = self._eval_node(comparator, context)
            cmp: Any
            if isinstance(op, ast.Lt):
                cmp = left < right
            elif isinstance(op, ast.LtE):
                cmp = left <= right
            elif isinstance(op, ast.Gt):
                cmp = left > right
            elif isinstance(op, ast.GtE):
                cmp = left >= right
            elif isinstance(op, ast.Eq):
                cmp = left == right
            elif isinstance(op, ast.NotEq):
                cmp = left != right
            else:
                raise UnsupportedNodeError(f"Unsupported compare op: {type(op).__name__}")
            result = cmp if result is None else (result & cmp)
            left = right
        return result

    def _eval_boolop(
        self,
        node: ast.BoolOp,
        context: Dict[str, Any],
    ) -> Any:
        values = [self._eval_node(v, context) for v in node.values]
        if isinstance(node.op, ast.And):
            result = values[0]
            for v in values[1:]:
                result = result & v
            return result
        if isinstance(node.op, ast.Or):
            result = values[0]
            for v in values[1:]:
                result = result | v
            return result
        raise UnsupportedNodeError(f"Unsupported bool op: {type(node.op).__name__}")

    # ------------------------------------------------------------------
    # Function call dispatcher
    # ------------------------------------------------------------------

    _BUILTINS = {
        "cumsum",
        "lag",
        "pv",
        "annuity",
        "max_series",
        "min_series",
        "scalar_to_series",
        "escalate",
        # Convenience math pass-throughs
        "abs",
        "min",
        "max",
        "round",
    }

    def _eval_call(
        self,
        node: ast.Call,
        context: Dict[str, Any],
    ) -> Any:
        # Resolve function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = self._dotted_name(node.func)
        else:
            raise UnsupportedNodeError("Complex function expression is not supported")

        if func_name not in self._BUILTINS:
            raise UnsupportedNodeError(
                f"Unknown function '{func_name}'. "
                f"Allowed functions: {sorted(self._BUILTINS)}"
            )

        args = [self._eval_node(a, context) for a in node.args]

        dispatch = {
            "cumsum": self._fn_cumsum,
            "lag": self._fn_lag,
            "pv": self._fn_pv,
            "annuity": self._fn_annuity,
            "max_series": self._fn_max_series,
            "min_series": self._fn_min_series,
            "scalar_to_series": self._fn_scalar_to_series,
            "escalate": self._fn_escalate,
            "abs": lambda *a: np.abs(a[0]),
            "min": lambda *a: np.minimum(a[0], a[1]) if len(a) == 2 else np.min(a[0]),
            "max": lambda *a: np.maximum(a[0], a[1]) if len(a) == 2 else np.max(a[0]),
            "round": lambda *a: np.round(a[0], int(a[1]) if len(a) > 1 else 0),
        }
        return dispatch[func_name](*args)

    # ------------------------------------------------------------------
    # Built-in function implementations
    # (all pure NumPy — no arithmetic performed by the evaluator itself)
    # ------------------------------------------------------------------

    def _fn_cumsum(self, series: np.ndarray) -> np.ndarray:
        """Cumulative sum along the time axis."""
        return np.cumsum(np.asarray(series, dtype=np.float64))

    def _fn_lag(self, series: np.ndarray, n: Any) -> np.ndarray:
        """
        Shift series right by n periods, padding with 0 on the left.
        lag(x, 1)[t] = x[t-1]  (i.e. previous period value)
        """
        n_shift = int(n)
        if n_shift < 0:
            raise ExpressionError(f"lag() shift must be >= 0, got {n_shift}")
        arr = np.asarray(series, dtype=np.float64)
        result = np.zeros(self.n_periods, dtype=np.float64)
        if n_shift < self.n_periods:
            end = self.n_periods - n_shift
            result[n_shift:] = arr[:end]
        return result

    def _fn_pv(self, series: np.ndarray, rate: Any) -> float:
        """
        Present value of a cash-flow series discounted to period 0.
        Period-end convention: CF at period t is discounted by (1+r)^(t+1).
        Returns a scalar.
        """
        arr = np.asarray(series, dtype=np.float64)
        r = float(rate)
        # t=0 is financial close; first cash flow lands at end of period 0
        t = np.arange(1, len(arr) + 1, dtype=np.float64)
        if r == 0.0:
            return float(np.sum(arr))
        discount = (1.0 + r) ** t
        return float(np.sum(arr / discount))

    def _fn_annuity(
        self,
        principal: Any,
        rate: Any,
        n_periods: Any,
    ) -> np.ndarray:
        """
        Level-annuity repayment schedule.

        Returns a numpy array of length self.n_periods.
        Positions [0 .. n-1] contain the equal periodic payment amount.
        Remaining positions are zero.

        Formula: PMT = PV × r(1+r)^n / ((1+r)^n − 1)
        The rate must be the per-period interest rate (annual_rate / periods_per_year).
        """
        pv = float(principal)
        r = float(rate)
        n = int(n_periods)
        result = np.zeros(self.n_periods, dtype=np.float64)
        if n <= 0 or pv <= 0.0:
            return result

        if r == 0.0:
            payment = pv / n
        else:
            payment = pv * r * (1.0 + r) ** n / ((1.0 + r) ** n - 1.0)

        slots = min(n, self.n_periods)
        result[:slots] = payment
        return result

    def _fn_max_series(self, a: Any, b: Any) -> np.ndarray:
        """Element-wise maximum."""
        return np.maximum(
            np.asarray(a, dtype=np.float64),
            np.asarray(b, dtype=np.float64),
        )

    def _fn_min_series(self, a: Any, b: Any) -> np.ndarray:
        """Element-wise minimum."""
        return np.minimum(
            np.asarray(a, dtype=np.float64),
            np.asarray(b, dtype=np.float64),
        )

    def _fn_scalar_to_series(self, value: Any) -> np.ndarray:
        """Broadcast a scalar to a full-length time-series array."""
        return np.full(self.n_periods, float(value), dtype=np.float64)

    def _fn_escalate(self, base: Any, rate: Any) -> np.ndarray:
        """
        Compound-growth escalation over the time axis.

        escalate(base, rate)[t] = base × (1 + rate)^(t / periods_per_year)

        Period 0 (financial close) → base × 1.0  (no escalation yet).
        """
        b = float(base)
        r = float(rate)
        return b * (1.0 + r) ** (self._period_index / self.periods_per_year)

    # ------------------------------------------------------------------
    # Static analysis helper
    # ------------------------------------------------------------------

    def _collect_refs(self, node: ast.expr, refs: List[str]) -> None:
        """Recursively collect all variable/attribute references in an AST."""
        if isinstance(node, ast.Name):
            if node.id != "period_index":
                refs.append(node.id)
        elif isinstance(node, ast.Attribute):
            dotted = self._dotted_name(node)
            if dotted:
                refs.append(dotted)
        elif isinstance(node, ast.Call):
            for arg in node.args:
                self._collect_refs(arg, refs)
        elif isinstance(node, ast.BinOp):
            self._collect_refs(node.left, refs)
            self._collect_refs(node.right, refs)
        elif isinstance(node, ast.UnaryOp):
            self._collect_refs(node.operand, refs)
        elif isinstance(node, ast.Compare):
            self._collect_refs(node.left, refs)
            for c in node.comparators:
                self._collect_refs(c, refs)
        elif isinstance(node, ast.BoolOp):
            for v in node.values:
                self._collect_refs(v, refs)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def evaluate(
    expr: str,
    context: Dict[str, Any],
    n_periods: int,
    periods_per_year: int = 4,
) -> Union[np.ndarray, float]:
    """
    One-shot evaluate without constructing an ExpressionEvaluator explicitly.
    Useful in tests and ad-hoc usage.
    """
    ev = ExpressionEvaluator(n_periods=n_periods, periods_per_year=periods_per_year)
    return ev.evaluate(expr, context)
