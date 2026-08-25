"""Executing model-generated pandas code, restricted two ways at once.

This is the one place in the app where an LLM's output does something other
than get parsed into a Pydantic model, so it gets two independent layers of
restriction rather than one, on the assumption that either could have a gap:

**Static**: :func:`validate_code` walks the parsed AST before anything runs and
rejects imports, function/class/lambda definitions, loops, `try`/`with`, any
name that was not assigned in this snippet or explicitly allowed (`df`, `pd`,
`np`, a short list of pure builtins), and any attribute access starting with
`_` -- which alone blocks every `__class__.__mro__`-style sandbox escape,
because reaching a dunder anywhere in the chain requires naming it.

**Dynamic**: even code that passes validation runs with `__builtins__` replaced
by a small explicit dict rather than the real one, against a *copy* of the
dataframe, with no `open`, `eval`, `exec`, `__import__`, `getattr`, or `input`
reachable by any name -- so a gap in the static pass still has nothing to call.

Execution also runs on a background thread with a join timeout, because pandas
has no cooperative cancellation: a runaway groupby cannot be interrupted, only
abandoned. The thread is a daemon and is not force-killed on timeout -- Python
has no safe way to do that -- so a timeout bounds how long the *user* waits, not
how long the interpreter keeps working. That is a stated limit of a pure-Python
sandbox, not an oversight.
"""

import ast
import threading
from typing import Any

import numpy as np
import pandas as pd

from app.core.exceptions import CodeExecutionError, CodeGenerationError, CodeTimeoutError

#: Names available to generated code without having been assigned by it.
_ALLOWED_GLOBAL_NAMES = frozenset({"df", "pd", "np", "result", "True", "False", "None"})

#: Builtins exposed to generated code. Nothing here can touch a file, the
#: network, the interpreter, or another object's internals.
_SAFE_BUILTINS: dict[str, Any] = {
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}

#: Statement and expression node types generated code may use. Notably absent:
#: Import, ImportFrom, FunctionDef, AsyncFunctionDef, ClassDef, Lambda, For,
#: While, Try, With, Raise, Assert, Global, Nonlocal, Delete.
_ALLOWED_NODE_TYPES: tuple[type[ast.AST], ...] = (
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.AugAssign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Attribute,
    ast.Call,
    ast.keyword,
    ast.Subscript,
    ast.Slice,
    ast.Index if hasattr(ast, "Index") else ast.Slice,  # py<3.9 compatibility, harmless on 3.11
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Set,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.comprehension,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
)


def _collect_assigned_names(tree: ast.AST) -> set[str]:
    """Every name that is a target of an assignment or comprehension binding.

    These are added to the allowed-names set for `Load` context, since code
    that computes `grouped = df.groupby(...)` needs to read `grouped` back.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def validate_code(code: str) -> ast.Module:
    """Parse `code` and reject anything outside the allowed subset.

    Args:
        code: The candidate pandas code.

    Returns:
        The parsed AST, for callers that want it (execution re-parses via
        `compile`, which is cheap and keeps this function side-effect-free).

    Raises:
        CodeGenerationError: The code is not valid Python, or uses a
            disallowed statement, name, or attribute.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise CodeGenerationError(f"The code has a syntax error: {exc}") from exc

    allowed_names = _ALLOWED_GLOBAL_NAMES | set(_SAFE_BUILTINS) | _collect_assigned_names(tree)

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise CodeGenerationError(
                f"The code uses `{type(node).__name__}`, which is not allowed here. "
                "Use only pandas expressions and simple assignments against `df`."
            )
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise CodeGenerationError(
                f"The code accesses `.{node.attr}`, which is not allowed. "
                "Only public pandas/numpy attributes may be used."
            )
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and (node.id.startswith("_") or node.id not in allowed_names)
        ):
            raise CodeGenerationError(
                f"The code references `{node.id}`, which is not defined. "
                "Only `df`, `pd`, `np`, and names the code itself assigns are available."
            )

    return tree


def _run_in_thread(code: str, exec_globals: dict[str, Any], timeout_seconds: float) -> None:
    """Compile and execute `code` against `exec_globals` on a joined thread.

    Raises:
        CodeExecutionError: The code raised while running.
        CodeTimeoutError: The thread did not finish within `timeout_seconds`.
    """
    outcome: dict[str, BaseException] = {}

    def _target() -> None:
        try:
            compiled = compile(code, "<generated>", "exec")
            exec(compiled, exec_globals)  # noqa: S102 - restricted globals, see module docstring
        except BaseException as exc:  # noqa: BLE001 - reported to the caller, not swallowed
            outcome["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        raise CodeTimeoutError(
            f"The analysis did not finish within {timeout_seconds:.0f}s. Try a narrower question."
        )

    error = outcome.get("error")
    if error is not None:
        raise CodeExecutionError(f"The code raised {type(error).__name__}: {error}") from error


def run_code(code: str, df: pd.DataFrame, *, timeout_seconds: float) -> Any:
    """Validate and execute generated code against a dataframe.

    Args:
        code: The candidate pandas code, assigning its answer to `result`.
        df: The dataset. A copy is executed against, so generated code cannot
            mutate the caller's dataframe.
        timeout_seconds: Wall-clock budget for execution.

    Returns:
        The value bound to `result` after execution.

    Raises:
        CodeGenerationError: The code failed static validation.
        CodeExecutionError: The code raised, or never assigned `result`.
        CodeTimeoutError: Execution exceeded the time budget.
    """
    validate_code(code)

    exec_globals: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "df": df.copy(),
    }

    _run_in_thread(code, exec_globals, timeout_seconds)

    if "result" not in exec_globals:
        raise CodeExecutionError("The code did not assign an answer to `result`.")

    return exec_globals["result"]
