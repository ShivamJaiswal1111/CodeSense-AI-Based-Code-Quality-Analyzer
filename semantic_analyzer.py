"""
CodeSense - Semantic Analyzer
Deep semantic analysis: unused variables, dead code, scope tracking, logic issues.
"""

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from logger import get_logger

logger = get_logger(__name__)


@dataclass
class SemanticIssue:
    line:       int
    issue_type: str
    message:    str
    suggestion: str
    severity:   str = "WARNING"   # INFO | WARNING | ERROR
    symbol:     str = ""


class PythonSemanticVisitor(ast.NodeVisitor):
    """
    Walks Python AST to collect semantic information:
    - Variable assignments and usages
    - Function definitions and calls
    - Import usage
    - Unreachable code after return/raise
    - Mutable default arguments
    - Broad exception catches
    """

    def __init__(self) -> None:
        self.issues:      List[SemanticIssue] = []
        self.scopes:      List[Dict[str, int]] = [{}]   # stack of {name: line}
        self.used_names:  Set[str]             = set()
        self.functions:   Dict[str, int]       = {}
        self.called_fns:  Set[str]             = set()
        self.imports:     Dict[str, int]       = {}
        self.used_imports: Set[str]            = set()

    # ── Scope helpers ──────────────────────────────────────────────────────

    def _current_scope(self) -> Dict[str, int]:
        return self.scopes[-1]

    def _define(self, name: str, lineno: int) -> None:
        self._current_scope()[name] = lineno

    def _use(self, name: str) -> None:
        self.used_names.add(name)

    # ── Visitor methods ────────────────────────────────────────────────────

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions[node.name] = node.lineno
        self._check_mutable_default(node)
        self.scopes.append({})
        for arg in node.args.args:
            self._define(arg.arg, node.lineno)
        self.generic_visit(node)
        self._check_unused_vars(node.lineno)
        self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scopes.append({})
        self.generic_visit(node)
        self.scopes.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._define(target.id, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            self._define(node.target.id, node.lineno)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._use(node.id)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self.imports[name] = node.lineno

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                return   # Wildcard import — skip
            name = alias.asname or alias.name
            self.imports[name] = node.lineno

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            self._use(node.value.id)
            self.used_imports.add(node.value.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.called_fns.add(node.func.id)
            self._use(node.func.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.issues.append(SemanticIssue(
                line=node.lineno,
                issue_type="BareExcept",
                message="Bare 'except:' catches ALL exceptions including KeyboardInterrupt and SystemExit.",
                suggestion="Catch specific exceptions, e.g., 'except (ValueError, TypeError):'",
                severity="WARNING",
            ))
        elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
            self.issues.append(SemanticIssue(
                line=node.lineno,
                issue_type="BroadExcept",
                message="Catching broad 'Exception' may hide unexpected errors.",
                suggestion="Catch the most specific exception type possible.",
                severity="INFO",
            ))
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.generic_visit(node)

    # ── Post-visit checks ─────────────────────────────────────────────────

    def _check_unused_vars(self, fn_line: int) -> None:
        scope = self._current_scope()
        skip  = {"self", "cls", "_"}
        for name, def_line in scope.items():
            if name.startswith("_") or name in skip:
                continue
            if name not in self.used_names:
                self.issues.append(SemanticIssue(
                    line=def_line,
                    issue_type="UnusedVariable",
                    message=f"Variable '{name}' is assigned but never used.",
                    suggestion=f"Remove the assignment or use '{name}' in your logic. Prefix with '_' to indicate intentional non-use.",
                    severity="WARNING",
                    symbol=name,
                ))

    def _check_mutable_default(self, node: ast.FunctionDef) -> None:
        for default in node.args.defaults + node.args.kw_defaults:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.issues.append(SemanticIssue(
                    line=node.lineno,
                    issue_type="MutableDefault",
                    message=f"Function '{node.name}' uses a mutable default argument (list/dict/set).",
                    suggestion="Use None as default and assign the mutable value inside the function:\n"
                               "  def fn(items=None):\n      if items is None: items = []",
                    severity="WARNING",
                    symbol=node.name,
                ))

    def finalize(self) -> None:
        """Check unused imports and functions after full traversal."""
        for name, lineno in self.imports.items():
            if name not in self.used_names and name not in self.used_imports:
                self.issues.append(SemanticIssue(
                    line=lineno,
                    issue_type="UnusedImport",
                    message=f"'{name}' is imported but never used.",
                    suggestion=f"Remove 'import {name}' or use it in your code.",
                    severity="WARNING",
                    symbol=name,
                ))

        for fn_name, lineno in self.functions.items():
            if fn_name not in self.called_fns and not fn_name.startswith(("test_", "__")):
                self.issues.append(SemanticIssue(
                    line=lineno,
                    issue_type="UnusedFunction",
                    message=f"Function '{fn_name}' is defined but never called.",
                    suggestion=f"Remove '{fn_name}' if unused, or call it where needed.",
                    severity="INFO",
                    symbol=fn_name,
                ))


def _detect_unreachable_python(code: str) -> List[SemanticIssue]:
    """Detect statements after return/raise at the same indentation level."""
    issues: List[SemanticIssue] = []
    lines = code.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if stripped.startswith(("return ", "return\n", "raise ", "raise\n")):
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                next_stripped = next_line.strip()
                if not next_stripped or next_stripped.startswith("#"):
                    j += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent == indent and next_stripped not in ("}", ")"):
                    issues.append(SemanticIssue(
                        line=j + 1,
                        issue_type="UnreachableCode",
                        message=f"Code at line {j + 1} is unreachable after 'return'/'raise' on line {i + 1}.",
                        suggestion="Remove the unreachable code or restructure your logic.",
                        severity="WARNING",
                    ))
                break
            i = j
        else:
            i += 1
    return issues


class SemanticAnalyzer:
    """Public interface for semantic analysis."""

    def analyze(self, code: str, language: str) -> Dict[str, Any]:
        if language == "python":
            return self._analyze_python(code)
        elif language in ("java", "cpp"):
            return self._analyze_generic(code, language)
        return {"issues": [], "summary": {}}

    def _analyze_python(self, code: str) -> Dict[str, Any]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"issues": [], "summary": {"note": "Semantic analysis skipped due to syntax errors."}}

        visitor = PythonSemanticVisitor()
        visitor.visit(tree)
        visitor.finalize()

        unreachable = _detect_unreachable_python(code)
        all_issues  = visitor.issues + unreachable

        return {
            "issues": [
                {
                    "line":        i.line,
                    "type":        i.issue_type,
                    "message":     i.message,
                    "suggestion":  i.suggestion,
                    "severity":    i.severity,
                    "symbol":      i.symbol,
                }
                for i in all_issues
            ],
            "summary": {
                "unused_variables": sum(1 for i in all_issues if i.issue_type == "UnusedVariable"),
                "unused_imports":   sum(1 for i in all_issues if i.issue_type == "UnusedImport"),
                "unused_functions": sum(1 for i in all_issues if i.issue_type == "UnusedFunction"),
                "unreachable_code": sum(1 for i in all_issues if i.issue_type == "UnreachableCode"),
                "mutable_defaults": sum(1 for i in all_issues if i.issue_type == "MutableDefault"),
                "broad_exceptions": sum(1 for i in all_issues if i.issue_type in ("BareExcept", "BroadExcept")),
                "total":            len(all_issues),
            },
        }

    def _analyze_generic(self, code: str, language: str) -> Dict[str, Any]:
        """Heuristic semantic checks for Java/C++."""
        issues: List[Dict] = []
        lines = code.splitlines()

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Null/nullptr dereference risk
            if language == "java" and re.search(r"\w+\s*=\s*null\s*;.*\.\w+\s*\(", stripped):
                issues.append({
                    "line": i, "type": "NullDereference",
                    "message": "Possible null dereference: variable assigned null then immediately accessed.",
                    "suggestion": "Add a null check before accessing this object.",
                    "severity": "WARNING", "symbol": "",
                })

            # Empty catch block
            if re.search(r"catch\s*\([^)]+\)\s*\{\s*\}", stripped):
                issues.append({
                    "line": i, "type": "EmptyCatch",
                    "message": "Empty catch block silently swallows exceptions.",
                    "suggestion": "Log the exception or rethrow it: e.g., System.err.println(e.getMessage());",
                    "severity": "WARNING", "symbol": "",
                })

        return {
            "issues": issues,
            "summary": {"total": len(issues)},
        }