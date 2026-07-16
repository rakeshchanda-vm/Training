from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable
from typing import Any, Dict, List, Literal, Mapping


ValidationSeverity = Literal["error", "warning"]


@dataclass
class ValidationIssue:
    """Single structured validation issue."""

    severity: ValidationSeverity
    message: str
    path: str | None = None
    code: str | None = None


@dataclass
class ValidationResult:
    """Structured output for validation checks."""

    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [
                {
                    "severity": issue.severity,
                    "message": issue.message,
                    **({"path": issue.path} if issue.path else {}),
                    **({"code": issue.code} if issue.code else {}),
                }
                for issue in self.issues
            ],
            "details": self.details,
        }


class AndromedaRuntimeError(Exception):
    """Base exception for runtime-level failures."""


class RunnableNotFoundError(AndromedaRuntimeError):
    """Raised when a runnable name cannot be resolved."""


class RunnableAmbiguousError(AndromedaRuntimeError):
    """Raised when a runnable name can match multiple entries."""


class WorkflowValidationError(AndromedaRuntimeError):
    """Raised when workflow definition is invalid."""


class AgentBuildError(AndromedaRuntimeError):
    """Raised when an agent entry cannot be built."""


class RunError(AndromedaRuntimeError):
    """Raised when execution fails."""

def error(message: str, path: str | None = None, code: str | None = None) -> ValidationIssue:
    return ValidationIssue(severity="error", message=message, path=path, code=code)


def warning(message: str, path: str | None = None, code: str | None = None) -> ValidationIssue:
    return ValidationIssue(severity="warning", message=message, path=path, code=code)


def result_from_issues(issues: Iterable[ValidationIssue], details: Mapping[str, Any] | None = None) -> ValidationResult:
    return ValidationResult(
        valid=not any(issue.severity == "error" for issue in issues),
        issues=list(issues),
        details=dict(details or {}),
    )
