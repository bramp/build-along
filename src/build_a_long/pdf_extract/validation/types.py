"""Validation types and data structures."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ValidationSeverity(Enum):
    """Severity level for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationIssue(BaseModel):
    """A single validation issue found during analysis.

    Attributes:
        severity: The severity level of the issue
        rule: Short identifier for the validation rule
        message: Human-readable description of the issue
        pages: List of affected PDF page numbers (1-indexed)
        details: Optional additional details
    """

    model_config = ConfigDict(frozen=True)

    severity: ValidationSeverity
    rule: str
    message: str
    pages: list[int] = Field(default_factory=list)
    details: str | None = None


class ValidationResult(BaseModel):
    """Collection of all validation issues found.

    Attributes:
        issues: List of all validation issues
    """

    issues: list[ValidationIssue] = Field(default_factory=list)

    def add(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)

    @property
    def error_count(self) -> int:
        """Count of ERROR severity issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of WARNING severity issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    @property
    def info_count(self) -> int:
        """Count of INFO severity issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.INFO)

    def has_issues(self) -> bool:
        """Check if any issues were found."""
        return len(self.issues) > 0
