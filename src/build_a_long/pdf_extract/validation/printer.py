"""Validation result printing and formatting."""

from .types import ValidationResult, ValidationSeverity

# ANSI color codes
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def print_validation(validation: ValidationResult, *, use_color: bool = True) -> None:
    """Print validation results in a human-readable format.

    Args:
        validation: The validation result to print
        use_color: Whether to use ANSI colors in output
    """
    if not validation.has_issues():
        print("✓ All validation checks passed")
        return

    print("=== Validation Issues ===")

    # Group by severity
    for severity in [
        ValidationSeverity.ERROR,
        ValidationSeverity.WARNING,
        ValidationSeverity.INFO,
    ]:
        issues = [i for i in validation.issues if i.severity == severity]
        if not issues:
            continue

        for issue in issues:
            # Choose symbol and color based on severity
            if severity == ValidationSeverity.ERROR:
                symbol = "✗"
                color = RED if use_color else ""
            elif severity == ValidationSeverity.WARNING:
                symbol = "⚠"
                color = YELLOW if use_color else ""
            else:
                symbol = "ℹ"
                color = ""

            reset = RESET if use_color and color else ""

            print(f"{color}{symbol} [{issue.rule}] {issue.message}{reset}")

            if issue.pages:
                pages_str = ", ".join(str(p) for p in issue.pages[:10])
                if len(issue.pages) > 10:
                    pages_str += f" ... ({len(issue.pages)} total)"
                print(f"    Pages: {pages_str}")

            if issue.details:
                print(f"    Details: {issue.details}")

    # Summary
    print()
    print(
        f"Summary: {validation.error_count} errors, "
        f"{validation.warning_count} warnings, "
        f"{validation.info_count} info"
    )
