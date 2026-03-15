"""Error analysis for Colab GPU execution results.

Classifies stderr output into actionable error categories and provides
user-friendly suggestions for resolution. Designed as a standalone module
with no dependencies on other project modules.

Key design:
- Immutable pattern definitions (tuple of tuples)
- Frozen dataclass for analysis results
- First-match-wins pattern scanning
- Never mutates input data
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# Maximum length of raw error text preserved in analysis results.
_RAW_ERROR_MAX_LENGTH = 500


class ErrorCategory(Enum):
    """Classification categories for Colab execution errors."""

    OOM = "oom"
    CUDA_ERROR = "cuda_error"
    IMPORT_ERROR = "import_error"
    TIMEOUT = "timeout"
    QUOTA_EXCEEDED = "quota_exceeded"
    RUNTIME_ERROR = "runtime_error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ErrorAnalysis:
    """Immutable analysis result for an execution error.

    Attributes:
        category: The classified error type.
        message: Human-readable description of the error.
        suggestion: Actionable advice to resolve the error.
        raw_error: Truncated original stderr text.
    """

    category: ErrorCategory
    message: str
    suggestion: str
    raw_error: str


# Each entry: (category, match_keywords, message, suggestion)
# Order matters — first match wins.
_ERROR_PATTERNS: tuple[tuple[ErrorCategory, tuple[str, ...], str, str], ...] = (
    (
        ErrorCategory.OOM,
        ("OutOfMemoryError", "CUDA out of memory", "ResourceExhausted", "OOM"),
        "GPU memory exhausted during execution",
        "Reduce batch_size, use gradient checkpointing, or upgrade to A100/H100",
    ),
    (
        ErrorCategory.CUDA_ERROR,
        ("CUDA error", "cudnn", "NCCL", "CudaError"),
        "CUDA runtime error occurred",
        "Check CUDA compatibility. Try restarting the runtime",
    ),
    (
        ErrorCategory.IMPORT_ERROR,
        ("ModuleNotFoundError", "ImportError"),
        "Required Python package not found",
        "Add '!pip install <package>' before your code",
    ),
    (
        ErrorCategory.TIMEOUT,
        ("Timed out waiting for kernel", "Timed out creating kernel session"),
        "Execution exceeded the time limit",
        "Increase the timeout parameter or simplify the computation",
    ),
    (
        ErrorCategory.QUOTA_EXCEEDED,
        ("GPU quota exceeded", "ResourcesExhausted", "quota"),
        "GPU quota limit reached",
        "Wait and retry, use a different account, or try a lower-tier GPU",
    ),
    (
        ErrorCategory.RUNTIME_ERROR,
        ("RuntimeError:", "ValueError:", "TypeError:", "ZeroDivisionError:"),
        "Python runtime error occurred",
        "Check the traceback for the specific error location and fix the code",
    ),
)

# Regex to extract module name from ImportError/ModuleNotFoundError messages.
_IMPORT_MODULE_RE = re.compile(r"No module named '(\w+)'")


def analyze_error(stderr: str, exit_code: int) -> ErrorAnalysis | None:
    """Classify stderr output into an actionable error analysis.

    Args:
        stderr: Standard error output from the Colab execution.
        exit_code: Process exit code (0 = success).

    Returns:
        An ErrorAnalysis if an error is detected, or None if the
        execution appears successful.
    """
    raw_error = stderr[:_RAW_ERROR_MAX_LENGTH]

    for category, keywords, message, suggestion in _ERROR_PATTERNS:
        if any(kw in stderr for kw in keywords):
            # Customize suggestion for import errors with package name.
            if category is ErrorCategory.IMPORT_ERROR:
                match = _IMPORT_MODULE_RE.search(stderr)
                if match:
                    package = match.group(1)
                    suggestion = f"Add '!pip install {package}' before your code"

            return ErrorAnalysis(
                category=category,
                message=message,
                suggestion=suggestion,
                raw_error=raw_error,
            )

    if exit_code != 0:
        return ErrorAnalysis(
            category=ErrorCategory.UNKNOWN,
            message="Execution failed with a non-zero exit code",
            suggestion="Check the stderr output for details",
            raw_error=raw_error,
        )

    return None


def enrich_result(result_dict: dict, stderr: str, exit_code: int) -> dict:
    """Return a new result dict enriched with error analysis if applicable.

    Creates a new dict rather than mutating the input (immutability principle).

    Args:
        result_dict: Original execution result dictionary.
        stderr: Standard error output from the Colab execution.
        exit_code: Process exit code (0 = success).

    Returns:
        A new dict with an ``error_analysis`` key added when an error
        is detected, or the original dict unchanged.
    """
    analysis = analyze_error(stderr, exit_code)
    if analysis is None:
        return result_dict

    return {
        **result_dict,
        "error_analysis": {
            "category": analysis.category.value,
            "message": analysis.message,
            "suggestion": analysis.suggestion,
        },
    }
