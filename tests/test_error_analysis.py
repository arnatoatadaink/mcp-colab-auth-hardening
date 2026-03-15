"""Tests for error_analysis module.

Covers:
- ErrorCategory enum values
- ErrorAnalysis frozen dataclass
- analyze_error: each category, None for success, UNKNOWN fallback, first-match-wins
- enrich_result: adds error_analysis key, immutability, passthrough on success
- ImportError package name extraction
"""

from __future__ import annotations

import pytest

from mcp_colab_gpu.error_analysis import (
    ErrorAnalysis,
    ErrorCategory,
    analyze_error,
    enrich_result,
)


# ---------------------------------------------------------------------------
# ErrorAnalysis frozen dataclass tests
# ---------------------------------------------------------------------------


class TestErrorAnalysis:
    def test_frozen(self):
        analysis = ErrorAnalysis(
            category=ErrorCategory.OOM,
            message="test",
            suggestion="fix it",
            raw_error="raw",
        )
        with pytest.raises(AttributeError):
            analysis.category = ErrorCategory.UNKNOWN  # type: ignore[misc]

    def test_fields(self):
        analysis = ErrorAnalysis(
            category=ErrorCategory.CUDA_ERROR,
            message="msg",
            suggestion="sug",
            raw_error="raw",
        )
        assert analysis.category == ErrorCategory.CUDA_ERROR
        assert analysis.message == "msg"
        assert analysis.suggestion == "sug"
        assert analysis.raw_error == "raw"


# ---------------------------------------------------------------------------
# analyze_error tests
# ---------------------------------------------------------------------------


class TestAnalyzeError:
    def test_success_returns_none(self):
        """exit_code=0 and no error patterns -> None."""
        result = analyze_error("some normal output", exit_code=0)
        assert result is None

    def test_empty_stderr_success_returns_none(self):
        result = analyze_error("", exit_code=0)
        assert result is None

    def test_oom_out_of_memory_error(self):
        stderr = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
        result = analyze_error(stderr, exit_code=1)
        assert result is not None
        assert result.category == ErrorCategory.OOM
        assert "batch_size" in result.suggestion

    def test_oom_resource_exhausted(self):
        stderr = "ResourceExhausted: OOM when allocating tensor"
        result = analyze_error(stderr, exit_code=1)
        assert result is not None
        assert result.category == ErrorCategory.OOM

    def test_cuda_error(self):
        stderr = "RuntimeError: CUDA error: device-side assert triggered"
        result = analyze_error(stderr, exit_code=1)
        assert result is not None
        assert result.category == ErrorCategory.CUDA_ERROR
        assert "CUDA compatibility" in result.suggestion

    def test_cuda_cudnn(self):
        stderr = "cudnn error: CUDNN_STATUS_INTERNAL_ERROR"
        result = analyze_error(stderr, exit_code=1)
        assert result is not None
        assert result.category == ErrorCategory.CUDA_ERROR

    def test_cuda_nccl(self):
        stderr = "NCCL error: unhandled system error"
        result = analyze_error(stderr, exit_code=1)
        assert result is not None
        assert result.category == ErrorCategory.CUDA_ERROR

    def test_import_error_with_package_name(self):
        stderr = "ModuleNotFoundError: No module named 'transformers'"
        result = analyze_error(stderr, exit_code=1)
        assert result is not None
        assert result.category == ErrorCategory.IMPORT_ERROR
        assert "!pip install transformers" in result.suggestion

    def test_import_error_without_package_name(self):
        stderr = "ImportError: cannot import name 'foo' from 'bar'"
        result = analyze_error(stderr, exit_code=1)
        assert result is not None
        assert result.category == ErrorCategory.IMPORT_ERROR
        assert "!pip install <package>" in result.suggestion

    def test_timeout_kernel(self):
        stderr = "Timed out waiting for kernel connection"
        result = analyze_error(stderr, exit_code=1)
        assert result is not None
        assert result.category == ErrorCategory.TIMEOUT
        assert "timeout" in result.suggestion.lower()

    def test_timeout_session(self):
        stderr = "Timed out creating kernel session"
        result = analyze_error(stderr, exit_code=1)
        assert result is not None
        assert result.category == ErrorCategory.TIMEOUT

    def test_quota_exceeded(self):
        stderr = "GPU quota exceeded for this account"
        result = analyze_error(stderr, exit_code=1)
        assert result is not None
        assert result.category == ErrorCategory.QUOTA_EXCEEDED
        assert "retry" in result.suggestion.lower()

    def test_quota_resources_exhausted(self):
        stderr = "ResourcesExhausted: no GPU available"
        result = analyze_error(stderr, exit_code=1)
        assert result is not None
        assert result.category == ErrorCategory.QUOTA_EXCEEDED

    def test_unknown_nonzero_exit(self):
        """Non-zero exit code with no matching pattern -> UNKNOWN."""
        stderr = "some random error that does not match any pattern"
        result = analyze_error(stderr, exit_code=1)
        assert result is not None
        assert result.category == ErrorCategory.UNKNOWN
        assert "non-zero exit code" in result.message

    def test_first_match_wins(self):
        """When stderr contains multiple patterns, first match wins (OOM before CUDA)."""
        stderr = "CUDA out of memory. Also CUDA error occurred."
        result = analyze_error(stderr, exit_code=1)
        assert result is not None
        assert result.category == ErrorCategory.OOM

    def test_raw_error_truncated(self):
        """raw_error should be truncated to 500 chars."""
        long_stderr = "x" * 1000
        result = analyze_error(long_stderr, exit_code=1)
        assert result is not None
        assert len(result.raw_error) == 500

    def test_oom_detected_even_with_exit_code_zero(self):
        """Pattern match takes precedence even when exit_code is 0."""
        stderr = "CUDA out of memory"
        result = analyze_error(stderr, exit_code=0)
        assert result is not None
        assert result.category == ErrorCategory.OOM


# ---------------------------------------------------------------------------
# enrich_result tests
# ---------------------------------------------------------------------------


class TestEnrichResult:
    def test_success_returns_unchanged(self):
        """No error -> original dict returned unchanged."""
        original = {"cells": [], "exit_code": 0, "stderr": ""}
        result = enrich_result(original, stderr="", exit_code=0)
        assert result is original
        assert "error_analysis" not in result

    def test_error_adds_analysis(self):
        original = {"cells": [], "exit_code": 1, "stderr": "CUDA out of memory"}
        result = enrich_result(original, stderr="CUDA out of memory", exit_code=1)
        assert "error_analysis" in result
        assert result["error_analysis"]["category"] == "oom"
        assert "message" in result["error_analysis"]
        assert "suggestion" in result["error_analysis"]

    def test_immutability(self):
        """enrich_result must NOT mutate the input dict."""
        original = {"cells": [], "exit_code": 1, "stderr": "OOM error"}
        original_copy = dict(original)
        result = enrich_result(original, stderr="CUDA out of memory", exit_code=1)
        # Original dict must be unchanged
        assert original == original_copy
        assert "error_analysis" not in original
        # Result is a new dict
        assert result is not original
        assert "error_analysis" in result

    def test_preserves_existing_keys(self):
        original = {"cells": [{"cell_num": 0}], "exit_code": 1, "custom_key": "value"}
        result = enrich_result(original, stderr="CUDA out of memory", exit_code=1)
        assert result["cells"] == [{"cell_num": 0}]
        assert result["custom_key"] == "value"
        assert result["exit_code"] == 1
        assert "error_analysis" in result
