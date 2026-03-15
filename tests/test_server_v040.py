"""Tests for v0.4.0 server features.

Covers:
- colab_cancel: success, unknown job, completed job
- colab_status: accelerator info, active job, recent executions
- session_info in colab_execute sync response
- error_analysis in colab_execute when execution fails
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from mcp_colab_gpu.background import JobStatus, JobStore


# ---------------------------------------------------------------------------
# colab_cancel tests
# ---------------------------------------------------------------------------


class TestColabCancel:
    @pytest.mark.asyncio
    async def test_cancel_active_job(self):
        from mcp_colab_gpu import server as srv

        srv._job_store = JobStore()
        # Create a job that stays active (we don't run it, just create in store)
        job_id = await srv._job_store.create_if_no_active("T4")
        result_str = await srv.colab_cancel(job_id=job_id)
        result = json.loads(result_str)
        assert result["status"] == "cancelled"
        assert result["job_id"] == job_id

    @pytest.mark.asyncio
    async def test_cancel_unknown_job(self):
        from mcp_colab_gpu import server as srv

        srv._job_store = JobStore()
        result_str = await srv.colab_cancel(job_id="nonexistent")
        result = json.loads(result_str)
        assert "error" in result
        assert "Unknown" in result["error"]

    @pytest.mark.asyncio
    async def test_cancel_completed_job(self):
        from mcp_colab_gpu import server as srv

        srv._job_store = JobStore()
        job_id = await srv._job_store.create_if_no_active("T4")
        await srv._job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        result_str = await srv.colab_cancel(job_id=job_id)
        result = json.loads(result_str)
        assert "error" in result
        assert "Cannot cancel" in result["error"]
        assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# colab_status tests
# ---------------------------------------------------------------------------


class TestColabStatus:
    @pytest.mark.asyncio
    async def test_status_has_accelerator_info(self):
        from mcp_colab_gpu import server as srv

        srv._job_store = JobStore()
        result_str = await srv.colab_status()
        result = json.loads(result_str)
        assert "supported_accelerators" in result
        assert "T4" in result["supported_accelerators"]
        assert "A100" in result["supported_accelerators"]
        assert result["supported_accelerators"]["T4"]["vram"] == "16 GB"

    @pytest.mark.asyncio
    async def test_status_no_active_job(self):
        from mcp_colab_gpu import server as srv

        srv._job_store = JobStore()
        result_str = await srv.colab_status()
        result = json.loads(result_str)
        assert result["active_job"] is None
        assert result["recent_executions"] == []

    @pytest.mark.asyncio
    async def test_status_with_active_job(self):
        from mcp_colab_gpu import server as srv

        srv._job_store = JobStore()
        job_id = await srv._job_store.create_if_no_active("A100")
        result_str = await srv.colab_status()
        result = json.loads(result_str)
        assert result["active_job"] is not None
        assert result["active_job"]["job_id"] == job_id
        assert result["active_job"]["accelerator"] == "A100"
        assert result["active_job"]["status"] == "starting"

    @pytest.mark.asyncio
    async def test_status_recent_executions(self):
        from mcp_colab_gpu import server as srv

        srv._job_store = JobStore()
        job_id = await srv._job_store.create_if_no_active("T4")
        now = datetime.now(timezone.utc).isoformat()
        await srv._job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=now,
        )
        result_str = await srv.colab_status()
        result = json.loads(result_str)
        assert len(result["recent_executions"]) == 1
        assert result["recent_executions"][0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_status_shows_cancelled_in_recent(self):
        from mcp_colab_gpu import server as srv

        srv._job_store = JobStore()
        job_id = await srv._job_store.create_if_no_active("T4")
        await srv._job_store.cancel(job_id)
        result_str = await srv.colab_status()
        result = json.loads(result_str)
        assert len(result["recent_executions"]) == 1
        assert result["recent_executions"][0]["status"] == "cancelled"


# ---------------------------------------------------------------------------
# session_info and error_analysis in colab_execute
# ---------------------------------------------------------------------------


class TestColabExecuteV040:
    @pytest.mark.asyncio
    async def test_session_info_in_sync_response(self):
        """Sync colab_execute should include session_info with elapsed and accelerator."""
        with patch("mcp_colab_gpu.server._run_on_colab") as mock_run:
            mock_run.return_value = (
                '===CELL_START_0===\nhello\n===CELL_END_0===',
                "",
                0,
                1.23,
            )
            from mcp_colab_gpu import server as srv

            srv._job_store = JobStore()
            result_str = await srv.colab_execute(
                code="print('hello')",
                accelerator="T4",
            )
            result = json.loads(result_str)
            assert "session_info" in result
            assert result["session_info"]["elapsed_seconds"] == 1.23
            assert result["session_info"]["accelerator"] == "T4"

    @pytest.mark.asyncio
    async def test_error_analysis_on_failure(self):
        """When execution fails with OOM, error_analysis should be present."""
        with patch("mcp_colab_gpu.server._run_on_colab") as mock_run:
            mock_run.return_value = (
                '===CELL_START_0===\n===CELL_END_0===',
                "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
                1,
                5.0,
            )
            from mcp_colab_gpu import server as srv

            srv._job_store = JobStore()
            result_str = await srv.colab_execute(
                code="import torch; x = torch.randn(100000, 100000)",
                accelerator="T4",
            )
            result = json.loads(result_str)
            assert "error_analysis" in result
            assert result["error_analysis"]["category"] == "oom"
            assert "batch_size" in result["error_analysis"]["suggestion"]

    @pytest.mark.asyncio
    async def test_no_error_analysis_on_success(self):
        """Successful execution should NOT have error_analysis."""
        with patch("mcp_colab_gpu.server._run_on_colab") as mock_run:
            mock_run.return_value = (
                '===CELL_START_0===\nok\n===CELL_END_0===',
                "",
                0,
                0.5,
            )
            from mcp_colab_gpu import server as srv

            srv._job_store = JobStore()
            result_str = await srv.colab_execute(
                code="print('ok')",
                accelerator="T4",
            )
            result = json.loads(result_str)
            assert "error_analysis" not in result


# ---------------------------------------------------------------------------
# colab_poll with CANCELLED status
# ---------------------------------------------------------------------------


class TestColabPollCancelled:
    @pytest.mark.asyncio
    async def test_poll_cancelled_job(self):
        from mcp_colab_gpu import server as srv

        srv._job_store = JobStore()
        job_id = await srv._job_store.create_if_no_active("T4")
        await srv._job_store.cancel(job_id)
        result_str = await srv.colab_poll(job_id=job_id)
        result = json.loads(result_str)
        assert result["status"] == "cancelled"
        assert "error" in result
        assert "cancelled" in result["error"].lower()
