"""Tests for colab_runtime credential handling (MCP-COLAB-HARDEN)."""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import requests
from google.oauth2.credentials import Credentials

from mcp_colab_gpu import colab_runtime as cr_mod
from mcp_colab_gpu.colab_runtime import get_credentials


class TestRefreshTokenHardening:
    """refresh_token must never be written to token.json on disk."""

    def test_save_strips_refresh_token_from_disk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cr_mod, "TOKEN_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(cr_mod, "TOKEN_CACHE_PATH", str(tmp_path / "token.json"))

        creds = MagicMock()
        creds.to_json.return_value = json.dumps({
            "token": "access-tok",
            "refresh_token": "real-refresh-tok",
            "client_id": "cid",
            "client_secret": "csecret",
            "scopes": cr_mod.SCOPES,
        })

        cr_mod._save_credentials(creds)

        with open(tmp_path / "token.json") as f:
            saved = json.load(f)

        assert saved["refresh_token"] is None
        assert saved["token"] == "access-tok"

    def test_cached_refresh_token_injected_when_disk_copy_lacks_one(self, monkeypatch):
        """token.json with refresh_token=null still allows a silent
        refresh within the same process via _cached_refresh_token."""
        monkeypatch.setattr(cr_mod, "_cached_refresh_token", "cached-refresh-tok")

        disk_creds = Credentials(
            token="stale-access-tok",
            refresh_token=None,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=cr_mod.CLIENT_CONFIG["installed"]["client_id"],
            client_secret=cr_mod.CLIENT_CONFIG["installed"]["client_secret"],
            scopes=cr_mod.SCOPES,
        )
        disk_creds.expiry = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)

        refreshed = {"called": False}

        def fake_refresh(self, request):
            refreshed["called"] = True
            self.token = "new-access-tok"
            self.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)

        with (
            patch("mcp_colab_gpu.colab_runtime.os.path.exists", return_value=True),
            patch(
                "mcp_colab_gpu.colab_runtime.Credentials.from_authorized_user_file",
                return_value=disk_creds,
            ),
            patch.object(Credentials, "refresh", fake_refresh),
            patch("mcp_colab_gpu.colab_runtime._save_credentials"),
        ):
            result = get_credentials()

        assert refreshed["called"] is True
        assert result.token == "new-access-tok"
        assert result.refresh_token == "cached-refresh-tok"

    def test_full_oauth_flow_caches_refresh_token_and_strips_from_disk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cr_mod, "TOKEN_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(cr_mod, "TOKEN_CACHE_PATH", str(tmp_path / "token.json"))
        monkeypatch.setattr(cr_mod, "_cached_refresh_token", None)

        new_creds = MagicMock()
        new_creds.refresh_token = "fresh-refresh-tok"
        new_creds.to_json.return_value = json.dumps({
            "token": "fresh-access-tok",
            "refresh_token": "fresh-refresh-tok",
            "client_id": cr_mod.CLIENT_CONFIG["installed"]["client_id"],
            "client_secret": cr_mod.CLIENT_CONFIG["installed"]["client_secret"],
            "scopes": cr_mod.SCOPES,
        })

        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = new_creds

        with (
            patch("mcp_colab_gpu.colab_runtime.os.path.exists", return_value=False),
            patch(
                "mcp_colab_gpu.colab_runtime.InstalledAppFlow.from_client_config",
                return_value=mock_flow,
            ),
        ):
            result = get_credentials()

        assert result is new_creds
        assert cr_mod._cached_refresh_token == "fresh-refresh-tok"

        with open(tmp_path / "token.json") as f:
            saved = json.load(f)

        assert saved["refresh_token"] is None
        assert saved["token"] == "fresh-access-tok"


class TestAuthModeOnline:
    """MCP-COLAB-HARDEN: MCP_COLAB_AUTH_MODE="online" alternative.

    No refresh token is ever issued by Google, so token.json has
    refresh_token=null naturally (vs. "hybrid" mode, which strips a
    refresh token that *was* issued). Trade-off: re-consent is required
    after the ~1h access token expires, since there is no refresh token
    to fall back on.
    """

    def test_default_auth_mode_is_hybrid(self):
        assert cr_mod.AUTH_MODE == "hybrid"

    def test_online_mode_requests_online_access_type(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cr_mod, "AUTH_MODE", "online")
        monkeypatch.setattr(cr_mod, "TOKEN_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(cr_mod, "TOKEN_CACHE_PATH", str(tmp_path / "token.json"))
        monkeypatch.setattr(cr_mod, "_cached_refresh_token", None)

        new_creds = MagicMock()
        new_creds.refresh_token = None
        new_creds.to_json.return_value = json.dumps({
            "token": "online-access-tok",
            "refresh_token": None,
            "client_id": cr_mod.CLIENT_CONFIG["installed"]["client_id"],
            "client_secret": cr_mod.CLIENT_CONFIG["installed"]["client_secret"],
            "scopes": cr_mod.SCOPES,
        })

        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = new_creds

        with (
            patch("mcp_colab_gpu.colab_runtime.os.path.exists", return_value=False),
            patch(
                "mcp_colab_gpu.colab_runtime.InstalledAppFlow.from_client_config",
                return_value=mock_flow,
            ),
        ):
            result = get_credentials()

        assert result is new_creds
        mock_flow.run_local_server.assert_called_once()
        assert mock_flow.run_local_server.call_args.kwargs["access_type"] == "online"

        # No refresh token was issued, so the in-memory cache stays empty
        # and the on-disk copy has refresh_token=null without any stripping.
        assert cr_mod._cached_refresh_token is None
        with open(tmp_path / "token.json") as f:
            saved = json.load(f)
        assert saved["refresh_token"] is None

    def test_hybrid_mode_requests_offline_access_type(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cr_mod, "AUTH_MODE", "hybrid")
        monkeypatch.setattr(cr_mod, "TOKEN_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(cr_mod, "TOKEN_CACHE_PATH", str(tmp_path / "token.json"))
        monkeypatch.setattr(cr_mod, "_cached_refresh_token", None)

        new_creds = MagicMock()
        new_creds.refresh_token = "fresh-refresh-tok"
        new_creds.to_json.return_value = json.dumps({
            "token": "hybrid-access-tok",
            "refresh_token": "fresh-refresh-tok",
            "client_id": cr_mod.CLIENT_CONFIG["installed"]["client_id"],
            "client_secret": cr_mod.CLIENT_CONFIG["installed"]["client_secret"],
            "scopes": cr_mod.SCOPES,
        })

        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = new_creds

        with (
            patch("mcp_colab_gpu.colab_runtime.os.path.exists", return_value=False),
            patch(
                "mcp_colab_gpu.colab_runtime.InstalledAppFlow.from_client_config",
                return_value=mock_flow,
            ),
        ):
            get_credentials()

        assert mock_flow.run_local_server.call_args.kwargs["access_type"] == "offline"

    def test_invalid_auth_mode_raises_on_import(self):
        env = dict(os.environ)
        env["MCP_COLAB_AUTH_MODE"] = "bogus"
        result = subprocess.run(
            [sys.executable, "-c", "import mcp_colab_gpu.colab_runtime"],
            capture_output=True, text=True, env=env,
        )
        assert result.returncode != 0
        assert "Invalid MCP_COLAB_AUTH_MODE" in result.stderr


class TestAllocateRuntimeAssignRetry:
    """allocate_runtime retries the /tun/m/assign POST on transient 5xx.

    Observed against real Colab: a cold allocation can return 503 after
    ~56s before a retry succeeds in a few seconds. ASSIGN_MAX_ATTEMPTS /
    ASSIGN_RETRY_DELAY / ASSIGN_POST_TIMEOUT bound this behavior.
    """

    @staticmethod
    def _xssi(payload: dict) -> str:
        return ")]}'\n" + json.dumps(payload)

    def _get_response(self):
        resp = MagicMock()
        resp.text = self._xssi({"token": "xsrf-token-abc"})
        resp.raise_for_status = MagicMock()
        return resp

    def _assignment_response(self):
        resp = MagicMock()
        resp.text = self._xssi({
            "endpoint": "gpu-t4-xyz",
            "runtimeProxyInfo": {"url": "https://proxy.example/", "token": "proxy-tok"},
        })
        resp.raise_for_status = MagicMock()
        return resp

    def _error_response(self, status_code: int):
        resp = MagicMock()
        resp.status_code = status_code
        resp.raise_for_status = MagicMock(
            side_effect=requests.exceptions.HTTPError(response=resp)
        )
        return resp

    def test_retries_on_503_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(cr_mod, "ASSIGN_RETRY_DELAY", 0)
        post_responses = [self._error_response(503), self._assignment_response()]

        with (
            patch("mcp_colab_gpu.colab_runtime.requests.get", return_value=self._get_response()),
            patch("mcp_colab_gpu.colab_runtime.requests.post", side_effect=post_responses) as mock_post,
            patch("mcp_colab_gpu.colab_runtime.time.sleep") as mock_sleep,
        ):
            result = cr_mod.allocate_runtime("tok", accelerator="T4")

        assert result["endpoint"] == "gpu-t4-xyz"
        assert result["proxy_url"] == "https://proxy.example"
        assert result["proxy_token"] == "proxy-tok"
        assert result["reused"] is False
        assert mock_post.call_count == 2
        assert mock_post.call_args_list[0].kwargs["timeout"] == cr_mod.ASSIGN_POST_TIMEOUT
        mock_sleep.assert_called_once()

    def test_raises_after_max_attempts_on_persistent_503(self, monkeypatch):
        monkeypatch.setattr(cr_mod, "ASSIGN_RETRY_DELAY", 0)
        post_responses = [self._error_response(503) for _ in range(cr_mod.ASSIGN_MAX_ATTEMPTS)]

        with (
            patch("mcp_colab_gpu.colab_runtime.requests.get", return_value=self._get_response()),
            patch("mcp_colab_gpu.colab_runtime.requests.post", side_effect=post_responses) as mock_post,
            patch("mcp_colab_gpu.colab_runtime.time.sleep"),
        ):
            try:
                cr_mod.allocate_runtime("tok", accelerator="T4")
                raise AssertionError("expected HTTPError")
            except requests.exceptions.HTTPError:
                pass

        assert mock_post.call_count == cr_mod.ASSIGN_MAX_ATTEMPTS

    def test_non_retryable_status_raises_immediately(self, monkeypatch):
        monkeypatch.setattr(cr_mod, "ASSIGN_RETRY_DELAY", 0)
        resp_401 = self._error_response(401)

        with (
            patch("mcp_colab_gpu.colab_runtime.requests.get", return_value=self._get_response()),
            patch("mcp_colab_gpu.colab_runtime.requests.post", return_value=resp_401) as mock_post,
            patch("mcp_colab_gpu.colab_runtime.time.sleep") as mock_sleep,
        ):
            try:
                cr_mod.allocate_runtime("tok", accelerator="T4")
                raise AssertionError("expected HTTPError")
            except requests.exceptions.HTTPError:
                pass

        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()

    def test_read_timeout_then_success(self, monkeypatch):
        monkeypatch.setattr(cr_mod, "ASSIGN_RETRY_DELAY", 0)

        with (
            patch("mcp_colab_gpu.colab_runtime.requests.get", return_value=self._get_response()),
            patch(
                "mcp_colab_gpu.colab_runtime.requests.post",
                side_effect=[requests.exceptions.ReadTimeout("timed out"), self._assignment_response()],
            ) as mock_post,
            patch("mcp_colab_gpu.colab_runtime.time.sleep") as mock_sleep,
        ):
            result = cr_mod.allocate_runtime("tok", accelerator="T4")

        assert result["endpoint"] == "gpu-t4-xyz"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once()
