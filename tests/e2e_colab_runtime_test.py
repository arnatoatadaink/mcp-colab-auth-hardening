#!/usr/bin/env python3
"""E2E smoke test for the Colab tunnel/runtime flow (Phase 1) against iptestserver.

Exercises allocate_runtime -> create_session -> execute_code -> unassign_runtime
without touching the real Colab/Jupyter APIs, using the iptestserver stub
described in MCP_COLAB_STUB_SPEC.md.

Usage:
    # 1. Start iptestserver (separate terminal):
    cd /mnt/d/Projects/claude_work/iptestserver
    .venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8002

    # 2. Place a stub token at ~/.config/colab-exec/token.json
    #    (see MCP_COLAB_STUB_SPEC.md for the JSON content)

    # 3. Run this test, pointed at the stub server:
    COLAB_API_BASE=http://127.0.0.1:8002 uv run python tests/e2e_colab_runtime_test.py

COLAB_API_BASE is required -- without it this script would hit the real
Colab API, which is not allowed for a dry-run smoke test.
"""

import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def _put_control(base: str, source: str, body: dict) -> None:
    req = urllib.request.Request(
        f"{base}/control/{source}",
        data=json.dumps(body).encode(),
        method="PUT",
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=5)


def test_allocate_create_execute_unassign(cr_mod) -> None:
    """Test 1: full allocate -> session -> execute -> unassign round-trip."""
    _separator("Test 1: allocate_runtime -> create_session -> execute_code -> unassign_runtime")

    creds = cr_mod.get_credentials()
    print(f"  [1a] credentials loaded: token={creds.token!r}, valid={creds.valid}")
    assert creds.valid, "stub credentials should be valid (unexpired, no refresh needed)"

    assignment = cr_mod.allocate_runtime(creds.token, accelerator="T4")
    print(f"  [1b] allocate_runtime: {assignment}")
    assert assignment["endpoint"], "missing endpoint in assignment"
    assert assignment["proxy_url"], "missing proxy_url in assignment"
    assert assignment["proxy_token"], "missing proxy_token in assignment"
    assert assignment["reused"] is True, "iptestserver stub should always report reused=True"

    kernel_id = cr_mod.create_session(assignment["proxy_url"], assignment["proxy_token"])
    print(f"  [1c] create_session: kernel_id={kernel_id}")
    assert kernel_id

    stdout, stderr, exit_code = cr_mod.execute_code(
        assignment["proxy_url"], assignment["proxy_token"], kernel_id,
        code="print('hello from smoke test')",
    )
    print(f"  [1d] execute_code: stdout={stdout!r} stderr={stderr!r} exit_code={exit_code}")
    assert exit_code == 0, f"expected success, got stderr={stderr!r}"
    assert "iptestserver stub output" in stdout

    ok = cr_mod.unassign_runtime(creds.token, assignment["endpoint"])
    print(f"  [1e] unassign_runtime: {ok}")
    assert ok is True

    print("\n  [PASS] Test 1 passed")


def test_execute_error_path(cr_mod, base: str) -> None:
    """Test 2: control plane forces an execute error -> had_error path."""
    _separator("Test 2: execute_code error path via /control/colab")

    creds = cr_mod.get_credentials()
    assignment = cr_mod.allocate_runtime(creds.token, accelerator="T4")
    kernel_id = cr_mod.create_session(assignment["proxy_url"], assignment["proxy_token"])

    try:
        _put_control(base, "colab", {"execute_error": True})
        print("  [2a] /control/colab execute_error=true set")

        stdout, stderr, exit_code = cr_mod.execute_code(
            assignment["proxy_url"], assignment["proxy_token"], kernel_id,
            code="raise RuntimeError('boom')",
        )
        print(f"  [2b] execute_code: stdout={stdout!r} stderr={stderr!r} exit_code={exit_code}")
        assert exit_code == 1
        assert "RuntimeError" in stderr
    finally:
        _put_control(base, "colab", {})
        print("  [2c] /control/colab reset")

    cr_mod.unassign_runtime(creds.token, assignment["endpoint"])
    print("\n  [PASS] Test 2 passed")


def main() -> None:
    base = os.environ.get("COLAB_API_BASE")
    if not base:
        print("ERROR: COLAB_API_BASE is not set.")
        print("Refusing to run against the real Colab API.")
        print("Set COLAB_API_BASE=http://127.0.0.1:8002 (iptestserver) and retry.")
        sys.exit(1)

    print("=" * 60)
    print("  mcp-colab-gpu E2E Colab Runtime (Phase 1) Smoke Test")
    print(f"  COLAB_API_BASE = {base}")
    print("=" * 60)

    from mcp_colab_gpu import colab_runtime as cr_mod

    assert cr_mod.COLAB_API == base, (
        f"colab_runtime.COLAB_API={cr_mod.COLAB_API!r} != COLAB_API_BASE={base!r} "
        "(module was imported before env var was set?)"
    )

    passed = 0
    failed = 0
    tests = [
        ("allocate/create/execute/unassign", lambda: test_allocate_create_execute_unassign(cr_mod)),
        ("execute_code error path", lambda: test_execute_error_path(cr_mod, base)),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    _separator("Results")
    print(f"  Passed: {passed}/{passed + failed}")
    if failed:
        print(f"  Failed: {failed}/{passed + failed}")
        sys.exit(1)
    else:
        print("  All tests passed!")


if __name__ == "__main__":
    main()
