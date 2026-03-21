# Privacy Policy — mcp-colab-gpu

**Last updated:** 2026-03-21

## Overview

mcp-colab-gpu is an open-source MCP (Model Context Protocol) server that
enables AI coding assistants to execute Python code on Google Colab
GPU/TPU runtimes. This privacy policy explains how the application
handles user data when accessing Google services.

## Data We Access

### Google Colab

- The application allocates and releases Colab runtimes on your behalf.
- Python code you provide is sent to the Colab runtime for execution.
- Execution output (stdout, stderr, exit codes) is returned to your
  local AI coding assistant.

### Google Drive (Optional)

- The application requests the **`drive.file`** scope, which provides
  access **only to files created or opened by this application**.
- The application **cannot** access your existing Google Drive files,
  photos, documents, or any other data not created by mcp-colab-gpu.
- Drive access is used solely to transfer files between your local
  machine and Colab runtimes (upload input data, download results).

## Data Storage

- **No server-side storage.** mcp-colab-gpu runs entirely on your local
  machine. No data is sent to any third-party server operated by the
  developers.
- **OAuth tokens** are cached locally at `~/.config/colab-exec/` with
  restrictive file permissions (`0600`). Tokens are short-lived and
  automatically refreshed.
- **No telemetry or analytics.** The application does not collect usage
  data, crash reports, or any other telemetry.

## Data Sharing

- The developers do not receive, store, or broker your code, files, or
  execution results.
- When you use Colab or Drive features, that data is transmitted directly
  from your local machine to Google's APIs over encrypted connections
  (HTTPS/WSS). Your use of Google services is governed by
  [Google's Privacy Policy](https://policies.google.com/privacy).

## Data Retention

- The application does not retain any user data beyond the local token
  cache.
- You can revoke access at any time by removing the cached tokens at
  `~/.config/colab-exec/` or by revoking the application's access in
  your [Google Account permissions](https://myaccount.google.com/permissions).

## Security

- All communication with Google APIs uses HTTPS/WSS encryption.
- OAuth tokens are stored with owner-only file permissions.
- The application uses the minimal required OAuth scope (`drive.file`)
  to limit access.

## Open Source

mcp-colab-gpu is open-source software licensed under the MIT License.
The source code is publicly available at:
[https://github.com/mio-github/mcp-colab-gpu](https://github.com/mio-github/mcp-colab-gpu)

## Contact

For questions or concerns about this privacy policy, please open an
issue on the GitHub repository:
[https://github.com/mio-github/mcp-colab-gpu/issues](https://github.com/mio-github/mcp-colab-gpu/issues)

## Changes

We may update this privacy policy from time to time. Changes will be
reflected in the "Last updated" date above and committed to the
repository.
