#!/usr/bin/env bash
# One-click environment install for the UNITE NSDI artifact (Wan2.1-T2V-14B only).
#
# Usage (from repo root):
#   bash scripts/install.sh
#   ENV_NAME=unite bash scripts/install.sh
#
# This is a thin wrapper around setup_env.sh so reviewers have a single entrypoint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT}/scripts/setup_env.sh" "$@"
