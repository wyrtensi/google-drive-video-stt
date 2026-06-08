Docker Cleanup: config init honors DATA_DIR + smoke script robustness

## Overview

Follow-up to `docs/plans/2026-06-08-docker-config-owned-deployment.md`. The core
Docker fixes (prompts as `src` package data, `DATA_DIR=/app/data` in the image)
are done and verified. This plan closes two remaining minor inconsistencies found
during verification. Deployment/packaging only - no Python feature behavior change.

Verified problems:

1. `gdstt config init` ignores `DATA_DIR`. With no `--config`/`GDSTT_CONFIG` and no
   `--local`, `init_config` (`src/config.py`) targets `_user_config_path()`, while
   the runtime resolver `resolve_config_file_path()` honors `DATA_DIR` (and the
   container bakes `DATA_DIR=/app/data`). So inside the container the default loop
   and `doctor` correctly auto-migrate into `/app/data`, but an operator who runs
   `gdstt config init` writes to `/root/.config/gdstt/config.yml` - outside the
   mounted volume and mismatched with where the runtime reads. Confirmed: with
   `DATA_DIR` set, `config init` wrote to the user path while `config path`
   resolved to `DATA_DIR/config.yml`.

2. `scripts/docker-smoke.sh` captures only stdout. `output="$(docker run ...)"`
   drops stderr, and under `set -e` a non-zero `gdstt doctor` exit aborts the
   script before the captured output is echoed, so a failing smoke run prints no
   diagnostic. The exit code still propagates (it fails correctly), but it is hard
   to debug.

## Context

- Files involved: `src/config.py` (`init_config`, `_resolve_config_file_path` /
  `resolve_config_file_path`, `_user_config_path`, `_local_config_path`),
  `scripts/docker-smoke.sh`, `tests/test_config.py` (and `tests/test_cli.py` if a
  CLI dispatch test is affected).
- Current `init_config` target selection: `config_path` arg wins; `--local`
  -> `./data/config.yml`; else `GDSTT_CONFIG` or `_user_config_path()`. It does NOT
  consult `DATA_DIR`.
- The runtime resolver `resolve_config_file_path()` priority is already
  `--config > GDSTT_CONFIG > DATA_DIR/config.yml (only when DATA_DIR set) >
  _user_config_path()`. The fix is to make `init_config`'s default target match
  that resolver so init writes exactly where the runtime reads.
- Out of scope: root-owned bind-mount files (no `USER` in Dockerfile) and the
  pre-existing `write_text`/`chmod(0600)` TOCTOU - both noted but intentionally not
  changed here to avoid uid-mapping and unrelated-scope risk.

## Development Approach

- Testing approach: Regular (code first, then tests), one test file per module, all
  external services mocked, no network.
- Complete each task fully before the next.
- CRITICAL: every task MUST include new/updated tests that fail before the fix and
  pass after.
- CRITICAL: `uv run pytest` and `uv run ruff check` must both be green before
  starting the next task.
- CRITICAL: do not change Python feature behavior; `--local` and explicit
  `--config`/`GDSTT_CONFIG` targeting must keep working exactly as before. Only the
  bare default (no flags) gains `DATA_DIR` awareness.

## Implementation Steps

### Task 1: `config init` default target honors DATA_DIR

**Files:**
- Modify: `src/config.py`, `tests/test_config.py`.

- [x] In `src/config.py::init_config`, change the default branch (no `config_path`,
      not `local`) so the target matches the runtime resolver: use
      `resolve_config_file_path()` (which already applies
      `GDSTT_CONFIG > DATA_DIR/config.yml when DATA_DIR set > _user_config_path()`)
      instead of `GDSTT_CONFIG or _user_config_path()`. Keep `--config` and
      `--local` behavior unchanged.
- [x] Add a test in `tests/test_config.py`: with `DATA_DIR` set (and no
      `--config`/`GDSTT_CONFIG`/`--local`), `init_config()` writes
      `config.yml` (and copies `prompts/`) under `DATA_DIR`, not the user path;
      and without `DATA_DIR` it still writes to the mocked user path. Confirm
      `--local` still targets `./data/config.yml` and an explicit `config_path`
      still wins. Mock `_user_config_path`; do not touch the real home dir.
- [x] Run `uv run pytest` and `uv run ruff check` - must pass before next task.

### Task 2: Make the Docker smoke script print diagnostics on failure

**Files:**
- Modify: `scripts/docker-smoke.sh`.

- [ ] Capture both stdout and stderr from the in-container `gdstt doctor`
      (`output="$(docker run ... 2>&1)"`) and `echo "$output"` BEFORE the
      `config:` path assertion, so a failing run shows the container's output for
      debugging. Keep `set -euo pipefail` and the existing pass/fail semantics
      (the script must still exit non-zero when the config path is wrong or the
      packaged prompt is unreachable). This task is a shell-script change; verify by
      `bash -n scripts/docker-smoke.sh` (syntax) and a manual read - it needs no
      pytest, but still run the suite to confirm nothing else regressed.
- [ ] Run `uv run pytest` and `uv run ruff check` - must pass.

## Verification

- `uv run pytest` fully green; `uv run ruff check` clean.
- With `DATA_DIR` set, `gdstt config init` writes into `DATA_DIR` (matching where
  the runtime/`doctor` read); without it, the per-user path is still used; `--local`
  and explicit `--config` unchanged.
- `bash -n scripts/docker-smoke.sh` passes and the script echoes container output
  before asserting.
- No Python feature behavior changed; all prior tests still pass.
