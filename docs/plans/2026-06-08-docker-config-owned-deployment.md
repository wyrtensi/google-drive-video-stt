Docker Deployment For The Config-Owned Preset DAG

## Overview

The config-owned preset DAG work (branch `openai-preset-dag`) reworked config
discovery, packaged the OpenAI prompts as assets, and moved Google auth into the
config. Two of those changes silently break the Docker deployment. This plan makes
the containerized service correct again without changing any Python feature
behavior - it is purely a deployment + packaging fix.

Verified problems (reproduced against the current code):

1. Config escapes the mounted volume. `docker-compose.yml` mounts
   `./data:/app/data` and runs `python -m src.main` in `/app` with neither
   `DATA_DIR` nor `GDSTT_CONFIG` set. After the discovery rework, with no
   `DATA_DIR`/`GDSTT_CONFIG` the resolver returns the OS user path
   `~/.config/gdstt/config.yml` -> in the container `/root/.config/gdstt/config.yml`,
   NOT `/app/data`. So `.env`->YAML auto-migration, the generated `config.yml`, and
   `credentials.json`/`token.json` are written outside the volume and lost on
   restart; the operator's mounted `./data` is ignored. Confirmed that
   `DATA_DIR=/app/data` restores resolution to `/app/data/config.yml`.

2. Prompt assets are unreachable in the container. `load_packaged_prompt` first
   tries `importlib.resources.files("src.assets.prompts")`, which raises
   `ModuleNotFoundError: No module named 'src.assets'` (it is not an importable
   package), so the loader always falls back to the repo `assets/prompts/`
   directory. The Dockerfile copies `src` and `config` but NOT `assets/`, so in the
   container the loader raises `ValueError: packaged prompt asset 'keypoints.md' is
   missing or empty` and the keypoints / OpenAI preset stage fails. This is also a
   latent packaging bug that affects any non-editable install.

Decisions for this plan:
- Fix the asset bug deeply: relocate the prompts to `src/assets/prompts/` as real
  Python package data so `importlib.resources` works in editable, wheel, and
  `uv tool install` modes, and so the Dockerfile's existing `COPY src ./src`
  ships them automatically. This removes the fragile force-include + repo-fallback.
- Bake `DATA_DIR=/app/data` into the image via `ENV` so the container is correct by
  default, even for a bare `docker run` without compose.

## Context

- Files involved (existing): `src/presets.py` (loader + `INSTRUCTIONS` +
  `PACKAGED_PROMPT_ASSETS`), `src/config.py` (`copy_prompt_assets`,
  `_resolve_config_file_path`, `_user_config_path`), `pyproject.toml`
  (`[tool.hatch.build.targets.wheel]` packages + force-include + sdist include),
  `Dockerfile`, `docker-compose.yml`, `README.md`, `AGENTS.md`,
  `tests/test_presets.py`, `tests/test_config.py`.
- Asset files to move: `assets/prompts/keypoints.md`,
  `assets/prompts/transcript-cleanup.md`, `assets/prompts/action-items.md`
  -> `src/assets/prompts/` (use `git mv`). Remove the now-empty top-level `assets/`.
- Files to create: `src/assets/__init__.py`, `src/assets/prompts/__init__.py`.
- Related code:
  - `src/presets.py`: `_PACKAGED_PROMPTS_PACKAGE = "src.assets.prompts"`,
    `_REPO_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "prompts"`,
    `load_packaged_prompt(name)`, the module-level `INSTRUCTIONS`, and
    `PACKAGED_PROMPT_ASSETS`.
  - `src/config.py`: `copy_prompt_assets(target)` (copies the packaged prompts
    beside a generated config), and the discovery chain
    `--config > GDSTT_CONFIG > DATA_DIR/config.yml (only when DATA_DIR set) >
    _user_config_path()`.
  - `Dockerfile`: `COPY src ./src`, `COPY config ./config`, `uv sync --frozen`,
    `CMD ["python", "-m", "src.main"]`.
- Dependencies: no new runtime deps. `ffmpeg` already installed in the image.
- Out of scope: no changes to STT/preset/auth feature behavior; no compose service
  redesign beyond the env/volume needed for correctness; no new deployment targets.

## Development Approach

- Testing approach: Regular (code first, then tests), one test file per module, all
  external services mocked, no network.
- Complete each task fully before the next.
- CRITICAL: every task MUST include new/updated tests that fail before the fix and
  pass after.
- CRITICAL: `uv run pytest` and `uv run ruff check` must both be green before
  starting the next task.
- CRITICAL: do not change any Python feature behavior; this is deployment/packaging
  only. Keep `STT_PROVIDER=""` MP3-only mode and the default `transcript ->
  keypoints` chain working.

## Implementation Steps

### Task 1: Ship prompt assets as real `src` package data

**Files:**
- Move: `assets/prompts/{keypoints,transcript-cleanup,action-items}.md`
  -> `src/assets/prompts/` (via `git mv`); delete the empty top-level `assets/`.
- Create: `src/assets/__init__.py`, `src/assets/prompts/__init__.py`.
- Modify: `src/presets.py`, `src/config.py`, `pyproject.toml`,
  `tests/test_presets.py`, `tests/test_config.py`.

- [x] `git mv` the three prompt `.md` files from `assets/prompts/` into
      `src/assets/prompts/`, and add empty `__init__.py` to `src/assets/` and
      `src/assets/prompts/` so `src.assets.prompts` is an importable package.
- [x] In `src/presets.py`, simplify `load_packaged_prompt` to read from
      `importlib.resources.files("src.assets.prompts")` (now a real package). Keep a
      single defensive fallback to a `Path(__file__)`-relative
      `src/assets/prompts` dir for source runs, and drop the old top-level
      `assets/prompts` repo fallback. Update `_PACKAGED_PROMPTS_PACKAGE` /
      `_REPO_PROMPTS_DIR` accordingly. `INSTRUCTIONS` and `PACKAGED_PROMPT_ASSETS`
      must keep resolving to the same text. (Renamed `_REPO_PROMPTS_DIR` ->
      `_SRC_PROMPTS_DIR`, now `Path(__file__).parent / "assets" / "prompts"`.)
- [x] In `pyproject.toml`, remove the
      `[tool.hatch.build.targets.wheel.force-include]` mapping (no longer needed)
      and update the sdist `include` to drop `assets/prompts`. Ensure the wheel
      ships `src/assets/prompts/*.md` (hatchling includes package files under
      `src`); if needed add an explicit `artifacts`/`include` entry for `*.md`.
      (Added `artifacts = ["src/assets/prompts/*.md"]` to the wheel target.)
- [x] Update `src/config.py::copy_prompt_assets` (and anything else reading the old
      `assets/prompts` location) to source the packaged prompts from the new
      package location via `load_packaged_prompt` / `importlib.resources`.
      (No change needed: `copy_prompt_assets` already sources via `load_packaged_prompt`.)
- [x] Add/adjust tests: `tests/test_presets.py` asserts `load_packaged_prompt(
      'keypoints.md')` works WITHOUT any top-level `assets/` directory present
      (simulating an installed/container layout, e.g. monkeypatch the source-dir
      fallback to a nonexistent path) and that `importlib.resources` is the path
      that succeeds; keep the `INSTRUCTIONS == keypoints.md` assertion.
- [x] Verify the built wheel includes the assets: `uv build --wheel` then confirm
      `src/assets/prompts/{keypoints,transcript-cleanup,action-items}.md` are inside
      the wheel; note the result in the task. (Confirmed: all three `.md` files plus
      `__init__.py` present in the built wheel.)
- [x] Run `uv run pytest` and `uv run ruff check` - must pass before next task.
      (567 passed, 2 skipped; ruff clean.)

### Task 2: Persist config and data inside the container volume

**Files:**
- Modify: `Dockerfile`, `docker-compose.yml`, `tests/test_config.py`.

- [x] In `Dockerfile`, add `ENV DATA_DIR=/app/data` (alongside the existing `ENV`
      block) so the resolver uses the mounted volume by default. Confirm the
      `WORKDIR /app` + volume mount `./data:/app/data` then make
      `gdstt`/`python -m src.main` read and write `config.yml`,
      `credentials.json`/`token.json` under the mount. (Added `DATA_DIR=/app/data`
      to the existing `ENV` block.)
- [x] In `docker-compose.yml`, document the same `DATA_DIR=/app/data` (either rely
      on the image `ENV` or set it explicitly in `environment:`), keeping the
      `./data:/app/data` volume and `env_file: .env` so `.env`->YAML
      auto-migration lands in the volume on first run. (Set `DATA_DIR: /app/data`
      explicitly in `environment:` with a clarifying comment.)
- [x] Add a test in `tests/test_config.py` that, with `DATA_DIR=/app/data` set and
      no `--config`/`GDSTT_CONFIG`, `resolve_config_file_path()` returns
      `/app/data/config.yml` (container resolution), and that without `DATA_DIR` it
      returns the user path (guarding against a regression of the mounted-volume
      fix). Mock the environment; do not touch the real filesystem.
      (Added `test_resolve_container_data_dir`.)
- [x] Run `uv run pytest` and `uv run ruff check` - must pass before next task.
      (568 passed, 2 skipped; ruff clean.)

### Task 3: Container verification and updated Docker docs

**Files:**
- Create: `scripts/docker-smoke.sh` (or document the steps inline in README).
- Modify: `README.md`, `AGENTS.md`, and `skills/gdstt-cli/SKILL.md` if it
  references Docker; `tests/test_skill_docs.py` only if it asserts new content.

- [ ] Add a documented container smoke check (a short `scripts/docker-smoke.sh` or
      a README snippet): `docker build` the image, then
      `docker run --rm -v "$PWD/data:/app/data" --env-file .env <image> gdstt doctor`
      and confirm the printed `config:` path is under `/app/data` and the preset DAG
      lists `keypoints` (prompts loaded). Note that it is a manual/CI check, not a
      pytest.
- [ ] Update the Docker section of `README.md` (and `AGENTS.md` Commands/Arch notes)
      for the config-owned model: `DATA_DIR=/app/data` persists everything in the
      volume; first run auto-migrates `.env` into `/app/data/config.yml`; prompts
      ship inside the package (no `assets/` copy needed); Google auth is inline-first
      in the config with file mode (`credentials.json`/`token.json` under the volume)
      as the opt-in; the config file is written `0600`.
- [ ] If `tests/test_skill_docs.py` or `SKILL.md` covers Docker/operator commands,
      keep them consistent and within the SKILL.md 400-line limit.
- [ ] Run `uv run pytest` and `uv run ruff check` - must pass before next task.

## Verification

- `uv run pytest` fully green; `uv run ruff check` clean.
- `uv build --wheel` produces a wheel containing `src/assets/prompts/*.md`.
- `docker build .` succeeds and `docker run ... gdstt doctor` reports a config path
  under `/app/data` and a non-empty preset DAG (`keypoints`), proving both the
  volume-persistence fix and the packaged-prompt fix inside the container.
- Existing behavior unchanged: default `transcript -> keypoints`, `STT_PROVIDER=""`
  MP3-only mode, and all current tests still pass.
