# AGENTS.md

Portable repo instructions for google-drive-video-stt. Prefer universal
Markdown and shared repo docs over editor-specific overlays.

## Source of truth layering

- `README.md` - human quickstart and deployment overview.
- `AGENTS.md` - canonical shared repo contract, commands, architecture, and conventions.
- `skills/gdstt-cli/SKILL.md` - canonical installable operator skill (single file).
- `CLAUDE.md` - thin compatibility shim that points back to `AGENTS.md`.

## Project snapshot

This repository polls Google Drive folders for MP4 files, extracts audio when
needed, transcribes with Deepgram, optionally post-processes the transcript and
runs a config-defined DAG of OpenAI presets (each preset writes its own sibling
artifact, e.g. Keypoints), and writes the results back to Drive or to a local
folder.

The code is configured through `data/config.yml` (loaded by `src/config.py`); the
main runtime lives in `src/main.py`, preset definitions in `src/presets.py`, the
DAG executor in `src/preset_pipeline.py`, and STT provider dispatch in
`src/stt/__init__.py`. `.env` is no longer a runtime source: on first run an
existing `.env`/environment is auto-migrated into `data/config.yml` and every
subsequent run reads only the YAML.

## Commands

```bash
uv tool install --editable .   # install the global gdstt command from this checkout
uv tool update-shell           # refresh PATH helpers for uv-installed tools
uv sync --extra dev          # install deps incl. pytest/ruff (use .venv)
uv run pytest                # run all tests
uv run pytest tests/test_config.py::test_name   # single test
uv run ruff check            # lint (line-length 100, target py311)
gdstt auth [--manual]        # OAuth refresh or recovery flow
uv run python -m src.auth    # module entry for the same OAuth flow
uv run python -m src.main    # run the polling loop locally
gdstt <auth|doctor|latest|run|run-once|process|transcribe|relabel|speakers|list|config>  # operator CLI (src/cli.py)
gdstt config migrate [--force]  # (re)write data/config.yml from the current .env/environment
gdstt --config PATH <command>   # point at a non-default config.yml (or set GDSTT_CONFIG)
docker compose up -d --build # containerized deployment (mounts ./data, DATA_DIR=/app/data)
scripts/docker-smoke.sh        # manual/CI: build + `gdstt doctor`, assert config under /app/data and load keypoints.md in-container
```

`ffmpeg` must be on PATH for local runs (bundled in the Docker image).

## Deployment (Docker)

The deployment is config-owned and persists everything in the mounted volume:

- The image bakes `ENV DATA_DIR=/app/data` and Compose mounts `./data:/app/data`,
  so the config resolver writes `config.yml` and the first-run `.env`->YAML
  migration under `./data`; any file-mode `credentials.json`/`token.json` resolve
  under the volume too. Without `DATA_DIR` the resolver would fall back to the OS
  user path (`~/.config/gdstt/...`) and escape the volume — keep `DATA_DIR=/app/data`
  set. `init_config()`'s default (no `--config`/`--local`) target follows the same
  bootstrap priority as the runtime resolver (`GDSTT_CONFIG` > `<DATA_DIR>/config.yml`
  > user path) so `gdstt config init` writes exactly where the runtime reads. Keep
  the two unified — do not reintroduce a separate `GDSTT_CONFIG or _user_config_path()`
  branch in `init`. Note it uses the non-pointer-following `_resolve_config_file_path()`
  (init creates a fresh file and must not dereference an existing forwarding pointer).
- Prompt assets ship inside the `src` package (`src/assets/prompts/*.md`), so the
  Dockerfile's `COPY src ./src` carries them; no separate `assets/` copy and no repo
  fallback. This is also why a wheel / `uv tool install` finds the prompts.
- Google auth is inline-first in `config.yml` (`google.credentials`/`google.token`),
  with file mode as the opt-in and a legacy `data/credentials.json`/`token.json`
  fallback. The generated `config.yml` is written `0600`.
- `scripts/docker-smoke.sh` is the manual/CI verification: it builds the image and
  runs `gdstt doctor` (asserting the `config:` path is under `/app/data`) and loads
  the packaged `keypoints` prompt inside the container (proving both the volume and
  packaged-prompt fixes).

## Architecture

Headless service: polls Google Drive folders, extracts audio from new MP4s via
ffmpeg, transcribes to a `.txt`, and optionally runs the enabled OpenAI presets to
write per-preset sibling artifacts (e.g. `.keypoints.md`). All flow is driven by
`Config` (`src/config.py`, frozen dataclass built by `load_config()` which reads
`data/config.yml`, validates provider-specific required values, and raises on
misconfiguration). `load_config()` resolves the config path from `--config PATH`,
the `GDSTT_CONFIG` env var, or `<data_dir>/config.yml`, and auto-migrates a
`.env`/environment into YAML when the file is missing or empty.

**Polling loop** (`src/main.py`): `main()` builds the Drive service once, then loops
`run_once()` every `POLL_INTERVAL` seconds. Per file, `process_item` computes two
independent needs — `needs_mp3` (no sibling `.mp3`) and `needs_txt` (STT enabled and no
sibling `.txt`) — so a file already having an MP3 can still get transcribed on a later
cycle. Idempotency comes from `drive.list_folder_state` reporting sibling presence by
basename. All work happens in a per-item `TemporaryDirectory`.

`process_target` (`src/main.py`) is the on-demand entry the CLI's `process` command uses:
it auto-detects file vs folder by `mimeType` (override with `is_folder`), then runs the same
`process_item` over a single file or every pending file in a folder. The `gdstt` CLI
(`src/cli.py`) wraps the same `load_config()`/Drive/STT layers behind argparse subcommands
without duplicating business logic.

**`latest` command** (`src/cli.py` + `drive.find_newest_mp4`): resolves a folder
(arg or first of `FOLDER_IDS`), finds the newest mp4 by `createdTime desc`, and
dispatches it through `process_target` (honoring `--dry-run`).

**Post-processing** runs in `process_item` after `transcribe_file` and before the
artifact is written, gated by `stt_postprocess` (local path, `src/postprocess.py`):
it cleans whitespace, parses interlocutor names from the file name, maps them onto
diarized `Speaker N` labels, and merges spurious extra speakers.

**Preset DAG** (`src/presets.py` + `src/preset_pipeline.py`): after the transcript
is written, `process_item` runs the enabled presets that are still missing an
artifact. Each preset is one OpenAI pass (`src/openai_pipeline.py`, sync or the
Batch API per preset) whose input is its dependency outputs concatenated with a
labeled separator, or the raw transcript when it has no dependencies. Independent
presets run in parallel (a `ThreadPoolExecutor` capped at `openai.max_parallel`),
and each non-empty output is written as `<base><artifact_suffix>` tagged
`artifact_type=<preset-name>`. Built-in presets ship in `BUILTIN_PRESETS` (at least
`keypoints`, producing `## Задачи` / `## Тезисы` / `## Открытые вопросы`, plain
text); `config.yml` presets override built-ins field-by-field, add new presets, and
disable a built-in with `enabled: false`. `validate_dag()` rejects unknown/disabled
dependencies and cycles. If a preset fails, its dependents are skipped while
independent branches still persist, then an aggregated error makes the file retry
and re-run only the still-missing presets on a later cycle.

**Output layer** (`src/output.py`): `write_artifact` writes each artifact either as
a Drive sibling (`OUTPUT_TARGET=drive` — upload, or update an existing sibling in
place via `drive.update_file`, its id flowing through `list_folder_state`) or into a
local `OUTPUT_DIR` (`OUTPUT_TARGET=folder`).

For deterministic, agent-driven speaker correction, `src/relabel_transcript.py`
(and the `gdstt relabel` command) rewrite `Speaker N` labels from a `MAP.json`
while preserving each utterance's words (whitespace is normalized).

**Error handling is tiered**: `RefreshError`/`AuthError` propagate up to `main()` and
cause `SystemExit(1)` so the container restarts (after re-running `src.auth`); all other
exceptions are logged + sent to Telegram via `notify.notify_error` and the loop continues.

`process_item()` also emits one process summary per worked file with provider,
`processing_mode`, outcome, retry count, and duration. `run-once()` emits one folder summary per
folder and one cycle summary with provider, overall outcome, folder count,
pending count, processed count, failed count, `retry_total`,
skipped-by-size count, folder-error count, dry-run flag, and duration.

**STT layer** (`src/stt/`): `get_provider(config)` dispatches on `STT_PROVIDER`,
which is Deepgram-only (`""`/`disabled` skips transcription). The base
`STTProvider` exposes `transcribe_full()`; Deepgram does whole-file transcription
and overrides it. `transcribe_file()` (`transcribe.py`) calls `transcribe_full` and
logs the best-effort Deepgram cost. Deepgram sends a full-file audio copy
(`m4a_copy`, `mp3_96k`, or `mp3_192k`); tuning stays provider-specific, but the CLI
workflow stays stable.

**Auth** (`src/auth.py`): single OAuth user credential covers Drive (scope `drive`)
— no service account. `gdstt auth` runs the OAuth flow for
initial setup, refresh, recovery, or headless/manual exchange.
`load_credentials` inspects the saved token's
`scopes` directly (because `from_authorized_user_file` echoes the requested scopes, not the
granted ones); a missing scope raises `AuthError` telling you to re-auth. Adding a scope to
`SCOPES` requires deleting `data/token.json` and re-running auth.

## Core invariants

- `stt.provider` is Deepgram-only. When it is absent, `load_config()` treats it as
  `deepgram`; set `stt.provider: disabled` (or empty) to skip transcription and only
  manage MP3 artifacts.
- Bootstrap and Drive-only commands use `load_config(validate_providers=False)`:
  `auth`, `doctor`, `list` / `status`, `speakers set`, and `config migrate`.
- Processing commands validate provider configuration and can spend credits:
  `run`, `run-once`, `process`, `latest`, and `transcribe`.
- `relabel` is a local file transform that touches no Drive and spends nothing.
- Configuration is `data/config.yml` (grouped `output`, `stt.deepgram`, `openai`,
  and a top-level `presets` map). `.env` is auto-migrated into YAML on first run;
  `gdstt config migrate [--force]` regenerates it explicitly. Resolve a non-default
  file with `gdstt --config PATH ...` or the `GDSTT_CONFIG` env var.
- Enabled presets run after the transcript is produced and require `openai.api_key`;
  each writes a `<base><artifact_suffix>` document tagged `artifact_type=<name>`.
  Having no enabled presets replaces the old `OPENAI_KEYPOINTS=false` gate.
- `output.target` selects where artifacts land: `drive` (siblings) or `folder`
  (`output.dir`, required when `folder`).
- Idempotency relies on `appProperties.source_video_id`, the per-artifact
  `appProperties.artifact_type` (so each preset's sibling is detected
  independently via `list_folder_state`'s `artifact_ids`), and sibling stem
  matching as a fallback. Existing `.keypoints.md` files carry
  `artifact_type=keypoints` and map onto the `keypoints` preset with no migration.
- New or changed operator behavior should be reflected in
  `tests/test_skill_docs.py` and `skills/gdstt-cli/SKILL.md`.

## Skill layering policy

- All repository skills use a Russian frontmatter `description`, but the agent responds in the interlocutor's language.
- The canonical operator skill is a single file: `skills/gdstt-cli/SKILL.md`.
- It must stay sufficient for the default workflow: auth, inspect, single-file
  processing, `latest`, reprocess, folder safety, relabeling, and the agent
  keypoints workflow.
- Keep Google Drive setup (OAuth, scopes, folder-id discovery) in `README.md`, not
  the skill — the skill points to it with one line.
- Use a separate skill only if it has a genuinely different trigger, audience, and
  decision tree. If that day comes, add it as a sibling folder under `skills/`, not
  as a nested subskill inside the `gdstt-cli` package.

## Current provider posture

Deepgram is the only STT provider. Keep the common CLI workflow stable and move
provider tuning into the Deepgram notes in `README.md` / `SKILL.md`.

Today that means:

- Deepgram is the default and only transcription path; `stt.provider=""` keeps
  MP3-only mode.
- The OpenAI preset DAG (the `presets` map, with `keypoints` shipped as a built-in)
  is documented as an optional layer on top of the transcript, independent of the
  STT path.

## Conventions

- Tests mock all external services (Drive, OpenAI, Deepgram, ffmpeg); one test file
  per `src` module. No network in tests.
- `from __future__ import annotations` + `X | None` style type hints throughout.
- Compute Drive-name stems with `drive.drive_stem` (`os.path.splitext`, string-based), never
  `Path(...).stem`/`.name` — Drive names may contain `/`, which `Path` would treat as a path
  separator and drop. Sanitize local temp filenames with `drive.safe_local_name`.
- Secrets and tokens live in `./data` (gitignored); never commit `credentials.json` / `token.json`.

## Portability policy

- Shared rules belong in `AGENTS.md`.
- Canonical portable operator guidance belongs in the single
  `skills/gdstt-cli/SKILL.md`.
- Copy `skills/gdstt-cli/SKILL.md` into each host's skills directory; do not
  commit host-specific `.agents/skills/` or `.claude/skills/` copies.
- Prefer universal docs over editor-specific overlays.
- When operator behavior changes, update `SKILL.md` and refresh its `version` and
  `last_updated` fields.
- If a tool-specific file is required, keep it as a thin pointer back to `AGENTS.md`
  or the main skill rather than redefining runtime behavior.
