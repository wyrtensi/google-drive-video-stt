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

The code is configured through `<GDSTT_HOME>/config.yml` (default `./data/config.yml`,
loaded by `src/config.py`); the main runtime lives in `src/main.py`, preset
definitions in `src/presets.py`, the DAG executor in `src/preset_pipeline.py`, and
STT provider dispatch in `src/stt/__init__.py`. There is no dotenv loader and no
migration step: the config must already exist (`gdstt config init`); a missing or
empty file is a clear setup error, and every run reads only the YAML.

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
gdstt <auth|doctor|latest|run|start|stop|run-once|process|reprocess|transcribe|relabel|speakers|list|config>  # operator CLI (src/cli.py)
gdstt reprocess <id> [STAGES]   # force-rerun chain stages by number (0=transcript, 1..N=presets; see `gdstt doctor`)
gdstt stop                      # set run.enabled=false so a running `gdstt run` loop pauses (idles) and stays paused across restarts
gdstt start                     # set run.enabled=true so a paused `gdstt run` loop resumes processing
gdstt config init [--force]     # create config.yml + prompts/ under GDSTT_HOME (default ./data)
gdstt --config PATH <command>   # one-shot override against a non-default config.yml
docker compose up -d --build # containerized deployment (mounts ./data, GDSTT_HOME=/app/data)
scripts/docker-smoke.sh      # manual/CI: clean config-only Docker smoke
scripts/release.sh [patch|minor|major]  # bump version, regen CHANGELOG, commit + tag
uv run bump-my-version show current_version   # read the current release version
uv run git-cliff --tag vX.Y.Z -o CHANGELOG.md # regenerate the changelog for a tag
```

`ffmpeg` must be on PATH for local runs (bundled in the Docker image).

## Deployment (Docker)

The deployment is config-owned and persists everything in the mounted volume:

- The image bakes `ENV GDSTT_HOME=/app/data` and Compose mounts `./data:/app/data`,
  so the config resolver writes `config.yml`, prompt copies,
  `deepgram-keyterms-example.txt`, and any file-mode `credentials.json`/`token.json`
  under the volume. Without `GDSTT_HOME` the resolver defaults the home to `./data`
  (relative to the working directory) — keep `GDSTT_HOME=/app/data` set so the
  instance directory is the volume. `init_config()`'s default (no `--config`) target
  follows the same resolver as the runtime (`<GDSTT_HOME>/config.yml`, else
  `./data/config.yml`) so `gdstt config init` writes exactly where the runtime reads.
  Keep the two unified — `_resolve_config_file_path()` is the single source of truth
  for both, and there are no OS-default paths or pointer files to dereference.
- Prompt assets ship inside the `src` package (`src/assets/prompts/*.md`), so the
  Dockerfile's `COPY src ./src` carries them; no separate `assets/` copy and no repo
  fallback. This is also why a wheel / `uv tool install` finds the prompts.
- Google auth is inline-first in `config.yml` (`google.credentials`/`google.token`),
  with file mode as the opt-in and a legacy `credentials.json`/`token.json`
  fallback under `data_dir`. The generated `config.yml` is written `0600` on POSIX.
- `scripts/docker-smoke.sh` is the manual/CI verification: it builds the image,
  initializes a clean `/app/data` volume via `gdstt config init`, runs `gdstt doctor`,
  validates provider config, and loads the packaged `keypoints` prompt inside the
  container.

## CI & releases

- **Tests** (`.github/workflows/tests.yml`): runs on pushes to `main` and on PRs.
  It installs deps with `uv sync --extra dev`, then runs `uv run ruff check` and
  `uv run pytest` on Python 3.11 and 3.12. Tests mock all external services and
  hit no network, so the job needs no secrets. Keep both commands green locally
  before pushing.
- **Release** (`scripts/release.sh` + `.github/workflows/release.yml`): the version
  lives only in `pyproject.toml`, bumped by `bump-my-version` (`[tool.bumpversion]`
  there — `commit`/`tag` are disabled so the script owns them). `scripts/release.sh
  [patch|minor|major]` bumps the version, regenerates `CHANGELOG.md` via `git-cliff
  --tag vX.Y.Z` (see `cliff.toml`), commits both as `chore: release vX.Y.Z`, and
  creates the tag. Push with `git push && git push --tags`; the tag push triggers
  `release.yml`, which publishes a GitHub Release with git-cliff notes for that tag.
- **Changelog**: `cliff.toml` drives grouping/ordering; a local pre-commit hook
  (`.pre-commit-config.yaml`) regenerates `CHANGELOG.md`. `chore: release …` and
  `task`/`todo`/`wip` commits are filtered out of the changelog.

## Architecture

Headless service: polls Google Drive folders, extracts audio from new MP4s via
ffmpeg, transcribes to a `.txt`, and optionally runs the enabled OpenAI presets to
write per-preset sibling artifacts (e.g. `.keypoints.md`). All flow is driven by
`Config` (`src/config.py`, frozen dataclass built by `load_config()` which reads
`<GDSTT_HOME>/config.yml`, validates provider-specific required values, and raises on
misconfiguration). `load_config()` resolves the config path from `--config PATH`
(one-shot), else `<GDSTT_HOME>/config.yml`, else `./data/config.yml`; a missing or
empty file raises a setup error pointing at `gdstt config init` (no auto-generation).

**Polling loop** (`src/main.py`): `main()` builds the Drive service once, then loops
`run_once()` every `POLL_INTERVAL` seconds. Per file, `process_item` computes two
independent needs — `needs_mp3` (no sibling `.mp3`) and `needs_txt` (STT enabled and no
sibling `.txt`) — so a file already having an MP3 can still get transcribed on a later
cycle. Idempotency comes from `drive.list_folder_state` reporting sibling presence by
basename. All work happens in a per-item `TemporaryDirectory`.

**Finding work** (`_discover` in `src/main.py`): a cycle either reads Drive's changes
feed from a saved cursor (`_discover_by_changes`) or sweeps every configured folder and
its subfolders (`_discover_by_walk`). Both return the same `(configured_folder_id,
items)` pairs, so everything downstream is unaware of which ran. The feed only says
*where* to look; the folder listing still decides *what* needs doing, which is what
keeps siblings, `source_video_id`, booking markers and preset backfill working
unchanged.

The cursor (`src/change_cursor.py`, `<data-dir>/changes_cursor.txt`) is the service's
only *discovery* state -- the booking journal and each artifact's appProperties are
durable too -- and is deliberately disposable: absent, unreadable, deleted, or
rejected by Drive with 404/410 all lead to the same branch — sweep, take a fresh
cursor, continue. Three orderings are load-bearing and each has a test: the cursor is
taken *before* a sweep (so a file landing mid-sweep is not stepped over), saved *after*
the work (so a failed cycle re-reads the same changes), and never moved by `--dry-run`.

`<data-dir>/changes_folders.txt` holds the folder ids the cursor was taken against and
is written only where the cursor is, under the same `cycle_drained` guard. A cursor
vouches for nothing outside that set: recordings already sitting in a folder added
since were never a change after it, so the feed will never name that folder. A
mismatch (or a missing file) therefore sweeps once -- `_cursor_covers_config`. That
guard matters: a config edited before the folder is actually shared fails to list,
which counts as a folder error, which holds both files where they are until the share
lands. `--mode changes` refuses outright on a mismatch rather than warning and
reading: that cycle would drain, and draining records the new folder set as vouched
for without it ever having been swept, which loses the backlog permanently.

`run.discovery` (`auto`|`walk`) picks the path the polling loop takes; the CLI's
`--mode` defaults to it rather than to `auto`, so a deployment pinned to `walk` is not
silently exercised on the other path. It exists because one assumption behind the feed
is still unproven: discovery via `changes.list` has only been exercised on folders the
account *owns*, and production watches folders shared *to* the service. Walking costs
a request per folder per cycle and stays inside quota at a thousand subfolders, so
`walk` is a real fallback, not a degraded mode.

`since` (`run.since`, `folders[].since`, `run-once --since`) puts older recordings out
of scope. Three things about it are load-bearing. It is compared against
`parse_meeting_start` of the recording's name, falling back to Drive's `createdTime`,
because the operator means the call and `createdTime` answers a different question.
Meet's own lag measured 0-2 hours across eight real recordings -- enough to carry a
late call past midnight -- and a copy or re-upload resets `createdTime` entirely. A recording neither
can date stays in scope -- fail open. And `cycle_skipped_old` is deliberately absent
from `cycle_drained`: an out-of-scope recording is a permanent skip like one over
`--max-size`, and counting it would hold the cursor on a backlog that is never going
to be processed. The filter runs before the settling check for the same reason -- an
old video Drive never finished processing would otherwise count as `deferred`, which
holds the cursor too. Absolute dates only; a rolling `max_age_days` would drop a
still-pending recording out of scope overnight with nothing having happened.

`_is_rejected_cursor` covers 404, 410 **and** a 400 whose error location is
`pageToken`. The 400 case was found live, not by reading: a corrupt cursor file made
Drive answer 400, which was handled as an ordinary feed failure, which held the
cursor, which was the same bad token next cycle -- the service failed every cycle for
good. Match the parameter, never 400 alone, or a genuinely broken request gets swept
over quietly. In changes mode, listings are merged per configured folder; each item
keeps its own `container_id`, so nothing about placement is lost.

Both paths accept a **shortcut** to a recording (`drive.names_a_recording`): Drive
reports the shortcut's own `application/vnd.google-apps.shortcut`, with the real type
only in `shortcutDetails.targetMimeType` (verified live, in the listing and in the
feed). Meet gives attendees shortcuts whatever their access, and the target keeps the
organizer's ACL. `list_folder_state` follows one only when `files.get` on the target
succeeds (403/404/trashed drop it), and the item then carries `media_id` -- what
`process_item` downloads -- and `target_parents`. `file` stays the shortcut, so
`source_video_id`, the bookkeeping appProperties and done-detection all key on the
shortcut's id; never write onto the organizer's file (verified live: appProperties
land on a shortcut with its `modifiedTime` kept). A shortcut with a transcript beside
it skips the target lookup (`target_parents=None`), or every walk would pay a request
per attended call for good.

`main._without_calls_the_organizer_covers` then drops items whose target lives under a
configured folder: that folder is the organizer's and processes the call, and
following the shortcut too would pay for it and post it twice. Every call site that
lists applies it -- walk, feed, `process_target`, `list`, `doctor` -- and they must stay
in step, or `list` contradicts a cycle again. `gdstt latest` is the deliberate exception:
`find_newest_mp4_in_tree` picks real recordings only, because an unreadable shortcut as
the newest file would make it fail. Every user-facing id stays the shortcut's (Drive's
own `webViewLink` for a shortcut is `/file/d/<shortcut id>/view`); putting the target's
id into the meta document would send `gdstt reprocess <video_id>` to the organizer's
folder, with no employee and likely no write access. A 403/404 while climbing means "not
configured" (every configured folder is readable, and so is what is inside it); any
other error raises and counts as a folder error, because guessing "not configured"
during an outage processes the call twice. Known prices, both documented: an organizer
configured after a call went through the shortcut processes it a second time, and a
target that becomes readable later is only found by a walk -- nothing changes in the
attendee's folder for the feed to report.

It is also held back entirely unless the cycle drained what it found. The feed names a
folder once, when something happens in it, and a recording that failed writes no
artifact -- so nothing there changes again and the feed never names it twice. A failed
file, an unlistable folder, an unresolvable parent, or a video left to settle therefore
all keep the cursor where it is. Deliberate permanent skips (no booking, over
`--max-size`) are *not* counted: those would freeze the cursor for good.

**Meeting subfolders** (`drive.list_folder_tree_state`): Google Meet files each meeting
into its own subfolder, so a configured folder is read together with its direct
subfolders — a union, not a mode, which is why a flat folder still behaves exactly as
before. Each item carries `container_id`, the folder the video actually lives in.
`folder_id` and `container_id` mean different things and must not be swapped: the
configured folder identifies the employee (every `config.folder_by_id`, and the
completion webhook's `file.folder_id`), the container is where artifacts are written.
`drive.find_configured_ancestor` translates a container back to its configured folder
for entry points that start from a file — `process_target` and the changes feed — and
returns `None` for folders nobody configured, which is a skip rather than an error.

Two rules that are easy to reintroduce, because each failed silently once. **Every
write goes to `container_id`** — the `.txt`, each preset artifact, `.meta.yml`, `.stt`
and the `.mp3`; the mp3 upload is its own call site and was the one left behind, which
put every artifact a level above its recording. **Every folder listing an operator
command makes reads the tree**, not one level: `list`, `latest`, `doctor --drive`,
`planfix sent` and `bookings restore-dates` all go through a `*_in_tree` helper,
because one level on a Meet root returns nothing and reads as "there is nothing here".

`process_target` (`src/main.py`) is the on-demand entry the CLI's `process` command uses:
it auto-detects file vs folder by `mimeType` (override with `is_folder`), then runs the same
`process_item` over a single file or every pending file in a folder. The `gdstt` CLI
(`src/cli.py`) wraps the same `load_config()`/Drive/STT layers behind argparse subcommands
without duplicating business logic.

**`latest` command** (`src/cli.py` + `drive.find_newest_mp4_in_tree`): resolves a folder
(arg or first configured `folders` entry), finds the newest mp4 across it and its
subfolders by `createdTime desc`, and dispatches it through `process_target` (honoring
`--dry-run`).

**Post-processing** runs in `process_item` after `transcribe_file` and before the
artifact is written, gated by `stt_postprocess` (local path, `src/postprocess.py`):
it cleans whitespace, parses interlocutor names from the file name, maps them onto
diarized `Speaker N` labels, and merges spurious extra speakers. Name parsing strips
calendar duration prefixes (`30-минутная онлайн-встреча …`) and parentheticals,
splits on `,`/`&`/`and`/`и`/`х`/`x`, and discards Google Meet room codes and the
`_ORG_TOKENS` org names. Two non-obvious rules: `_ORG_TOKENS` is a code constant, not
config (an operator's own org name would be read as a person), and the latin `x`
separator is matched case-sensitively so an uppercase `X` stays a middle initial.

Names are looked for in Meet's own transcript first (`src/meet_transcript.py`): a
Google Doc sits beside each recording with an `Attendees` block and `Name: turn` lines.
Both are read — the block is complete but unordered and includes a shared screen as
`<name>'s Presentation`, while the turns say who actually spoke but omit anyone silent.
The result feeds `_resolve_speaker_names` as its `candidates`,
together with the document text. Every failure path (no transcript, no access, an
unfamiliar shape, fewer than two names) returns `None` and leaves file-name parsing in
charge: losing the names is a worse transcript, losing the recording is an outage.

`speaker_roles.resolve` gives the model everything at once: the candidates, the
transcript from its first speech for `WINDOW_SECONDS` (consecutive lines of one
speaker merged, capped by `MAX_*_CHARS`), Meet's turns for the blocks overlapping that
window (`meet_transcript.turns`, each carrying its block's start), the folder owner,
and the manager the calendar title marks (`postprocess.split_participants`). The
window is time-based on purpose: `word_speaker` splits a line on every voice change,
so the old 30-line sample was a minute of mic checks. A reply of `{}` means "cannot
tell" and returns `None`, as does anything outside the candidates. **Once a model
could be asked, no name is bound to a speaker by position**: `_resolve_speaker_names`
returns `[]` for "asked, not answered", and the transcript keeps `Speaker N`. Neither
Meet's speaking order nor the file name's is diarization's -- on a real call Meet's
swapped the labels. `None` means the model was never asked (no key, fewer than two
names) and keeps the old positional binding from the file name. `_run_preset_stage`
still gets the names as `participant_names` (Meet's, or the file name's via `None`):
`build_prompt` lists them "in no particular order", so they carry no swap there.
Without an OpenAI key Meet's transcript is not read at all.

**Preset DAG** (`src/presets.py` + `src/preset_pipeline.py`): after the transcript
is written, `process_item` runs the enabled presets that are still missing an
artifact. Each preset is one OpenAI pass (`src/openai_pipeline.py`, sync or the
Batch API per preset) whose input is its dependency outputs concatenated with a
labeled separator, or the raw transcript when it has no dependencies. Independent
presets run in parallel (a `ThreadPoolExecutor` capped at `openai.max_parallel`),
and each non-empty output is written as `<base><artifact_suffix>` tagged
`artifact_type=<preset-name>`. Built-in presets ship in `BUILTIN_PRESETS`:
`keypoints` (producing `## Задачи` / `## Тезисы` / `## Открытые вопросы`, plain
text) and `meta` (a `.meta.md` YAML-frontmatter artifact whose fields come from
`meta.entities` in `config.yml` — config-shaped data, `src/meta_entity.py` —
parsed back into a `dict[str, str | list[str]]` by `src/meta.py`). The generated
default chain is `transcript-cleanup -> keypoints + meta`; the third packaged
preset, `action-items`, ships disabled (its output duplicates `keypoints`' `##
Задачи` section) but its prompt asset stays packaged, so re-enabling it is a
`presets:` edit, not a code change.
Built-ins carry `depends_on=()` — a hardcoded dependency on a *config* preset like
`transcript-cleanup` would make `validate_dag` reject every config that omits it;
the default chain is wired in `_default_config_dict` instead. `meta` additionally
ships `enabled=False`: a default-enabled built-in joins every config that predates
it, which would trip the `openai.api_key` gate on an STT-only deployment and feed
`meta` the raw diarized transcript wherever `depends_on` was never wired. The
generated config turns it on explicitly. `config.yml` presets
override built-ins field-by-field, add new presets, and disable a built-in with
`enabled: false`. `validate_dag()` rejects unknown/disabled dependencies and cycles. If a preset fails, its dependents are skipped while
independent branches still persist, then an aggregated error makes the file retry
and re-run only the still-missing presets on a later cycle.

**`src/meta_entity.py`** — what the `meta` preset extracts, as config-shaped
data (the `MetaEntity` dataclass, YAML parsing, validation, and the
`{{entities}}` renderer). A leaf module: `config`, `meta`, and `meta_doc` all
import it, and it imports none of them.

**Output layer** (`src/output.py`): `write_artifact` writes each artifact either as
a Drive sibling (`output.target=drive` — upload, or update an existing sibling in
place via `drive.update_file`, its id flowing through `list_folder_state`) or into a
local `output.dir` (`output.target=folder`).

For deterministic, agent-driven speaker correction, `src/relabel_transcript.py`
(and the `gdstt relabel` command) rewrite `Speaker N` labels from a `MAP.json`
while preserving each utterance's words (whitespace is normalized).

**Completion webhook** (`src/webhook.py`): when `webhook.url` is set, `process_item`
POSTs `{file, employee, transcript, artifacts}` on the success path only, after
every artifact is written — normally once per file, and again if a later cycle
re-feeds its transcript to produce a newly added preset (`file.id` is the
receiver's dedupe key). The employee comes from
`config.folder_by_id(folder_id)`; `artifacts` carries each preset's raw text keyed
by name, except `meta`, which `meta.parse_meta` turns into one key per configured
entity (`config.meta_entities`).
`notify_complete` mirrors `notify.notify_error`'s contract and never raises: a
blank url is a `logger.debug` no-op, and any failure logs only the exception *type*
so the token cannot leak. The whole block is additionally wrapped in `try/except` —
a payload-build bug must not undo an already-uploaded file.

**Call bookings and Planfix** (`src/booking_server.py`, `src/call_booking.py`,
`src/meeting_time.py`, `src/booking_gate.py`, `src/planfix.py`): when
`call_booking.enabled` is set, `main()` starts a stdlib HTTP receiver in a daemon
thread. `POST /` (bearer-authenticated, `{start_time, task_id, manager_email}`)
appends a booking to `<GDSTT_HOME>/call_bookings.jsonl`; `GET /health` answers 200.
`run_once` resolves each pending mp4 against that journal via `booking_gate.resolve`:
the meeting start time is parsed out of the Drive file name (not `createdTime`,
which is the *upload* time — start plus the call's length), matched against the
folder employee's email within `call_booking.threshold_minutes`, nearest booking
wins. On the success path `process_item` posts the `planfix.presets` artifacts, each
under its own heading, into the matched task via `planfix.create_comment_url`, and
posts the same summary into the folder's `telegram` chat when it has one.

**Error handling is tiered**: `RefreshError`/`AuthError` propagate up to `main()` and
cause `SystemExit(1)` so the container restarts (after re-running `src.auth`); all other
exceptions are logged + sent to Telegram via `notify.notify_error` and the loop continues.

`process_item()` also emits one process summary per worked file with provider,
`processing_mode`, outcome, retry count, and duration. `run-once()` emits one folder summary per
folder and one cycle summary with provider, overall outcome, folder count,
pending count, processed count, failed count, `retry_total`,
skipped-by-size count, folder-error count, dry-run flag, and duration.

**STT layer** (`src/stt/`): `get_provider(config)` dispatches on `stt.provider`,
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
  `auth`, `doctor`, `list` / `status`, `changes`, `cursor`, and `speakers set`.
- `gdstt changes` is read-only and must stay that way: consuming the feed there would
  leave the next cycle with nothing to find. `gdstt cursor reset` is the supported way
  to force a full sweep, and `run-once --mode walk` sweeps without moving the cursor.
- A video with no `videoMediaMetadata` is left for a later cycle, but only within
  `_MEDIA_SETTLING_GRACE`. The bound is the point: a video that never gets metadata
  must still be transcribed rather than waited on forever.
- Processing commands validate provider configuration and can spend credits:
  `run`, `run-once`, `process`, `reprocess`, `latest`, and `transcribe`.
- `relabel` is a local file transform that touches no Drive and spends nothing.
- `reprocess <id> [STAGES]` force-reruns chain stages by number: `0` = transcript
  (re-runs STT + everything downstream), `1..N` = enabled presets in
  `preset_pipeline.topological_order`. It threads `reprocess_presets` through
  `process_target` -> `process_item` -> `_run_preset_stage(only_presets=...)`,
  reusing dependency artifacts as `precomputed`. `gdstt doctor` prints the numbers.
- `run.enabled` (config flag, default true) controls the polling loop: `gdstt stop`
  sets it false and the loop **pauses** — `main()` re-reads it each cycle via
  `is_run_enabled()` and idles (sleeps then re-checks) instead of exiting, so the
  container stays up. The stop is sticky: `main()` never auto-enables the flag at
  startup, so it survives Docker `restart: unless-stopped` without auto-resuming.
  Resume with `gdstt start`/`gdstt run` (both set it true); `docker compose stop`
  halts the container itself.
- Configuration is `<GDSTT_HOME>/config.yml` (default `./data/config.yml`; grouped
  `output`, `stt.deepgram`, `openai`, `tags`, `webhook`, and a top-level `presets`
  map). The file must
  exist — create it with `gdstt config init` (no dotenv, no migration, no
  auto-generation). Resolve a one-shot non-default file with `gdstt --config PATH ...`.
- `openai.reasoning_effort` (`low`/`medium`/`high`) is the global reasoning effort;
  a preset's own `reasoning_effort` overrides it, exactly like `model`
  (`preset.reasoning_effort or config.openai_reasoning_effort`). Unset is the
  default and a fourth state, not a synonym for `medium`: the request omits the
  `reasoning` field entirely, so a config written before this key produces
  byte-identical API calls. Both the global and the per-preset value go through
  `presets.parse_reasoning_effort` — the vocabulary lives in the leaf module
  because `config` imports `presets`, never the other way round.
- Enabled presets run after the transcript is produced and require `openai.api_key`;
  each writes a `<base><artifact_suffix>` document tagged `artifact_type=<name>`.
  Having no enabled presets replaces the old `openai.keypoints=false` gate.
- Drive folders are configured as `folders: [{folder_id, name, email, telegram}]` —
  one entry per employee, every field but `folder_id` optional. The old `folder_ids: [str]` is **removed**:
  a config still carrying that key raises a setup `ValueError` quoting the `folders`
  shape (the presence of the key is the trigger, so an empty `folder_ids: []` fails
  too). Iteration sites read `folder_id` off `config.folders` directly; there is no
  `Config.folder_ids` compatibility shim (it would revive the exact name the
  migration error bans). `Config.folder_by_id()` resolves an entry back to its
  `EmployeeFolder`.
- `meta.entities` in `config.yml` is the single source of what the `meta` preset
  extracts; each entity is `name`/`prompt` (required) plus optional `type`
  (`text`/`enum`), `multiple`, `allowed`, `label`, `requires`. An `enum` entity's
  `allowed` list is the only vocabulary the model may use for that field, and it
  reaches the prompt through the single `{{entities}}` placeholder rendered at
  load time by `_resolve_prompt_text` (inline, file, and packaged prompts alike).
  `meta.parse_meta` enforces it again: it drops an `enum` value outside `allowed`,
  and empties any entity whose `requires` target came back empty. A
  missing/malformed block degrades to every entity empty (a dict, not a `Meta()` —
  that dataclass is gone) — a bad LLM reply must not fail the file. The top-level
  `tags.allowed`/`referrals.allowed` keys are deprecated: read only while
  `meta.entities` is absent, feeding the four built-in entities
  (`meta_entity.default_entities`); once `meta.entities` is declared, the old keys
  are ignored and logged at startup.
- The completion webhook (`webhook.url`, optional `webhook.token`) is
  fire-and-forget: it fires on the success path only (once per transcription, plus
  once per later preset-backfill cycle), and any failure is logged without failing
  the file. Its payload carries PII (employee email, full
  transcript), so failure logs must never include the body or the token.
- `output.target` selects where artifacts land: `drive` (siblings) or `folder`
  (`output.dir`, required when `folder`). `output.also_drive` publishes only the
  combined `<stem>.stt` (keypoints + meta + transcript, sections chosen by
  `output.stt_presets`) as a Drive sibling on top of `folder` mode — never a copy
  of every artifact. Every artifact still lands in `output.dir` regardless; never
  flip a live `folder` deployment to `target: drive` instead, because the local
  artifacts are what mark a recording processed and the whole backlog would be
  re-transcribed.
- Idempotency relies on `appProperties.source_video_id`, the per-artifact
  `appProperties.artifact_type` (so each preset's sibling is detected
  independently via `list_folder_state`'s `artifact_ids`), and sibling stem
  matching as a fallback. Existing `.keypoints.md` files carry
  `artifact_type=keypoints` and map onto the `keypoints` preset with no migration.
- New or changed operator behavior should be reflected in
  `tests/test_skill_docs.py` and `skills/gdstt-cli/SKILL.md`.
- The booking gate applies to the polling loop only. `run_once` resolves the
  decision, blocks the file, and counts `skipped_unmatched`; the manual commands
  (`process`, `latest`, `transcribe`, `reprocess`) let `process_item` resolve its own
  decision, which yields the `task_id` without the gate. Processing a marked file by
  hand is the supported way to revive it, alongside `gdstt bookings rematch`.
- `call_booking.name_rules` (`{regex, task_id}`, both required, compiled at load
  time so a broken pattern is a startup error) is checked at the very top of
  `booking_gate.resolve` — **before** the `call_booking_enabled` short-circuit and
  before the journal. First match wins, `re.search` against the Drive name, no
  implicit `IGNORECASE`. The placement is the point: a rule is a routing mechanism
  of its own, so it must produce a `task_id` even on a deployment with the receiver
  off (where every decision is otherwise `disabled` with no task and no Planfix
  comment), and it must outrank a real booking that happened to line up with a demo
  call. Its decision is `matched` with `reason="name-rule"`, so everything
  downstream — processing, the Planfix comment, `planfix_task_url` in the meta
  document — needs no special case.
- `booking_match=none` is written **only** while `booking_server.is_running()`. If
  the receiver never bound its port, every recording looks unmatched, and marking
  them would silently retire the whole backlog; with the receiver down the loop skips
  without marking and the files wait.
- The Planfix comment is idempotent through the `planfix_comment_task_id`
  appProperty, written only after a successful POST. `process_item` reaches its
  success path again whenever a later cycle backfills a newly configured preset, and
  without the marker that pass would post a duplicate comment.
- `load_config` rejects `call_booking.enabled` without an `authorization_token`, and
  `disable_recognition` while any `folders` entry lacks both an `email` and a
  `telegram` chat — that folder could never match a booking, so it would never be
  transcribed again. A folder with a chat is exempt: it is recognized regardless.
- `folders[].telegram` is a chat id, and setting it does two things. `run_once` never
  skips (or marks) that folder's unmatched recordings — the chat, not a booking, is
  what the folder is watched for — and `_send_telegram_summary` posts the same header
  + `planfix.presets` sections the CRM comment carries, as **plain text**
  (`_to_plain_text`), split into Telegram-sized chunks by `notify.split_message`.
  Plain text is deliberate: a parse mode fails the whole `sendMessage` on one
  unbalanced `*` or `<` in transcript-derived text, and a rejected message is a lost
  summary. Both channels render from `_summary_sections`, so they never drift.
  Idempotency is the `telegram_sent_chat_id` appProperty, written only after every
  chunk was accepted. A failed send is logged, never escalated through
  `notify.notify_error` — that is the same API that just failed.
- `planfix.ignore_telegram_when_planfix` (default false) turns the chat into a
  fallback: a `matched` recording with `planfix.create_comment_url` configured stays
  out of it. Off by default because the two are independent channels.
- `load_config` rejects a `folders[].telegram` while `notifications.telegram.bot_token`
  is empty: `notify.send_message` returns quietly with no token, so the recordings
  would be transcribed at full cost and the chat would stay empty.
- `bookings list` / `bookings rematch` / `bookings restore-dates` use
  `load_config(validate_providers=False)`; `rematch` and `restore-dates` touch only
  Drive metadata and spend nothing.

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
- The OpenAI preset DAG (the `presets` map, with `keypoints` and `meta` shipped as
  built-ins) is documented as an optional layer on top of the transcript,
  independent of the STT path.

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
