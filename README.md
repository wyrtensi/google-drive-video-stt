# google-drive-video-stt

Monitors Google Drive folders for new MP4 files, optionally extracts MP3 audio
with ffmpeg, transcribes them with Deepgram Nova-3 (speaker diarization), and
writes a speaker-named transcript — plus an optional Keypoints document — back to
Google Drive or to a local folder. Designed as a headless preprocessing step for
speech-to-text pipelines.

## Features

- Polls one or more Google Drive folders on a configurable interval
- Idempotent: skips already-created artifacts, linked by source file id metadata
  when available and by sibling name as a legacy fallback
- Audio extraction via ffmpeg (`libmp3lame`, configurable bitrate)
- Deepgram Nova-3 transcription with speaker diarization, full-file (no chunking)
- Optional Telegram error notifications (success is silent)
- Operator CLI (`gdstt`) wrapping auth, the polling loop, on-demand processing,
  newest-file processing, local-file transcription, deterministic speaker
  relabeling, and folder-state inspection
- Local post-processing that maps diarized `Speaker N` labels to the interlocutor
  names parsed from the file name
- Config-defined DAG of OpenAI presets (each writes its own sibling artifact, e.g.
  the built-in Keypoints pass: `## Задачи` / `## Тезисы` / `## Открытые вопросы`),
  with independent presets run in parallel via the OpenAI Responses API
- Output to Google Drive siblings or to a local folder (`output.target`)
- Sibling `.mp3`/`.txt` names preserve the full Drive file name, including `/`
  characters
- Explicit speaker names can be stored on the Drive MP4 when the filename is not
  enough for reliable speaker mapping
- Docker-first deployment, all mutable state in `./data`

## Requirements

- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv) for local development
- `ffmpeg` available on `PATH` for local runs (already included in the Docker image)
- Google Cloud project with the Drive API enabled and OAuth client metadata in
  `data/credentials.json`
- A Deepgram API key for transcription (`stt.provider: deepgram`)
- Optional: an OpenAI API key when any OpenAI preset is enabled (e.g. `keypoints`)
- Optional: a Telegram bot token + chat ID for error notifications
- Optional: an HTTP(S) or SOCKS proxy via `PROXY_URL`; SOCKS support is included
  through the `requests[socks]` dependency

## Setup

For an operator-style local install, install the global CLI first:

```bash
uv tool install --editable .
uv tool update-shell
```

For development in this checkout, install the editable environment too:

```bash
uv sync --extra dev
```

Configuration lives in a single `config.yml`. For a fresh global install, generate
one from the packaged defaults (the default `transcript -> keypoints` preset chain
works out of the box):

```bash
gdstt config init      # writes config.yml + prompts/ to the resolved target (see below)
gdstt config path      # print the resolved config.yml path
```

`config init` (and `config link`/auto-migration) writes to an OS-specific per-user
location unless you pass `--config`/`GDSTT_CONFIG` or set `DATA_DIR`:

| OS | Default `config.yml` |
| --- | --- |
| Linux | `${XDG_CONFIG_HOME:-~/.config}/gdstt/config.yml` |
| macOS | `~/Library/Application Support/gdstt/config.yml` |
| Windows | `%APPDATA%\gdstt\config.yml` |

The packaged prompt assets (`keypoints.md`, `transcript-cleanup.md`,
`action-items.md`) are copied into `<config_dir>/prompts/` beside the config, and
each preset's `prompt_file` points at them. There is no hidden runtime prompt in
Python — the prompt text is owned by these `.md` files.

Alternatively, migrate from an existing `.env`:

```bash
cp .env.example .env
# Set at least FOLDER_IDS, DEEPGRAM_API_KEY, and (if used) OUTPUT_DIR / OPENAI_API_KEY
gdstt config migrate   # writes config.yml from .env; first run also does this
```

`.env` is only read during this one-time migration; afterwards every command reads
`config.yml` exclusively. You can also author `config.yml` by hand (see
[Configuration](#configuration)). Point at a non-default file with
`gdstt --config PATH ...` or the `GDSTT_CONFIG` env var.

After `data/credentials.json` is in place (see below), authenticate once and
verify access with the safe operator flow:

```bash
gdstt auth
gdstt doctor --drive
gdstt list
gdstt process <file-id> --dry-run
```

## Google Drive setup

This app authenticates with a single Google OAuth user credential covering Drive.

Google auth is **config-owned and inline-first**. By default the OAuth client JSON
lives inline in the config under `google.credentials` and the saved token under
`google.token`, both inside `config.yml`. Import the downloaded client with
`gdstt auth import-credentials <path>`, then run `gdstt auth` to write the token.

File mode is an explicit opt-in: `gdstt auth use-files --credentials-file PATH
[--token-file PATH]` sets `google.credentials_file`/`google.token_file` and clears
the inline copies. When neither inline mappings nor file pointers are set, the
loader falls back to `data/credentials.json` and `data/token.json` for
back-compatibility (the legacy layout used by the examples below).

Inline tokens, `client_secret`, and `refresh_token` are secrets: in a shared
(synced) config, keep them in file mode (or a local, non-synced part of the
config) rather than inline where others can read the config.

### Option A — gcloud / Application Default Credentials

Install and initialize the Google Cloud CLI first:

```bash
gcloud init
gcloud auth login --enable-gdrive-access
```

Use an existing project or create a dedicated one:

```bash
gcloud config set project <project-id>
# or
gcloud projects create <project-id> \
  --name="google-drive-video-stt" \
  --set-as-default
```

Enable the Drive API:

```bash
gcloud services enable drive.googleapis.com
```

This app uses an installed-app OAuth client JSON at `data/credentials.json`.
If you have initialized gcloud Application Default Credentials, create that file
from the local gcloud client metadata:

```powershell
gcloud auth application-default login `
  --scopes=https://www.googleapis.com/auth/drive

New-Item -ItemType Directory -Force data | Out-Null
$adcPath = Join-Path $env:APPDATA "gcloud\application_default_credentials.json"
$adc = Get-Content $adcPath | ConvertFrom-Json
$client = @{
  installed = @{
    client_id = $adc.client_id
    client_secret = $adc.client_secret
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    redirect_uris = @("http://localhost")
  }
}
$client | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 data\credentials.json
```

The generated `data/credentials.json` contains only the OAuth client metadata.
Do not copy the ADC `refresh_token` into it. The app creates its own
`data/token.json` when you run `gdstt auth`.

### Option B — Google Cloud Console OAuth client

If the ADC flow is not available in your environment, use the Console fallback:
APIs & Services -> Credentials -> Create Credentials -> OAuth client ID ->
Desktop app, then save the downloaded JSON as `data/credentials.json`. If Google
asks for an OAuth consent screen first, configure it in external test mode and add
your own Google account as a test user. The app requests the
`https://www.googleapis.com/auth/drive` scope.

### Authenticating

```bash
gdstt auth
gdstt auth --manual
gdstt auth "http://localhost/?code=4/abc123&scope=..."
```

### Finding folder ids

To create a Drive folder from the command line, use the Drive API with the gcloud
access token:

```powershell
$token = gcloud auth print-access-token
$body = @{
  name = "google-drive-video-stt"
  mimeType = "application/vnd.google-apps.folder"
} | ConvertTo-Json
Invoke-RestMethod `
  -Method Post `
  -Uri "https://www.googleapis.com/drive/v3/files" `
  -Headers @{ Authorization = "Bearer $token" } `
  -ContentType "application/json" `
  -Body $body
```

Copy the returned `id` into `.env` as `FOLDER_IDS=<folder-id>`. For an existing
Drive folder, the folder id is the last path segment in the browser URL:
`https://drive.google.com/drive/folders/<folder-id>`.

## Configuration

All configuration lives in `data/config.yml`. It is grouped under `output`, `stt`
(with a nested `deepgram` block), and `openai`, plus a top-level `presets` map. On
first run (or via `gdstt config migrate`) the file is auto-generated from the
`.env`/environment described by the table below; afterwards `.env` is no longer
read. Resolve a non-default file with `gdstt --config PATH ...` or `GDSTT_CONFIG`.

```yaml
folder_ids: [abc, def]
poll_interval: 600
bitrate: 96k
data_dir: data
proxy_url: ""
output:
  target: drive          # drive | folder
  dir: null              # required when target=folder
stt:
  provider: deepgram     # "" / disabled => MP3-only
  language: ru
  postprocess: true
  drive_mp3_artifact: false
  deepgram:
    api_key: "..."
    model: nova-3
    diarize_model: latest
    audio_source: m4a_copy
    txt_formatter: word_speaker
    keyterms_enabled: true
    keyterms_file: config/deepgram-keyterms.txt
openai:
  api_key: "..."
  model: gpt-5.4-mini    # global default model for presets
  batch: false           # global default batch mode for presets
  batch_wait: true       # wait synchronously for batch results (default)
  max_parallel: 4        # cap on presets run concurrently
google: {}               # inline-first auth; empty => data_dir fallback
presets:
  transcript-cleanup:
    prompt_file: prompts/transcript-cleanup.md   # packaged asset, copied beside the config
    batch: false                                 # keep an upstream stage off batch (see below)
  keypoints:
    depends_on: [transcript-cleanup]   # overrides the built-in keypoints preset
  expertizeme-managers:
    depends_on: [transcript-cleanup]
    instructions: "Extract per-manager action items..."   # inline prompt instead of a file
```

Presets define the OpenAI post-processing DAG (see
[Preset DAG](#preset-dag-keypoints-and-beyond)). A preset's prompt comes from
`instructions` (inline) **or** `prompt_file`; supplying neither is an error. The
env-var names below are still recognized by the one-time migration and map onto
these YAML keys:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FOLDER_IDS` | (required) | Comma-separated Google Drive folder IDs to monitor |
| `POLL_INTERVAL` | `600` | Seconds between poll cycles |
| `BITRATE` | `96k` | MP3 audio bitrate passed to ffmpeg |
| `DRIVE_MP3_ARTIFACT` | auto | Upload an MP3 artifact to Drive. Defaults to `false` for `DEEPGRAM_AUDIO_SOURCE=m4a_copy`; `true` otherwise |
| `TELEGRAM_BOT_TOKEN` | (empty) | If set with chat ID, errors are posted to Telegram |
| `TELEGRAM_CHAT_ID` | (empty) | Telegram chat to receive error notifications |
| `DATA_DIR` | `data` | Directory holding `credentials.json` and `token.json` |
| `PROXY_URL` | (empty) | Optional `http`/`https`/`socks5` proxy for Telegram, Deepgram, and OpenAI |
| `STT_PROVIDER` | `deepgram` | `deepgram` by default. Set `disabled` (or empty) to skip transcription and only manage MP3 artifacts |
| `STT_LANGUAGE` | (empty) | Language hint. `deepgram`: empty defaults to `ru` |
| `STT_POSTPROCESS` | `true` | Clean the transcript and map diarized `Speaker N` labels to the interlocutor names parsed from the file name, merging spurious extra speakers |
| `OUTPUT_TARGET` | `drive` | Where artifacts are written: `drive` (sibling files) or `folder` (local `OUTPUT_DIR`) |
| `OUTPUT_DIR` | — | Required when `OUTPUT_TARGET=folder`; local directory for transcript/keypoints files |
| `OPENAI_KEYPOINTS` | `false` | Generate a `<base>.keypoints.md` Keypoints document via the OpenAI Responses API after transcription. Requires `OPENAI_API_KEY` |
| `OPENAI_API_KEY` | — | Required when `OPENAI_KEYPOINTS=true` |
| `OPENAI_MODEL` | `gpt-5.4-mini` | Global default model for presets |
| `OPENAI_BATCH` | `false` | Global default batch mode. Batch API is ~50% cheaper but slower (not higher quality); batch on an upstream preset delays its downstream presets |
| `OPENAI_BATCH_WAIT` | `true` | Wait synchronously for batch results (the async path is unsupported) |
| `OPENAI_MAX_PARALLEL` | `4` | Max number of independent presets run concurrently (maps to `openai.max_parallel`) |
| `DEEPGRAM_API_KEY` | — | Required when `STT_PROVIDER=deepgram` unless `DEEPGRAM_API_KEY_FILE` is set |
| `DEEPGRAM_API_KEY_FILE` | — | Optional file containing a raw Deepgram token or JSON with `api_key`, `deepgram_api_key`, or `DEEPGRAM_API_KEY` |
| `DEEPGRAM_MODEL` | `nova-3` | Deepgram model name |
| `DEEPGRAM_DIARIZE_MODEL` | `latest` | Deepgram diarization model: `latest` or `v1` |
| `DEEPGRAM_AUDIO_SOURCE` | `m4a_copy` | Audio sent to Deepgram: `m4a_copy`, `mp3_96k`, or `mp3_192k` |
| `DEEPGRAM_TXT_FORMATTER` | `word_speaker` | Deepgram TXT formatter: `word_speaker` or `utterance` |
| `DEEPGRAM_KEYTERMS_ENABLED` | `true` | Enables Nova-3 keyterm prompting |
| `DEEPGRAM_KEYTERMS_FILE` | `config/deepgram-keyterms.txt` | Keyterms file, one term per line, max 100 |

## Speech-to-text

With `STT_PROVIDER=deepgram` (the default) each pending recording is transcribed
through Deepgram and a sibling `<basename>.txt` is written next to the MP4 (or into
`OUTPUT_DIR` when `OUTPUT_TARGET=folder`). Set `STT_PROVIDER=disabled` to skip
transcription entirely and only manage the optional MP3 artifact.

### Deepgram Nova-3 (diarization)

The `deepgram` provider submits a full-file audio copy to Deepgram's pre-recorded
`/v1/listen` endpoint using Nova-3, Russian language, and `diarize_model=latest`
by default. It is the recommended provider for Russian speaker diarization. It
does not require the Deepgram SDK; the provider uses the existing `requests` HTTP
client dependency.

Setup:

1. Create a Deepgram API key.
2. Set `STT_PROVIDER=deepgram` and either `DEEPGRAM_API_KEY` or
   `DEEPGRAM_API_KEY_FILE` in `.env`.

`DEEPGRAM_API_KEY_FILE` may contain either the raw token or JSON with one of these
fields: `api_key`, `deepgram_api_key`, or `DEEPGRAM_API_KEY`. The API key is never
logged. After each successful Deepgram transcription, the service logs the request
id, duration, and best-effort request cost in USD when Deepgram's usage API has
recorded it.

The production defaults are:

```
DEEPGRAM_MODEL=nova-3
STT_LANGUAGE=ru
DEEPGRAM_DIARIZE_MODEL=latest
DEEPGRAM_AUDIO_SOURCE=m4a_copy  # m4a_copy, mp3_96k, or mp3_192k
DEEPGRAM_TXT_FORMATTER=word_speaker
DEEPGRAM_KEYTERMS_ENABLED=true
DEEPGRAM_KEYTERMS_FILE=config/deepgram-keyterms.txt
```

`m4a_copy` extracts a temporary AAC/M4A audio copy from the source MP4 for
Deepgram without re-encoding. Use `mp3_96k` or `mp3_192k` to send a temporary MP3
instead. With the Deepgram `m4a_copy` default, no extra Drive MP3 is uploaded
unless `DRIVE_MP3_ARTIFACT=true` is set. If an MP3 already exists but TXT is
missing, Deepgram downloads the MP4 again so it can use the selected high-quality
audio source.

`word_speaker` is a Deepgram-only TXT formatter. It uses `utterances` for readable
timing, but splits a line when `words[].speaker` changes inside the utterance.
Set `DEEPGRAM_TXT_FORMATTER=utterance` to use the older utterance-level formatter.

Keyterms are read from `DEEPGRAM_KEYTERMS_FILE`, one term per line. Blank lines and
lines beginning with `#` are ignored. At most 100 keyterms are allowed, and they
are sent only when `DEEPGRAM_MODEL=nova-3`.

Sample output:

```
[00:00:00] Speaker 1: Привет, коллеги.
[00:00:05] Speaker 2: Добрый день.
```

Deepgram sync pre-recorded requests have a processing-time limit: Nova/Base/Enhanced
requests that process for more than 10 minutes may return `504 Gateway Timeout`.
Callback mode would avoid that for long files, but it requires a public callback
endpoint and is intentionally not implemented. In practice this limit has not been
hit: a 1.5-hour (~90 minute) recording has been transcribed through the sync
endpoint without triggering a `504`, so the documented caveat above is a worst-case
warning rather than a hard ceiling observed in real use.

### Transcript post-processing

By default (`STT_POSTPROCESS=true`) the transcript is post-processed before it is
written rather than stored as raw STT output. The local post-processor
(`src/postprocess.py`) normalizes whitespace, parses the interlocutor names from
the recording file name (e.g. `Alice and Bob - 2026/05/28 ... .mp4` → `Alice`,
`Bob`), maps them onto the diarized `Speaker N` labels by order of appearance, and
merges any extra (spurious) diarization speakers into the real one whose turns they
continue.

When a sibling `.txt` already exists, normal polling skips it to avoid spending STT
credits repeatedly. Use `gdstt process <file-id> --reprocess-txt` when you
intentionally want to run STT again and overwrite the existing `.txt` in place. New
`.txt` and `.mp3` artifacts are tagged with the source MP4 id, so future source
renames do not break artifact detection.

### Preset DAG (Keypoints and beyond)

After the transcript is produced, the service runs the **enabled presets** defined
in `data/config.yml` (`src/presets.py` + `src/preset_pipeline.py`, OpenAI Responses
API). Each preset is one OpenAI pass with its own `instructions`; it feeds on the
concatenated outputs of its `depends_on` presets, or the raw transcript when it has
none, and writes its own sibling artifact `<base><artifact_suffix>` (default
`.<name>.md`) tagged `artifact_type=<name>`. Independent presets run in parallel up
to `openai.max_parallel`, and each preset may set its own `model`/`batch`, falling
back to the `openai` defaults.

A built-in `keypoints` preset ships with the code and produces a `<base>.keypoints.md`
document containing `## Задачи` (grouped by `### Ответственный`), `## Тезисы`, and
`## Открытые вопросы` in plain text. The default chain `transcript -> keypoints`
works out of the box. Config presets override built-ins field-by-field, add new
presets, and disable a built-in with `enabled: false`. Running any enabled preset
requires `openai.api_key` and honors `proxy_url`. The canonical example chain is
`transcript-cleanup -> keypoints + expertizeme-managers`.

**Prompt source priority.** Each preset's instructions are resolved as
`instructions` (inline text in the YAML) > `prompt_file` > error. A `prompt_file`
is resolved in order: the path as written, then relative to the config file's
directory, then the packaged asset by base name. A `prompt_file` that is missing,
unreadable, or empty is an error, and a preset that defines neither `instructions`
nor `prompt_file` is an error. The packaged prompt assets (`keypoints.md`,
`transcript-cleanup.md`, `action-items.md`) are copied into `<config_dir>/prompts/`
beside the config — there is no hidden runtime prompt in Python.

**Adding a DAG stage (no Python changes).** Add an entry under `presets:` with an
`instructions` or `prompt_file`, a `depends_on` chain, and optionally
`model`/`batch`/`artifact_suffix`. To insert a stage in the middle, point the
`depends_on` of the downstream presets at the new stage. Everything is config —
editing `config.yml` is the only step.

**Conflict rules (rejected at load time).** Duplicate YAML keys (including two
presets sharing a name) are rejected; enabled presets must use unique
`artifact_suffix` values so their sibling artifacts don't collide on disk; a
`prompt_file` must resolve to a readable, non-empty file; and `depends_on` on an
unknown/disabled preset or a cycle in the DAG is rejected.

**Batch is cheaper and slower, not higher quality.** Setting `openai.batch: true`
(globally or per preset) submits the pass through the OpenAI Batch API (~50%
cheaper at the cost of higher latency); it does not change output quality. A
recommended layout is global `openai.batch: true` with
`transcript-cleanup.batch: false`, because batch on an upstream stage delays every
downstream stage (they wait for its artifact). `openai.batch_wait: true` is the
default (the service waits synchronously for batch results; the async path is not
supported). Per-preset `model`/`batch`/`batch_wait` override the global
`openai.*` defaults.

Idempotency is per preset: `list_folder_state` reports an `artifact_ids` map keyed
by `artifact_type`, so only the presets still missing an artifact are produced on a
later cycle. Existing `.keypoints.md` files map onto the `keypoints` preset with no
migration. `gdstt doctor` prints the resolved config path and the resolved preset
DAG (names, dependencies, enabled state).

For an agent-driven path (reason about speakers, confirm the mapping, relabel
deterministically, and write the Keypoints document by hand), see
[`skills/gdstt-cli/SKILL.md`](skills/gdstt-cli/SKILL.md).

### Output destination

`OUTPUT_TARGET` controls where the transcript and keypoints files land. With the
default `drive`, they are written as siblings of the source MP4 and uploaded (or
updated in place when one already exists). With `folder`, the service writes
`<output_dir>/<base_name>.txt` (and `.keypoints.md`), creating `OUTPUT_DIR` if it
is missing. `OUTPUT_DIR` is required when `OUTPUT_TARGET=folder`.

## Usage

Local run (after `src/auth` has produced a token):

```bash
uv run python -m src.main
```

The process loops forever, sleeping `POLL_INTERVAL` seconds between cycles.

### CLI

`uv sync` installs a `gdstt` console script that wraps every operation
(equivalently `uv run python -m src.cli`). All commands read configuration from
`data/config.yml` via `load_config()` (auto-migrated from `.env` on first run);
pass `gdstt --config PATH ...` or set `GDSTT_CONFIG` to use a non-default file.

Safe operator flow: `gdstt doctor` -> `gdstt list` -> `gdstt process <file-id> --dry-run`
-> `gdstt process <file-id>`. Move to `run-once` or continuous `run` only after
that single-file path looks correct.

```bash
gdstt auth [response_url]   # one-time interactive OAuth → google.token (or token.json in file mode)
gdstt auth import-credentials <path>   # store an OAuth client JSON inline (google.credentials)
gdstt auth use-files --credentials-file PATH [--token-file PATH]   # switch to file mode
gdstt doctor [--drive]      # check Drive/OAuth configuration without changing it
gdstt latest [--folder ID] [--dry-run] [--max-size SIZE] [--confirm-large]   # process the newest mp4 in a folder
gdstt run                   # continuous polling; can spend STT credits across all pending configured folders
gdstt run-once [--dry-run] [--max-size SIZE] [--confirm-large]   # single cycle; use --dry-run first
gdstt process <id> [--folder] [--reprocess-txt] [--dry-run] [--max-size SIZE] [--confirm-large]   # single target or folder; use --dry-run first
gdstt speakers set <file-id> "Alice" "Bob"   # store explicit speaker names on an MP4
gdstt transcribe <audio> [-o out.txt]   # STT-only on a local file; prints to stdout by default
gdstt relabel --in SRC --out OUT --map MAP.json [--no-header]   # deterministic local speaker relabeling
gdstt list [--folder ID]   # show sibling mp3/txt state without doing work (alias: status)
gdstt config init [--local] [--data-dir DIR] [--output-dir DIR] [--prompt-dir DIR] [--force]   # fresh config.yml + prompts/ from packaged defaults
gdstt config migrate [--force]   # (re)write config.yml from the current .env/environment
gdstt config path           # print the resolved config.yml path (no secrets required)
gdstt config link DIR [--copy-prompts] [--force]   # move config into DIR and leave a pointer behind
gdstt config get [KEY] [--show-secrets]   # print the config/KEY with secrets masked (or revealed)
gdstt config set KEY VALUE  # set a dotted KEY and validate
gdstt config unset KEY      # remove an optional dotted KEY
gdstt --config PATH <command>    # use a non-default config.yml (or set GDSTT_CONFIG)
```

`process` auto-detects whether the ID is a file or a folder; pass `--folder` to
force folder handling. `latest` resolves the folder from `--folder` or the first of
`FOLDER_IDS` and processes the newest (most recently created) mp4. `list`/`status`
defaults to the configured `FOLDER_IDS` when `--folder` is omitted.
`--reprocess-txt` intentionally spends STT provider credits again and overwrites
the linked `.txt` when one exists. `speakers set` affects future local
post-processing; combine it with `process <file-id> --reprocess-txt` when an
already-uploaded transcript needs to be regenerated with corrected names.

`relabel` is a local file transform — it reads a transcript and a `MAP.json`
(`default` label → name plus verbatim-text `exceptions`), merges consecutive
same-speaker turns, preserves each utterance's words (whitespace is normalized),
and reports unmapped labels on stderr. It touches no Drive and spends nothing.

Use `doctor` first when setting up a new agent or machine: it reports `DATA_DIR`,
credentials/token presence, the `FOLDER_IDS` count, and `STT_PROVIDER` without
validating provider secrets. Add `--drive` only when you want it to authenticate
and list the configured folders. Use `--dry-run` on `run-once`, `latest`, or folder
`process` to preview pending work without downloads, uploads, or STT calls.
`--max-size` is off unless you pass it. Use it as an optional manual safety limit
before processing folders, for example `--max-size 50MB`; files larger than the
limit are skipped unless you also pass `--confirm-large`.

`run` has no preview mode and is intentionally the least safe operator entrypoint:
it keeps polling and can continue spending STT credits until you stop it. Use it
only after the single-file or `run-once --dry-run` path already matches expectations.

#### Managing the config file

There is always exactly one active `config.yml`; `gdstt config path` prints where
it is. `config init` creates one from the packaged defaults and copies the prompt
assets into `<config_dir>/prompts/`. With no `--config`/`--local`, init resolves its
target the same way the runtime reads it — `GDSTT_CONFIG` > `<DATA_DIR>/config.yml`
(when `DATA_DIR` is set) > the OS-default path — so it writes exactly where the
runtime will look (relevant under Docker, where the image bakes `DATA_DIR=/app/data`).
`config get`/`set`/`unset` read (with secrets masked) or edit a dotted key (e.g.
`openai.model`, `stt.deepgram.api_key`) in place, validating the result. The
OS-default locations are listed in [Setup](#setup).

`config link DIR` moves the effective full config into `DIR/config.yml` and leaves
a forwarding pointer (`config_file: <DIR/config.yml>`) behind at the OS-default
path; if no full config exists yet, it creates one there from the defaults. Pass
`--copy-prompts` to copy the prompt assets into `DIR/prompts/`. This enables a
**two-layer shared-folder setup**: keep the full `config.yml` + `prompts/` in a
shared (synced) folder, while each machine's OS-default path holds only a pointer
to it — the pointer path differs per OS, but every machine resolves the same shared
config. `load_config()` follows the pointer chain to the real file. Keep secrets
(`openai.api_key`, `stt.deepgram.api_key`, inline `google.token`/`credentials`) out
of a shared config — use file mode or a local, non-synced copy.

Synced notes folders are just ordinary user-chosen paths: to land artifacts in
your (possibly synced) notes folder, set `output.target: folder` + `output.dir`,
and select a non-default config or prompt directory with `gdstt --config PATH`,
`GDSTT_CONFIG`, or `config init --prompt-dir`.

### Runtime reliability and summaries

The runtime treats incomplete output as failure instead of silently uploading it:

- Empty provider transcripts raise an STT error; a blank `.txt` is not written.
- Transient Drive metadata lookups, folder-state listings, and downloads retry with
  bounded backoff. Uploads are not retried automatically.
- Downloads are checked against Drive metadata size; mismatched partial temp files
  are removed before retry or recovery.
- `FOLDER_IDS` containing only commas or whitespace fails configuration loading
  instead of producing a misleading no-op run.

`run-once` logs one process summary per worked file, one folder summary per folder,
and one cycle summary. The cycle summary includes pending, processed, failed,
`retry_total`, skipped-by-size, folder-error, and duration fields. Each process
summary also records the Deepgram request cost (USD, when the usage API has
recorded it) and the OpenAI keypoints token usage.

After `process` and `latest`, the CLI prints a short **spend summary** for the
worked files: the Deepgram cost (or `pending` when the usage API has not recorded
it yet) and, when keypoints ran, the OpenAI token counts. `transcribe` prints the
Deepgram cost after a local-file run.

### Agent-facing documentation

Shared repository instructions live in [`AGENTS.md`](AGENTS.md). The operator
skill is a single file, [`skills/gdstt-cli/SKILL.md`](skills/gdstt-cli/SKILL.md);
copy it into your agent's skills directory to use it. The skill also documents an
optional, fill-in-the-blanks Vault integration layer (wikilinks, vault output
paths, and a sensitive-fragment redaction step); the default output stays plain.

## Tests

```bash
uv run pytest
uv run ruff check
```

Deepgram has a gated live smoke test that can spend a small amount of credit. It is
skipped unless explicitly enabled:

```bash
RUN_DEEPGRAM_LIVE_TESTS=1 \
DEEPGRAM_API_KEY_FILE=/path/to/deepgram_api_secret.json \
DEEPGRAM_LIVE_AUDIO_PATH=/path/to/short-audio-or-video.mp4 \
uv run pytest tests/test_stt_deepgram_live.py -s
```

For MP4/MOV/M4V inputs, the live test extracts only the first 30 seconds to a
temporary MP3. It prints the transcript preview, Deepgram request id, duration, and
best-effort USD cost when Deepgram's usage API has recorded it.

## Docker deployment

Build and run with the bundled Compose file:

```bash
docker compose up -d --build
```

The image bakes `DATA_DIR=/app/data` and mounts `./data` there, so the config
resolver keeps **all mutable state inside the volume**: `config.yml`, the first-run
`.env`->YAML auto-migration, and `credentials.json`/`token.json` are written under
`./data` and survive restarts. The Compose file also sets `DATA_DIR=/app/data`
explicitly for clarity, and a bare `docker run` is correct by default thanks to the
image `ENV`. Logs are JSON-file with a 10 MB / 3-file rotation. Restart policy is
`unless-stopped`.

The prompt assets ship **inside the `src` package** (`src/assets/prompts/*.md`), so
the `COPY src ./src` in the `Dockerfile` carries them automatically — there is no
separate `assets/` copy and the keypoints / OpenAI preset stage works in the
container with no extra setup.

Google auth follows the config-owned model (see
[Google Drive setup](#google-drive-setup)): it is inline-first in `config.yml`
(`google.credentials` / `google.token`), with file mode
(`gdstt auth use-files --credentials-file data/credentials.json`, which points
`config.yml` at an operator-supplied `credentials.json`/`token.json` under the
volume rather than creating them) as the explicit opt-in, and a legacy fallback to
`data/credentials.json` / `data/token.json`. The generated `config.yml` is written `0600` because it can hold
inline secrets.

For a fresh VPS:

1. Copy the repo, `.env`, and `data/` (with `credentials.json` and `token.json`) to the host.
2. `docker compose up -d --build`
3. Tail logs with `docker compose logs -f` and verify a poll cycle completes.

### Container smoke check

After a build, confirm the two deployment-critical fixes — volume persistence and
packaged prompts — with the bundled script (a manual/CI check, not a pytest; it
only runs `gdstt doctor` and spends nothing):

```bash
scripts/docker-smoke.sh            # builds google-drive-video-stt:smoke, then runs doctor
scripts/docker-smoke.sh my-image   # use an existing tag instead
```

Equivalently, by hand:

```bash
docker build -t google-drive-video-stt:latest .
docker run --rm -v "$PWD/data:/app/data" --env-file .env \
  google-drive-video-stt:latest gdstt doctor
```

A healthy run prints a `config:` path under `/app/data/config.yml` (volume
persistence). It also proves packaged prompts load: importing the code reads
`keypoints.md` from the `src` package, so a missing asset would abort `doctor`
with a `ValueError` before it prints anything. (Whether `keypoints` appears in
doctor's preset DAG is a *config* property — a migrated `.env` only enables it
when `OPENAI_KEYPOINTS=true` — so it does not prove packaging.) The
`docker-smoke.sh` script additionally loads the prompt explicitly inside the
container to assert packaging directly.

## Project layout

```
src/
  auth.py        OAuth flow + Drive service builder
  config.py      Env var loading
  drive.py       List / download / upload helpers
  extractor.py   ffmpeg MP4 → MP3/M4A wrappers
  notify.py      Telegram error notifier
  main.py        Polling loop + on-demand process_target entry points
  cli.py         gdstt operator CLI (argparse subcommands)
  output.py      Output destination layer (Drive sibling or local folder)
  postprocess.py Local transcript cleanup + speaker-name mapping
  openai_pipeline.py OpenAI Responses keypoints generation (sync + batch)
  relabel_transcript.py Deterministic speaker relabeling from a MAP.json
  stt/
    __init__.py        get_provider() dispatch (Deepgram-only)
    base.py            STTProvider ABC (transcribe_full hook)
    transcribe.py      Full-file transcription call + cost logging
    deepgram_provider.py Deepgram Nova-3 + diarization
    deepgram_usage.py  Best-effort Deepgram usage/cost lookup
tests/           Unit tests (mock external services)
data/            Tokens, credentials, gitignored
```

## License

MIT — see `LICENSE`.
