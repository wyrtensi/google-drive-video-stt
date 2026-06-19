# gdstt Config Home Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the remaining `.env`/migration/bootstrap model with one explicit config home directory, defaulting to `./data`, while keeping `config.yml` as the only runtime source of application settings.

**Architecture:** `GDSTT_HOME` points at a gdstt instance directory; the active config file is always `<GDSTT_HOME>/config.yml`. If `GDSTT_HOME` is unset, the default home is `./data`, so repo and VPS deployments use `./data/config.yml` without hidden OS config paths. `--config PATH` remains a one-shot file override for debugging/tests, but it is not persisted and is not routed through `os.environ`.

**Tech Stack:** Python 3.11, PyYAML, argparse CLI, Docker Compose, pytest, ruff.

---

## Analysis

### Current Env Entry Points

- `.env`/process env is still parsed in `src/config.py` by `_config_from_env()`.
- Missing or empty config triggers auto-migration inside `load_config()`.
- `gdstt config migrate` exposes explicit `.env -> config.yml` conversion.
- `python-dotenv` is still a runtime dependency.
- `GDSTT_CONFIG` points at an arbitrary config file.
- `DATA_DIR` points at a directory and also influences runtime config lookup.
- `src.cli.main()` implements `--config` by writing `os.environ["GDSTT_CONFIG"]`.
- `config_file:` pointer files plus `config link` provide another persistent locator.
- `src.notify.notify_error()` still reads `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` directly from env.
- Docker currently sets `DATA_DIR=/app/data`.

### Target Model

Use exactly one persistent bootstrap knob:

```text
GDSTT_HOME=/app/data
/app/data/config.yml
/app/data/credentials.json
/app/data/token.json
/app/data/prompts/*.md
/app/data/config/deepgram-keyterms.txt
```

Resolver priority:

```text
--config PATH                 # one-shot file override
GDSTT_HOME/config.yml         # persistent instance directory
./data/config.yml             # default when GDSTT_HOME is unset
```

No `.env` is read. No config is generated implicitly by `load_config()`. Missing config is a clear setup error that points to `gdstt config init`.

### Why Directory Env, Not File Env

`GDSTT_HOME` names the whole instance root, not just the YAML file. This keeps `config.yml`, OAuth files, prompt assets, and Deepgram keyterms in one mounted directory. It avoids the old ambiguity where `DATA_DIR` was both a lookup hint and a runtime `data_dir` setting.

### Breaking Changes

- Old `.env` deployments no longer auto-migrate.
- `gdstt config migrate` is removed.
- `GDSTT_CONFIG` is removed as a persistent locator.
- `DATA_DIR` is removed as a config locator.
- OS-default config paths and pointer files are removed.
- Existing users must run `gdstt config init` and fill `data/config.yml`, or set `GDSTT_HOME` to the directory that contains `config.yml`.

### Compatibility Decision

Keep `--config PATH` as a one-shot override because it is useful for tests, local experiments, and emergency diagnostics. Do not keep `GDSTT_CONFIG`; otherwise there are two persistent locators again.

---

## File Map

- Modify `src/config.py`: resolver, missing-config behavior, config init target, removal of `.env` migration, and removal of pointer/link APIs.
- Modify `src/cli.py`: pass `args.config` explicitly; remove `config migrate`; remove `config link`; remove `config init --local`; update help text.
- Modify `src/main.py`: accept `config_path` and pass it to `load_config()` / `is_run_enabled()`.
- Modify `src/notify.py`: stop reading Telegram credentials from env; use config-owned notification settings.
- Modify `pyproject.toml` / `uv.lock`: remove `python-dotenv`.
- Modify `Dockerfile`: replace `DATA_DIR=/app/data` with `GDSTT_HOME=/app/data`.
- Modify `docker-compose.yml`: replace `DATA_DIR` with `GDSTT_HOME`.
- Modify `scripts/docker-smoke.sh`: verify `GDSTT_HOME` and clean `config init`.
- Delete `.env.example`: it advertises the old model. Do not replace it with `config.example.yml`; `gdstt config init` is the only generated-config template.
- Modify `README.md`, `AGENTS.md`, `skills/gdstt-cli/SKILL.md`: document `GDSTT_HOME`, `./data/config.yml`, and no auto-migration.
- Modify `tests/test_config.py`: resolver, init, missing config, no migration, no pointer.
- Modify `tests/test_cli.py`: explicit config-path plumbing, removed commands, help text.
- Modify `tests/test_notify.py`: config-owned notification behavior.
- Modify `tests/test_docker_deploy.py`: Docker env is `GDSTT_HOME`, not `DATA_DIR`.

---

## Task 1: Lock Resolver Semantics With Tests

**Files:**
- Modify: `tests/test_config.py`

- [ ] Replace resolver tests that mention `GDSTT_CONFIG`, `DATA_DIR`, OS user paths, and `.env` anchoring.

Expected tests:

```python
def test_resolve_config_file_path_defaults_to_cwd_data(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GDSTT_HOME", raising=False)

    assert resolve_config_file_path() == Path("data") / CONFIG_FILE_NAME


def test_resolve_config_file_path_honors_gdstt_home(monkeypatch, tmp_path):
    home = tmp_path / "instance"
    monkeypatch.setenv("GDSTT_HOME", str(home))

    assert resolve_config_file_path() == home / CONFIG_FILE_NAME


def test_resolve_config_file_path_expands_home_and_envvars(monkeypatch, tmp_path):
    root = tmp_path / "root"
    monkeypatch.setenv("GDSTT_ROOT", str(root))
    monkeypatch.setenv("GDSTT_HOME", "$GDSTT_ROOT/instance")

    assert resolve_config_file_path() == root / "instance" / CONFIG_FILE_NAME


def test_resolve_config_file_path_prefers_explicit_file(monkeypatch, tmp_path):
    monkeypatch.setenv("GDSTT_HOME", str(tmp_path / "instance"))
    explicit = tmp_path / "custom.yml"

    assert resolve_config_file_path(explicit) == explicit
```

- [ ] Add missing-config tests:

```python
def test_load_config_missing_file_tells_operator_to_init(tmp_path):
    config_file = tmp_path / "missing.yml"

    with pytest.raises(ValueError, match="gdstt config init"):
        load_config(config_path=config_file, validate_providers=False)

    assert not config_file.exists()


def test_load_config_empty_file_tells_operator_to_init(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("  \n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_config(config_path=config_file, validate_providers=False)
```

- [ ] Delete tests under the `auto-migration` and `migrate_config` sections after the missing-config tests above are in place.

- [ ] Run:

```bash
uv run pytest tests/test_config.py -k "resolve_config_file_path or missing_file or empty_file"
```

Expected: new tests fail before implementation.

---

## Task 2: Implement `GDSTT_HOME` Resolver And Remove `.env` Loader

**Files:**
- Modify: `src/config.py`
- Modify: `pyproject.toml`
- Update: `uv.lock`

- [ ] Remove `python-dotenv` import and dependency.

- [ ] Replace bootstrap constants:

```python
CONFIG_HOME_ENV_VAR = "GDSTT_HOME"
DEFAULT_CONFIG_HOME = Path("data")
```

- [ ] Replace `_resolve_config_file_path()` with directory-home semantics:

```python
def _expand_config_home(raw: str) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(raw.strip()))
    return Path(expanded)


def _resolve_config_file_path(config_path: str | Path | None = None) -> Path:
    if config_path:
        return Path(config_path)
    home_raw = os.environ.get(CONFIG_HOME_ENV_VAR, "").strip()
    home = _expand_config_home(home_raw) if home_raw else DEFAULT_CONFIG_HOME
    return home / CONFIG_FILE_NAME
```

- [ ] Keep `resolve_effective_config_path()` as a compatibility wrapper that returns `(path, path)`. This avoids broad call-site churn while removing pointer behavior.

```python
def resolve_effective_config_path(
    config_path: str | Path | None = None,
) -> tuple[Path, Path]:
    path = _resolve_config_file_path(config_path)
    return path, path
```

- [ ] Remove `_dotenv_path()`, `_resolve_relative_to_dotenv()`, `resolve_config_path()`, `_config_from_env()`, `_uses_default_env_keyterms_file()`, `_config_to_owned_yaml_dict()`, and `migrate_config()`.

- [ ] Replace missing/empty behavior in `load_config()`:

```python
def _missing_config_error(path: Path, *, empty: bool = False) -> ValueError:
    state = "empty" if empty else "missing"
    return ValueError(
        f"{path} is {state}; run `gdstt config init` to create a config.yml."
    )
```

```python
resolved = resolve_effective_config_path(config_path)[1]
text = _read_config_text(resolved) if resolved.exists() else ""
if not resolved.exists():
    raise _missing_config_error(resolved)
if not text.strip():
    raise _missing_config_error(resolved, empty=True)
```

- [ ] Run:

```bash
uv lock
uv run pytest tests/test_config.py -k "resolve_config_file_path or missing_file or empty_file"
uv run ruff check src/config.py tests/test_config.py
```

Expected: resolver/missing tests pass; unrelated migration tests fail until removed in later tasks.

---

## Task 3: Make Config Mutation Require An Existing Full Config

**Files:**
- Modify: `src/config.py`
- Modify: `tests/test_config.py`

- [ ] Change `_load_effective_yaml_dict()` so missing or empty config raises the same setup error instead of returning `{}`.

```python
def _load_effective_yaml_dict(config_path: str | Path | None = None) -> tuple[Path, dict]:
    effective = resolve_effective_config_path(config_path)[1]
    if not effective.exists():
        raise _missing_config_error(effective)
    text = _read_config_text(effective)
    if not text.strip():
        raise _missing_config_error(effective, empty=True)
    raw = _parse_config_yaml(text)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{effective} must contain a YAML mapping, got: {type(raw).__name__}"
        )
    return effective, raw
```

- [ ] Keep `is_run_enabled()` defensive: missing/unreadable config returns `True` so a transient read issue does not pause a running loop.

- [ ] Add tests:

```python
def test_config_set_missing_config_requires_init(tmp_path):
    with pytest.raises(ValueError, match="gdstt config init"):
        config_set("run.enabled", "false", config_path=tmp_path / "missing.yml")


def test_config_get_missing_config_requires_init(tmp_path):
    with pytest.raises(ValueError, match="gdstt config init"):
        config_get(config_path=tmp_path / "missing.yml")
```

- [ ] Run:

```bash
uv run pytest tests/test_config.py -k "config_set_missing_config or config_get_missing_config or run_enabled"
```

Expected: all selected tests pass.

---

## Task 4: Update `config init` And Remove Pointer/Link Flow

**Files:**
- Modify: `src/config.py`
- Modify: `src/cli.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_cli.py`

- [ ] Make `init_config()` default to the same resolver: `GDSTT_HOME/config.yml` or `./data/config.yml`.

- [ ] Remove `init_config(local=True)` and the CLI `config init --local` flag. The default target is now already `./data/config.yml`, so `--local` is redundant.

- [ ] Remove pointer parsing helpers:

```text
_read_pointer_target
_resolve_pointer_target
POINTER_KEY
link_config
```

- [ ] Remove CLI command:

```bash
gdstt config link
```

- [ ] Replace link tests with init/home tests:

```python
def test_init_default_writes_to_gdstt_home(monkeypatch, tmp_path):
    home = tmp_path / "instance"
    monkeypatch.setenv("GDSTT_HOME", str(home))

    path = init_config()

    assert path == home / CONFIG_FILE_NAME
    assert path.is_file()
    assert (home / "prompts" / "keypoints.md").is_file()
    assert (home / "config" / "deepgram-keyterms.txt").is_file()
```

- [ ] Run:

```bash
uv run pytest tests/test_config.py -k "init_default or pointer or link"
uv run pytest tests/test_cli.py -k "config_link or config_init"
```

Expected: no pointer/link tests remain; init tests pass.

---

## Task 5: Remove `config migrate` And Auto-Migration CLI Assumptions

**Files:**
- Modify: `src/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_config.py`

- [ ] Remove `migrate_config` import and `cmd_config_migrate()`.

- [ ] Remove `config migrate` parser branch.

- [ ] Update `cmd_run()` comment and behavior. It must validate the config before setting `run.enabled`, and it must not mention migration.

```python
def cmd_run(args: argparse.Namespace) -> None:
    load_config(config_path=args.config)
    set_run_enabled(True, config_path=args.config)
    main_module.main(config_path=args.config)
```

- [ ] Update test:

```python
def test_run_dispatch_validates_enables_then_calls_main(mocker):
    calls = []
    mocker.patch("src.cli.load_config", side_effect=lambda *a, **k: calls.append("load"))
    mocker.patch("src.cli.set_run_enabled", side_effect=lambda *a, **k: calls.append("set"))
    mocker.patch("src.cli.main_module.main", side_effect=lambda *a, **k: calls.append("main"))

    cli.main(["run"])

    assert calls == ["load", "set", "main"]
```

- [ ] Remove tests named `test_config_migrate_*` and `test_run_migration_runs_before_run_enabled_keeps_env`.

- [ ] Run:

```bash
uv run pytest tests/test_cli.py -k "run_dispatch or config_migrate"
uv run pytest tests/test_config.py -k "migration or migrate"
```

Expected: migrate tests are gone; run dispatch passes.

---

## Task 6: Stop Routing `--config` Through Env

**Files:**
- Modify: `src/cli.py`
- Modify: `src/main.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_main.py`

- [ ] Remove this behavior from `src.cli.main()`:

```python
os.environ[CONFIG_PATH_ENV_VAR] = args.config
```

- [ ] Pass `args.config` explicitly to every command that reads or writes config:

```python
config = load_config(config_path=args.config, validate_providers=False)
path = set_run_enabled(True, config_path=args.config)
path = init_config(config_path=args.config, ...)
```

- [ ] Change `src.main.main()` signature:

```python
def main(*, config_path: str | Path | None = None) -> None:
    config = load_config(config_path=config_path)
    ...
    if not is_run_enabled(config_path=config_path):
        ...
```

- [ ] Update CLI tests so `--config` is asserted via function arguments, not env mutation:

```python
def test_config_flag_is_passed_to_doctor(mocker, capsys, tmp_path):
    target = tmp_path / "custom.yml"
    cfg = make_config(folder_ids=["f1"], data_dir=tmp_path)
    load = mocker.patch("src.cli.load_config", return_value=cfg)

    cli.main(["--config", str(target), "doctor"])

    load.assert_called_once_with(config_path=str(target), validate_providers=False)
```

- [ ] Run:

```bash
uv run pytest tests/test_cli.py -k "config_flag or doctor or start_command or stop_command"
uv run pytest tests/test_main.py -k "run_enabled"
```

Expected: CLI no longer mutates env; config path is passed directly.

---

## Task 7: Move Telegram Notification Settings Into Config

**Files:**
- Modify: `src/config.py`
- Modify: `src/notify.py`
- Modify: `src/main.py`
- Modify: `tests/test_notify.py`
- Modify: `tests/test_config.py`

- [ ] Add config fields:

```python
telegram_bot_token: str = ""
telegram_chat_id: str = ""
```

- [ ] Add default YAML block:

```yaml
notifications:
  telegram:
    bot_token: ""
    chat_id: ""
```

- [ ] Parse it in `_config_from_yaml()` from `notifications.telegram`.

- [ ] Change `notify_error()` signature:

```python
def notify_error(
    text: str,
    *,
    telegram_bot_token: str = "",
    telegram_chat_id: str = "",
    proxy_url: str = "",
) -> None:
```

- [ ] Update `src.main` call sites:

```python
notify.notify_error(
    message,
    telegram_bot_token=config.telegram_bot_token,
    telegram_chat_id=config.telegram_chat_id,
    proxy_url=config.proxy_url,
)
```

- [ ] Update tests to pass token/chat explicitly instead of `monkeypatch.setenv()`.

- [ ] Run:

```bash
uv run pytest tests/test_notify.py tests/test_main.py -k "notify"
uv run pytest tests/test_config.py -k "notification or telegram"
```

Expected: no runtime env reads remain in notify code.

---

## Task 8: Update Docker And Smoke Tests

**Files:**
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `scripts/docker-smoke.sh`
- Modify: `tests/test_docker_deploy.py`

- [ ] Replace Docker env:

```dockerfile
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    GDSTT_HOME=/app/data
```

- [ ] Replace Compose env:

```yaml
environment:
  GDSTT_HOME: /app/data
```

- [ ] Keep volume:

```yaml
volumes:
  - ./data:/app/data
```

- [ ] Update smoke script assertions to avoid `.env`, `DATA_DIR`, and `--env-file`.

- [ ] Update Docker deployment test:

```python
assert "env_file" not in service
assert service["environment"]["GDSTT_HOME"] == "/app/data"
assert "DATA_DIR" not in service["environment"]
```

- [ ] Run:

```bash
uv run pytest tests/test_docker_deploy.py
docker compose config
docker build -t google-drive-video-stt:config-home .
```

Expected: compose shows `GDSTT_HOME=/app/data` and no `DATA_DIR`.

---

## Task 9: Update Docs, Skill, And Example Artifacts

**Files:**
- Delete: `.env.example`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `skills/gdstt-cli/SKILL.md`

- [ ] Remove all instructions that say `.env`, `config migrate`, `GDSTT_CONFIG`, `DATA_DIR`, OS default config, pointer config, or `config link`.

- [ ] Document local default:

```bash
gdstt config init --force
gdstt config set stt.deepgram.api_key ...
gdstt config set openai.api_key ...
gdstt doctor
```

This writes `./data/config.yml` when `GDSTT_HOME` is unset.

- [ ] Document custom instance directory:

```bash
export GDSTT_HOME=/srv/gdstt
gdstt config init --force
gdstt doctor
```

- [ ] Document Docker:

```bash
mkdir -p data
docker compose run --rm google-drive-video-stt gdstt config init --force
# edit ./data/config.yml
docker compose up -d --build
```

- [ ] Update skill frontmatter to the next patch version and keep `skills/gdstt-cli/SKILL.md` at 400 lines or fewer.

- [ ] Run:

```bash
uv run pytest tests/test_skill_docs.py
rg -n "\\.env|config migrate|GDSTT_CONFIG|DATA_DIR|config link|config_file:" README.md AGENTS.md skills/gdstt-cli/SKILL.md Dockerfile docker-compose.yml scripts
rg -n "load_dotenv|_config_from_env|migrate_config|CONFIG_PATH_ENV_VAR|DATA_DIR_ENV_VAR|TELEGRAM_BOT_TOKEN|TELEGRAM_CHAT_ID|env_file|--env-file" src tests pyproject.toml
```

Expected: no matches, except unrelated test-only environment variables (`RUN_*`, `GDSTT_E2E_*`, `DEEPGRAM_LIVE_*`) and OAuth's `OAUTHLIB_INSECURE_TRANSPORT`.

---

## Task 10: Full Verification

**Files:**
- All touched files

- [ ] Run:

```bash
uv run pytest
uv run ruff check
docker compose config
docker build -t google-drive-video-stt:config-home .
```

- [ ] Manual clean Docker smoke:

```powershell
$image = "google-drive-video-stt:config-home"
$smoke = Join-Path ([System.IO.Path]::GetTempPath()) ("gdstt-home-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $smoke | Out-Null
docker run --rm -e GDSTT_HOME=/app/data -v "${smoke}:/app/data" $image gdstt config init --force
docker run --rm -e GDSTT_HOME=/app/data -v "${smoke}:/app/data" $image gdstt config set stt.deepgram.api_key smoke-deepgram-key
docker run --rm -e GDSTT_HOME=/app/data -v "${smoke}:/app/data" $image gdstt config set openai.api_key smoke-openai-key
docker run --rm -e GDSTT_HOME=/app/data -v "${smoke}:/app/data" $image gdstt doctor
docker run --rm -e GDSTT_HOME=/app/data -v "${smoke}:/app/data" $image python -c "from src.config import load_config; cfg = load_config(); assert cfg.config_file.as_posix() == '/app/data/config.yml'; assert cfg.deepgram_keyterms"
Remove-Item -LiteralPath $smoke -Recurse -Force
```

Expected: clean Docker instance uses `/app/data/config.yml`, keyterms load from `/app/data/config/deepgram-keyterms.txt`, and no `.env` file is needed.

---

## Self-Review

- Spec coverage: The plan removes `.env` conversion, removes `DATA_DIR`/`GDSTT_CONFIG` persistent locators, introduces one env directory locator, keeps `./data/config.yml` as default, and updates Docker/docs/tests.
- Placeholder scan: No placeholder markers or unspecified error-handling steps are present.
- Type consistency: The plan consistently uses `GDSTT_HOME` as a directory and `config_path` as a one-shot file path.
