from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from src.presets import (
    BUILTIN_PRESETS,
    PACKAGED_PROMPT_ASSETS,
    Preset,
    default_artifact_suffix,
    load_packaged_prompt,
    merge_presets,
    validate_dag,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logger = logging.getLogger(__name__)

CONFIG_FILE_NAME = "config.yml"
# Subdirectory (relative to a config file) where ``config init``/``link`` and
# auto-migration copy the packaged prompt assets and where generated configs point
# their ``prompt_file`` entries by default.
PROMPTS_DIR_NAME = "prompts"
CONFIG_PATH_ENV_VAR = "GDSTT_CONFIG"
DATA_DIR_ENV_VAR = "DATA_DIR"
APP_DIR_NAME = "gdstt"
# Keys a forwarding pointer file may carry alongside ``config_file``. A pointer is
# meant to do one thing — redirect to the real config — so any runtime key (e.g.
# ``folder_ids`` or ``stt``) appearing next to ``config_file`` is rejected.
POINTER_KEY = "config_file"

SUPPORTED_STT_PROVIDERS = ("", "deepgram")
OUTPUT_TARGETS = ("drive", "folder")
DEEPGRAM_DIARIZE_MODELS = ("latest", "v1")
DEEPGRAM_AUDIO_SOURCES = ("m4a_copy", "mp3_96k", "mp3_192k")
DEEPGRAM_TXT_FORMATTERS = ("word_speaker", "utterance")
DEEPGRAM_DEFAULT_KEYTERMS_FILE = Path("config/deepgram-keyterms.txt")
DEEPGRAM_MAX_KEYTERMS = 100
CHECKOUT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    folder_ids: list[str]
    poll_interval: int
    bitrate: str
    data_dir: Path
    proxy_url: str
    stt_provider: str
    openai_api_key: str
    deepgram_api_key: str
    stt_language: str
    stt_postprocess: bool = True
    drive_mp3_artifact: bool = True
    output_target: str = "drive"
    output_dir: Path | None = None
    openai_keypoints: bool = False
    openai_model: str = "gpt-5.4-mini"
    openai_batch: bool = False
    openai_batch_wait: bool = True
    openai_max_parallel: int = 4
    deepgram_model: str = "nova-3"
    deepgram_diarize_model: str = "latest"
    deepgram_audio_source: str = "m4a_copy"
    deepgram_txt_formatter: str = "word_speaker"
    deepgram_keyterms_enabled: bool = True
    deepgram_keyterms_file: Path = DEEPGRAM_DEFAULT_KEYTERMS_FILE
    deepgram_keyterms: tuple[str, ...] = ()
    presets: tuple[Preset, ...] = ()
    # Google OAuth is config-owned and inline-first. ``google_credentials``/
    # ``google_token`` hold inline mappings (the OAuth client JSON and the saved
    # token); the ``*_file`` paths point at on-disk copies instead. When all four
    # are unset the loaders fall back to ``data_dir/credentials.json`` and
    # ``data_dir/token.json`` for back-compat. The config file path is carried so
    # that refreshing an inline token can be persisted back into the YAML.
    google_credentials: dict | None = None
    google_token: dict | None = None
    google_credentials_file: Path | None = None
    google_token_file: Path | None = None
    config_file: Path | None = None


class UniqueKeyLoader(yaml.SafeLoader):
    """A ``SafeLoader`` that rejects duplicate keys in any YAML mapping.

    PyYAML's default loader silently keeps the last value when a mapping repeats a
    key, which would let ``config.yml`` hide a second preset under the same name (or
    a second top-level key) without warning. This loader raises a ``ValueError`` the
    moment a duplicate key is constructed, covering nested maps such as two presets
    sharing a name under ``presets:``.
    """

    def construct_mapping(self, node, deep=False):  # type: ignore[override]
        mapping: dict = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(
                    f"duplicate key {key!r} in {CONFIG_FILE_NAME} mapping "
                    f"(line {key_node.start_mark.line + 1})"
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _parse_config_yaml(text: str) -> object:
    """Parse config YAML text, rejecting duplicate mapping keys at parse time."""
    return yaml.load(text, Loader=UniqueKeyLoader)


def _parse_folder_ids(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_prompt_text(preset: Preset, config_file: Path | None) -> str:
    """Resolve a preset's final prompt text from instructions or prompt_file.

    Resolution priority: inline ``instructions`` win; otherwise ``prompt_file`` is
    resolved to text in this order — the path as written if readable on this OS,
    then ``<config_dir>/<prompt_file>`` relative to the config file's parent (when a
    config file exists), then the packaged asset by base name. A ``prompt_file``
    that resolves but is missing/unreadable/empty raises ``ValueError``; a preset
    with neither instructions nor prompt_file also raises.
    """
    if preset.instructions.strip():
        return preset.instructions
    if not preset.prompt_file:
        raise ValueError(
            f"preset {preset.name!r} must define instructions or prompt_file"
        )

    candidates: list[Path] = [Path(preset.prompt_file)]
    if config_file is not None:
        candidates.append(config_file.parent / preset.prompt_file)
    for candidate in candidates:
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8-sig")
            if not text.strip():
                raise ValueError(
                    f"preset {preset.name!r} prompt_file {preset.prompt_file!r} "
                    f"is empty: {candidate}"
                )
            return text

    try:
        return load_packaged_prompt(os.path.basename(preset.prompt_file))
    except ValueError as exc:
        raise ValueError(
            f"preset {preset.name!r} prompt_file {preset.prompt_file!r} "
            f"could not be resolved: {exc}"
        ) from exc


def _resolve_presets(
    config_presets: dict | None,
    config_file: Path | None = None,
) -> tuple[Preset, ...]:
    """Merge config presets over built-ins, resolve prompts, validate, and freeze."""
    merged = merge_presets(BUILTIN_PRESETS, config_presets)
    resolved = {
        name: replace(preset, instructions=_resolve_prompt_text(preset, config_file))
        for name, preset in merged.items()
    }
    validate_dag(resolved)
    _validate_unique_artifact_suffixes(resolved.values())
    return tuple(resolved.values())


def _validate_unique_artifact_suffixes(presets: object) -> None:
    """Reject two enabled presets that would write the same sibling artifact.

    Presets may share a ``prompt_file`` but must differ in ``name`` (already enforced
    by the mapping) and in ``artifact_suffix`` so their outputs don't collide on
    disk. ``merge_presets`` returns only enabled presets, so every preset passed here
    is enabled.
    """
    seen: dict[str, str] = {}
    for preset in presets:
        owner = seen.get(preset.artifact_suffix)
        if owner is not None:
            raise ValueError(
                f"presets {owner!r} and {preset.name!r} both use artifact_suffix "
                f"{preset.artifact_suffix!r}; enabled presets must use distinct suffixes"
            )
        seen[preset.artifact_suffix] = preset.name


def _dotenv_path() -> Path:
    cwd_path = Path(".env")
    if cwd_path.exists():
        return cwd_path
    checkout_path = CHECKOUT_ROOT / ".env"
    return checkout_path if checkout_path.exists() else cwd_path


def _resolve_relative_to_dotenv(raw: str, dotenv_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute() or dotenv_path == Path(".env"):
        return path
    return dotenv_path.parent / path


def resolve_config_path(raw: str) -> Path:
    return _resolve_relative_to_dotenv(raw, _dotenv_path())


def _load_deepgram_api_key(api_key: str, api_key_file: str) -> str:
    api_key = api_key.strip()
    if api_key:
        return api_key

    api_key_file = api_key_file.strip()
    if not api_key_file:
        return ""

    path = Path(api_key_file)
    try:
        raw = path.read_text(encoding="utf-8-sig").strip()
    except OSError as exc:
        raise ValueError(f"DEEPGRAM_API_KEY_FILE could not be read: {path}") from exc
    if not raw:
        return ""

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, dict):
        for key in ("api_key", "deepgram_api_key", "DEEPGRAM_API_KEY"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    return ""


def _parse_max_parallel(raw: object, *, default: int) -> int:
    """Parse a positive ``openai.max_parallel`` worker cap (env string or YAML int)."""
    if raw is None:
        return default
    # bool is an int subclass: int(True) == 1 would silently accept a YAML bool.
    if isinstance(raw, bool):
        raise ValueError(f"openai.max_parallel must be an integer, got: {raw!r}")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return default
        raw = text
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError(f"openai.max_parallel must be an integer, got: {raw!r}")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"openai.max_parallel must be an integer, got: {raw!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"openai.max_parallel must be positive, got: {value}")
    return value


def _parse_bool(raw: str, *, default: bool) -> bool:
    value = raw.strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got: {raw!r}")


def _load_deepgram_keyterms(enabled: bool, keyterms_file: Path) -> tuple[str, ...]:
    if not enabled:
        return ()

    try:
        raw_lines = keyterms_file.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ValueError(f"DEEPGRAM_KEYTERMS_FILE could not be read: {keyterms_file}") from exc

    keyterms = tuple(
        line.strip()
        for line in raw_lines
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(keyterms) > DEEPGRAM_MAX_KEYTERMS:
        raise ValueError(
            f"DEEPGRAM_KEYTERMS_FILE may contain at most {DEEPGRAM_MAX_KEYTERMS} "
            f"keyterms, got: {len(keyterms)}"
        )
    return keyterms


def _config_from_env(*, validate_providers: bool = True) -> Config:
    """Build a Config from `.env`/environment variables (auto-migration source)."""
    dotenv_path = _dotenv_path()
    if load_dotenv is not None:
        load_dotenv(dotenv_path=dotenv_path, override=False, encoding="utf-8-sig")
    folder_ids_raw = os.environ.get("FOLDER_IDS", "")
    folder_ids = _parse_folder_ids(folder_ids_raw)
    if folder_ids_raw and not folder_ids:
        raise ValueError(
            "FOLDER_IDS was set but does not contain any non-empty folder ids"
        )

    poll_raw = os.environ.get("POLL_INTERVAL", "600").strip() or "600"
    try:
        poll_interval = int(poll_raw)
    except ValueError as exc:
        raise ValueError(f"POLL_INTERVAL must be an integer, got: {poll_raw!r}") from exc
    if poll_interval <= 0:
        raise ValueError(f"POLL_INTERVAL must be positive, got: {poll_interval}")

    bitrate = os.environ.get("BITRATE", "96k").strip() or "96k"
    data_dir = _resolve_relative_to_dotenv(
        os.environ.get("DATA_DIR", "data").strip() or "data",
        dotenv_path,
    )
    proxy_url = os.environ.get("PROXY_URL", "").strip()

    stt_provider_raw = os.environ.get("STT_PROVIDER")
    if stt_provider_raw is None:
        stt_provider = "deepgram"
    else:
        stt_provider = stt_provider_raw.strip().lower()
    if stt_provider == "disabled":
        stt_provider = ""
    if stt_provider not in SUPPORTED_STT_PROVIDERS:
        raise ValueError(
            f"STT_PROVIDER must be one of {SUPPORTED_STT_PROVIDERS!r}, got: {stt_provider!r}"
        )
    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    deepgram_api_key = ""
    deepgram_model = "nova-3"
    deepgram_diarize_model = "latest"
    deepgram_audio_source = "m4a_copy"
    deepgram_txt_formatter = "word_speaker"
    deepgram_keyterms_enabled = True
    deepgram_keyterms_file = DEEPGRAM_DEFAULT_KEYTERMS_FILE
    deepgram_keyterms: tuple[str, ...] = ()
    # Parse Deepgram settings whenever the provider is selected so auto-migration
    # and `config migrate` serialize them faithfully into config.yml. Gating the
    # parse on validate_providers (as before) made a validate_providers=False load
    # — e.g. `gdstt doctor` / `gdstt config migrate` — persist empty/default
    # Deepgram values, after which the next processing run failed from the
    # migrated YAML. Validation (raising on missing/invalid values) stays gated on
    # validate_providers below; for inspection-only loads (`gdstt auth`,
    # `gdstt list`) a bad file or flag falls back to the default rather than
    # blocking the command.
    if stt_provider == "deepgram":
        deepgram_model = os.environ.get("DEEPGRAM_MODEL", "nova-3").strip() or "nova-3"
        deepgram_diarize_model = (
            os.environ.get("DEEPGRAM_DIARIZE_MODEL", "latest").strip().lower()
            or "latest"
        )
        deepgram_audio_source = (
            os.environ.get("DEEPGRAM_AUDIO_SOURCE", "m4a_copy").strip().lower()
            or "m4a_copy"
        )
        deepgram_txt_formatter = (
            os.environ.get("DEEPGRAM_TXT_FORMATTER", "word_speaker").strip().lower()
            or "word_speaker"
        )
        try:
            deepgram_keyterms_enabled = _parse_bool(
                os.environ.get("DEEPGRAM_KEYTERMS_ENABLED", ""),
                default=True,
            )
        except ValueError:
            if validate_providers:
                raise
            deepgram_keyterms_enabled = True
        deepgram_keyterms_file = _resolve_relative_to_dotenv(
            os.environ.get("DEEPGRAM_KEYTERMS_FILE", str(DEEPGRAM_DEFAULT_KEYTERMS_FILE))
            .strip()
            or str(DEEPGRAM_DEFAULT_KEYTERMS_FILE),
            dotenv_path,
        )
        deepgram_api_key_file = os.environ.get("DEEPGRAM_API_KEY_FILE", "").strip()
        try:
            deepgram_api_key = _load_deepgram_api_key(
                os.environ.get("DEEPGRAM_API_KEY", ""),
                (
                    str(_resolve_relative_to_dotenv(deepgram_api_key_file, dotenv_path))
                    if deepgram_api_key_file
                    else ""
                ),
            )
        except ValueError:
            if validate_providers:
                raise
            deepgram_api_key = ""
    stt_language = os.environ.get("STT_LANGUAGE", "").strip()
    if stt_provider == "deepgram" and not stt_language:
        stt_language = "ru"

    stt_postprocess = _parse_bool(
        os.environ.get("STT_POSTPROCESS", ""), default=True
    )
    drive_mp3_artifact_raw = os.environ.get("DRIVE_MP3_ARTIFACT", "")
    drive_mp3_artifact_default = not (
        stt_provider == "deepgram" and deepgram_audio_source == "m4a_copy"
    )
    drive_mp3_artifact = _parse_bool(
        drive_mp3_artifact_raw, default=drive_mp3_artifact_default
    )

    openai_model = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini").strip() or "gpt-5.4-mini"
    openai_keypoints = _parse_bool(
        os.environ.get("OPENAI_KEYPOINTS", ""), default=False
    )
    openai_batch = _parse_bool(os.environ.get("OPENAI_BATCH", ""), default=False)
    openai_batch_wait = _parse_bool(
        os.environ.get("OPENAI_BATCH_WAIT", ""), default=True
    )
    openai_max_parallel = _parse_max_parallel(
        os.environ.get("OPENAI_MAX_PARALLEL", ""), default=4
    )
    # Env config has no presets map; the built-in keypoints pass is gated by the
    # legacy OPENAI_KEYPOINTS flag so the migrated YAML stays behavior-compatible.
    presets = _resolve_presets({"keypoints": {"enabled": openai_keypoints}}, None)

    output_target = (
        os.environ.get("OUTPUT_TARGET", "drive").strip().lower() or "drive"
    )
    if output_target not in OUTPUT_TARGETS:
        raise ValueError(
            f"OUTPUT_TARGET must be one of {OUTPUT_TARGETS!r}, got: {output_target!r}"
        )
    output_dir_raw = os.environ.get("OUTPUT_DIR", "").strip()
    output_dir = (
        _resolve_relative_to_dotenv(output_dir_raw, dotenv_path)
        if output_dir_raw
        else None
    )
    if output_target == "folder" and output_dir is None:
        raise ValueError("OUTPUT_DIR is required when OUTPUT_TARGET=folder")

    # Provider-secret validation is skipped for commands that only need Drive/
    # data-dir settings (e.g. `gdstt auth`, `gdstt list`), so an unconfigured STT
    # provider can't block bootstrap/inspection commands.
    if validate_providers:
        if presets and not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when any OpenAI preset is enabled")
        if stt_provider == "deepgram":
            if not deepgram_api_key:
                raise ValueError(
                    "DEEPGRAM_API_KEY is required when STT_PROVIDER=deepgram. "
                    "Add DEEPGRAM_API_KEY to your .env."
                )
            if deepgram_diarize_model not in DEEPGRAM_DIARIZE_MODELS:
                raise ValueError(
                    f"DEEPGRAM_DIARIZE_MODEL must be one of {DEEPGRAM_DIARIZE_MODELS!r}, "
                    f"got: {deepgram_diarize_model!r}"
                )
            if deepgram_audio_source not in DEEPGRAM_AUDIO_SOURCES:
                raise ValueError(
                    f"DEEPGRAM_AUDIO_SOURCE must be one of {DEEPGRAM_AUDIO_SOURCES!r}, "
                    f"got: {deepgram_audio_source!r}"
                )
            if deepgram_txt_formatter not in DEEPGRAM_TXT_FORMATTERS:
                raise ValueError(
                    f"DEEPGRAM_TXT_FORMATTER must be one of {DEEPGRAM_TXT_FORMATTERS!r}, "
                    f"got: {deepgram_txt_formatter!r}"
                )
            deepgram_keyterms = _load_deepgram_keyterms(
                deepgram_keyterms_enabled,
                deepgram_keyterms_file,
            )

    return Config(
        folder_ids=folder_ids,
        poll_interval=poll_interval,
        bitrate=bitrate,
        data_dir=data_dir,
        proxy_url=proxy_url,
        stt_provider=stt_provider,
        openai_api_key=openai_api_key,
        deepgram_api_key=deepgram_api_key,
        stt_language=stt_language,
        stt_postprocess=stt_postprocess,
        drive_mp3_artifact=drive_mp3_artifact,
        output_target=output_target,
        output_dir=output_dir,
        openai_keypoints=openai_keypoints,
        openai_model=openai_model,
        openai_batch=openai_batch,
        openai_batch_wait=openai_batch_wait,
        openai_max_parallel=openai_max_parallel,
        deepgram_model=deepgram_model,
        deepgram_diarize_model=deepgram_diarize_model,
        deepgram_audio_source=deepgram_audio_source,
        deepgram_txt_formatter=deepgram_txt_formatter,
        deepgram_keyterms_enabled=deepgram_keyterms_enabled,
        deepgram_keyterms_file=deepgram_keyterms_file,
        deepgram_keyterms=deepgram_keyterms,
        presets=presets,
    )


def _resolve_relative_to(raw: str, base: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return base / path


def _resolve_google_auth(
    google: dict,
    base: Path | None,
) -> tuple[dict | None, dict | None, Path | None, Path | None]:
    """Resolve the ``google:`` block into (credentials, token, creds_file, token_file).

    Inline ``credentials``/``token`` mappings win; otherwise ``credentials_file``/
    ``token_file`` paths are returned (resolved relative to ``base`` — the config
    file's parent — unless absolute). Supplying BOTH an inline mapping and a file
    pointer for the same object is rejected so generated configs and ``config set``
    writes never carry an ambiguous source.
    """
    credentials = google.get("credentials")
    if credentials is not None and not isinstance(credentials, dict):
        raise ValueError("google.credentials must be a mapping in " + CONFIG_FILE_NAME)
    token = google.get("token")
    if token is not None and not isinstance(token, dict):
        raise ValueError("google.token must be a mapping in " + CONFIG_FILE_NAME)

    credentials_file_raw = _yaml_str(google.get("credentials_file"))
    token_file_raw = _yaml_str(google.get("token_file"))

    if credentials is not None and credentials_file_raw:
        raise ValueError(
            "google.credentials and google.credentials_file are both set; "
            "use inline credentials or a file, not both."
        )
    if token is not None and token_file_raw:
        raise ValueError(
            "google.token and google.token_file are both set; "
            "use an inline token or a file, not both."
        )

    def _resolve(raw: str) -> Path:
        return _resolve_relative_to(raw, base) if base is not None else Path(raw)

    credentials_file = _resolve(credentials_file_raw) if credentials_file_raw else None
    token_file = _resolve(token_file_raw) if token_file_raw else None
    return credentials, token, credentials_file, token_file


def _user_config_path() -> Path:
    """Return the OS-specific per-user config path for a global install.

    * Windows: ``%APPDATA%/gdstt/config.yml`` (falling back to
      ``~/AppData/Roaming`` when ``%APPDATA%`` is unset).
    * macOS: ``~/Library/Application Support/gdstt/config.yml``.
    * Linux/other: ``${XDG_CONFIG_HOME:-~/.config}/gdstt/config.yml``.

    Pure and testable by monkeypatching ``sys.platform`` and the relevant
    environment variables; ``~`` is always expanded.
    """
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "").strip()
        base = Path(appdata) if appdata else Path("~/AppData/Roaming")
    elif sys.platform == "darwin":
        base = Path("~/Library/Application Support")
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
        base = Path(xdg) if xdg else Path("~/.config")
    return (base / APP_DIR_NAME / CONFIG_FILE_NAME).expanduser()


def _resolve_config_file_path(config_path: str | Path | None = None) -> Path:
    """Resolve the bootstrap config.yml path (before any pointer is followed).

    Priority: explicit ``--config`` arg > ``GDSTT_CONFIG`` env > ``<DATA_DIR>/
    config.yml`` *only when* ``DATA_DIR`` is explicitly set > the per-user config
    path (:func:`_user_config_path`). The current working directory's ``./data``
    is no longer auto-selected: a global install must opt in via ``DATA_DIR`` or
    ``GDSTT_CONFIG``.
    """
    if config_path:
        return Path(config_path)
    env_path = os.environ.get(CONFIG_PATH_ENV_VAR, "").strip()
    if env_path:
        return Path(env_path)
    data_dir_raw = os.environ.get(DATA_DIR_ENV_VAR)
    if data_dir_raw is not None and data_dir_raw.strip():
        data_dir = _resolve_relative_to_dotenv(data_dir_raw.strip(), _dotenv_path())
        return data_dir / CONFIG_FILE_NAME
    return _user_config_path()


def _read_pointer_target(path: Path) -> str | None:
    """Return the ``config_file`` target if ``path`` is a forwarding pointer.

    A pointer file is a mapping whose *only* key is ``config_file``. If the file
    is missing, empty, or not such a pointer, ``None`` is returned and the file is
    treated as a normal config. A file that carries ``config_file`` alongside any
    other key is rejected with ``ValueError``.
    """
    if not path.exists():
        return None
    text = _read_config_text(path)
    if not text.strip():
        return None
    raw = _parse_config_yaml(text)
    if not isinstance(raw, dict) or POINTER_KEY not in raw:
        return None
    extra = sorted(key for key in raw if key != POINTER_KEY)
    if extra:
        raise ValueError(
            f"pointer config {path} may only contain {POINTER_KEY!r}; "
            f"found extra keys: {', '.join(map(str, extra))}"
        )
    target = raw[POINTER_KEY]
    if not isinstance(target, str) or not target.strip():
        raise ValueError(
            f"pointer config {path} {POINTER_KEY!r} must be a non-empty path string"
        )
    return target.strip()


def _resolve_pointer_target(target: str, pointer_dir: Path) -> Path:
    """Expand ``~``/``$VAR`` in a pointer target and anchor it to the pointer dir."""
    expanded = os.path.expanduser(os.path.expandvars(target))
    path = Path(expanded)
    if not path.is_absolute():
        path = pointer_dir / path
    return path


def resolve_effective_config_path(
    config_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve (bootstrap_path, effective_path), following forwarding pointers.

    ``bootstrap_path`` is where lookup starts (CLI flag/env/DATA_DIR/user path).
    A config file may instead contain only ``config_file: <path>``; the resolver
    follows that chain (relative paths anchored to the pointer's directory, with
    ``~``/``$VAR`` expansion) until it reaches a non-pointer file, which becomes
    ``effective_path``. Pointer loops (including self-reference) are rejected.
    """
    bootstrap = _resolve_config_file_path(config_path)
    effective = bootstrap
    seen: list[Path] = []
    while True:
        try:
            resolved_key = effective.resolve()
        except OSError:
            resolved_key = effective
        if resolved_key in seen:
            chain = " -> ".join(str(p) for p in [*seen, resolved_key])
            raise ValueError(f"pointer config loop detected: {chain}")
        seen.append(resolved_key)
        target = _read_pointer_target(effective)
        if target is None:
            return bootstrap, effective
        effective = _resolve_pointer_target(target, effective.parent)


def resolve_config_file_path(config_path: str | Path | None = None) -> Path:
    """Public resolver for the effective config.yml path (CLI flag/env/user path).

    Follows forwarding pointers so callers like the CLI's ``doctor`` report the
    real file ``load_config`` reads, not an intermediate pointer.
    """
    return resolve_effective_config_path(config_path)[1]


def _as_mapping(value: object, label: str) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise ValueError(f"{label} must be a mapping in {CONFIG_FILE_NAME}")


def _yaml_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _yaml_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _parse_bool(value, default=default)
    raise ValueError(f"Expected boolean value, got: {value!r}")


def _config_from_yaml(
    raw: dict,
    config_file: Path,
    *,
    validate_providers: bool = True,
) -> Config:
    """Build a Config from the grouped `data/config.yml` schema."""
    base = config_file.parent
    output = _as_mapping(raw.get("output"), "output")
    stt = _as_mapping(raw.get("stt"), "stt")
    deepgram = _as_mapping(stt.get("deepgram"), "stt.deepgram")
    openai = _as_mapping(raw.get("openai"), "openai")
    google = _as_mapping(raw.get("google"), "google")
    config_presets = _as_mapping(raw.get("presets"), "presets")

    (
        google_credentials,
        google_token,
        google_credentials_file,
        google_token_file,
    ) = _resolve_google_auth(google, base)

    folder_ids_raw = raw.get("folder_ids") or []
    if isinstance(folder_ids_raw, str):
        folder_ids = _parse_folder_ids(folder_ids_raw)
    elif isinstance(folder_ids_raw, (list, tuple)):
        folder_ids = [str(item).strip() for item in folder_ids_raw if str(item).strip()]
    else:
        raise ValueError("folder_ids must be a list or comma-separated string")

    poll_raw = raw.get("poll_interval", 600)
    try:
        poll_interval = int(poll_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"poll_interval must be an integer, got: {poll_raw!r}") from exc
    if poll_interval <= 0:
        raise ValueError(f"poll_interval must be positive, got: {poll_interval}")

    bitrate = _yaml_str(raw.get("bitrate"), "96k") or "96k"
    data_dir = _resolve_relative_to(
        _yaml_str(raw.get("data_dir"), "data") or "data", base
    )
    proxy_url = _yaml_str(raw.get("proxy_url"))

    stt_provider = _yaml_str(stt.get("provider"), "deepgram").lower()
    if stt_provider == "disabled":
        stt_provider = ""
    if stt_provider not in SUPPORTED_STT_PROVIDERS:
        raise ValueError(
            f"stt.provider must be one of {SUPPORTED_STT_PROVIDERS!r}, got: {stt_provider!r}"
        )

    stt_language = _yaml_str(stt.get("language"))
    if stt_provider == "deepgram" and not stt_language:
        stt_language = "ru"
    stt_postprocess = _yaml_bool(stt.get("postprocess"), default=True)

    openai_api_key = _yaml_str(openai.get("api_key"))
    openai_model = _yaml_str(openai.get("model"), "gpt-5.4-mini") or "gpt-5.4-mini"
    openai_keypoints = _yaml_bool(openai.get("keypoints"), default=False)
    openai_batch = _yaml_bool(openai.get("batch"), default=False)
    openai_batch_wait = _yaml_bool(openai.get("batch_wait"), default=True)
    openai_max_parallel = _parse_max_parallel(openai.get("max_parallel"), default=4)
    presets = _resolve_presets(config_presets, config_file)

    deepgram_api_key = ""
    deepgram_model = _yaml_str(deepgram.get("model"), "nova-3") or "nova-3"
    deepgram_diarize_model = (
        _yaml_str(deepgram.get("diarize_model"), "latest").lower() or "latest"
    )
    deepgram_audio_source = (
        _yaml_str(deepgram.get("audio_source"), "m4a_copy").lower() or "m4a_copy"
    )
    deepgram_txt_formatter = (
        _yaml_str(deepgram.get("txt_formatter"), "word_speaker").lower() or "word_speaker"
    )
    deepgram_keyterms_enabled = _yaml_bool(
        deepgram.get("keyterms_enabled"), default=True
    )
    deepgram_keyterms_file = _resolve_relative_to(
        _yaml_str(deepgram.get("keyterms_file"), str(DEEPGRAM_DEFAULT_KEYTERMS_FILE))
        or str(DEEPGRAM_DEFAULT_KEYTERMS_FILE),
        base,
    )
    deepgram_keyterms: tuple[str, ...] = ()

    drive_mp3_artifact_default = not (
        stt_provider == "deepgram" and deepgram_audio_source == "m4a_copy"
    )
    drive_mp3_artifact = _yaml_bool(
        stt.get("drive_mp3_artifact"), default=drive_mp3_artifact_default
    )

    output_target = _yaml_str(output.get("target"), "drive").lower() or "drive"
    if output_target not in OUTPUT_TARGETS:
        raise ValueError(
            f"output.target must be one of {OUTPUT_TARGETS!r}, got: {output_target!r}"
        )
    output_dir_raw = _yaml_str(output.get("dir"))
    output_dir = _resolve_relative_to(output_dir_raw, base) if output_dir_raw else None
    if output_target == "folder" and output_dir is None:
        raise ValueError("output.dir is required when output.target=folder")

    if validate_providers:
        if presets and not openai_api_key:
            raise ValueError(
                "openai.api_key is required when any OpenAI preset is enabled"
            )
        if stt_provider == "deepgram":
            api_key_file = _yaml_str(deepgram.get("api_key_file"))
            deepgram_api_key = _load_deepgram_api_key(
                _yaml_str(deepgram.get("api_key")),
                str(_resolve_relative_to(api_key_file, base)) if api_key_file else "",
            )
            if not deepgram_api_key:
                raise ValueError(
                    "deepgram.api_key is required when stt.provider=deepgram. "
                    "Add stt.deepgram.api_key to your config.yml."
                )
            if deepgram_diarize_model not in DEEPGRAM_DIARIZE_MODELS:
                raise ValueError(
                    f"stt.deepgram.diarize_model must be one of "
                    f"{DEEPGRAM_DIARIZE_MODELS!r}, got: {deepgram_diarize_model!r}"
                )
            if deepgram_audio_source not in DEEPGRAM_AUDIO_SOURCES:
                raise ValueError(
                    f"stt.deepgram.audio_source must be one of "
                    f"{DEEPGRAM_AUDIO_SOURCES!r}, got: {deepgram_audio_source!r}"
                )
            if deepgram_txt_formatter not in DEEPGRAM_TXT_FORMATTERS:
                raise ValueError(
                    f"stt.deepgram.txt_formatter must be one of "
                    f"{DEEPGRAM_TXT_FORMATTERS!r}, got: {deepgram_txt_formatter!r}"
                )
            deepgram_keyterms = _load_deepgram_keyterms(
                deepgram_keyterms_enabled,
                deepgram_keyterms_file,
            )

    return Config(
        folder_ids=folder_ids,
        poll_interval=poll_interval,
        bitrate=bitrate,
        data_dir=data_dir,
        proxy_url=proxy_url,
        stt_provider=stt_provider,
        openai_api_key=openai_api_key,
        deepgram_api_key=deepgram_api_key,
        stt_language=stt_language,
        stt_postprocess=stt_postprocess,
        drive_mp3_artifact=drive_mp3_artifact,
        output_target=output_target,
        output_dir=output_dir,
        openai_keypoints=openai_keypoints,
        openai_model=openai_model,
        openai_batch=openai_batch,
        openai_batch_wait=openai_batch_wait,
        openai_max_parallel=openai_max_parallel,
        deepgram_model=deepgram_model,
        deepgram_diarize_model=deepgram_diarize_model,
        deepgram_audio_source=deepgram_audio_source,
        deepgram_txt_formatter=deepgram_txt_formatter,
        deepgram_keyterms_enabled=deepgram_keyterms_enabled,
        deepgram_keyterms_file=deepgram_keyterms_file,
        deepgram_keyterms=deepgram_keyterms,
        presets=presets,
        google_credentials=google_credentials,
        google_token=google_token,
        google_credentials_file=google_credentials_file,
        google_token_file=google_token_file,
        config_file=config_file,
    )


def _relpath_for_config(path: Path | None, config_file: Path | None) -> str | None:
    """Serialize a filesystem path so it round-trips through ``config.yml``.

    ``_config_from_yaml`` resolves relative paths against the config file's parent
    directory, so any path written to the YAML must be expressed relative to that
    directory. Writing the bare ``str(path)`` instead made paths like
    ``output.dir`` and ``deepgram.keyterms_file`` re-resolve under ``<data_dir>``
    on the next load (e.g. ``out`` -> ``data/out``), breaking the second run.
    """
    if path is None:
        return None
    if config_file is None:
        return Path(path).as_posix()
    try:
        return Path(os.path.relpath(path, config_file.parent)).as_posix()
    except ValueError:
        # Windows raises when path and base sit on different drives; fall back to the
        # absolute path (still POSIX-normalized for portable YAML).
        return Path(path).as_posix()


def _default_prompt_file(name: str) -> str:
    """Return the default ``prompt_file`` for a preset (``prompts/<name>.md``).

    Always uses ``/`` separators so generated YAML stays portable across OSes; the
    loader resolves it relative to the config file's parent directory.
    """
    return f"{PROMPTS_DIR_NAME}/{name}.md"


def _preset_to_yaml_entry(preset: Preset) -> dict:
    """Serialize one resolved preset into a ``presets:`` entry.

    The inline ``instructions`` carried by a resolved preset are deliberately not
    written back; the prompt text is owned by the ``prompt_file`` (.md asset) so the
    YAML stays a thin pointer. Non-default ``depends_on``/``model``/``batch``/
    ``artifact_suffix`` are emitted only when they diverge from the built-in defaults.
    """
    entry: dict[str, object] = {"enabled": preset.enabled}
    # The prompt text is owned by an .md asset copied beside the config under
    # ``prompts/``. A bare basename (the built-in default, e.g. ``keypoints.md``) is
    # rewritten to ``prompts/<name>.md`` so it points at the copied asset; an explicit
    # path with directories is preserved as-is (already POSIX-normalized below).
    prompt_file = preset.prompt_file or _default_prompt_file(preset.name)
    if "/" not in prompt_file and "\\" not in prompt_file:
        prompt_file = f"{PROMPTS_DIR_NAME}/{prompt_file}"
    entry["prompt_file"] = prompt_file
    if preset.depends_on:
        entry["depends_on"] = list(preset.depends_on)
    if preset.model is not None:
        entry["model"] = preset.model
    if preset.batch is not None:
        entry["batch"] = preset.batch
    # batch_wait is omitted unless explicitly set so the generated YAML stays a thin
    # pointer that inherits the global openai.batch_wait default.
    if preset.batch_wait is not None:
        entry["batch_wait"] = preset.batch_wait
    if preset.artifact_suffix != default_artifact_suffix(preset.name):
        entry["artifact_suffix"] = preset.artifact_suffix
    return entry


def _presets_to_yaml_dict(config: Config) -> dict:
    """Build the ``presets:`` block for a serialized Config.

    ``config.presets`` only holds enabled presets, so the built-in ``keypoints``
    pass is emitted explicitly (with ``enabled: false`` but a real ``prompt_file``)
    whenever it is disabled, keeping the default chain one edit away from re-enabled.
    """
    presets: dict[str, dict] = {}
    for preset in config.presets:
        presets[preset.name] = _preset_to_yaml_entry(preset)
    for builtin in BUILTIN_PRESETS:
        if builtin.name not in presets:
            presets[builtin.name] = {
                "enabled": False,
                "prompt_file": _default_prompt_file(builtin.name),
            }
    return presets


def _default_config_dict(
    *,
    data_dir: str | None = None,
    output_target: str | None = None,
    output_dir: str | None = None,
    prompt_dir: str | None = None,
) -> dict:
    """Build a full default ``config.yml`` mapping for ``config init``/``link``.

    The default preset chain is ``transcript -> keypoints`` with only ``keypoints``
    enabled; every prompt_file uses ``/``-style relative paths so the generated YAML
    is portable. ``prompt_dir`` (when given) is a ``/``-joined path the prompts were
    copied to and that the prompt_file entries point at; otherwise the default
    ``prompts/<name>.md`` layout is used. ``data_dir``/``output_*`` override the
    matching fields.
    """
    def prompt_path(name: str) -> str:
        if prompt_dir:
            return f"{prompt_dir.rstrip('/')}/{name}.md"
        return _default_prompt_file(name)

    presets: dict[str, dict] = {}
    for builtin in BUILTIN_PRESETS:
        presets[builtin.name] = {
            "enabled": builtin.name == "keypoints",
            "prompt_file": prompt_path(builtin.name),
        }

    config: dict[str, object] = {
        "folder_ids": [],
        "poll_interval": 600,
        "bitrate": "96k",
        "data_dir": data_dir or "data",
        "proxy_url": "",
        "output": {
            "target": (output_target or "drive"),
            "dir": output_dir,
        },
        "stt": {
            "provider": "deepgram",
            "language": "ru",
            "postprocess": True,
            "deepgram": {
                "api_key": "",
                "model": "nova-3",
                "diarize_model": "latest",
                "audio_source": "m4a_copy",
                "txt_formatter": "word_speaker",
                "keyterms_enabled": True,
                "keyterms_file": str(DEEPGRAM_DEFAULT_KEYTERMS_FILE),
            },
        },
        "openai": {
            "api_key": "",
            "model": "gpt-5.4-mini",
            "batch": False,
            "max_parallel": 4,
        },
        # Google auth is inline-first and config-owned. The generated config ships an
        # empty block (no *_file pointers) so the data_dir fallback applies until the
        # operator runs `gdstt auth import-credentials` / `auth use-files`.
        "google": {},
        "presets": presets,
    }
    return config


def copy_prompt_assets(target_dir: Path, *, overwrite: bool = False) -> list[Path]:
    """Copy the packaged prompt assets into ``target_dir``.

    Each asset in :data:`PACKAGED_PROMPT_ASSETS` is written under ``target_dir`` by
    file name. Existing files are left untouched unless ``overwrite`` is set. The
    directory is created if needed. Returns the paths actually written.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in PACKAGED_PROMPT_ASSETS:
        dest = target_dir / name
        if dest.exists() and not overwrite:
            continue
        dest.write_text(load_packaged_prompt(name), encoding="utf-8")
        written.append(dest)
    return written


def _google_to_yaml_dict(config: Config, config_file: Path | None) -> dict:
    """Serialize the Google auth block, inline-first.

    Inline ``credentials``/``token`` mappings are written verbatim (masking happens
    at display time, not on disk). The ``*_file`` pointers are only emitted in file
    mode (i.e. when no inline mapping is present and a file path is set); they are
    written relative to the config file's parent so they round-trip. When nothing is
    configured an empty mapping is returned so the loader's data_dir fallback applies.
    """
    block: dict[str, object] = {}
    if config.google_credentials is not None:
        block["credentials"] = config.google_credentials
    elif config.google_credentials_file is not None:
        block["credentials_file"] = _relpath_for_config(
            config.google_credentials_file, config_file
        )
    if config.google_token is not None:
        block["token"] = config.google_token
    elif config.google_token_file is not None:
        block["token_file"] = _relpath_for_config(config.google_token_file, config_file)
    return block


def _config_to_yaml_dict(config: Config, config_file: Path | None = None) -> dict:
    """Serialize a Config into the grouped `config.yml` schema.

    Filesystem paths (``data_dir``, ``output.dir``, ``deepgram.keyterms_file``)
    are written relative to the config file's parent directory so that re-reading
    the YAML (which resolves relative paths against that parent) yields the same
    locations. See :func:`_relpath_for_config`.
    """
    return {
        "folder_ids": list(config.folder_ids),
        "poll_interval": config.poll_interval,
        "bitrate": config.bitrate,
        "data_dir": _relpath_for_config(config.data_dir, config_file),
        "proxy_url": config.proxy_url,
        "output": {
            "target": config.output_target,
            "dir": _relpath_for_config(config.output_dir, config_file),
        },
        "stt": {
            "provider": config.stt_provider,
            "language": config.stt_language,
            "postprocess": config.stt_postprocess,
            "drive_mp3_artifact": config.drive_mp3_artifact,
            "deepgram": {
                "api_key": config.deepgram_api_key,
                "model": config.deepgram_model,
                "diarize_model": config.deepgram_diarize_model,
                "audio_source": config.deepgram_audio_source,
                "txt_formatter": config.deepgram_txt_formatter,
                "keyterms_enabled": config.deepgram_keyterms_enabled,
                "keyterms_file": _relpath_for_config(
                    config.deepgram_keyterms_file, config_file
                ),
            },
        },
        "openai": {
            "api_key": config.openai_api_key,
            "model": config.openai_model,
            "batch": config.openai_batch,
            # batch_wait defaults to true and is omitted unless explicitly disabled,
            # keeping the generated YAML free of the (unsupported) async path.
            **(
                {}
                if config.openai_batch_wait
                else {"batch_wait": config.openai_batch_wait}
            ),
            "max_parallel": config.openai_max_parallel,
            "keypoints": config.openai_keypoints,
        },
        "google": _google_to_yaml_dict(config, config_file),
        # Serialize the resolved preset DAG. Each entry carries a ``prompt_file`` so
        # the prompt text stays owned by the .md assets; disabled built-ins (e.g.
        # keypoints under OPENAI_KEYPOINTS=false) are still written with their
        # prompt_file so the default chain is one edit away from re-enabled.
        "presets": _presets_to_yaml_dict(config),
    }


def _dump_yaml(data: dict) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _write_config_text(path: Path, text: str) -> None:
    """Write the effective config and restrict it to owner-only (0600).

    The config file is the primary store for inline secrets (``openai.api_key``,
    ``stt.deepgram.api_key``, ``google.credentials.*.client_secret``,
    ``google.token.*``), so it is created/rewritten with the same owner-only
    permissions as ``token.json`` rather than the process umask, which on a shared
    host would otherwise leave secrets group/world-readable. ``chmod`` is best-effort
    (a no-op on platforms that do not support POSIX modes).
    """
    path.write_text(text, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _read_config_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except OSError:
        return ""


def load_config(
    *,
    validate_providers: bool = True,
    config_path: str | Path | None = None,
) -> Config:
    """Load configuration from `data/config.yml`, auto-migrating from `.env` once.

    When the resolved config file is missing or empty, build the configuration from
    the existing `.env`/environment, persist it as YAML for future runs, and return
    the in-memory values. Otherwise read settings solely from the YAML file.

    Forwarding pointers (a file containing only ``config_file: <path>``) are
    followed so the effective target file is what gets read and written.
    """
    resolved = resolve_effective_config_path(config_path)[1]
    text = _read_config_text(resolved) if resolved.exists() else ""
    if text.strip():
        raw = _parse_config_yaml(text)
        if not isinstance(raw, dict):
            raise ValueError(
                f"{resolved} must contain a YAML mapping, got: {type(raw).__name__}"
            )
        return _config_from_yaml(raw, resolved, validate_providers=validate_providers)

    # Auto-migration: build from env, persist best-effort, then use in-memory values.
    config = _config_from_env(validate_providers=validate_providers)
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        _write_config_text(resolved, _dump_yaml(_config_to_yaml_dict(config, resolved)))
        copy_prompt_assets(resolved.parent / PROMPTS_DIR_NAME)
        logger.info("Migrated configuration from environment to %s", resolved)
    except OSError as exc:
        logger.warning("Could not write migrated config file %s: %s", resolved, exc)
    return config


def migrate_config(
    *,
    force: bool = False,
    config_path: str | Path | None = None,
) -> Path:
    """Write `data/config.yml` from the current `.env`/environment.

    Raises if the target already exists and ``force`` is False. Validation of
    provider secrets is skipped so migration works for inspection-only setups.
    Forwarding pointers are followed so migration writes the effective target.
    """
    resolved = resolve_effective_config_path(config_path)[1]
    if resolved.exists() and _read_config_text(resolved).strip() and not force:
        raise ValueError(
            f"{resolved} already exists; pass --force to overwrite it."
        )
    config = _config_from_env(validate_providers=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    _write_config_text(resolved, _dump_yaml(_config_to_yaml_dict(config, resolved)))
    copy_prompt_assets(resolved.parent / PROMPTS_DIR_NAME)
    return resolved


def _local_config_path() -> Path:
    """Return ``./data/config.yml`` under the current working directory."""
    return Path("data") / CONFIG_FILE_NAME


def _rel_posix(path: Path, base: Path) -> str:
    """Express ``path`` relative to ``base`` using ``/`` separators (portable YAML)."""
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:
        # Different Windows drives: no relative path exists, keep the absolute one.
        return Path(path).as_posix()


def init_config(
    *,
    config_path: str | Path | None = None,
    local: bool = False,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    prompt_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Create a fresh full ``config.yml`` from the packaged defaults.

    Target selection: explicit ``config_path`` wins; ``local`` writes
    ``./data/config.yml`` in the cwd; otherwise the runtime resolver picks the
    target (``GDSTT_CONFIG`` > ``<DATA_DIR>/config.yml`` when ``DATA_DIR`` is set >
    the cross-platform user config path) so init writes where the runtime reads.
    The default preset chain is ``transcript -> keypoints`` with only
    ``keypoints`` enabled. Prompt assets are always copied beside the config: into
    ``prompt_dir`` when given (and the ``prompt_file`` entries point there), else into
    ``<config_dir>/prompts/``. Refuses to overwrite a non-empty existing config unless
    ``force``.
    """
    if config_path is not None:
        target = Path(config_path)
    elif local:
        target = _local_config_path()
    else:
        # Match the runtime resolver's bootstrap target so init writes where the
        # runtime reads: GDSTT_CONFIG > DATA_DIR/config.yml (when DATA_DIR set) >
        # user path. Use the bootstrap resolver, not the public pointer-following
        # one: `init` creates a fresh config at that location and must not silently
        # dereference an existing forwarding pointer (keeps GDSTT_CONFIG/user-path
        # targeting exactly as before, only adding DATA_DIR awareness).
        target = _resolve_config_file_path()

    if target.exists() and _read_config_text(target).strip() and not force:
        raise ValueError(
            f"{target} already exists; pass --force to overwrite it."
        )

    config_dir = target.parent
    config_dir.mkdir(parents=True, exist_ok=True)

    if prompt_dir is not None:
        prompts_target = Path(prompt_dir)
        if not prompts_target.is_absolute():
            prompts_target = config_dir / prompts_target
        prompt_rel = _rel_posix(prompts_target, config_dir)
    else:
        prompts_target = config_dir / PROMPTS_DIR_NAME
        prompt_rel = None

    copy_prompt_assets(prompts_target)

    data_dir_value = (
        _rel_posix(Path(data_dir), config_dir) if data_dir is not None else None
    )
    output_target = None
    output_dir_value = None
    if output_dir is not None:
        output_target = "folder"
        output_dir_value = _rel_posix(Path(output_dir), config_dir)

    data = _default_config_dict(
        data_dir=data_dir_value,
        output_target=output_target,
        output_dir=output_dir_value,
        prompt_dir=prompt_rel,
    )
    _write_config_text(target, _dump_yaml(data))
    return target


def link_config(
    target_dir: str | Path,
    *,
    copy_prompts: bool = False,
    force: bool = False,
    config_path: str | Path | None = None,
) -> Path:
    """Move/create the effective full config into ``target_dir/config.yml``.

    Resolves the current bootstrap and effective config. If a full config already
    lives at the bootstrap path (the OS-default or ``--config``/``GDSTT_CONFIG``
    target), its settings are moved into ``target_dir/config.yml`` and the bootstrap
    is replaced with a forwarding pointer (``config_file: <target>``). If no full
    config exists yet, a fresh one is created from the packaged defaults. Refuses to
    overwrite an existing ``target_dir/config.yml`` unless ``force``. ``copy_prompts``
    copies the packaged prompt assets into ``target_dir/prompts/`` when missing.
    """
    dest_dir = Path(target_dir)
    dest = dest_dir / CONFIG_FILE_NAME

    bootstrap, effective = resolve_effective_config_path(config_path)

    if dest.exists() and _read_config_text(dest).strip() and not force:
        raise ValueError(
            f"{dest} already exists; pass --force to overwrite it."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)

    effective_text = _read_config_text(effective) if effective.exists() else ""
    if effective_text.strip():
        # Move the existing full config's settings into the destination.
        _write_config_text(dest, effective_text)
        if effective.resolve() != dest.resolve():
            # Replace the source with a pointer to the destination so the OS-default
            # path keeps resolving to the moved config.
            pointer_target = _rel_posix(dest, effective.parent)
            effective.parent.mkdir(parents=True, exist_ok=True)
            effective.write_text(
                _dump_yaml({POINTER_KEY: pointer_target}), encoding="utf-8"
            )
    else:
        # No full config yet: create one from defaults at the destination.
        prompts_target = dest_dir / PROMPTS_DIR_NAME
        copy_prompt_assets(prompts_target)
        _write_config_text(dest, _dump_yaml(_default_config_dict()))
        # Point the bootstrap path at the new destination unless it is the same file.
        if bootstrap.resolve() != dest.resolve():
            pointer_target = _rel_posix(dest, bootstrap.parent)
            bootstrap.parent.mkdir(parents=True, exist_ok=True)
            bootstrap.write_text(
                _dump_yaml({POINTER_KEY: pointer_target}), encoding="utf-8"
            )

    if copy_prompts:
        copy_prompt_assets(dest_dir / PROMPTS_DIR_NAME)

    return dest


# --- config get / set / unset -----------------------------------------------

# Secret-bearing keys are masked when the whole config is dumped via ``config get``
# (with no KEY). Each entry is a tuple of nested mapping keys leading to the value.
MASKED_KEY_PATHS: tuple[tuple[str, ...], ...] = (
    ("openai", "api_key"),
    ("stt", "deepgram", "api_key"),
    ("google", "client_secret"),
    ("google", "refresh_token"),
)
# Leaf key names whose values are masked wherever they appear (covers nested
# credentials/token blocks copied verbatim into the YAML).
MASKED_LEAF_KEYS: frozenset[str] = frozenset(
    {"client_secret", "refresh_token", "token", "access_token", "api_key"}
)
MASK = "***"


def _load_effective_yaml_dict(config_path: str | Path | None = None) -> tuple[Path, dict]:
    """Return the (effective_path, parsed-mapping) for the active config file.

    Follows forwarding pointers so reads/writes land on the real config, never a
    pointer. An empty or missing file yields an empty mapping.
    """
    effective = resolve_effective_config_path(config_path)[1]
    text = _read_config_text(effective) if effective.exists() else ""
    if not text.strip():
        return effective, {}
    raw = _parse_config_yaml(text)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{effective} must contain a YAML mapping, got: {type(raw).__name__}"
        )
    return effective, raw


def _mask_value(value: object) -> object:
    if isinstance(value, dict):
        return {k: _mask_value(MASK if k in MASKED_LEAF_KEYS else v) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask_value(item) for item in value]
    return value


def _mask_config_dict(data: dict) -> dict:
    """Return a deep copy of ``data`` with secret values replaced by ``MASK``."""
    masked = _mask_value(data)
    assert isinstance(masked, dict)  # noqa: S101 - top-level is always a mapping
    for path in MASKED_KEY_PATHS:
        node: object = masked
        for key in path[:-1]:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        if isinstance(node, dict) and path[-1] in node and node[path[-1]] not in (None, ""):
            node[path[-1]] = MASK
    return masked


def _split_key(key: str) -> list[str]:
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError("config key must be a non-empty dotted path")
    return parts


def _get_nested(data: dict, parts: list[str]) -> object:
    node: object = data
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(".".join(parts))
        node = node[part]
    return node


def _set_nested(data: dict, parts: list[str], value: object) -> None:
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _unset_nested(data: dict, parts: list[str]) -> bool:
    node: object = data
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        return False
    del node[parts[-1]]
    return True


def _format_get_value(value: object) -> str:
    if isinstance(value, (dict, list)):
        return _dump_yaml(value).rstrip("\n")
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _mask_get_value(parts: list[str], value: object) -> object:
    """Mask a single ``config get KEY`` value so secrets do not leak to stdout/logs.

    A secret scalar leaf (api keys, tokens, client_secret, refresh_token) is replaced
    wholesale; a mapping/list value (e.g. ``google.credentials`` / ``google.token``)
    is deep-masked so nested secret leaves are hidden while structure stays visible.
    """
    if isinstance(value, (dict, list)):
        return _mask_value(value)
    if parts[-1] in MASKED_LEAF_KEYS and value not in (None, ""):
        return MASK
    return value


def config_get(
    key: str | None = None,
    *,
    config_path: str | Path | None = None,
    show_secrets: bool = False,
) -> str:
    """Return a printable view of the effective config (whole) or one value.

    Secrets (api keys, tokens, client_secret, refresh_token) are masked by default
    for both the whole-config dump and a single-key lookup; pass ``show_secrets`` to
    reveal them. This keeps ``config get google.credentials`` / ``google.token`` from
    leaking the OAuth secret or refresh token to terminal scrollback or CI logs.
    """
    _, data = _load_effective_yaml_dict(config_path)
    if not key:
        return _dump_yaml(data if show_secrets else _mask_config_dict(data)).rstrip("\n")
    parts = _split_key(key)
    try:
        value = _get_nested(data, parts)
    except KeyError as exc:
        raise ValueError(f"config key {key!r} is not set") from exc
    if not show_secrets:
        value = _mask_get_value(parts, value)
    return _format_get_value(value)


def _parse_set_value(parts: list[str], raw: str) -> object:
    """Coerce a raw CLI string into the value type implied by its dotted key."""
    leaf = parts[-1]
    # depends_on accepts a JSON list or a comma/space-separated list of names.
    if leaf == "depends_on":
        text = raw.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"depends_on must be a JSON list, got: {raw!r}") from exc
            if not isinstance(parsed, list):
                raise ValueError(f"depends_on must be a JSON list, got: {raw!r}")
            return [str(item).strip() for item in parsed if str(item).strip()]
        items = [item for item in re.split(r"[,\s]+", text) if item]
        return items
    # Booleans for known boolean leaves.
    if leaf in {"enabled", "batch", "batch_wait", "drive", "postprocess", "keyterms_enabled"}:
        return _parse_bool(raw, default=False)
    # Integers for known numeric leaves.
    if leaf in {"poll_interval", "max_parallel"}:
        try:
            return int(raw.strip())
        except ValueError as exc:
            raise ValueError(f"{leaf} must be an integer, got: {raw!r}") from exc
    return raw


def _atomic_write_validated(
    effective: Path, data: dict, *, config_path: str | Path | None
) -> None:
    """Write ``data`` to ``effective``, validating; leave the file unchanged on failure.

    The original text is snapshotted first; the new YAML is written, then
    ``load_config(validate_providers=False)`` is run against the effective path. Any
    validation error restores the original bytes (or removes a freshly created file)
    and re-raises, so a bad ``set``/``unset`` never corrupts the on-disk config.
    """
    existed = effective.exists()
    original = effective.read_text(encoding="utf-8-sig") if existed else None
    effective.parent.mkdir(parents=True, exist_ok=True)
    _write_config_text(effective, _dump_yaml(data))
    try:
        load_config(validate_providers=False, config_path=config_path)
    except Exception:
        if original is not None:
            _write_config_text(effective, original)
        else:
            effective.unlink(missing_ok=True)
        raise


def config_set(key: str, value: str, *, config_path: str | Path | None = None) -> Path:
    """Set a dotted ``key`` to ``value`` in the effective config and validate.

    Booleans/ints/lists are parsed from the raw string per key. ``output.dir PATH``
    also sets ``output.target: folder``; ``output.drive true`` sets
    ``output.target: drive``; ``output.drive false`` requires an existing
    ``output.dir`` (else an error tells the operator to set it first). The resulting
    full config is validated and the file is left unchanged if validation fails.
    """
    effective, data = _load_effective_yaml_dict(config_path)
    parts = _split_key(key)

    if parts == ["output", "drive"]:
        as_drive = _parse_bool(value, default=False)
        output = data.setdefault("output", {})
        if not isinstance(output, dict):
            output = {}
            data["output"] = output
        if as_drive:
            output["target"] = "drive"
        else:
            if not _yaml_str(output.get("dir")):
                raise ValueError(
                    "output.drive false needs a local folder; run "
                    "`config set output.dir PATH` first."
                )
            output["target"] = "folder"
    else:
        parsed = _parse_set_value(parts, value)
        _set_nested(data, parts, parsed)
        if parts == ["output", "dir"]:
            output = data.setdefault("output", {})
            if isinstance(output, dict):
                output["target"] = "folder"

    _atomic_write_validated(effective, data, config_path=config_path)
    return effective


def config_unset(key: str, *, config_path: str | Path | None = None) -> Path:
    """Remove a dotted ``key`` from the effective config and validate the result."""
    effective, data = _load_effective_yaml_dict(config_path)
    parts = _split_key(key)
    if not _unset_nested(data, parts):
        raise ValueError(f"config key {key!r} is not set")
    _atomic_write_validated(effective, data, config_path=config_path)
    return effective


# --- google auth config helpers ---------------------------------------------


def import_google_credentials(
    credentials_path: str | Path, *, config_path: str | Path | None = None
) -> Path:
    """Read an OAuth client JSON and store it inline under ``google.credentials``.

    The file's parsed structure (the Desktop-app client config, e.g. the
    ``{"installed": {...}}`` mapping) is written verbatim into the effective config
    so it is config-owned. Any pre-existing ``google.credentials_file`` pointer is
    cleared so the two sources never both apply. Validation rejects a malformed JSON
    file and any inline/file conflict before the write lands.
    """
    src = Path(credentials_path)
    try:
        parsed = json.loads(src.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"OAuth client credentials at {src} could not be read: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"OAuth client credentials at {src} must be a JSON object."
        )

    effective, data = _load_effective_yaml_dict(config_path)
    google = data.setdefault("google", {})
    if not isinstance(google, dict):
        google = {}
        data["google"] = google
    google["credentials"] = parsed
    google.pop("credentials_file", None)
    _atomic_write_validated(effective, data, config_path=config_path)
    return effective


def use_google_files(
    credentials_file: str | Path,
    *,
    token_file: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Path:
    """Switch Google auth to file mode and clear any inline credentials/token.

    Writes ``google.credentials_file`` (and ``google.token_file``) and removes the
    inline ``google.credentials``/``google.token`` mappings so the file pointers are
    the single source. ``token_file`` defaults to ``<credentials parent>/token.json``.
    """
    creds_path = Path(credentials_file)
    if token_file is None:
        token_path = creds_path.parent / "token.json"
    else:
        token_path = Path(token_file)

    effective, data = _load_effective_yaml_dict(config_path)
    google = data.setdefault("google", {})
    if not isinstance(google, dict):
        google = {}
        data["google"] = google
    google.pop("credentials", None)
    google.pop("token", None)
    google["credentials_file"] = str(creds_path)
    google["token_file"] = str(token_path)
    _atomic_write_validated(effective, data, config_path=config_path)
    return effective
