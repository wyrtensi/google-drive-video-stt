from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from src.presets import (
    BUILTIN_PRESETS,
    Preset,
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
    openai_max_parallel: int = 4
    deepgram_model: str = "nova-3"
    deepgram_diarize_model: str = "latest"
    deepgram_audio_source: str = "m4a_copy"
    deepgram_txt_formatter: str = "word_speaker"
    deepgram_keyterms_enabled: bool = True
    deepgram_keyterms_file: Path = DEEPGRAM_DEFAULT_KEYTERMS_FILE
    deepgram_keyterms: tuple[str, ...] = ()
    presets: tuple[Preset, ...] = ()


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
    config_presets = _as_mapping(raw.get("presets"), "presets")

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
        return str(path)
    return os.path.relpath(path, config_file.parent)


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
            "max_parallel": config.openai_max_parallel,
            "keypoints": config.openai_keypoints,
        },
        # Seed a presets block from the built-in keypoints pass. The DAG executor
        # (later tasks) reads this map; here it captures the current single-pass gate.
        "presets": {
            "keypoints": {"enabled": config.openai_keypoints},
        },
    }


def _dump_yaml(data: dict) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


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
        resolved.write_text(
            _dump_yaml(_config_to_yaml_dict(config, resolved)), encoding="utf-8"
        )
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
    resolved.write_text(
        _dump_yaml(_config_to_yaml_dict(config, resolved)), encoding="utf-8"
    )
    return resolved
