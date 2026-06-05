from __future__ import annotations

from dataclasses import dataclass, field
import logging
import json
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError
import requests

from src import drive, notify, output, postprocess, preset_pipeline
from src.auth import AuthError, build_drive_service
from src.config import Config, load_config
from src.extractor import extract_m4a_copy, extract_mp3
from src.presets import Preset
from src.stt.transcribe import transcribe_file

logger = logging.getLogger(__name__)

_TRANSIENT_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
_TRANSIENT_RETRY_ATTEMPTS = 3
_TRANSIENT_RETRY_DELAYS = (1.0, 2.0)


@dataclass
class _RetryState:
    retry_count: int = 0


@dataclass
class _ProcessTelemetry:
    provider: str
    processing_mode: str
    retry_count: int
    duration_s: float
    mp3_uploaded: bool = False
    txt_uploaded: bool = False
    cost_usd: dict[str, float | None] = field(default_factory=dict)
    usage: dict[str, dict[str, int]] = field(default_factory=dict)


def _http_status_code(exc: Exception) -> int | None:
    if isinstance(exc, HttpError):
        return getattr(exc.resp, "status", None)
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code
    return None


def _is_transient_runtime_error(exc: Exception) -> bool:
    if isinstance(exc, (RefreshError, AuthError)):
        return False
    if isinstance(exc, drive.DownloadIntegrityError):
        return True
    if isinstance(exc, (TimeoutError, requests.ConnectionError, requests.Timeout)):
        return True
    status = _http_status_code(exc)
    return status in _TRANSIENT_HTTP_STATUS_CODES


def _call_with_transient_retries(operation, *, description: str, retry_state: _RetryState | None = None):
    for attempt in range(1, _TRANSIENT_RETRY_ATTEMPTS + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            if attempt >= _TRANSIENT_RETRY_ATTEMPTS or not _is_transient_runtime_error(exc):
                raise
            if retry_state is not None:
                retry_state.retry_count += 1
            delay = _TRANSIENT_RETRY_DELAYS[min(attempt - 1, len(_TRANSIENT_RETRY_DELAYS) - 1)]
            logger.warning(
                "Transient error during %s (attempt %d/%d): %s; retrying in %.1fs",
                description,
                attempt,
                _TRANSIENT_RETRY_ATTEMPTS,
                exc,
                delay,
            )
            time.sleep(delay)


def _save_and_upload_txt(
    service: Any,
    source_file_id: str,
    mp4_name: str,
    text: str,
    folder_id: str,
    tmp_dir: Path,
    config: Config,
    *,
    txt_id: str | None = None,
) -> None:
    stem = drive.drive_stem(mp4_name)
    app_properties = {
        drive.SOURCE_VIDEO_ID_PROPERTY: source_file_id,
        drive.ARTIFACT_TYPE_PROPERTY: "txt",
    }
    output.write_artifact(
        service,
        base_name=stem,
        suffix=".txt",
        text=text,
        folder_id=folder_id,
        config=config,
        tmp_dir=tmp_dir,
        existing_id=txt_id,
        app_properties=app_properties,
        mime_type=drive.TXT_MIME,
    )


def _save_and_upload_preset(
    service: Any,
    source_file_id: str,
    mp4_name: str,
    preset: Preset,
    text: str,
    folder_id: str,
    tmp_dir: Path,
    config: Config,
    *,
    existing_id: str | None = None,
) -> None:
    stem = drive.drive_stem(mp4_name)
    app_properties = {
        drive.SOURCE_VIDEO_ID_PROPERTY: source_file_id,
        drive.ARTIFACT_TYPE_PROPERTY: preset.name,
    }
    output.write_artifact(
        service,
        base_name=stem,
        suffix=preset.artifact_suffix,
        text=text,
        folder_id=folder_id,
        config=config,
        tmp_dir=tmp_dir,
        existing_id=existing_id,
        app_properties=app_properties,
        mime_type=drive.MD_MIME,
    )


def _run_preset_stage(
    service: Any,
    source_file_id: str,
    mp4_name: str,
    transcript: str,
    folder_id: str,
    tmp_dir: Path,
    config: Config,
    *,
    speaker_names: list[str] | None,
    artifact_ids: dict[str, str],
    reprocess: bool,
    usage: dict[str, dict[str, int]],
) -> None:
    """Run the enabled preset DAG over a transcript and persist each new artifact.

    Only presets still missing an artifact are produced (``reprocess`` re-runs them
    all, overwriting in place). Successful, non-empty outputs are written as soon as
    the stage returns; if any preset failed, an aggregated error is raised. For
    Drive targets the file is re-selected on a later cycle (its ``.txt`` sibling is
    re-fed without re-running STT) so only the still-missing presets retry. Folder
    targets write preset artifacts to local disk, which ``list_folder_state`` does
    not track, so their preset stage runs once per transcription only.
    """
    preset_by_name = {preset.name: preset for preset in config.presets}
    if not preset_by_name:
        return
    if reprocess:
        missing = list(preset_by_name)
    else:
        missing = [name for name in preset_by_name if name not in artifact_ids]
    if not missing:
        return

    # Reuse dependency artifacts already persisted on Drive so a retry re-runs
    # only the still-missing presets (per the plan): a dependency that completed
    # on an earlier cycle is re-fed from its artifact instead of being re-run,
    # which avoids extra OpenAI spend and keeps dependent siblings consistent with
    # the dependency output that produced the earlier ones.
    precomputed: dict[str, str] = {}
    if not reprocess:
        for dep in preset_pipeline.dependency_names(config.presets, missing):
            existing_id = artifact_ids.get(dep)
            if existing_id is None:
                continue
            precomputed[dep] = _call_with_transient_retries(
                lambda existing_id=existing_id: drive.download_text(
                    service, existing_id
                ),
                description=f"download {dep} artifact for {mp4_name}",
            )

    results = preset_pipeline.run_presets(
        transcript,
        mp4_name,
        config,
        config.presets,
        speaker_names=speaker_names,
        only=missing,
        precomputed=precomputed,
    )
    for name in missing:
        result = results.get(name)
        if result is None or not result.ok or not result.text.strip():
            continue
        if result.usage:
            usage[f"openai_{name}"] = dict(result.usage)
        _save_and_upload_preset(
            service,
            source_file_id,
            mp4_name,
            preset_by_name[name],
            result.text,
            folder_id,
            tmp_dir,
            config,
            existing_id=artifact_ids.get(name),
        )

    aggregated = preset_pipeline.aggregate_error(results)
    if aggregated:
        raise RuntimeError(aggregated)


def _prepare_deepgram_audio(mp4_path: Path, config: Config) -> Path:
    if config.deepgram_audio_source == "m4a_copy":
        return extract_m4a_copy(mp4_path)
    if config.deepgram_audio_source == "mp3_96k":
        return extract_mp3(mp4_path, bitrate="96k")
    if config.deepgram_audio_source == "mp3_192k":
        return extract_mp3(mp4_path, bitrate="192k")
    raise RuntimeError(f"Unknown Deepgram audio source: {config.deepgram_audio_source}")


def _should_make_mp3_artifact(config: Config) -> bool:
    return config.drive_mp3_artifact


def _missing_preset_names(item: dict, config: Config) -> list[str]:
    """Enabled presets that have no artifact yet for this item."""
    artifact_ids = item.get("artifact_ids") or {}
    return [preset.name for preset in config.presets if preset.name not in artifact_ids]


def _needs_preset_reprocess(item: dict, config: Config, *, needs_txt: bool) -> bool:
    """Whether to re-run missing presets from an existing Drive transcript.

    Only applies when a Drive ``.txt`` sibling already exists (``txt_id``) and the
    transcript is not being regenerated this pass. Folder-mode transcripts have no
    ``txt_id`` (the ``.txt`` lives on local disk, not as a Drive sibling), so they
    are excluded — their preset artifacts are not tracked in ``artifact_ids`` and
    would otherwise reprocess on every cycle.
    """
    if needs_txt or not config.presets:
        return False
    if item.get("txt_id") is None:
        return False
    return bool(_missing_preset_names(item, config))


def _apply_local_output_state(items: list[dict], config: Config) -> list[dict]:
    """Reflect local artifacts in sibling flags when OUTPUT_TARGET=folder.

    In folder mode the .txt is written to OUTPUT_DIR instead of as a Drive
    sibling, so the Drive-derived ``has_txt`` flag never flips to True. Without
    this, the daemon would re-select the same source on every poll and re-run
    Deepgram (and OpenAI keypoints) indefinitely. Mark ``has_txt`` from the
    local output file so processing stays idempotent.
    """
    if config.output_target != "folder" or config.output_dir is None:
        return items
    for item in items:
        if item.get("has_txt"):
            continue
        stem = drive.drive_stem(item["file"]["name"])
        local_txt = config.output_dir / (drive.safe_local_name(stem) + ".txt")
        if local_txt.exists():
            item["has_txt"] = True
    return items


def _speaker_names_from_file_info(file_info: dict) -> list[str] | None:
    raw = file_info.get("appProperties", {}).get(drive.SPEAKER_NAMES_PROPERTY)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Ignoring invalid speaker_names appProperty on %s", file_info.get("id"))
        return None
    if not isinstance(parsed, list):
        return None
    names = [item.strip() for item in parsed if isinstance(item, str) and item.strip()]
    return names or None


def _coerce_size_bytes(raw: Any) -> int | None:
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _processing_provider(config: Config, *, needs_txt: bool) -> str:
    if needs_txt and config.stt_provider:
        return config.stt_provider
    return "artifact-only"


def _processing_mode(*, needs_mp3: bool, needs_txt: bool) -> str:
    if needs_mp3 and needs_txt:
        return "artifact-and-txt"
    if needs_mp3:
        return "artifact-only"
    return "txt-only"


def _processing_outcome(exc: Exception | None) -> str:
    if exc is None:
        return "success"
    return "failed"


def _cycle_outcome(*, dry_run: bool, failed: int, folder_errors: int) -> str:
    if dry_run:
        return "dry_run"
    if failed or folder_errors:
        return "partial_failure"
    return "success"


def _retry_count_from_process_result(result: Any) -> int:
    retry_count = getattr(result, "retry_count", None)
    return retry_count if isinstance(retry_count, int) else 0


def _retry_count_from_exception(exc: Exception) -> int:
    retry_count = getattr(exc, "gdstt_retry_count", None)
    return retry_count if isinstance(retry_count, int) else 0


def process_item(
    service: Any,
    item: dict,
    folder_id: str,
    config: Config,
    *,
    reprocess_txt: bool = False,
) -> _ProcessTelemetry | None:
    file_info = item["file"]
    file_id = file_info["id"]
    file_name = file_info["name"]
    file_size = _coerce_size_bytes(file_info.get("size"))
    has_mp3 = item.get("has_mp3", False)
    has_txt = item.get("has_txt", False)

    stt_enabled = bool(config.stt_provider)
    needs_mp3 = _should_make_mp3_artifact(config) and not has_mp3
    needs_txt = stt_enabled and (reprocess_txt or not has_txt)
    needs_presets = _needs_preset_reprocess(item, config, needs_txt=needs_txt)

    if not needs_mp3 and not needs_txt and not needs_presets:
        return

    provider = _processing_provider(config, needs_txt=needs_txt)
    processing_mode = _processing_mode(needs_mp3=needs_mp3, needs_txt=needs_txt)
    retry_state = _RetryState()
    started_at = time.monotonic()
    error: Exception | None = None

    logger.info(
        "Processing %s (id=%s) in folder %s [mp3=%s, txt=%s]",
        file_name, file_id, folder_id, "make" if needs_mp3 else "skip",
        "make" if needs_txt else "skip",
    )

    duration_s = 0.0
    cost_usd: dict[str, float | None] = {}
    usage: dict[str, dict[str, int]] = {}
    mp3_uploaded = False
    txt_uploaded = False

    try:
        with tempfile.TemporaryDirectory(prefix="gd-stt-") as tmp:
            tmp_dir = Path(tmp)
            mp4_path: Path | None = None
            mp3_path: Path | None = None

            if needs_mp3:
                mp4_path = _call_with_transient_retries(
                    lambda: drive.download(
                        service,
                        file_id,
                        tmp_dir,
                        file_name,
                        expected_size_bytes=file_size,
                    ),
                    description=f"download source file {file_name} ({file_id})",
                    retry_state=retry_state,
                )
                mp3_path = extract_mp3(mp4_path, bitrate=config.bitrate)
                mp3_drive_name = drive.drive_stem(file_name) + ".mp3"
                drive.upload(
                    service,
                    mp3_path,
                    folder_id,
                    mime_type=drive.MP3_MIME,
                    name=mp3_drive_name,
                    app_properties={
                        drive.SOURCE_VIDEO_ID_PROPERTY: file_id,
                        drive.ARTIFACT_TYPE_PROPERTY: "mp3",
                    },
                )
                mp3_uploaded = True
                logger.info("Uploaded %s to folder %s", mp3_drive_name, folder_id)

            if needs_txt:
                if mp4_path is None:
                    mp4_path = _call_with_transient_retries(
                        lambda: drive.download(
                            service,
                            file_id,
                            tmp_dir,
                            file_name,
                            expected_size_bytes=file_size,
                        ),
                        description=f"download source file {file_name} ({file_id})",
                        retry_state=retry_state,
                    )
                stt_audio_path = _prepare_deepgram_audio(mp4_path, config)
                text = transcribe_file(stt_audio_path, config, cost_usd=cost_usd)
                speaker_names = _speaker_names_from_file_info(file_info)
                if config.stt_postprocess:
                    text = postprocess.postprocess_transcript(
                        text,
                        file_name,
                        speaker_names=speaker_names,
                    )
                _save_and_upload_txt(
                    service, file_id, file_name, text, folder_id, tmp_dir, config,
                    txt_id=item.get("txt_id"),
                )
                txt_uploaded = True

                _run_preset_stage(
                    service,
                    file_id,
                    file_name,
                    text,
                    folder_id,
                    tmp_dir,
                    config,
                    speaker_names=speaker_names,
                    artifact_ids=item.get("artifact_ids") or {},
                    reprocess=reprocess_txt,
                    usage=usage,
                )
            elif needs_presets:
                # The transcript already exists on Drive; re-feed it to produce the
                # still-missing presets (a failed earlier preset or a newly added
                # one) without re-running STT.
                text = _call_with_transient_retries(
                    lambda: drive.download_text(service, item["txt_id"]),
                    description=f"download transcript for {file_name} ({file_id})",
                    retry_state=retry_state,
                )
                speaker_names = _speaker_names_from_file_info(file_info)
                _run_preset_stage(
                    service,
                    file_id,
                    file_name,
                    text,
                    folder_id,
                    tmp_dir,
                    config,
                    speaker_names=speaker_names,
                    artifact_ids=item.get("artifact_ids") or {},
                    reprocess=False,
                    usage=usage,
                )
    except Exception as exc:
        error = exc
        setattr(exc, "gdstt_retry_count", retry_state.retry_count)
        raise
    finally:
        duration_s = time.monotonic() - started_at
        logger.info(
            "Process summary [file=%s, file_id=%s, folder=%s, provider=%s, processing_mode=%s, "
            "outcome=%s, retry_count=%d, duration_s=%.3f, cost_usd=%s, usage=%s]",
            file_name,
            file_id,
            folder_id,
            provider,
            processing_mode,
            _processing_outcome(error),
            retry_state.retry_count,
            duration_s,
            cost_usd,
            usage,
        )

    return _ProcessTelemetry(
        provider=provider,
        processing_mode=processing_mode,
        retry_count=retry_state.retry_count,
        duration_s=duration_s,
        mp3_uploaded=mp3_uploaded,
        txt_uploaded=txt_uploaded,
        cost_usd=cost_usd,
        usage=usage,
    )


def _pending_items(items: list[dict], config: Config) -> list[dict]:
    stt_enabled = bool(config.stt_provider)
    pending = []
    for item in items:
        needs_txt = stt_enabled and not item.get("has_txt")
        if (
            (_should_make_mp3_artifact(config) and not item.get("has_mp3"))
            or needs_txt
            or _needs_preset_reprocess(item, config, needs_txt=needs_txt)
        ):
            pending.append(item)
    return pending


def _file_size_bytes(item: dict) -> int | None:
    return _coerce_size_bytes(item.get("file", {}).get("size"))


def _format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    amount = float(value)
    for unit in units:
        if amount < 1000 or unit == units[-1]:
            if unit == "B":
                return f"{value} B"
            return f"{amount:.1f} {unit}"
        amount /= 1000
    return f"{value} B"


def _items_allowed_by_size(
    items: list[dict],
    *,
    max_size_bytes: int | None,
    confirm_large: bool,
) -> list[dict]:
    if max_size_bytes is None or confirm_large:
        return items

    allowed = []
    for item in items:
        file_info = item.get("file", {})
        size = _file_size_bytes(item)
        if size is not None and size > max_size_bytes:
            logger.warning(
                "Skipping %s (id=%s): size %s exceeds --max-size %s; "
                "pass --confirm-large to process it",
                file_info.get("name"),
                file_info.get("id"),
                _format_bytes(size),
                _format_bytes(max_size_bytes),
            )
            continue
        allowed.append(item)
    return allowed


def _dry_run_preset_names(item: dict, config: Config, *, needs_txt: bool, reprocess_txt: bool) -> list[str]:
    """Preset artifacts a real run would generate for this item.

    Mirrors :func:`_run_preset_stage`: ``--reprocess-txt`` regenerates every
    enabled preset, a fresh or reprocessable transcript generates the presets
    still missing an artifact, and otherwise no preset work happens.
    """
    if not config.presets:
        return []
    if reprocess_txt:
        return [preset.name for preset in config.presets]
    if needs_txt or _needs_preset_reprocess(item, config, needs_txt=needs_txt):
        return _missing_preset_names(item, config)
    return []


def _log_dry_run(folder_id: str, item: dict, config: Config, *, reprocess_txt: bool) -> None:
    file_info = item["file"]
    has_mp3 = item.get("has_mp3", False)
    has_txt = item.get("has_txt", False)
    needs_mp3 = _should_make_mp3_artifact(config) and not has_mp3
    needs_txt = bool(config.stt_provider) and (reprocess_txt or not has_txt)
    preset_names = _dry_run_preset_names(
        item, config, needs_txt=needs_txt, reprocess_txt=reprocess_txt
    )
    logger.info(
        "DRY RUN: would process %s (id=%s) in folder %s [mp3=%s, txt=%s, presets=%s]",
        file_info["name"],
        file_info["id"],
        folder_id,
        "make" if needs_mp3 else "skip",
        "make" if needs_txt else "skip",
        ",".join(preset_names) if preset_names else "skip",
    )


def process_target(
    service: Any,
    target_id: str,
    config: Config,
    *,
    is_folder: bool | None = None,
    reprocess_txt: bool = False,
    dry_run: bool = False,
    max_size_bytes: int | None = None,
    confirm_large: bool = False,
) -> list[_ProcessTelemetry]:
    """Process a single Drive file or every pending file in a folder, on demand."""
    meta = _call_with_transient_retries(
        lambda: drive.get_file_metadata(service, target_id),
        description=f"get metadata for {target_id}",
    )
    mime = meta.get("mimeType", "")
    treat_as_folder = is_folder if is_folder is not None else mime == drive.FOLDER_MIME

    if treat_as_folder:
        telemetry: list[_ProcessTelemetry] = []
        items = _call_with_transient_retries(
            lambda: drive.list_folder_state(service, target_id),
            description=f"list folder state for {target_id}",
        )
        _apply_local_output_state(items, config)
        pending = items if reprocess_txt else _pending_items(items, config)
        pending = _items_allowed_by_size(
            pending,
            max_size_bytes=max_size_bytes,
            confirm_large=confirm_large,
        )
        logger.info("Folder %s: %d pending file(s)", target_id, len(pending))
        if dry_run:
            for item in pending:
                _log_dry_run(target_id, item, config, reprocess_txt=reprocess_txt)
            return telemetry
        for item in pending:
            result = process_item(
                service,
                item,
                target_id,
                config,
                reprocess_txt=reprocess_txt,
            )
            if result is not None:
                telemetry.append(result)
        return telemetry

    parents = meta.get("parents") or []
    if not parents:
        raise RuntimeError(f"File {target_id} has no parent folder")
    folder_id = parents[0]
    items = _call_with_transient_retries(
        lambda: drive.list_folder_state(service, folder_id),
        description=f"list folder state for {folder_id}",
    )
    _apply_local_output_state(items, config)
    match = next(
        (it for it in items if it["file"]["id"] == target_id), None
    )
    if match is None:
        raise RuntimeError(
            f"File {target_id} is not an MP4 in folder {folder_id}"
        )
    allowed = _items_allowed_by_size(
        [match],
        max_size_bytes=max_size_bytes,
        confirm_large=confirm_large,
    )
    if not allowed:
        return []
    if dry_run:
        _log_dry_run(folder_id, match, config, reprocess_txt=reprocess_txt)
        return []
    result = process_item(service, match, folder_id, config, reprocess_txt=reprocess_txt)
    return [result] if result is not None else []


def run_once(
    service: Any,
    config: Config,
    *,
    dry_run: bool = False,
    max_size_bytes: int | None = None,
    confirm_large: bool = False,
) -> None:
    cycle_started_at = time.monotonic()
    cycle_pending = 0
    cycle_processed = 0
    cycle_failed = 0
    cycle_retry_total = 0
    cycle_skipped_size = 0
    cycle_folder_errors = 0

    for folder_id in config.folder_ids:
        listing_retry_state = _RetryState()
        try:
            items = _call_with_transient_retries(
                lambda: drive.list_folder_state(service, folder_id),
                description=f"list folder state for {folder_id}",
                retry_state=listing_retry_state,
            )
        except (RefreshError, AuthError):
            raise
        except Exception as exc:
            cycle_folder_errors += 1
            logger.exception("Failed to list folder %s", folder_id)
            notify.notify_error(
                f"Failed to list folder {folder_id}: {exc}\n{traceback.format_exc()}",
                proxy_url=config.proxy_url,
            )
            continue
        finally:
            cycle_retry_total += listing_retry_state.retry_count

        _apply_local_output_state(items, config)
        pending = _pending_items(items, config)
        pending_before_size = len(pending)
        pending = _items_allowed_by_size(
            pending,
            max_size_bytes=max_size_bytes,
            confirm_large=confirm_large,
        )
        skipped_size = pending_before_size - len(pending)
        cycle_pending += len(pending)
        cycle_skipped_size += skipped_size
        logger.info(
            "Folder %s summary [total=%d, pending=%d, skipped_size=%d, dry_run=%s]",
            folder_id,
            len(items),
            len(pending),
            skipped_size,
            dry_run,
        )
        if dry_run:
            for item in pending:
                _log_dry_run(folder_id, item, config, reprocess_txt=False)
            continue
        for item in pending:
            try:
                telemetry = process_item(service, item, folder_id, config)
                cycle_processed += 1
                cycle_retry_total += _retry_count_from_process_result(telemetry)
            except (RefreshError, AuthError):
                raise
            except Exception as exc:
                cycle_failed += 1
                cycle_retry_total += _retry_count_from_exception(exc)
                file_name = item.get("file", {}).get("name")
                logger.exception(
                    "Failed to process %s in folder %s", file_name, folder_id
                )
                notify.notify_error(
                    f"Failed to process {file_name} in {folder_id}: {exc}\n"
                    f"{traceback.format_exc()}",
                    proxy_url=config.proxy_url,
                )

    logger.info(
        "Cycle summary [provider=%s, outcome=%s, folders=%d, pending=%d, processed=%d, failed=%d, "
        "retry_total=%d, skipped_size=%d, folder_errors=%d, dry_run=%s, duration_s=%.3f]",
        config.stt_provider or "artifact-only",
        _cycle_outcome(
            dry_run=dry_run,
            failed=cycle_failed,
            folder_errors=cycle_folder_errors,
        ),
        len(config.folder_ids),
        cycle_pending,
        cycle_processed,
        cycle_failed,
        cycle_retry_total,
        cycle_skipped_size,
        cycle_folder_errors,
        dry_run,
        time.monotonic() - cycle_started_at,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = load_config()
    if not config.folder_ids:
        logger.error("FOLDER_IDS is empty; set it in the environment to start polling")
        raise SystemExit(1)

    try:
        service = build_drive_service(config=config)
    except (RefreshError, AuthError) as exc:
        logger.exception("OAuth bootstrap failed; exiting for restart")
        notify.notify_error(
            f"OAuth bootstrap failed; container will exit so it can be restarted "
            f"after re-running `python -m src.auth`: {exc}",
            proxy_url=config.proxy_url,
        )
        raise SystemExit(1) from exc

    while True:
        try:
            run_once(service, config)
        except (RefreshError, AuthError) as exc:
            logger.exception("OAuth refresh failed; exiting for restart")
            notify.notify_error(
                f"OAuth refresh failed; container will exit so it can be restarted "
                f"after re-running `python -m src.auth`: {exc}",
                proxy_url=config.proxy_url,
            )
            raise SystemExit(1) from exc
        except Exception as exc:
            logger.exception("Cycle failed")
            notify.notify_error(
                f"Cycle failed: {exc}\n{traceback.format_exc()}",
                proxy_url=config.proxy_url,
            )
        time.sleep(config.poll_interval)


if __name__ == "__main__":
    main()
