from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import logging
import json
import re
import ssl
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError
import requests

from src import (
    booking_gate,
    booking_server,
    change_cursor,
    drive,
    meet_transcript as meet_transcript_module,
    meta as meta_module,
    meta_doc,
    meta_entity,
    notify,
    output,
    planfix,
    planfix_html,
    postprocess,
    preset_pipeline,
    speaker_roles,
    stt_document,
    webhook,
)
from src.auth import AuthError, build_drive_service
from src.config import Config, is_run_enabled, load_config, parse_since
from src.meeting_time import parse_meeting_start
from src.extractor import extract_m4a_copy, extract_mp3
from src.openai_pipeline import OpenAIPipeline
from src.presets import Preset
from src.stt.transcribe import transcribe_file

logger = logging.getLogger(__name__)

_TRANSIENT_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
# Drive's answer when a saved cursor has aged out of its journal. Not a failure to
# report: it is the documented way of being told to start over.
_STALE_CURSOR_HTTP_STATUS_CODES = {404, 410}
# How long a video with no videoMediaMetadata is assumed to be still uploading rather
# than simply never getting any. Generous on purpose: the cost of waiting is one more
# cycle, the cost of giving up too early is a download of a half-written file.
_MEDIA_SETTLING_GRACE = timedelta(hours=2)
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
    transcript: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    # The merged meta document `_write_call_documents` built this cycle (None when no
    # preset stage ran). Task 6 reads this to quote the meta fields into the Planfix
    # comment instead of re-parsing the `meta` artifact a second time.
    meta_document: dict[str, object] | None = None


def _http_status_code(exc: Exception) -> int | None:
    if isinstance(exc, HttpError):
        return getattr(exc.resp, "status", None)
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code
    return None


def _is_rejected_cursor(exc: Exception) -> bool:
    """Whether Drive refused the saved cursor itself, rather than failing the read.

    An expired cursor gets 404 or 410. A malformed one gets 400 instead, with the
    error pinned to the ``pageToken`` parameter -- found by writing a corrupt token
    into the cursor file on a live Drive. Read as an ordinary feed failure, that 400
    held the cursor, and a held cursor is the same bad token on the next cycle: the
    service failed every cycle for good while the module promised that a corrupt
    cursor costs one sweep.

    Matching the parameter rather than 400 alone keeps a genuinely broken request
    -- a bad ``fields`` after a code change, say -- surfacing as the failure it is
    instead of being swept over quietly every cycle.
    """
    status = _http_status_code(exc)
    if status in _STALE_CURSOR_HTTP_STATUS_CODES:
        return True
    if status != 400 or not isinstance(exc, HttpError):
        return False
    try:
        details = json.loads(exc.content.decode("utf-8"))["error"]["errors"]
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return any(
        isinstance(detail, dict) and detail.get("location") == "pageToken"
        for detail in details
    )


def _is_transient_runtime_error(exc: Exception) -> bool:
    if isinstance(exc, (RefreshError, AuthError)):
        return False
    if isinstance(exc, drive.DownloadIntegrityError):
        return True
    # requests' exceptions only cover our own HTTP calls. The Google API client runs on
    # httplib2, which lets socket and TLS failures through as builtins -- a dropped
    # keep-alive connection arrives as BrokenPipeError, unrelated to requests.ConnectionError.
    # Without the builtins here those never reach the retry path and a routine reconnect
    # escalates into a failed cycle and an alert.
    if isinstance(
        exc,
        (TimeoutError, ConnectionError, ssl.SSLError, requests.ConnectionError, requests.Timeout),
    ):
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
    container_id: str,
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
        folder_id=container_id,
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
    container_id: str,
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
        folder_id=container_id,
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
    container_id: str,
    tmp_dir: Path,
    config: Config,
    *,
    speaker_names: list[str] | None,
    artifact_ids: dict[str, str],
    reprocess: bool,
    usage: dict[str, dict[str, int]],
    unproduced: set[str],
    local_artifact_paths: dict[str, Path] | None = None,
    only_presets: list[str] | None = None,
) -> dict[str, str]:
    """Run the enabled preset DAG over a transcript and persist each new artifact.

    Returns every enabled preset's text keyed by preset name — freshly produced ones
    plus any that completed on an earlier cycle, read back from their artifacts — so
    callers (the completion webhook) ship a file's full set of outputs even though
    only the still-missing presets were run. The earlier-cycle backfill costs a Drive
    read per artifact and only the webhook consumes it, so it is skipped entirely
    when ``webhook.url`` is unset; the return is then this cycle's presets alone.

    Only presets still missing an artifact are produced (``reprocess`` re-runs them
    all, overwriting in place). Successful, non-empty outputs are written as soon as
    the stage returns; if any preset failed, an aggregated error is raised. For
    Drive targets the file is re-selected on a later cycle (its ``.txt`` sibling is
    re-fed without re-running STT) so only the still-missing presets retry. Folder
    targets write preset artifacts to local disk, which ``list_folder_state`` does
    not track, so their preset stage runs once per transcription only.

    ``unproduced`` collects presets that ran without error but returned blank text, so
    no artifact was written. Those files come back next cycle, so the caller must not
    report this pass as the file's completion.
    """
    preset_by_name = {preset.name: preset for preset in config.presets}
    if not preset_by_name:
        return {}
    local_artifact_paths = local_artifact_paths or {}
    existing_names = set(artifact_ids) | set(local_artifact_paths)
    if only_presets is not None:
        # Force-rerun an explicit set of stages (``gdstt reprocess``); their
        # dependencies are reused from existing artifacts below rather than re-run.
        missing = [name for name in only_presets if name in preset_by_name]
    elif reprocess:
        missing = list(preset_by_name)
    else:
        missing = [name for name in preset_by_name if name not in existing_names]

    # Reuse dependency artifacts already persisted on Drive so a retry re-runs
    # only the still-missing presets (per the plan): a dependency that completed
    # on an earlier cycle is re-fed from its artifact instead of being re-run,
    # which avoids extra OpenAI spend and keeps dependent siblings consistent with
    # the dependency output that produced the earlier ones.
    def load_existing(name: str) -> str | None:
        existing_id = artifact_ids.get(name)
        if existing_id is not None:
            return _call_with_transient_retries(
                lambda: drive.download_text(service, existing_id),
                description=f"download {name} artifact for {mp4_name}",
            )
        local_path = local_artifact_paths.get(name)
        if local_path is not None:
            return local_path.read_text(encoding="utf-8")
        return None

    # Every webhook POST carries a file's full artifact set, so a preset that
    # succeeded on an earlier cycle — and is therefore not re-run here — still has
    # to reach the receiver.
    # Backfilling it costs a Drive download apiece, and the completion webhook is
    # this data's only consumer, so skip the reads outright when no receiver is
    # configured (``notify_complete`` would discard them on its blank-URL return).
    # These reads only enrich the payload, so they must never fail the file: every
    # artifact is already persisted by the time this runs, and raising here would
    # both alert on a good record and — since the next cycle sees no missing presets
    # — leave the webhook permanently undelivered. Degrade to a partial payload.
    def backfill(
        produced: dict[str, str], precomputed: dict[str, str]
    ) -> dict[str, str]:
        if not config.webhook_url.strip():
            return produced
        for name in preset_by_name:
            if name in produced:
                continue
            text = precomputed.get(name)
            if text is None:
                try:
                    text = load_existing(name)
                except Exception as exc:
                    logger.warning(
                        "Webhook backfill skipped [preset=%s, file=%s]: %s",
                        name,
                        mp4_name,
                        type(exc).__name__,
                    )
                    continue
            if text is not None and text.strip():
                produced[name] = text
        return produced

    if not missing:
        # Every preset already has an artifact, so nothing is re-run — but the file
        # can still reach the webhook (its ``.txt`` was regenerated this cycle), and
        # the receiver expects the full set, so the artifacts are read back.
        return backfill({}, {})

    precomputed: dict[str, str] = {}
    if not reprocess:
        for dep in preset_pipeline.dependency_names(config.presets, missing):
            text = load_existing(dep)
            if text is not None:
                precomputed[dep] = text

    employee = config.folder_by_id(folder_id)
    results = preset_pipeline.run_presets(
        transcript,
        mp4_name,
        config,
        config.presets,
        speaker_names=speaker_names,
        manager_name=employee.name if employee else "",
        only=missing,
        precomputed=precomputed,
    )
    generated_names = set(results) - set(precomputed)
    names_to_save = set(missing) | (generated_names - existing_names)
    ordered_names = [
        name
        for name in preset_pipeline.topological_order(config.presets)
        if name in names_to_save
    ]
    for name in ordered_names:
        result = results.get(name)
        if result is None or not result.ok:
            continue
        if not result.text.strip():
            # Blank output writes no artifact (by design — a blank doc is worthless),
            # so the preset stays "missing" and the file is re-selected next cycle.
            # Record it: the webhook must not treat this pass as the file's completion
            # and re-POST the transcript on every cycle from here on.
            unproduced.add(name)
            logger.warning(
                "Preset %s returned empty output for %s; no artifact written",
                name,
                mp4_name,
            )
            continue
        if result.usage:
            usage[f"openai_{name}"] = dict(result.usage)
        _save_and_upload_preset(
            service,
            source_file_id,
            mp4_name,
            preset_by_name[name],
            result.text,
            container_id,
            tmp_dir,
            config,
            existing_id=artifact_ids.get(name),
        )

    aggregated = preset_pipeline.aggregate_error(results)
    if aggregated:
        raise RuntimeError(aggregated)

    produced = {
        name: result.text
        for name, result in results.items()
        if result.ok and result.text.strip()
    }
    return backfill(produced, precomputed)


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


def _local_artifact_path(config: Config, mp4_name: str, suffix: str) -> Path | None:
    if config.output_target != "folder" or config.output_dir is None:
        return None
    stem = drive.drive_stem(mp4_name)
    return config.output_dir / (drive.safe_local_name(stem) + suffix)


def _artifact_text(
    name: str, artifacts: dict[str, str], config: Config, mp4_name: str
) -> str:
    """This cycle's text for a preset, or the artifact an earlier cycle left on disk.

    A cycle that re-ran only the still-missing presets returns just those, so the
    document would otherwise lose the sections that completed earlier.
    """
    text = artifacts.get(name, "")
    if text.strip():
        return text
    preset = next((p for p in config.presets if p.name == name), None)
    if preset is None:
        return ""
    path = _local_artifact_path(config, mp4_name, preset.artifact_suffix)
    if path is None or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read %s for the .stt document: %s", path, type(exc).__name__)
        return ""


def _write_call_documents(
    service: Any,
    file_id: str,
    file_name: str,
    folder_id: str,
    container_id: str,
    transcript: str,
    artifacts: dict[str, str],
    config: Config,
    tmp_dir: Path,
    *,
    item: dict,
    booking_decision: booking_gate.BookingDecision,
) -> dict[str, object] | None:
    """Write ``<stem>.meta.yml`` and ``<stem>.stt`` for one recording.

    Returns the meta document so the Planfix comment can quote from it. Both files are
    written from artifacts that already exist, so this costs no model call. Neither file
    takes part in the processed/pending bookkeeping: only ``.txt`` and the preset
    artifacts decide whether a recording still needs work, and a deleted ``.stt`` must
    never put a recording back into the transcription queue.
    """
    stem = drive.drive_stem(file_name)
    values = meta_module.parse_meta(
        _artifact_text("meta", artifacts, config, file_name),
        config.meta_entities,
    )
    task_id = booking_decision.task_id or str(item.get("planfix_comment_task_id") or "")
    document = meta_doc.build(
        values=values,
        file_id=file_id,
        file_name=file_name,
        folder_id=folder_id,
        config=config,
        transcript=transcript,
        planfix_task_id=task_id,
        processed_at=datetime.now(timezone.utc),
    )
    meta_yaml = meta_doc.to_yaml(document, config.meta_entities)

    body = _artifact_text("transcript-cleanup", artifacts, config, file_name) or transcript
    text = stt_document.assemble(
        title=stem,
        sections=[
            _artifact_text(name, artifacts, config, file_name) for name in config.stt_presets
        ],
        meta_yaml=meta_yaml,
        transcript=body,
    )

    output.write_artifact(
        service, base_name=stem, suffix=".meta.yml", text=meta_yaml,
        folder_id=container_id, config=config, tmp_dir=tmp_dir,
        existing_id=item.get("meta_yml_id"),
    )
    output.write_artifact(
        service, base_name=stem, suffix=".stt", text=text, folder_id=container_id,
        config=config, tmp_dir=tmp_dir,
        existing_id=item.get("stt_id"),
        # No source_video_id: the transcript (`.txt`) is looked up on Drive by that
        # same appProperty, and `.stt` also uploads as text/plain, so carrying it
        # here would risk the `.stt` winning that lookup and being fed to the preset
        # stage -- or overwritten -- as if it were the transcript. `drive.py` also
        # excludes `.stt`/`.meta.yml` from that lookup by name, belt and suspenders.
        app_properties={drive.ARTIFACT_TYPE_PROPERTY: "stt"},
        mime_type=drive.TXT_MIME,
    )
    return document


def _try_write_call_documents(
    service: Any,
    file_id: str,
    file_name: str,
    folder_id: str,
    container_id: str,
    transcript: str,
    artifacts: dict[str, str],
    config: Config,
    tmp_dir: Path,
    *,
    item: dict,
    booking_decision: booking_gate.BookingDecision,
) -> dict[str, object] | None:
    """Call ``_write_call_documents``, degrading to no document on failure.

    The ``.stt``/``.meta.yml`` write happens after the ``.txt`` and every preset
    artifact are already persisted. Letting a write failure here propagate would
    re-raise out of ``process_item`` and leave the recording looking fully
    processed on the next cycle (``has_txt`` true, no missing presets) -- so the
    webhook and the Planfix comment, which run after this returns, would never
    fire, and no later cycle would retry them. A document that "takes no part in
    the bookkeeping" by design must not be able to fail a recording that already
    transcribed successfully.
    """
    try:
        return _write_call_documents(
            service, file_id, file_name, folder_id, container_id, transcript, artifacts,
            config, tmp_dir, item=item, booking_decision=booking_decision,
        )
    except Exception as exc:
        logger.warning(
            "Could not write the .stt/.meta.yml documents for %s: %s",
            file_name, type(exc).__name__,
        )
        return None


def _local_artifact_paths(item: dict) -> dict[str, Path]:
    paths = item.get("local_artifact_paths") or {}
    return {name: Path(path) for name, path in paths.items()}


def _existing_preset_names(item: dict) -> set[str]:
    artifact_ids = item.get("artifact_ids") or {}
    return set(artifact_ids) | set(_local_artifact_paths(item))


def _missing_preset_names(item: dict, config: Config) -> list[str]:
    """Enabled presets that have no artifact yet for this item."""
    existing = _existing_preset_names(item)
    return [preset.name for preset in config.presets if preset.name not in existing]


def _has_existing_transcript(item: dict) -> bool:
    return item.get("txt_id") is not None or item.get("local_txt_path") is not None


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
    if not _has_existing_transcript(item):
        return False
    return bool(_missing_preset_names(item, config))


def _apply_local_output_state(items: list[dict], config: Config) -> list[dict]:
    """Reflect local artifacts in sibling flags when output.target=folder.

    In folder mode the .txt is written to output.dir instead of as a Drive
    sibling, so the Drive-derived ``has_txt`` flag never flips to True. Without
    this, the daemon would re-select the same source on every poll and re-run
    Deepgram (and OpenAI keypoints) indefinitely. Mark ``has_txt`` from the
    local output file so processing stays idempotent.
    """
    if config.output_target != "folder" or config.output_dir is None:
        return items
    for item in items:
        file_name = item["file"]["name"]
        local_txt = _local_artifact_path(config, file_name, ".txt")
        if local_txt.exists():
            item["has_txt"] = True
            item["local_txt_path"] = local_txt
        local_paths = _local_artifact_paths(item)
        for preset in config.presets:
            local_artifact = _local_artifact_path(config, file_name, preset.artifact_suffix)
            if local_artifact is not None and local_artifact.exists():
                local_paths[preset.name] = local_artifact
        if local_paths:
            item["local_artifact_paths"] = local_paths
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


def _read_meet_transcript(
    service: Any, container_id: str, file_name: str
) -> tuple[list[str], str] | None:
    """Who Meet says was on this call and the transcript itself, or ``None``.

    Meet writes a transcript next to every recording and names the people in it. That
    closes the one gap the recording's own name cannot: a call started outside the
    calendar is named after the meeting room, so there is nothing in it to read and the
    speakers stay ``Speaker 1`` / ``Speaker 2``. Even when the name does carry names it
    carries the ones the calendar invite used, which is how "Viktoriia" arrives without
    a surname. The text travels with the names because its turns are the model's best
    evidence of who is who.

    Failure here is not failure of the recording: no transcript, no access to it, or a
    shape this cannot read all return ``None`` and leave the existing name parsing in
    charge.
    """
    try:
        doc = drive.find_meet_transcript(service, container_id, file_name)
        if doc is None:
            return None
        text = drive.export_document_text(service, doc["id"])
    except (RefreshError, AuthError):
        raise
    except Exception:
        logger.info(
            "Could not read Meet's transcript beside %s; falling back to the file name",
            file_name,
            exc_info=True,
        )
        return None

    names = meet_transcript_module.participants(text)
    if len(names) < 2:
        return None
    logger.info("Meet's transcript names %s for %s", names, file_name)
    return names, text


def _resolve_speaker_names(
    transcript: str,
    file_name: str,
    folder_id: str,
    config: Config,
    *,
    usage: dict[str, dict[str, int]] | None = None,
    candidates: list[str] | None = None,
    meet_text: str = "",
) -> list[str] | None:
    """Ask the model which diarized speaker is which participant.

    Without this the names extracted from the file name are bound to speakers by who
    talks first, which silently swaps the pair on every call the client opens. The
    model gets the opening minutes of the transcript, Meet's own turns for the same
    minutes when there are any, the folder's owner and the name the calendar title
    marked with the company.

    Returns the names in speaker order when the model placed them, and ``[]`` -- leave
    the speakers numbered -- when it was asked and did not: binding the names by
    position instead would be right only when the manager happens to speak first, and
    wrong silently. ``None`` means the model was never asked (no key, fewer than two
    names), and the caller keeps binding the file name's names by position, as it
    always did without a model.
    """
    if not config.openai_api_key:
        return None
    if candidates is None:
        candidates = postprocess.extract_interlocutor_names(file_name)
    if len(candidates) < 2:
        return None

    employee = config.folder_by_id(folder_id)
    calendar_manager, _ = postprocess.split_participants(file_name)
    pipeline = OpenAIPipeline(
        api_key=config.openai_api_key,
        model=config.openai_model,
        proxy_url=config.proxy_url,
    )
    try:
        names = speaker_roles.resolve(
            postprocess.clean_transcript(transcript),
            candidates=candidates,
            manager_name=employee.name if employee else "",
            run=pipeline.run,
            meet_text=meet_text,
            calendar_manager=calendar_manager,
        )
    finally:
        if usage is not None and pipeline.last_usage:
            usage["openai_speaker_roles"] = dict(pipeline.last_usage)
        pipeline.close()

    if names is None:
        logger.info("Speaker roles unresolved for %s; leaving the speakers numbered", file_name)
        return []
    logger.info("Speaker roles resolved for %s", file_name)
    return names


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


def _webhook_payload(
    file_id: str,
    file_name: str,
    folder_id: str,
    config: Config,
    transcript: str,
    artifacts: dict[str, str],
) -> dict:
    """Build the completion-webhook body.

    Non-``meta`` presets pass through as raw text keyed by preset name, so adding a
    preset to config.yml extends the payload with no code change. ``meta`` is parsed
    into one key per configured entity (``config.meta_entities``) -- the built-in
    ``{subject, tags, referral, referral_note}`` for an operator who hasn't declared
    ``meta.entities``, or whatever else that config names instead. Enum values are
    filtered to each entity's allow-list. An unknown employee sends empty strings
    rather than omitting the key.
    """
    employee = config.folder_by_id(folder_id)
    payload_artifacts: dict[str, object] = dict(artifacts)
    meta_text = artifacts.get("meta")
    if meta_text is not None:
        parsed = meta_module.parse_meta(meta_text, config.meta_entities)
        payload_artifacts["meta"] = dict(parsed)

    return {
        "file": {"id": file_id, "name": file_name, "folder_id": folder_id},
        "employee": {
            "name": employee.name if employee else "",
            "email": employee.email if employee else "",
        },
        "transcript": transcript,
        "artifacts": payload_artifacts,
    }


# Labels for the meta-document fields the code fills in itself. Entity labels come
# from the entities -- see MetaEntity.planfix_label.
_PLANFIX_CODE_LABELS = {
    "manager": "Менеджер",
    "client": "Клиент",
    "date": "Дата",
    "duration": "Длительность",
    "video_url": "Запись",
}


def _planfix_labels(
    entities: tuple[meta_entity.MetaEntity, ...],
) -> dict[str, str]:
    """Every field that may appear in the comment header, mapped to its label."""
    labels = dict(_PLANFIX_CODE_LABELS)
    for entity in entities:
        labels[entity.name] = entity.planfix_label
    return labels


# Markers prepended to the keypoints headings in the Planfix comment, so the three
# sections are tellable apart while scrolling a CRM feed. They live here and not in the
# preset prompt on purpose: the `.keypoints.md` artifact and the `.stt` document are
# read as documents and parsed by other tools, where a symbol in a heading is noise.
# A heading this map does not name is left exactly as the preset wrote it.
_PLANFIX_SECTION_MARKERS = {
    "Задачи": "☑️",
    "Тезисы": "📝",
    "Открытые вопросы": "❓",
}

_MARKDOWN_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})[ \t]+(?P<title>.+?)[ \t]*$", re.MULTILINE)


def _mark_planfix_sections(text: str) -> str:
    """Mark the known keypoints headings and give each one air, for the CRM comment only.

    A marked heading is surrounded by blank lines: run together, the three sections read
    as one wall of text in a Planfix feed. Sub-headings the preset emits per assignee
    (``### Mels``) are names, not section titles, so they miss the map and pass through
    untouched, tight against the section they belong to.
    """
    gap = planfix_html.SECTION_BREAK

    def _mark(match: re.Match[str]) -> str:
        marker = _PLANFIX_SECTION_MARKERS.get(match.group("title"))
        if marker is None:
            return match.group(0)
        heading = f"{match.group('hashes')} {marker} {match.group('title')}"
        return f"{gap}\n\n{heading}\n\n{gap}"

    return _MARKDOWN_HEADING_RE.sub(_mark, text)


def _planfix_meta_lines(
    document: dict[str, object] | None,
    fields: tuple[str, ...],
    entities: tuple[meta_entity.MetaEntity, ...],
) -> list[str]:
    """Render the selected meta fields as Markdown lines for the comment header.

    A field that is empty, or is not in ``labels`` at all (an unknown name -- e.g.
    a stale entry left in ``planfix.meta_fields`` after an entity was removed), is
    skipped silently, so shortening the configured list or a call with no referral
    never leaves a dangling label. That is a different case from a field whose
    *label* is the empty string: such a field is rendered bold with no label
    instead of being skipped; the first one encountered in ``fields`` is hoisted to
    the top as the comment's heading, and any further ones follow in place as bold
    lines.
    """
    if not document:
        return []
    labels = _planfix_labels(entities)
    lines: list[str] = []
    heading_taken = False
    for field_name in fields:
        if field_name not in labels:
            continue
        value = document.get(field_name)
        if isinstance(value, list):
            value = ", ".join(str(entry) for entry in value)
        # Collapse embedded newlines (free LLM text can carry them): markdown_to_html
        # splits on "\n", so an unnormalised value would fracture the header into
        # extra, label-less paragraphs -- or a stray bullet/heading if the
        # continuation happened to start with "- " or "#".
        text = " ".join(str(value or "").split())
        if not text:
            continue
        label = labels[field_name]
        if not label:
            if heading_taken:
                lines.append(f"**{text}**")
            else:
                lines.insert(0, f"**{text}**")
                heading_taken = True
        elif field_name == "video_url":
            # source_name is excluded from the default field list *because* it becomes
            # this anchor's text instead of a line of its own; fall back to the fixed
            # label when the document carries no source_name (or an empty one).
            anchor = str(document.get("source_name") or "").strip()
            anchor = " ".join(anchor.split()) or label
            lines.append(f"[{anchor}]({text})")
        else:
            lines.append(f"**{label}:** {text}")
    return lines


_MARKDOWN_LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<url>[^)\s]+)\)")
_MARKDOWN_BOLD_RE = re.compile(r"\*\*(?P<text>.+?)\*\*")


def _summary_sections(
    artifacts: dict[str, str], preset_names: tuple[str, ...]
) -> list[str]:
    """The preset artifacts that carry text, in configured order.

    Shared by both delivery channels so a call reads the same in Planfix and in
    Telegram; only the rendering below them differs.
    """
    return [
        artifacts[name].strip()
        for name in preset_names
        if artifacts.get(name, "").strip()
    ]


def _to_plain_text(markdown: str) -> str:
    """Flatten the Markdown a preset emits into text Telegram can show as-is.

    Telegram's parse modes are all-or-nothing: one unbalanced ``*`` or ``<`` anywhere
    in a transcript-derived document fails the whole ``sendMessage`` call, so the
    summary goes out unparsed and the markup has to come off here instead. Headings
    keep their title, links become "text: url", bold loses its asterisks; list dashes
    stay, because they read fine as plain text.
    """
    text = _MARKDOWN_LINK_RE.sub(
        lambda m: f"{m.group('text')}: {m.group('url')}", markdown
    )
    text = _MARKDOWN_HEADING_RE.sub(lambda m: m.group("title"), text)
    return _MARKDOWN_BOLD_RE.sub(lambda m: m.group("text"), text)


def _telegram_summary(
    artifacts: dict[str, str],
    preset_names: tuple[str, ...],
    meta_document: dict[str, object] | None = None,
    meta_fields: tuple[str, ...] = (),
    meta_entities: tuple[meta_entity.MetaEntity, ...] = (),
) -> str:
    """Render the same header + preset sections as the Planfix comment, as plain text.

    Returns an empty string when no preset produced anything, matching
    ``_planfix_description``: a header alone (a duration and a link) is not a summary
    worth posting into someone's chat.
    """
    sections = _summary_sections(artifacts, preset_names)
    if not sections:
        return ""
    header = "\n".join(_planfix_meta_lines(meta_document, meta_fields, meta_entities))
    blocks = [header] if header else []
    blocks.extend(sections)
    return _to_plain_text("\n\n".join(blocks)).strip()


def _planfix_description(
    artifacts: dict[str, str],
    preset_names: tuple[str, ...],
    meta_document: dict[str, object] | None = None,
    meta_fields: tuple[str, ...] = (),
    meta_entities: tuple[meta_entity.MetaEntity, ...] = (),
) -> str:
    """Concatenate the meta header and the configured preset artifacts into one body.

    The header (subject, tags, referral, ...) is rendered first from ``meta_document``
    and ``meta_fields``, so a manager reading the comment sees what the call was about
    before scrolling into the preset sections. Presets are joined in configured order
    and a preset with no artifact is skipped rather than emitting an empty section.
    The preset's own name is not printed: "keypoints" is how the pipeline spells a
    stage, not something a manager reading a CRM comment needs to see, and it landed
    as a stray English word between the Russian header and the Russian content. The
    header renders only alongside at least one preset section -- a header with no
    sections returns an empty string, which callers rely on to mean "nothing to
    comment".

    The result is HTML, not the Markdown the presets emit: Planfix stores comments as
    HTML and renders ``##`` and ``-`` as literal characters. Conversion happens once,
    on the assembled document, so headings and lists nest the same way they read (and
    the body stays a single line, since Planfix rewrites every newline as ``<br>``).
    """
    sections = [
        _mark_planfix_sections(section)
        for section in _summary_sections(artifacts, preset_names)
    ]
    # The header alone is never enough: a document carrying only a duration and a
    # link but no preset section is not something worth commenting, and
    # `_send_planfix_comment`'s `if not description` guard relies on an empty
    # return here to skip both the POST and the `planfix_comment_task_id` marker.
    if not sections:
        return ""

    header = "\n".join(
        _planfix_meta_lines(meta_document, meta_fields, meta_entities)
    )
    blocks: list[str] = [header] if header else []
    for section in sections:
        if blocks:
            # Where the preset's name used to sit. Doubling with the gap a marked
            # heading brings is harmless -- the converter collapses them.
            blocks.append(planfix_html.SECTION_BREAK)
        blocks.append(section)
    return planfix_html.markdown_to_html("\n\n".join(blocks))


def _send_planfix_comment(
    service: Any,
    item: dict,
    file_id: str,
    config: Config,
    artifacts: dict[str, str],
    booking_decision: booking_gate.BookingDecision,
    meta_document: dict[str, object] | None = None,
) -> None:
    """Post the meeting summary into the matched Planfix task, exactly once.

    `process_item` can legitimately reach its success path more than once per file — a
    later cycle that backfills a newly configured preset re-feeds the transcript — so
    the `planfix_comment_task_id` marker, written only after a successful POST, is what
    keeps a second pass from posting a duplicate comment into the task.

    ``meta_document`` (the merged document Task 4's ``_write_call_documents`` built
    this cycle) opens the comment with a header drawn from ``config.planfix_meta_fields``
    before the preset sections; it defaults to ``None`` so a caller with no document
    still gets the plain preset-only comment.
    """
    if not booking_decision.is_matched:
        return
    if not config.planfix_create_comment_url:
        return
    if item.get("planfix_comment_task_id"):
        logger.debug("Planfix comment already sent for %s, skipping", file_id)
        return

    description = _planfix_description(
        artifacts,
        config.planfix_presets,
        meta_document,
        config.planfix_meta_fields,
        meta_entities=config.meta_entities,
    )
    if not description:
        logger.warning(
            "No configured Planfix preset produced text for %s; nothing to comment",
            file_id,
        )
        return

    sent = planfix.send_comment(
        url=config.planfix_create_comment_url,
        token=config.planfix_token,
        proxy_url=config.proxy_url,
        task_id=booking_decision.task_id,
        description=description,
    )
    if sent:
        drive.set_file_app_properties(
            service,
            file_id,
            {drive.PLANFIX_COMMENT_TASK_ID_PROPERTY: booking_decision.task_id},
        )
        return

    # Unlike the completion webhook, a lost CRM comment is invisible to a human, so it
    # escalates. No marker is written, so `gdstt reprocess` can resend it.
    notify.notify_error(
        f"Failed to create the Planfix comment on task {booking_decision.task_id} "
        f"for {item.get('file', {}).get('name')}; rerun `gdstt reprocess {file_id}`",
        telegram_bot_token=config.telegram_bot_token,
        telegram_chat_id=config.telegram_chat_id,
        proxy_url=config.proxy_url,
    )


def folder_telegram_chat(config: Config, folder_id: str) -> str:
    """The chat a folder's summaries go to, or "" when it has none.

    Also the "recognize unconditionally" predicate: a folder with a chat is watched for
    its own sake, so ``run_once`` must not skip its recordings for want of a booking.
    """
    folder = config.folder_by_id(folder_id)
    return folder.telegram.strip() if folder else ""


def _send_telegram_summary(
    service: Any,
    item: dict,
    file_id: str,
    folder_id: str,
    config: Config,
    artifacts: dict[str, str],
    booking_decision: booking_gate.BookingDecision,
    meta_document: dict[str, object] | None = None,
) -> None:
    """Post the meeting summary into the folder's Telegram chat, exactly once.

    Independent of Planfix by default -- a folder that asked for a chat gets every
    call in it, matched or not. ``planfix.ignore_telegram_when_planfix`` turns that
    into a fallback: a call the CRM already recorded stays out of the chat.

    Like the Planfix marker, ``telegram_sent_chat_id`` is written only after a
    successful send, so a later cycle backfilling a newly configured preset does not
    re-post the whole summary.
    """
    chat_id = folder_telegram_chat(config, folder_id)
    if not chat_id:
        return
    if (
        config.planfix_ignore_telegram_when_planfix
        and booking_decision.is_matched
        and config.planfix_create_comment_url
    ):
        logger.debug(
            "Planfix covers %s and ignore_telegram_when_planfix is set; "
            "skipping the Telegram summary",
            file_id,
        )
        return
    if item.get("telegram_sent_chat_id"):
        logger.debug("Telegram summary already sent for %s, skipping", file_id)
        return

    text = _telegram_summary(
        artifacts,
        config.planfix_presets,
        meta_document,
        config.planfix_meta_fields,
        meta_entities=config.meta_entities,
    )
    if not text:
        logger.warning(
            "No configured preset produced text for %s; nothing to send to Telegram",
            file_id,
        )
        return

    sent = notify.send_message(
        text,
        bot_token=config.telegram_bot_token,
        chat_id=chat_id,
        proxy_url=config.proxy_url,
    )
    if sent:
        drive.set_file_app_properties(
            service,
            file_id,
            {drive.TELEGRAM_SENT_CHAT_ID_PROPERTY: chat_id},
        )
        return

    # No ``notify_error`` here: the error channel is the same Telegram API that just
    # failed, so the escalation would most likely be lost too. No marker is written,
    # so `gdstt reprocess` can resend it.
    logger.warning(
        "Failed to send the Telegram summary for %s to %s; rerun "
        "`gdstt reprocess %s`",
        item.get("file", {}).get("name"), chat_id, file_id,
    )


def process_item(
    service: Any,
    item: dict,
    folder_id: str,
    config: Config,
    *,
    reprocess_txt: bool = False,
    reprocess_presets: list[str] | None = None,
    booking_decision: booking_gate.BookingDecision | None = None,
) -> _ProcessTelemetry | None:
    file_info = item["file"]
    file_id = file_info["id"]
    file_name = file_info["name"]
    file_size = _coerce_size_bytes(file_info.get("size"))
    has_mp3 = item.get("has_mp3", False)
    has_txt = item.get("has_txt", False)
    # Artifacts belong beside the video, which with a subfolder per meeting is no
    # longer the configured folder. Falling back to ``folder_id`` keeps a caller that
    # built an item by hand working, and is exactly right for a flat folder.
    container_id = item.get("container_id") or folder_id
    # What to download. The same file for a recording in this folder; for a call
    # followed through a shortcut, the organizer's recording the shortcut points at,
    # while ``file_id`` -- the shortcut -- keeps pairing artifacts and bookkeeping.
    media_id = item.get("media_id") or file_id

    stt_enabled = bool(config.stt_provider)
    preset_only_reprocess = reprocess_presets is not None and not reprocess_txt
    needs_mp3 = (
        not preset_only_reprocess
        and _should_make_mp3_artifact(config)
        and not has_mp3
    )
    needs_txt = stt_enabled and (
        reprocess_txt or (not has_txt and not preset_only_reprocess)
    )
    needs_presets = _needs_preset_reprocess(item, config, needs_txt=needs_txt)
    # `gdstt reprocess <stages>` force-reruns explicit presets from an existing
    # transcript even when their artifacts already exist.
    if reprocess_presets and not needs_txt and _has_existing_transcript(item):
        needs_presets = True

    if not needs_mp3 and not needs_txt and not needs_presets:
        return

    # `run_once` resolves this itself so it can gate and count; the manual commands do
    # not, and get a decision here purely so a matched call still reaches Planfix.
    if booking_decision is None:
        booking_decision = booking_gate.resolve(file_info, folder_id, config)

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
    transcript = ""
    artifacts: dict[str, str] = {}
    unproduced: set[str] = set()
    meta_document: dict[str, object] | None = None

    try:
        with tempfile.TemporaryDirectory(prefix="gd-stt-") as tmp:
            tmp_dir = Path(tmp)
            mp4_path: Path | None = None
            mp3_path: Path | None = None

            if needs_mp3:
                mp4_path = _call_with_transient_retries(
                    lambda: drive.download(
                        service,
                        media_id,
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
                    container_id,
                    mime_type=drive.MP3_MIME,
                    name=mp3_drive_name,
                    app_properties={
                        drive.SOURCE_VIDEO_ID_PROPERTY: file_id,
                        drive.ARTIFACT_TYPE_PROPERTY: "mp3",
                    },
                )
                mp3_uploaded = True
                logger.info("Uploaded %s to folder %s", mp3_drive_name, container_id)

            if needs_txt:
                if mp4_path is None:
                    mp4_path = _call_with_transient_retries(
                        lambda: drive.download(
                            service,
                            media_id,
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
                # Who the presets are told was on the call. The same names as the
                # transcript's labels, except when Meet named people nobody could place
                # on a speaker: presets take them "in no particular order", so they are
                # still worth knowing.
                participant_names = speaker_names
                if config.stt_postprocess:
                    if speaker_names is None and config.openai_api_key:
                        # Meet's own transcript knows the participants even when the
                        # recording's name does not, and knows them in full when the
                        # name only has a first name from the calendar invite. Without
                        # a model nothing could use it, so it is not read.
                        meet = _read_meet_transcript(service, container_id, file_name)
                        # An answer the model would not stand behind leaves the
                        # speakers numbered (``[]``). No name is ever bound to a
                        # speaker by order once a model could be asked: neither Meet's
                        # order nor the file name's is diarization's, and on a real call
                        # Meet's swapped the labels.
                        speaker_names = _resolve_speaker_names(
                            text, file_name, folder_id, config, usage=usage,
                            candidates=meet[0] if meet else None,
                            meet_text=meet[1] if meet else "",
                        )
                        participant_names = speaker_names or (meet[0] if meet else None)
                    text = postprocess.postprocess_transcript(
                        text,
                        file_name,
                        speaker_names=speaker_names,
                    )
                _save_and_upload_txt(
                    service, file_id, file_name, text, container_id, tmp_dir, config,
                    txt_id=item.get("txt_id"),
                )
                txt_uploaded = True
                transcript = text

                artifacts = _run_preset_stage(
                    service,
                    file_id,
                    file_name,
                    text,
                    folder_id,
                    container_id,
                    tmp_dir,
                    config,
                    speaker_names=participant_names,
                    artifact_ids=item.get("artifact_ids") or {},
                    reprocess=reprocess_txt,
                    usage=usage,
                    unproduced=unproduced,
                    local_artifact_paths=_local_artifact_paths(item),
                    only_presets=reprocess_presets,
                )
                meta_document = _try_write_call_documents(
                    service, file_id, file_name, folder_id, container_id, text, artifacts,
                    config, tmp_dir, item=item, booking_decision=booking_decision,
                )
            elif needs_presets:
                # The transcript already exists on Drive; re-feed it to produce the
                # still-missing presets (a failed earlier preset or a newly added
                # one) without re-running STT.
                if item.get("txt_id") is not None:
                    text = _call_with_transient_retries(
                        lambda: drive.download_text(service, item["txt_id"]),
                        description=f"download transcript for {file_name} ({file_id})",
                        retry_state=retry_state,
                    )
                else:
                    text = Path(item["local_txt_path"]).read_text(encoding="utf-8")
                transcript = text
                speaker_names = _speaker_names_from_file_info(file_info)
                artifacts = _run_preset_stage(
                    service,
                    file_id,
                    file_name,
                    text,
                    folder_id,
                    container_id,
                    tmp_dir,
                    config,
                    speaker_names=speaker_names,
                    artifact_ids=item.get("artifact_ids") or {},
                    reprocess=False,
                    usage=usage,
                    unproduced=unproduced,
                    local_artifact_paths=_local_artifact_paths(item),
                    only_presets=reprocess_presets,
                )
                meta_document = _try_write_call_documents(
                    service, file_id, file_name, folder_id, container_id, text, artifacts,
                    config, tmp_dir, item=item, booking_decision=booking_decision,
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

    # Success path only, after every artifact is written. An mp3-only pass produces
    # nothing a receiver can use, so it stays silent rather than POSTing blanks over
    # a good record. Fire-and-forget: the whole block is guarded because a file that
    # transcribed and uploaded must count as processed even if the payload or the
    # receiver misbehaves.
    #
    # A preset that returned blank wrote no artifact, so this file is still pending and
    # comes back next cycle. A file may notify more than once (a later cycle can add a
    # newly configured preset), but that re-delivery is bounded — a blank preset never
    # settles, so firing here would re-POST the transcript on *every* cycle forever.
    # Staying silent keeps re-delivery bounded; the receiver gets no retry either way.
    if unproduced:
        logger.warning(
            "Completion webhook withheld for %s: presets produced no artifact (%s)",
            file_name,
            ", ".join(sorted(unproduced)),
        )
    elif txt_uploaded or artifacts:
        try:
            webhook.notify_complete(
                url=config.webhook_url,
                token=config.webhook_token,
                proxy_url=config.proxy_url,
                payload=_webhook_payload(
                    file_id, file_name, folder_id, config, transcript, artifacts
                ),
            )
        except Exception as exc:
            logger.warning("Completion webhook failed: %s", type(exc).__name__)

        try:
            _send_planfix_comment(
                service, item, file_id, config, artifacts, booking_decision,
                meta_document=meta_document,
            )
        except Exception as exc:
            # A file that transcribed and uploaded must count as processed even if the
            # CRM hand-off misbehaves.
            logger.warning("Planfix comment failed: %s", type(exc).__name__)

        try:
            _send_telegram_summary(
                service, item, file_id, folder_id, config, artifacts,
                booking_decision, meta_document=meta_document,
            )
        except Exception as exc:
            logger.warning("Telegram summary failed: %s", type(exc).__name__)

    return _ProcessTelemetry(
        provider=provider,
        processing_mode=processing_mode,
        retry_count=retry_state.retry_count,
        duration_s=duration_s,
        mp3_uploaded=mp3_uploaded,
        txt_uploaded=txt_uploaded,
        cost_usd=cost_usd,
        usage=usage,
        transcript=transcript,
        artifacts=artifacts,
        meta_document=meta_document,
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _is_still_settling(item: dict, now: datetime) -> bool:
    """True while Drive looks like it has not finished processing this upload.

    Meet's recording lands in Drive well after the meeting folder does -- around an
    hour for an hour-long call -- and `videoMediaMetadata` is filled once Drive has
    processed it. Skipping a video that has no metadata yet costs one cycle;
    downloading one costs a transfer and an STT run that may have to be redone.

    The grace window is the important half. A video that never gets metadata still
    has to be transcribed, and waiting on it indefinitely would lose the recording
    quietly -- the exact failure this whole change exists to remove. So the wait is
    bounded, and anything without a readable age is processed rather than held.
    """
    if item.get("has_media_metadata", True):
        return False
    created_raw = item.get("file", {}).get("createdTime")
    if not created_raw:
        return False
    try:
        created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
    except ValueError:
        logger.info("Unreadable createdTime %r; not waiting on it", created_raw)
        return False
    return now - created < _MEDIA_SETTLING_GRACE


def _recording_datetime(item: dict) -> datetime | None:
    """When the call happened, as well as it can be known.

    The name first: Meet writes the meeting time into it, and that is what an
    operator means by "calls from the 12th". Drive's ``createdTime`` is the fallback
    rather than the source because it answers a different question -- when this file
    appeared -- and the two come apart in both small ways and large. Measured across
    eight real recordings, Meet's own lag ran 0-2 hours, enough to push a late call
    past midnight into the next day. Copying or re-uploading a recording resets
    ``createdTime`` outright: the examples this was built against were three days
    adrift for exactly that reason.

    ``None`` when neither is readable, which the caller treats as in scope. Dropping
    a recording nobody can date would be a silent loss, and silent loss is the
    failure this whole area exists to remove.
    """
    file_info = item.get("file", {})
    meeting = parse_meeting_start(file_info.get("name", ""))
    if meeting is not None:
        return meeting
    created_raw = file_info.get("createdTime")
    if not created_raw:
        return None
    try:
        return datetime.fromisoformat(str(created_raw).replace("Z", "+00:00"))
    except ValueError:
        logger.info("Unreadable createdTime %r; treating it as in scope", created_raw)
        return None


def _items_in_date_scope(
    items: list[dict], cutoff: datetime | None, *, dry_run: bool
) -> tuple[list[dict], int]:
    """Split off recordings of calls older than ``cutoff``.

    Applied before everything else in the cycle, and deliberately so: an old
    recording that Drive never finished processing would otherwise be counted as
    deferred, and deferred holds the changes cursor -- the backlog an operator asked
    to ignore would freeze the feed instead.

    Counted per folder rather than logged per file: a folder with a year of history
    would print its whole backlog every ten minutes. ``--dry-run`` names them, which
    is where an operator goes to see what a cutoff will actually do.
    """
    if cutoff is None:
        return items, 0
    kept: list[dict] = []
    skipped = 0
    for item in items:
        when = _recording_datetime(item)
        if when is not None and when < cutoff:
            skipped += 1
            if dry_run:
                logger.info(
                    "DRY RUN: %s is from %s, before %s; not in scope",
                    item.get("file", {}).get("name"),
                    when.isoformat(),
                    cutoff.isoformat(),
                )
            continue
        kept.append(item)
    return kept, skipped


def _pending_items(items: list[dict], config: Config) -> list[dict]:
    stt_enabled = bool(config.stt_provider)
    now = _utcnow()
    pending = []
    for item in items:
        if _is_still_settling(item, now):
            logger.info(
                "Drive has not finished processing %s yet; leaving it for a later cycle",
                item.get("file", {}).get("name"),
            )
            continue
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


def _dry_run_preset_names(
    item: dict,
    config: Config,
    *,
    needs_txt: bool,
    reprocess_txt: bool,
    reprocess_presets: list[str] | None = None,
) -> list[str]:
    """Preset artifacts a real run would generate for this item.

    Mirrors :func:`_run_preset_stage`: ``--reprocess-txt`` regenerates every
    enabled preset, an explicit ``reprocess_presets`` set force-reruns those stages
    from an existing transcript, a fresh or reprocessable transcript generates the
    presets still missing an artifact, and otherwise no preset work happens.
    """
    if not config.presets:
        return []
    if reprocess_txt:
        return [preset.name for preset in config.presets]
    enabled = {preset.name for preset in config.presets}
    if reprocess_presets and not needs_txt and _has_existing_transcript(item):
        requested = [name for name in reprocess_presets if name in enabled]
        dependencies = preset_pipeline.dependency_names(config.presets, requested)
        existing = _existing_preset_names(item)
        names = set(requested) | (dependencies - existing)
        return [
            name
            for name in preset_pipeline.topological_order(config.presets)
            if name in names
        ]
    if needs_txt or _needs_preset_reprocess(item, config, needs_txt=needs_txt):
        return _missing_preset_names(item, config)
    return []


def _log_dry_run(
    folder_id: str,
    item: dict,
    config: Config,
    *,
    reprocess_txt: bool,
    reprocess_presets: list[str] | None = None,
) -> None:
    file_info = item["file"]
    has_mp3 = item.get("has_mp3", False)
    has_txt = item.get("has_txt", False)
    preset_only_reprocess = reprocess_presets is not None and not reprocess_txt
    needs_mp3 = (
        not preset_only_reprocess
        and _should_make_mp3_artifact(config)
        and not has_mp3
    )
    needs_txt = bool(config.stt_provider) and (
        reprocess_txt or (not has_txt and not preset_only_reprocess)
    )
    preset_names = _dry_run_preset_names(
        item, config, needs_txt=needs_txt, reprocess_txt=reprocess_txt,
        reprocess_presets=reprocess_presets,
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


def _configured_folder_for(service: Any, container_id: str, config: Config) -> str:
    """Which configured folder a container belongs to, falling back to itself.

    `process` and `reprocess` start from an id an operator typed, which may be a
    per-meeting subfolder the configuration has never named. Without this the
    employee, the Planfix routing and the folder's Telegram chat all resolve to
    nothing -- silently, because `folder_by_id` returns None rather than raising.

    The fallback keeps the old behaviour for an id that belongs to no configured
    folder at all: it is still processed, just without an employee, exactly as a
    hand-made folder was before subfolders existed.
    """
    configured = drive.find_configured_ancestor(
        service, container_id, {folder.folder_id for folder in config.folders}
    )
    if configured is None:
        return container_id
    if configured != container_id:
        logger.info(
            "Folder %s belongs to configured folder %s", container_id, configured
        )
    return configured


def process_target(
    service: Any,
    target_id: str,
    config: Config,
    *,
    is_folder: bool | None = None,
    reprocess_txt: bool = False,
    reprocess_presets: list[str] | None = None,
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
    configured_ids = {folder.folder_id for folder in config.folders}
    ancestors: dict[str, str | None] = {}

    if treat_as_folder:
        telemetry: list[_ProcessTelemetry] = []
        items = _call_with_transient_retries(
            lambda: _without_calls_the_organizer_covers(
                service,
                drive.list_folder_tree_state(service, target_id),
                configured_ids,
                ancestors,
            ),
            description=f"list folder state for {target_id}",
        )
        configured_id = _configured_folder_for(service, target_id, config)
        _apply_local_output_state(items, config)
        if reprocess_txt:
            pending = items
        elif reprocess_presets:
            pending = [item for item in items if _has_existing_transcript(item)]
        else:
            pending = _pending_items(items, config)
        pending = _items_allowed_by_size(
            pending,
            max_size_bytes=max_size_bytes,
            confirm_large=confirm_large,
        )
        logger.info("Folder %s: %d pending file(s)", target_id, len(pending))
        if dry_run:
            for item in pending:
                _log_dry_run(
                    configured_id, item, config,
                    reprocess_txt=reprocess_txt,
                    reprocess_presets=reprocess_presets,
                )
            return telemetry
        for item in pending:
            result = process_item(
                service,
                item,
                configured_id,
                config,
                reprocess_txt=reprocess_txt,
                reprocess_presets=reprocess_presets,
            )
            if result is not None:
                telemetry.append(result)
        return telemetry

    parents = meta.get("parents") or []
    if not parents:
        raise RuntimeError(f"File {target_id} has no parent folder")
    container_id = parents[0]
    folder_id = _configured_folder_for(service, container_id, config)
    listed = _call_with_transient_retries(
        lambda: drive.list_folder_state(service, container_id),
        description=f"list folder state for {container_id}",
    )
    items = _without_calls_the_organizer_covers(
        service, listed, configured_ids, ancestors
    )
    _apply_local_output_state(items, config)
    match = next(
        (it for it in items if it["file"]["id"] == target_id), None
    )
    if match is None:
        if any(it["file"]["id"] == target_id for it in listed):
            raise RuntimeError(
                f"File {target_id} is a shortcut to a recording in a configured folder; "
                "that folder processes the call"
            )
        raise RuntimeError(
            f"File {target_id} is not an MP4 in folder {container_id}"
        )
    allowed = _items_allowed_by_size(
        [match],
        max_size_bytes=max_size_bytes,
        confirm_large=confirm_large,
    )
    if not allowed:
        return []
    if dry_run:
        _log_dry_run(
            folder_id, match, config,
            reprocess_txt=reprocess_txt,
            reprocess_presets=reprocess_presets,
        )
        return []
    result = process_item(
        service, match, folder_id, config,
        reprocess_txt=reprocess_txt,
        reprocess_presets=reprocess_presets,
    )
    return [result] if result is not None else []


@dataclass
class _Discovery:
    """What one cycle found, and what to remember for the next one."""

    listings: list[tuple[str, list[dict]]]
    cursor: str | None
    retries: int = 0
    folder_errors: int = 0


def _notify_listing_failure(what: str, exc: Exception, config: Config) -> None:
    logger.exception("Failed to list %s", what)
    notify.notify_error(
        f"Failed to list {what}: {exc}\n{traceback.format_exc()}",
        telegram_bot_token=config.telegram_bot_token,
        telegram_chat_id=config.telegram_chat_id,
        proxy_url=config.proxy_url,
    )


def _is_in_a_configured_folder(
    service: Any,
    parents: list[str] | None,
    configured_ids: set[str] | frozenset[str],
    cache: dict[str, str | None],
) -> bool:
    """Whether a followed shortcut's recording lives under a configured folder.

    No parents -- not looked up because the call was already done here, or a recording
    shared on its own without its folder -- is "no". So is a 403/404 on the way up:
    every configured folder is readable, and so is everything inside it, so a folder
    this account cannot open is not one of them. Anything else is raised, because
    guessing "no" during an outage would process a call its organizer's folder is
    about to process too.
    """
    if not parents:
        return False
    try:
        owner = drive.find_configured_ancestor(
            service, parents[0], configured_ids, cache=cache
        )
    except HttpError as exc:
        if getattr(exc.resp, "status", None) not in (403, 404):
            raise
        cache[parents[0]] = None
        return False
    return owner is not None


def _without_calls_the_organizer_covers(
    service: Any,
    items: list[dict],
    configured_ids: set[str] | frozenset[str],
    cache: dict[str, str | None],
) -> list[dict]:
    """Drop followed shortcuts whose recording sits in a configured folder.

    Meet gives the organizer the recording and each attendee a shortcut to it. When
    the organizer's folder is watched too, that folder processes the call; following
    the shortcut as well would transcribe, summarize and deliver it twice. When it is
    not -- a client's call, or a colleague nobody configured -- the shortcut is the
    only way in.

    Decided every cycle from where the recording lives, never remembered. Configuring
    an organizer later moves each of their calls not yet processed over to their own
    folder; a call already processed through a shortcut stays processed, and a second
    pass from the organizer's folder is the known price of that order of events.
    """
    kept: list[dict] = []
    for item in items:
        if _is_in_a_configured_folder(
            service, item.get("target_parents"), configured_ids, cache
        ):
            logger.debug(
                "Leaving %s to the organizer's configured folder",
                item.get("file", {}).get("name"),
            )
            continue
        kept.append(item)
    return kept


def _discover_by_walk(service: Any, config: Config) -> _Discovery:
    """Read every configured folder and its meeting subfolders.

    The complete answer, and the expensive one: a request per folder per cycle. It
    runs on the first cycle and whenever the cursor is gone, which is what makes
    losing the cursor a cost rather than a loss.

    The cursor is taken *before* the sweep. Anything that lands while the sweep is
    running then shows up in the next feed read; taking it afterwards would open a
    window whose files no cycle ever looks at again.
    """
    cursor: str | None = None
    retries = 0
    try:
        cursor = drive.get_start_page_token(service)
    except (RefreshError, AuthError):
        raise
    except Exception:
        # A sweep with no cursor still processes everything; it just has to sweep
        # again next time. Refusing to sweep would be the worse trade.
        logger.exception("Could not take a changes cursor; this cycle will sweep again")

    listings: list[tuple[str, list[dict]]] = []
    folder_errors = 0
    configured_ids = {folder.folder_id for folder in config.folders}
    ancestors: dict[str, str | None] = {}
    for folder in config.folders:
        folder_id = folder.folder_id
        listing_retry_state = _RetryState()
        try:
            items = _call_with_transient_retries(
                lambda: _without_calls_the_organizer_covers(
                    service,
                    drive.list_folder_tree_state(service, folder_id),
                    configured_ids,
                    ancestors,
                ),
                description=f"list folder state for {folder_id}",
                retry_state=listing_retry_state,
            )
        except (RefreshError, AuthError):
            raise
        except Exception as exc:
            folder_errors += 1
            _notify_listing_failure(f"folder {folder_id}", exc, config)
            continue
        finally:
            retries += listing_retry_state.retry_count
        listings.append((folder_id, items))
    return _Discovery(listings, cursor, retries, folder_errors)


def _discover_by_changes(service: Any, config: Config, cursor: str) -> _Discovery | None:
    """Read Drive's own journal and look only where something happened.

    One request answers "has anything changed", however many folders are watched and
    however many meeting subfolders have piled up in them. Only the folders the
    journal names are then listed, and the listing -- not the journal -- still decides
    what needs doing, so every existing rule about siblings, markers and reprocessing
    keeps working untouched.

    Returns ``None`` when the cursor is no longer usable, which is the caller's signal
    to sweep and take a fresh one.
    """
    retry_state = _RetryState()
    try:
        entries, new_cursor = _call_with_transient_retries(
            lambda: drive.list_changes(service, cursor),
            description="read the changes feed",
            retry_state=retry_state,
        )
    except (RefreshError, AuthError):
        raise
    except Exception as exc:
        if _is_rejected_cursor(exc):
            logger.info("The changes cursor is no longer valid; sweeping instead")
            return None
        _notify_listing_failure("the changes feed", exc, config)
        return _Discovery([], cursor, retry_state.retry_count, folder_errors=1)

    configured_ids = {folder.folder_id for folder in config.folders}
    ancestors: dict[str, str | None] = {}
    containers: dict[str, str] = {}
    unresolved = 0
    for entry in entries:
        if entry.get("removed"):
            continue
        file_info = entry.get("file") or {}
        if file_info.get("trashed"):
            continue
        # Our own uploads come through here too. Judging by the entry alone is what
        # keeps the feed to a single request: no files.get to find out what something
        # is. An attended call arrives as a shortcut, so that counts as a recording.
        if not drive.names_a_recording(file_info):
            continue
        parents = file_info.get("parents") or []
        if not parents:
            continue
        container_id = parents[0]
        if container_id in containers:
            continue
        try:
            owner = drive.find_configured_ancestor(
                service, container_id, configured_ids, cache=ancestors
            )
        except (RefreshError, AuthError):
            raise
        except Exception as exc:
            # Not knowing whose folder this is must not read as "nobody's". Counting
            # it holds the cursor, so the same change is read again next cycle.
            unresolved += 1
            _notify_listing_failure(f"the folder above {container_id}", exc, config)
            continue
        if owner is None:
            # The account can see folders nobody configured, and the feed reports
            # those too.
            continue
        containers[container_id] = owner

    by_owner: dict[str, list[dict]] = {}
    folder_errors = 0
    for container_id, owner in containers.items():
        listing_retry_state = _RetryState()
        try:
            items = _call_with_transient_retries(
                lambda: _without_calls_the_organizer_covers(
                    service,
                    drive.list_folder_state(service, container_id),
                    configured_ids,
                    ancestors,
                ),
                description=f"list folder state for {container_id}",
                retry_state=listing_retry_state,
            )
        except (RefreshError, AuthError):
            raise
        except Exception as exc:
            folder_errors += 1
            _notify_listing_failure(f"folder {container_id}", exc, config)
            continue
        finally:
            retry_state.retry_count += listing_retry_state.retry_count
        # Each item already carries its own `container_id`, so merging them under
        # the configured folder loses nothing about where the files live.
        by_owner.setdefault(owner, []).extend(items)

    listings = list(by_owner.items())
    logger.info(
        "Changes feed [entries=%d, meeting_folders=%d, folders=%d]",
        len(entries),
        len(containers),
        len(listings),
    )
    return _Discovery(
        listings, new_cursor, retry_state.retry_count, folder_errors + unresolved
    )


def _cursor_covers_config(config: Config, *, mode: str) -> bool:
    """Whether the saved cursor can vouch for the folders now being watched.

    A cursor means "nothing has happened since" only for folders that were already
    in the config when it was taken. A folder added afterwards -- which is how an
    employee gets onboarded, not some one-off migration -- brings recordings that
    were never a change after that cursor, so the feed will never name it and its
    backlog would stay invisible until someone reset the cursor by hand. One sweep
    is the whole cost of noticing.

    ``changes`` mode refuses instead of sweeping -- that is its contract, and it is
    the only safe answer here. Reading the feed anyway would let the cycle drain and
    record the new folder set as vouched for without it ever having been swept, so
    the backlog would be invisible from then on.
    """
    watched = change_cursor.fingerprint(
        folder.folder_id for folder in config.folders
    )
    vouched = change_cursor.read_folders(
        change_cursor.folders_path_for(config.data_dir)
    )
    if vouched == watched:
        return True
    if mode == "changes":
        raise SystemExit(
            "The watched folders changed since the cursor was taken; the feed cannot "
            "report recordings that were already in a folder added since. Run a "
            "normal cycle or `gdstt run-once --mode walk` first."
        )
    logger.info(
        "The watched folders changed since the cursor was taken; sweeping once so a "
        "newly added folder's existing recordings are not missed"
    )
    return False


def _discover(service: Any, config: Config, *, mode: str = "auto") -> _Discovery:
    """Take the cheap path when a cursor says where to resume, the full one otherwise.

    ``walk`` forces the sweep and leaves the cursor where it is, which is what makes
    it a safe "check everything now" for an operator: the feed picks up afterwards
    exactly where it was, and anything the sweep already handled is simply found
    done. ``changes`` refuses to fall back, so it can answer whether the feed itself
    works without waiting for a cycle.
    """
    if mode == "walk":
        found = _discover_by_walk(service, config)
        # Leave the saved cursor alone: this was a look, not a new starting point.
        return replace(found, cursor=None)

    saved = change_cursor.read(change_cursor.path_for(config.data_dir))
    if mode == "changes" and saved is None:
        raise SystemExit(
            "No changes cursor saved yet; run `gdstt run-once --mode walk` or a "
            "normal cycle first."
        )
    if saved is not None and not _cursor_covers_config(config, mode=mode):
        saved = None
    if saved is not None:
        found = _discover_by_changes(service, config, saved)
        if found is not None:
            return found
        if mode == "changes":
            raise SystemExit(
                "The saved changes cursor is no longer valid; a normal cycle would "
                "sweep and take a fresh one."
            )
    return _discover_by_walk(service, config)


def run_once(
    service: Any,
    config: Config,
    *,
    dry_run: bool = False,
    max_size_bytes: int | None = None,
    confirm_large: bool = False,
    mode: str = "auto",
    since: str = "",
) -> None:
    cycle_started_at = time.monotonic()
    cycle_pending = 0
    cycle_processed = 0
    cycle_failed = 0
    cycle_retry_total = 0
    cycle_skipped_size = 0
    cycle_skipped_unmatched = 0
    cycle_folder_errors = 0
    cycle_deferred = 0
    cycle_skipped_old = 0
    settle_check_time = _utcnow()

    discovery = _discover(service, config, mode=mode)
    cycle_retry_total += discovery.retries
    cycle_folder_errors += discovery.folder_errors

    for folder_id, items in discovery.listings:
        _apply_local_output_state(items, config)
        total_seen = len(items)
        items, skipped_old = _items_in_date_scope(
            items,
            parse_since(since or config.since_for(folder_id), source="since"),
            dry_run=dry_run,
        )
        cycle_skipped_old += skipped_old
        cycle_deferred += sum(
            1 for item in items if _is_still_settling(item, settle_check_time)
        )
        pending = _pending_items(items, config)
        # A marked recording is settled: reconsidering it every cycle would re-log and
        # re-decide forever. `gdstt bookings rematch` or any manual command revives it.
        pending = [
            item for item in pending
            if item.get("booking_match") != drive.BOOKING_MATCH_NONE
        ]
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
            "Folder %s summary [total=%d, pending=%d, skipped_size=%d, "
            "skipped_old=%d, dry_run=%s]",
            folder_id,
            total_seen,
            len(pending),
            skipped_size,
            skipped_old,
            dry_run,
        )
        if dry_run:
            for item in pending:
                _log_dry_run(folder_id, item, config, reprocess_txt=False)
            continue
        for item in pending:
            decision = booking_gate.resolve(item["file"], folder_id, config)
            if (
                decision.state == booking_gate.UNMATCHED
                and config.call_booking_disable_recognition
                # A folder with a Telegram chat is recognized unconditionally: the
                # chat is the destination, so "no booking" is not a reason to skip --
                # and marking the file unmatched would park it for good.
                and not folder_telegram_chat(config, folder_id)
            ):
                file_name = item.get("file", {}).get("name")
                if booking_server.is_running():
                    # Permanent by design: the booking arrives before the call, so a
                    # recording with no booking is not a client call.
                    try:
                        booking_gate.mark_unmatched(service, item["file"]["id"])
                    except (RefreshError, AuthError):
                        raise
                    except Exception:
                        # A transient Drive failure here must not kill the polling
                        # loop; the file stays unmarked and is retried next cycle.
                        logger.exception(
                            "Failed to mark %s in folder %s as unmatched; will "
                            "retry next cycle",
                            file_name, folder_id,
                        )
                    else:
                        logger.info(
                            "Skipping %s in folder %s: no booked call (%s); marked "
                            "so it is not reconsidered (undo with `gdstt bookings "
                            "rematch`)",
                            file_name, folder_id, decision.reason,
                        )
                else:
                    logger.warning(
                        "Skipping %s in folder %s: no booked call (%s), but the "
                        "booking receiver is not listening, so it is not marked",
                        file_name, folder_id, decision.reason,
                    )
                cycle_skipped_unmatched += 1
                continue
            try:
                telemetry = process_item(
                    service, item, folder_id, config, booking_decision=decision
                )
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
                    telegram_bot_token=config.telegram_bot_token,
                    telegram_chat_id=config.telegram_chat_id,
                    proxy_url=config.proxy_url,
                )

    # Only a cycle that actually drained what it found may move the cursor, and only
    # after the work. The changes feed reports a folder once, when something happens
    # in it; a recording this cycle failed on, or deliberately left for later, will
    # produce no second change of its own. Stepping over it would lose it for good --
    # the very failure this whole change exists to remove. Re-reading changes instead
    # is free, because the folder listing decides what still needs doing.
    # `cycle_skipped_old` is deliberately absent: a recording left out by `since` is
    # a permanent skip by design, like one over `--max-size`. Counting it would hold
    # the cursor on a backlog that is never going to be processed.
    cycle_drained = not (cycle_failed or cycle_folder_errors or cycle_deferred)
    if not dry_run and discovery.cursor and cycle_drained:
        change_cursor.write(change_cursor.path_for(config.data_dir), discovery.cursor)
        # Saved with the cursor, never apart from it: a cursor whose folder set is
        # missing cannot be vouched for and would sweep every cycle. The same
        # `cycle_drained` guard is what keeps a config edited before the folder was
        # actually shared from being recorded as seen -- that listing fails, which
        # counts as a folder error, which holds both files where they are.
        change_cursor.write_folders(
            change_cursor.folders_path_for(config.data_dir),
            change_cursor.fingerprint(
                folder.folder_id for folder in config.folders
            ),
        )
    elif not dry_run and discovery.cursor:
        logger.info(
            "Holding the changes cursor [failed=%d, folder_errors=%d, deferred=%d]; "
            "the next cycle reads the same changes again",
            cycle_failed, cycle_folder_errors, cycle_deferred,
        )

    logger.info(
        "Cycle summary [provider=%s, outcome=%s, folders=%d, pending=%d, processed=%d, failed=%d, "
        "retry_total=%d, skipped_size=%d, skipped_unmatched=%d, skipped_old=%d, "
        "folder_errors=%d, deferred=%d, cursor_moved=%s, dry_run=%s, "
        "duration_s=%.3f]",
        config.stt_provider or "artifact-only",
        _cycle_outcome(
            dry_run=dry_run,
            failed=cycle_failed,
            folder_errors=cycle_folder_errors,
        ),
        len(config.folders),
        cycle_pending,
        cycle_processed,
        cycle_failed,
        cycle_retry_total,
        cycle_skipped_size,
        cycle_skipped_unmatched,
        cycle_skipped_old,
        cycle_folder_errors,
        cycle_deferred,
        bool(discovery.cursor) and cycle_drained and not dry_run,
        dry_run,
        time.monotonic() - cycle_started_at,
    )


def main(*, config_path: str | Path | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = load_config(config_path=config_path)
    if not config.folders:
        logger.error("folders is empty; configure it in config.yml to start polling")
        raise SystemExit(1)

    try:
        service = build_drive_service(config=config)
    except (RefreshError, AuthError) as exc:
        logger.exception("OAuth bootstrap failed; exiting for restart")
        notify.notify_error(
            f"OAuth bootstrap failed; container will exit so it can be restarted "
            f"after re-running `python -m src.auth`: {exc}",
            telegram_bot_token=config.telegram_bot_token,
            telegram_chat_id=config.telegram_chat_id,
            proxy_url=config.proxy_url,
        )
        raise SystemExit(1) from exc

    try:
        booking_server.start(config)
    except OSError as exc:
        # Degrade, do not exit: transcription is the primary job. With the receiver
        # down the gate refuses to mark anything (see `run_once`), so nothing is lost --
        # unmatched files simply wait.
        logger.exception("Booking receiver failed to start; continuing without it")
        notify.notify_error(
            f"Booking receiver failed to start on "
            f"{config.call_booking_listen_host}:{config.call_booking_listen_port}: "
            f"{exc}. Call bookings are not being received; recordings will not be "
            f"marked as unmatched until it is back.",
            telegram_bot_token=config.telegram_bot_token,
            telegram_chat_id=config.telegram_chat_id,
            proxy_url=config.proxy_url,
        )

    paused_logged = False
    while True:
        if not is_run_enabled(config_path=config_path):
            # `gdstt stop` sets run.enabled=false. Stay up but idle so a Docker
            # `restart: unless-stopped` policy does not crash-loop and the stop
            # survives restarts without auto-resuming. Resume with `gdstt start`.
            if not paused_logged:
                logger.info(
                    "run.enabled is false (gdstt stop); polling loop paused "
                    "(resume with `gdstt start` or `gdstt run`)"
                )
                paused_logged = True
            time.sleep(config.poll_interval)
            continue
        paused_logged = False
        try:
            run_once(service, config, mode=config.run_discovery)
        except (RefreshError, AuthError) as exc:
            logger.exception("OAuth refresh failed; exiting for restart")
            notify.notify_error(
                f"OAuth refresh failed; container will exit so it can be restarted "
                f"after re-running `python -m src.auth`: {exc}",
                telegram_bot_token=config.telegram_bot_token,
                telegram_chat_id=config.telegram_chat_id,
                proxy_url=config.proxy_url,
            )
            raise SystemExit(1) from exc
        except Exception as exc:
            logger.exception("Cycle failed")
            notify.notify_error(
                f"Cycle failed: {exc}\n{traceback.format_exc()}",
                telegram_bot_token=config.telegram_bot_token,
                telegram_chat_id=config.telegram_chat_id,
                proxy_url=config.proxy_url,
            )
        time.sleep(config.poll_interval)


if __name__ == "__main__":
    main()
