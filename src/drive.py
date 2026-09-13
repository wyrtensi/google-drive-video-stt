from __future__ import annotations

import io
import logging
import os
import re
from pathlib import Path
from typing import Any

from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

logger = logging.getLogger(__name__)

MP4_MIME = "video/mp4"
MP3_MIME = "audio/mpeg"
TXT_MIME = "text/plain"
MD_MIME = "text/markdown"
FOLDER_MIME = "application/vnd.google-apps.folder"
GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
PAGE_SIZE = 1000
# Meet nests meeting folders one level under the root; the slack absorbs an
# unexpected layer without letting a circular parent chain run away.
MAX_ANCESTOR_DEPTH = 4
SOURCE_VIDEO_ID_PROPERTY = "source_video_id"
ARTIFACT_TYPE_PROPERTY = "artifact_type"
SPEAKER_NAMES_PROPERTY = "speaker_names"
BOOKING_MATCH_PROPERTY = "booking_match"
PLANFIX_COMMENT_TASK_ID_PROPERTY = "planfix_comment_task_id"
TELEGRAM_SENT_CHAT_ID_PROPERTY = "telegram_sent_chat_id"
# The single value ``booking_match`` ever takes: this recording matched no booked
# call, so the polling loop must leave it alone.
BOOKING_MATCH_NONE = "none"
_LOCAL_FILENAME_UNSAFE_RE = re.compile(r'[<>:"/\\|?*\0]')


class DownloadIntegrityError(RuntimeError):
    """Raised when a Drive download completes with an unexpected local size."""


def drive_stem(name: str) -> str:
    """Filename minus its extension, treating the Drive name as a flat string.

    Drive names may contain ``/`` (Drive allows it in file names); ``Path(...).stem``
    would treat that as a path separator and drop everything before the last ``/``.
    ``os.path.splitext`` only splits on the final dot, so the full name is preserved.
    """
    return os.path.splitext(name)[0]


def safe_local_name(name: str) -> str:
    """Sanitize a Drive name into a filesystem-safe local filename.

    Drive accepts characters that Windows does not allow in local filenames.
    Replace those characters so temp downloads/extractions work on every platform.
    """
    return _LOCAL_FILENAME_UNSAFE_RE.sub("_", name)


def get_file_metadata(service: Any, file_id: str) -> dict:
    """Return id/name/mimeType/parents/size/appProperties for a single Drive file."""
    return (
        service.files()
        .get(
            fileId=file_id,
            fields="id, name, mimeType, parents, size, appProperties",
            supportsAllDrives=True,
        )
        .execute()
    )


def _next_page_token(response: Any) -> str | None:
    """The next page token, or ``None`` when there is not one.

    Every listing here loops until Drive stops handing out tokens, so the loop's exit
    depends on a value that arrives untyped from outside. Insisting on a non-empty
    string makes a malformed answer end the listing instead of spinning on it -- the
    difference between a short read and a process that never returns.
    """
    token = response.get("nextPageToken")
    if isinstance(token, str) and token:
        return token
    return None


def _list_files_by_mime(service: Any, folder_id: str, mime_type: str) -> list[dict]:
    return _list_files_by_mimes(service, folder_id, (mime_type,))


def _list_files_by_mimes(
    service: Any, folder_id: str, mime_types: tuple[str, ...]
) -> list[dict]:
    """List a folder's files of any of ``mime_types`` in a single query.

    Drive charges a round trip per request, not per mime type, and with a subfolder
    per meeting the old one-request-per-mime shape multiplied by the number of
    meetings on every cycle. Asking for all four at once and splitting the answer by
    ``mimeType`` -- which the response already carries -- costs one round trip per
    folder regardless of how many types the caller wants.
    """
    files: list[dict] = []
    page_token: str | None = None
    mime_clause = " or ".join(f"mimeType = '{mime}'" for mime in mime_types)
    query = f"'{folder_id}' in parents and ({mime_clause}) and trashed = false"
    while True:
        response = (
            service.files()
            .list(
                q=query,
                fields=(
                    "nextPageToken, files(id, name, mimeType, size, createdTime, "
                    # Only whether Drive has finished with the video is read, so one
                    # cheap sub-field stands in for the whole object on a listing the
                    # polling loop makes for every folder, every cycle.
                    "videoMediaMetadata(durationMillis), appProperties, "
                    "shortcutDetails(targetId, targetMimeType))"
                ),
                pageSize=PAGE_SIZE,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files.extend(response.get("files", []))
        page_token = _next_page_token(response)
        if page_token is None:
            break
    return files


def list_mp4_timestamps(service: Any, folder_id: str) -> list[dict]:
    """Return every recording in a folder with its timestamps and appProperties.

    ``_list_files_by_mime`` keeps its field list small because the polling loop calls
    it every cycle. Only the date repair needs ``createdTime``/``modifiedTime``, so it
    asks for them here rather than widening the hot path.

    A shortcut to a recording counts: a call followed through one keeps its markers on
    the shortcut, and a report that skipped shortcuts would say those calls were never
    sent anywhere.
    """
    files: list[dict] = []
    page_token: str | None = None
    query = (
        f"'{folder_id}' in parents and (mimeType = '{MP4_MIME}' or "
        f"(mimeType = '{SHORTCUT_MIME}' and "
        f"shortcutDetails.targetMimeType = '{MP4_MIME}')) and trashed = false"
    )
    while True:
        response = (
            service.files()
            .list(
                q=query,
                fields=(
                    "nextPageToken, files(id, name, mimeType, createdTime, "
                    "modifiedTime, appProperties, shortcutDetails(targetId, targetMimeType))"
                ),
                pageSize=PAGE_SIZE,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files.extend(
            f for f in response.get("files", [])
            # The query already says this; repeating it keeps the answer right on a
            # Drive -- or a fake -- that ignores the shortcutDetails clause.
            if f.get("mimeType") != SHORTCUT_MIME or names_a_recording(f)
        )
        page_token = _next_page_token(response)
        if page_token is None:
            break
    return files


def list_mp4_timestamps_in_tree(service: Any, folder_id: str) -> list[dict]:
    """``list_mp4_timestamps`` for a folder and each of its subfolders.

    The operator-facing reports run over this. Asking only the configured folder
    answers "no recordings at all" on a Google Meet root, which reads as "nothing was
    ever sent to Planfix" or "nothing to restore" -- confidently, and wrongly.
    """
    files = list_mp4_timestamps(service, folder_id)
    for subfolder in list_subfolders(service, folder_id):
        files.extend(list_mp4_timestamps(service, subfolder["id"]))
    return files


def set_file_modified_time(service: Any, file_id: str, modified_time: str) -> dict:
    """Set a file's modifiedTime, leaving appProperties and content untouched.

    The body carries only the date: the ``booking_match`` marks must survive, or the
    polling loop would reconsider the whole backlog and re-transcribe it.
    """
    return (
        service.files()
        .update(
            fileId=file_id,
            body={"modifiedTime": modified_time},
            fields="id, name, modifiedTime",
            supportsAllDrives=True,
        )
        .execute()
    )


def find_newest_mp4(service: Any, folder_id: str) -> dict | None:
    """Return the most recently created mp4 directly in a folder, or None when empty."""
    query = (
        f"'{folder_id}' in parents and mimeType = '{MP4_MIME}' and trashed = false"
    )
    response = (
        service.files()
        .list(
            q=query,
            fields="files(id, name, mimeType, size, createdTime, appProperties)",
            orderBy="createdTime desc",
            pageSize=1,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    files = response.get("files", [])
    if not files:
        return None
    return {**files[0], "container_id": folder_id}


def find_newest_mp4_in_tree(service: Any, folder_id: str) -> dict | None:
    """Return the newest mp4 in a folder or any of its subfolders, with its container.

    Pointed at a Google Meet root, the single-folder lookup answers "no mp4 files":
    the root holds only per-meeting subfolders. That is the worst answer an operator
    can get from `gdstt latest`, because it looks like an empty folder rather than a
    command that stopped understanding the folder layout.

    Asks each folder separately instead of joining every parent into one query: the
    joined form grows with the number of meetings and would eventually outgrow the
    query, while this is a hand-run command where a few extra round trips cost
    nothing.
    """
    newest: dict | None = None
    for candidate_folder in [folder_id, *(f["id"] for f in list_subfolders(service, folder_id))]:
        candidate = find_newest_mp4(service, candidate_folder)
        if candidate is None:
            continue
        if newest is None or candidate.get("createdTime", "") > newest.get("createdTime", ""):
            newest = candidate
    return newest


def meet_transcript_name(video_name: str) -> str:
    """The name Meet gives the transcript sitting beside ``video_name``.

    Meet names the pair from one base. A call booked in the calendar gets
    ``<title> - <when> - Recording`` and ``<title> - <when> - Transcript``; a call
    started outside it gets ``<room> (<when>)`` and ``<room> (<when>) - Transcript``.
    Dropping a trailing ``- Recording`` covers both.
    """
    base = video_name
    for extension in (".mp4", ".MP4"):
        if base.endswith(extension):
            base = base[: -len(extension)]
            break
    if base.endswith(" - Recording"):
        base = base[: -len(" - Recording")]
    return f"{base} - Transcript"


def find_meet_transcript(service: Any, folder_id: str, video_name: str) -> dict | None:
    """The Google Doc transcript belonging to one recording, or ``None``.

    Matched by name rather than by being the only document in the folder: a recurring
    meeting keeps every instance in the same subfolder, so "the transcript here" is
    not a question with one answer.

    An attendee's meeting folder holds a shortcut to the organizer's transcript rather
    than the document. A shortcut has nothing to export, so for one of those the
    target's id is returned -- whether it opens is the caller's question.
    """
    wanted = meet_transcript_name(video_name)
    for doc in _list_files_by_mimes(service, folder_id, (GOOGLE_DOC_MIME, SHORTCUT_MIME)):
        if doc.get("name") != wanted:
            continue
        if doc.get("mimeType") != SHORTCUT_MIME:
            return doc
        details = doc.get("shortcutDetails") or {}
        if details.get("targetMimeType") == GOOGLE_DOC_MIME and details.get("targetId"):
            return {"id": details["targetId"], "name": doc.get("name")}
    return None


def export_document_text(service: Any, file_id: str) -> str:
    """Read a Google Doc as plain text.

    A Google Doc has no bytes to download -- it has to be exported -- which is also
    why this service never saw Meet's transcripts before: they are invisible to a
    listing that asks for ``text/plain``.
    """
    data = (
        service.files()
        .export(fileId=file_id, mimeType="text/plain")
        .execute()
    )
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def describe_folder(service: Any, folder_id: str) -> dict:
    """Return ``{id, name, parents, trashed}`` for a folder.

    What the diagnostics were missing. Counting files in a configured folder answered
    "can I reach it", and the answer stayed yes for two months after Google moved the
    recordings elsewhere: the old folder was still there, still readable, and simply
    never got anything new again. The name is what gives that away at a glance --
    a folder that now reads `Legacy Meet Recordings` is the whole diagnosis.
    """
    return (
        service.files()
        .get(
            fileId=folder_id,
            fields="id, name, parents, trashed",
            supportsAllDrives=True,
        )
        .execute()
    )


def get_start_page_token(service: Any) -> str:
    """Return a cursor marking "everything up to now has been seen"."""
    # Only `supportsAllDrives` here: `getStartPageToken` does not take
    # `includeItemsFromAllDrives`, and passing it is a TypeError from the client
    # rather than an API error -- which a mock accepts happily and a real Drive does
    # not. Without the token the service silently falls back to sweeping every folder
    # on every cycle, for good.
    response = (
        service.changes()
        .getStartPageToken(supportsAllDrives=True)
        .execute()
    )
    return response.get("startPageToken", "")


def list_changes(service: Any, page_token: str) -> tuple[list[dict], str]:
    """Return everything that changed since ``page_token``, and the next cursor.

    Drive keeps this journal itself -- it is what the Activity panel shows -- so one
    request answers "has anything happened" regardless of how many folders are
    watched or how many meeting subfolders have accumulated in them. Walking folders
    costs a request per folder per cycle; this costs one.

    Every page is read before the new cursor is returned. Reporting a cursor from a
    partial read would skip whatever sat on the pages never asked for, and nothing
    would bring those files back.

    The field list is what makes the feed cheap: with ``mimeType``, ``parents`` and
    ``trashed`` on the entry itself, the caller can discard everything that is not a
    live video of ours without a single ``files.get``. ``shortcutDetails`` belongs to
    that list too: an attended call arrives as a shortcut, and only its target's type
    says it is a recording.
    """
    entries: list[dict] = []
    cursor = page_token
    while True:
        response = (
            service.changes()
            .list(
                pageToken=cursor,
                fields=(
                    "nextPageToken, newStartPageToken, "
                    "changes(fileId, removed, "
                    "file(id, name, mimeType, parents, trashed, "
                    "shortcutDetails(targetMimeType)))"
                ),
                pageSize=PAGE_SIZE,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                includeRemoved=True,
            )
            .execute()
        )
        entries.extend(response.get("changes", []))
        next_page = _next_page_token(response)
        if next_page is None:
            new_cursor = response.get("newStartPageToken")
            return entries, new_cursor if isinstance(new_cursor, str) else cursor
        cursor = next_page


def find_configured_ancestor(
    service: Any,
    container_id: str,
    configured_ids: set[str] | frozenset[str],
    *,
    cache: dict[str, str | None] | None = None,
) -> str | None:
    """Return which configured folder ``container_id`` belongs to, or ``None``.

    A file's own folder no longer identifies the employee: it may be a per-meeting
    subfolder the configuration has never heard of. Everything that starts from a
    file rather than from the configuration needs this translation -- processing one
    file by id, and reading the changes feed, which reports every file the token can
    see and not only the ones we watch.

    ``None`` means "not ours" and must be treated as a skip, not as an error: the
    account sees folders nobody configured.

    Drive failures are raised, never folded into that ``None``. An expired token or a
    502 would otherwise be indistinguishable from "belongs to nobody", and the caller
    would skip a real recording believing it had decided something.

    Walks at most ``MAX_ANCESTOR_DEPTH`` levels. Meet nests meeting folders one level
    under the root, so the bound is slack rather than a limit, and it keeps a
    malformed or circular parent chain from costing unbounded requests.
    """
    if cache is not None and container_id in cache:
        return cache[container_id]

    found: str | None = None
    current = container_id
    for _ in range(MAX_ANCESTOR_DEPTH):
        if current in configured_ids:
            found = current
            break
        metadata = (
            service.files()
            .get(fileId=current, fields="id, parents", supportsAllDrives=True)
            .execute()
        )
        parents = metadata.get("parents") or []
        if not parents:
            break
        current = parents[0]

    if cache is not None:
        cache[container_id] = found
    return found


def names_a_recording(file_info: dict) -> bool:
    """Whether a listed file or change entry stands for a recording.

    Meet gives the organizer the recording and every other participant a shortcut to
    it, and a shortcut stays a shortcut whoever looks at it: Drive reports its own
    mime type and keeps the real one in ``shortcutDetails.targetMimeType``.
    """
    mime = file_info.get("mimeType")
    if mime == MP4_MIME:
        return True
    details = file_info.get("shortcutDetails") or {}
    return mime == SHORTCUT_MIME and details.get("targetMimeType") == MP4_MIME


def list_recording_shortcuts(service: Any, folder_id: str) -> list[dict]:
    """Shortcuts to recordings in a folder and its meeting subfolders.

    Meet files a call into every participant's folder, but only the organizer gets
    the recording itself -- everyone else gets a shortcut to it. A shortcut is followed
    only when its target opens, and whether it does depends on the organizer's sharing,
    not the folder's: on the first real employee folder checked, the account the folder
    was shared with could open none of them. ``doctor --drive`` uses this to say how
    many of a folder's attended calls it processes, and why not the rest.

    Returns ``[{id, name, container_id, target_id}]``; nothing is resolved here.
    """
    found: list[dict] = []
    for container_id in [folder_id, *(f["id"] for f in list_subfolders(service, folder_id))]:
        page_token: str | None = None
        while True:
            response = (
                service.files()
                .list(
                    q=(
                        f"'{container_id}' in parents and mimeType = '{SHORTCUT_MIME}' "
                        "and trashed = false"
                    ),
                    fields="nextPageToken, files(id, name, shortcutDetails)",
                    pageSize=PAGE_SIZE,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            for shortcut in response.get("files", []):
                details = shortcut.get("shortcutDetails") or {}
                if details.get("targetMimeType") != MP4_MIME:
                    continue
                found.append({
                    "id": shortcut.get("id"),
                    "name": shortcut.get("name", ""),
                    "container_id": container_id,
                    "target_id": details.get("targetId"),
                })
            page_token = _next_page_token(response)
            if not page_token:
                break
    return found


def get_shortcut_target(service: Any, target_id: str) -> dict | None:
    """The file a shortcut points at, or ``None`` when this account cannot open it.

    Drive answers a file you may not see with 404, exactly as it answers one that
    does not exist; for a shortcut's target those mean the same thing here, and so does
    a target in the bin. Anything else -- an outage, a revoked token -- is raised: folded
    into ``None`` it would drop a real call from the listing as if by decision.

    The fields are the ones a shortcut lacks: size, readiness and age describe the
    recording, and ``parents`` says whose folder it lives in.
    """
    try:
        target = (
            service.files()
            .get(
                fileId=target_id,
                fields=(
                    "id, name, mimeType, size, createdTime, parents, trashed, "
                    "videoMediaMetadata(durationMillis)"
                ),
                supportsAllDrives=True,
            )
            .execute()
        )
    except HttpError as exc:
        if getattr(exc.resp, "status", None) in (403, 404):
            return None
        raise
    if target.get("trashed"):
        return None
    return target


def _follow_recording_shortcut(
    service: Any, shortcut: dict, *, transcribed_here: bool
) -> tuple[dict, dict] | None:
    """A shortcut to a recording as ``(file, extra item fields)``, or ``None``.

    The file keeps the shortcut's id, name and appProperties: it is the object in this
    folder, so bookkeeping is written onto it and artifacts are paired with it -- the
    organizer's own file is theirs, and may not even be writable. Size, readiness and
    age come from the target, which is also what gets downloaded (``media_id``).

    A shortcut that already has a transcript beside it was followed before. Walking
    re-lists every folder every cycle, so its target is not asked about again: that
    would cost a request per attended call for good, for an answer nothing needs.
    """
    details = shortcut.get("shortcutDetails") or {}
    target_id = details.get("targetId")
    if not target_id:
        return None
    file = {
        "id": shortcut["id"],
        "name": shortcut.get("name", ""),
        "mimeType": SHORTCUT_MIME,
        "createdTime": shortcut.get("createdTime"),
        "appProperties": shortcut.get("appProperties") or {},
    }
    if transcribed_here:
        # Already done here, so the readiness gate has nothing left to wait on, and
        # whose folder the target lives in was settled when it was followed.
        return file, {
            "media_id": target_id, "target_parents": None, "has_media_metadata": True,
        }

    target = get_shortcut_target(service, target_id)
    if target is None:
        logger.debug(
            "Shortcut %s points at a recording this account cannot open", shortcut["id"]
        )
        return None
    file["size"] = target.get("size")
    file["createdTime"] = target.get("createdTime") or file["createdTime"]
    if target.get("videoMediaMetadata"):
        file["videoMediaMetadata"] = target["videoMediaMetadata"]
    return file, {"media_id": target_id, "target_parents": target.get("parents") or []}


def list_subfolders(service: Any, folder_id: str) -> list[dict]:
    """Return the direct subfolders of ``folder_id`` as ``[{id, name}]``.

    One level only, not a tree walk: Google Meet files every meeting into its own
    subfolder directly under the account's ``Google Meet`` folder, so there is no
    deeper nesting to chase and recursing would only invite cycles through shortcuts.
    """
    folders: list[dict] = []
    page_token: str | None = None
    query = (
        f"'{folder_id}' in parents and mimeType = '{FOLDER_MIME}' and trashed = false"
    )
    while True:
        response = (
            service.files()
            .list(
                q=query,
                fields="nextPageToken, files(id, name)",
                pageSize=PAGE_SIZE,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        folders.extend(response.get("files", []))
        page_token = _next_page_token(response)
        if page_token is None:
            break
    return folders


def list_folder_tree_state(service: Any, folder_id: str) -> list[dict]:
    """Return ``list_folder_state`` for ``folder_id`` and for each of its subfolders.

    A union, not a mode. A flat folder has no subfolders and yields exactly what it
    did before; a Google Meet root holds no mp4 of its own and yields only its
    per-meeting subfolders; a folder holding both yields both. Because the two shapes
    share one path, nothing in the configuration has to declare which kind a folder
    is, and the existing flat-folder tests stay honest as the regression guard.
    """
    items = list_folder_state(service, folder_id)
    for subfolder in list_subfolders(service, folder_id):
        items.extend(list_folder_state(service, subfolder["id"]))
    return items


def list_folder_state(service: Any, folder_id: str) -> list[dict]:
    """Return mp4 files with sibling flags.

    Each item is ``{file, has_mp3, has_txt, mp3_id, mp3_name, txt_id, stt_id,
    meta_yml_id, artifact_ids}`` where ``artifact_ids`` maps each produced preset's
    ``artifact_type`` appProperty to its Drive file id (e.g. ``{"keypoints": "k1"}``).
    The OpenAI stage consults this to skip presets that already have an artifact.
    Legacy ``<video-stem>.keypoints.md`` files uploaded before the appProperty
    existed are folded onto the ``keypoints`` preset by stem.

    A shortcut to a recording this account can open is an item too, and only such an
    item carries ``media_id`` (the file to download) and ``target_parents`` (where
    the recording lives, ``None`` when it was not looked up). Whether the organizer's
    folder is configured -- in which case that folder processes the call -- is the
    caller's decision; this module does not know the configuration.
    """
    found = _list_files_by_mimes(
        service, folder_id, (MP4_MIME, MP3_MIME, TXT_MIME, MD_MIME, SHORTCUT_MIME)
    )
    mp4_files = [f for f in found if f.get("mimeType") == MP4_MIME]
    recording_shortcuts = [
        f for f in found if f.get("mimeType") == SHORTCUT_MIME and names_a_recording(f)
    ]
    mp3_files = [f for f in found if f.get("mimeType") == MP3_MIME]
    text_files = [f for f in found if f.get("mimeType") == TXT_MIME]
    md_files = [f for f in found if f.get("mimeType") == MD_MIME]

    # ``text/plain`` is shared by the transcript (``.txt``), the assembled call
    # document (``.stt``), and the merged meta document (``.meta.yml``). Splitting
    # by name suffix keeps the transcript lookup (below) scoped to real transcripts:
    # without this, ``drive_stem("video.stt") == "video"`` would collide with
    # ``video.txt`` in the legacy stem index, and a ``.stt`` could get fed to the
    # preset stage as if it were the transcript, or overwritten by one.
    txt_files = [f for f in text_files if f["name"].endswith(".txt")]
    stt_files = [f for f in text_files if f["name"].endswith(".stt")]
    meta_yml_files = [f for f in text_files if f["name"].endswith(".meta.yml")]

    mp3_by_stem = {drive_stem(f["name"]): f for f in mp3_files}
    txt_by_stem = {drive_stem(f["name"]): f for f in txt_files}
    # `.stt`/`.meta.yml` carry no `source_video_id` (dropped from the `.stt` upload
    # specifically to keep it out of `txt_by_source_id` below; `.meta.yml` never had
    # one), so a reprocess is matched by stem alone -- same as the legacy keypoints
    # fallback, and reliable here because both names are always exactly
    # ``<stem>.stt``/``<stem>.meta.yml``.
    stt_by_stem = {drive_stem(f["name"]): f for f in stt_files}
    meta_yml_by_stem = {
        _strip_suffix(f["name"], ".meta.yml"): f for f in meta_yml_files
    }
    # Keypoints are uploaded as ``<video-stem>.keypoints.md``; strip the
    # ``.keypoints`` suffix so they match back to the source video stem.
    #
    # The robust link is the ``source_video_id`` appProperty (see
    # ``artifacts_by_source_id``); this bare-stem fallback only exists for
    # legacy artifacts uploaded before that property was set. To avoid
    # false-matching a user-authored ``<something>.keypoints.md`` file as the
    # generated artifact of ``<something>.mp4`` (which would wrongly skip
    # regeneration), only fold a ``.md`` file into the stem index when it both
    # uses the ``.keypoints.md`` naming convention AND carries no
    # ``source_video_id`` pointing at some other video.
    keypoints_by_stem: dict[str, dict] = {}
    for f in md_files:
        stem = drive_stem(f["name"])
        if not stem.endswith(".keypoints"):
            continue
        if f.get("appProperties", {}).get(SOURCE_VIDEO_ID_PROPERTY):
            # Authoritatively linked via source_video_id; handled separately.
            continue
        keypoints_by_stem[_strip_keypoints_suffix(stem)] = f
    mp3_by_source_id = _files_by_source_video_id(mp3_files)
    txt_by_source_id = _files_by_source_video_id(txt_files)
    artifacts_by_source_id = _artifacts_by_source_video_id(md_files)

    recordings: list[tuple[dict, dict]] = [(mp4, {}) for mp4 in mp4_files]
    for shortcut in recording_shortcuts:
        transcribed_here = (
            txt_by_source_id.get(shortcut["id"])
            or txt_by_stem.get(drive_stem(shortcut.get("name", "")))
        ) is not None
        followed = _follow_recording_shortcut(
            service, shortcut, transcribed_here=transcribed_here
        )
        if followed is not None:
            recordings.append(followed)

    items: list[dict] = []
    for mp4, extra in recordings:
        stem = drive_stem(mp4["name"])
        mp3 = mp3_by_source_id.get(mp4["id"]) or mp3_by_stem.get(stem)
        txt = txt_by_source_id.get(mp4["id"]) or txt_by_stem.get(stem)
        stt = stt_by_stem.get(stem)
        meta_yml = meta_yml_by_stem.get(stem)

        artifact_ids: dict[str, str] = {}
        # Legacy bare-stem keypoints first; an authoritative source_video_id
        # match (below) overrides it for the same artifact_type.
        legacy_keypoints = keypoints_by_stem.get(stem)
        if legacy_keypoints is not None:
            artifact_ids["keypoints"] = legacy_keypoints["id"]
        for artifact_type, f in artifacts_by_source_id.get(mp4["id"], {}).items():
            artifact_ids[artifact_type] = f["id"]

        mp4_props = mp4.get("appProperties", {}) or {}
        items.append({
            "file": mp4,
            # The folder this file actually lives in, and therefore the folder its
            # artifacts must be written back to. Once subfolders are walked this is
            # no longer the configured folder the caller started from, and the two
            # must not be confused: the configured one identifies the employee,
            # this one addresses the files.
            "container_id": folder_id,
            # Drive fills videoMediaMetadata once it has finished processing an
            # upload. Its absence is the cheapest available "still settling" signal;
            # the caller decides how long to honour it, because a video that never
            # gets metadata must not wait forever.
            "has_media_metadata": bool(mp4.get("videoMediaMetadata")),
            "has_mp3": mp3 is not None,
            "has_txt": txt is not None,
            "mp3_id": mp3["id"] if mp3 else None,
            "mp3_name": mp3["name"] if mp3 else None,
            "txt_id": txt["id"] if txt else None,
            # Not consulted by any pending/processed decision (see
            # ``_write_call_documents`` in main.py): a reprocess uses these purely
            # to overwrite the previous ``.stt``/``.meta.yml`` in place on Drive
            # instead of leaving a duplicate behind.
            "stt_id": stt["id"] if stt else None,
            "meta_yml_id": meta_yml["id"] if meta_yml else None,
            "artifact_ids": artifact_ids,
            "booking_match": mp4_props.get(BOOKING_MATCH_PROPERTY, ""),
            "planfix_comment_task_id": mp4_props.get(
                PLANFIX_COMMENT_TASK_ID_PROPERTY, ""
            ),
            "telegram_sent_chat_id": mp4_props.get(
                TELEGRAM_SENT_CHAT_ID_PROPERTY, ""
            ),
            **extra,
        })
    return items


def _strip_suffix(name: str, suffix: str) -> str:
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _strip_keypoints_suffix(stem: str) -> str:
    return stem[: -len(".keypoints")] if stem.endswith(".keypoints") else stem


def _files_by_source_video_id(files: list[dict]) -> dict[str, dict]:
    by_source_id: dict[str, dict] = {}
    for item in files:
        source_id = item.get("appProperties", {}).get(SOURCE_VIDEO_ID_PROPERTY)
        if source_id:
            by_source_id[source_id] = item
    return by_source_id


def _artifacts_by_source_video_id(files: list[dict]) -> dict[str, dict[str, dict]]:
    """Group preset artifacts by source video id, then by ``artifact_type``.

    Returns ``{source_video_id: {artifact_type: file}}``. A markdown artifact
    carrying ``source_video_id`` but no ``artifact_type`` is assumed to be a legacy
    keypoints document (the only artifact type that predates multi-preset support).
    """
    by_source: dict[str, dict[str, dict]] = {}
    for item in files:
        props = item.get("appProperties", {})
        source_id = props.get(SOURCE_VIDEO_ID_PROPERTY)
        if not source_id:
            continue
        artifact_type = props.get(ARTIFACT_TYPE_PROPERTY) or "keypoints"
        by_source.setdefault(source_id, {})[artifact_type] = item
    return by_source


def download(
    service: Any,
    file_id: str,
    dest_dir: Path,
    file_name: str,
    *,
    expected_size_bytes: int | None = None,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_local_name(file_name)
    if not safe_name or safe_name in {".", ".."}:
        raise ValueError(f"Invalid file name from Drive: {file_name!r}")
    dest_path = dest_dir / safe_name

    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with io.FileIO(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status is not None:
                logger.debug(
                    "Downloading %s: %d%%", safe_name, int(status.progress() * 100)
                )
    if expected_size_bytes is not None:
        actual_size = dest_path.stat().st_size
        if actual_size != expected_size_bytes:
            dest_path.unlink(missing_ok=True)
            raise DownloadIntegrityError(
                f"Downloaded file size mismatch for {file_name}: expected "
                f"{expected_size_bytes} bytes, got {actual_size}"
            )
    return dest_path


def download_text(service: Any, file_id: str) -> str:
    """Download a small text/markdown Drive file's content into memory.

    Used to re-feed an existing transcript into the preset stage without
    re-running STT when a Drive ``.txt`` sibling already exists but some preset
    artifact is still missing.
    """
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _status, done = downloader.next_chunk()
    return buffer.getvalue().decode("utf-8-sig")


def upload(
    service: Any,
    local_path: Path,
    folder_id: str,
    mime_type: str = MP3_MIME,
    name: str | None = None,
    app_properties: dict[str, str] | None = None,
) -> dict:
    metadata = {"name": name or local_path.name, "parents": [folder_id]}
    if app_properties:
        metadata["appProperties"] = app_properties
    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
    response = (
        service.files()
        .create(
            body=metadata,
            media_body=media,
            fields="id, name, parents",
            supportsAllDrives=True,
        )
        .execute()
    )
    return response


def update_file(
    service: Any,
    file_id: str,
    local_path: Path,
    mime_type: str = TXT_MIME,
    app_properties: dict[str, str] | None = None,
) -> dict:
    """Overwrite an existing Drive file's content in place (keeps id and name)."""
    metadata = {"appProperties": app_properties} if app_properties else None
    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
    kwargs: dict[str, Any] = {
        "fileId": file_id,
        "media_body": media,
        "fields": "id, name",
        "supportsAllDrives": True,
    }
    if metadata:
        kwargs["body"] = metadata
    response = (
        service.files()
        .update(**kwargs)
        .execute()
    )
    return response


def set_file_app_properties(
    service: Any,
    file_id: str,
    app_properties: dict[str, str | None],
) -> dict:
    """Merge appProperties onto a Drive file without changing its content.

    Drive counts every ``files.update`` as an edit: it moves ``modifiedTime``, sets
    ``lastModifyingUser`` and appends "You edited an item" to the activity feed.
    These properties are our own bookkeeping, not a user edit, and people sort these
    shared folders by "Last modified" -- so the date has to survive the write.
    Reading the current value and sending it straight back in the same request keeps
    it exactly where it was.

    Preservation is unconditional rather than opt-in: every call site writes
    bookkeeping, and a flag is something a future call site forgets to pass.
    """
    current = (
        service.files()
        .get(fileId=file_id, fields="modifiedTime", supportsAllDrives=True)
        .execute()
    )
    body: dict[str, Any] = {"appProperties": app_properties}
    modified_time = current.get("modifiedTime")
    if modified_time:
        # A blank value would clear the date rather than preserve it, so only send
        # one Drive actually gave us.
        body["modifiedTime"] = modified_time
    return (
        service.files()
        .update(
            fileId=file_id,
            body=body,
            fields="id, name, appProperties",
            supportsAllDrives=True,
        )
        .execute()
    )
