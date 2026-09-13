from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from src import drive


# A fixture file that belongs to whichever folder is being queried. Tests that
# predate subfolders say "these files are in the folder under test" and mean it; only
# tests that exercise the parent filter itself need to name real parents.
_ANY_PARENT = "*"

_PARENT_RE = re.compile(r"'([^']+)' in parents")
_MIME_RE = re.compile(r"mimeType\s*=\s*'([^']+)'")


def _make_drive_service(
    files: list[dict],
    *,
    changes: list[dict] | None = None,
    start_page_token: str = "tok-start",
    new_start_page_token: str = "tok-next",
    unreadable: tuple[str, ...] = (),
) -> MagicMock:
    """A tiny in-memory Drive: ``files().list`` really filters by parent and mimeType.

    The fake this replaced dispatched on a substring of ``q`` and answered anything it
    did not recognise with an empty list. That is the worst possible default for a
    fake: a query for subfolders fell through to it, so a test walking into subfolders
    passed while discovering none of them. Parsing the query instead means an
    unsupported filter shows up as a wrong answer, not a silently empty one.

    Every file in ``files`` carries ``parents``; ``_ANY_PARENT`` matches whatever
    folder is asked about, which is what the pre-subfolder tests assume.
    """
    service = MagicMock()
    files_resource = MagicMock()
    service.files.return_value = files_resource

    def matches(f: dict, parent: str | None, mimes: set[str], want_untrashed: bool) -> bool:
        if want_untrashed and f.get("trashed"):
            return False
        if mimes and f.get("mimeType") not in mimes:
            return False
        if parent is not None:
            owners = f.get("parents", [_ANY_PARENT])
            if _ANY_PARENT not in owners and parent not in owners:
                return False
        return True

    def list_side_effect(**kwargs):
        q = kwargs.get("q", "")
        parent_match = _PARENT_RE.search(q)
        parent = parent_match.group(1) if parent_match else None
        mimes = set(_MIME_RE.findall(q))
        want_untrashed = "trashed = false" in q

        found = [f for f in files if matches(f, parent, mimes, want_untrashed)]

        order_by = kwargs.get("orderBy") or ""
        if order_by.startswith("createdTime desc"):
            found.sort(key=lambda f: f.get("createdTime", ""), reverse=True)

        page_size = kwargs.get("pageSize") or len(found) or 1
        start = int(kwargs.get("pageToken") or 0)
        page = found[start : start + page_size]
        body: dict = {"files": page}
        if start + page_size < len(found):
            body["nextPageToken"] = str(start + page_size)

        request = MagicMock()
        request.execute.return_value = body
        return request

    files_resource.list.side_effect = list_side_effect

    by_id = {f["id"]: f for f in files}

    def get_side_effect(**kwargs):
        request = MagicMock()
        if kwargs["fileId"] in unreadable:
            # What Drive answers for a file this account may not see.
            from googleapiclient.errors import HttpError

            request.execute.side_effect = HttpError(MagicMock(status=404), b"")
        else:
            request.execute.return_value = by_id.get(kwargs["fileId"], {})
        return request

    files_resource.get.side_effect = get_side_effect

    changes_resource = MagicMock()
    service.changes.return_value = changes_resource
    changes_resource.getStartPageToken.return_value.execute.return_value = {
        "startPageToken": start_page_token
    }

    def changes_list_side_effect(**kwargs):
        # `changes` may be a flat list (one page) or a list of pages.
        pages = changes or []
        if pages and not isinstance(pages[0], list):
            pages = [pages]
        # The first cursor is an opaque token from Drive; only the fake's own
        # continuation tokens are page indexes.
        raw = str(kwargs.get("pageToken") or "")
        index = int(raw) if raw.isdigit() else 0
        body: dict = {"changes": list(pages[index]) if index < len(pages) else []}
        if index + 1 < len(pages):
            body["nextPageToken"] = str(index + 1)
        else:
            body["newStartPageToken"] = new_start_page_token
        request = MagicMock()
        request.execute.return_value = body
        return request

    changes_resource.list.side_effect = changes_list_side_effect
    return service


def _make_list_service(pages_by_query: dict[str, list[dict]]) -> MagicMock:
    """Build the tiny Drive from the old per-mime buckets, for tests that predate parents."""
    mime_by_key = {
        "mp4": drive.MP4_MIME,
        "mp3": drive.MP3_MIME,
        "txt": drive.TXT_MIME,
        "md": drive.MD_MIME,
    }
    files: list[dict] = []
    for key, mime in mime_by_key.items():
        for f in pages_by_query.get(key, []):
            files.append({"mimeType": mime, "parents": [_ANY_PARENT], **f})
    return _make_drive_service(files)


def test_download_writes_file_to_dest_dir(tmp_path, mocker):
    service = MagicMock()
    request = MagicMock()
    service.files.return_value.get_media.return_value = request

    chunks_done = [False, False, True]

    def make_downloader(fh, _request):
        downloader = MagicMock()
        # Write something to the file handle on each call
        state = {"i": 0}

        def next_chunk():
            i = state["i"]
            state["i"] += 1
            fh.write(b"chunk")
            status = MagicMock()
            status.progress.return_value = (i + 1) / len(chunks_done)
            return status, chunks_done[i]

        downloader.next_chunk.side_effect = next_chunk
        return downloader

    mocker.patch("src.drive.MediaIoBaseDownload", side_effect=make_downloader)

    dest = tmp_path / "downloads"
    result = drive.download(service, "fileid123", dest, "video.mp4")

    assert result == dest / "video.mp4"
    assert result.exists()
    assert result.read_bytes() == b"chunk" * 3
    service.files.return_value.get_media.assert_called_once_with(
        fileId="fileid123", supportsAllDrives=True
    )


def test_download_creates_missing_dest_dir(tmp_path, mocker):
    service = MagicMock()
    service.files.return_value.get_media.return_value = MagicMock()

    def make_downloader(fh, _request):
        downloader = MagicMock()
        downloader.next_chunk.return_value = (None, True)
        return downloader

    mocker.patch("src.drive.MediaIoBaseDownload", side_effect=make_downloader)

    dest = tmp_path / "new" / "nested" / "dir"
    result = drive.download(service, "fid", dest, "f.mp4")

    assert dest.is_dir()
    assert result.parent == dest


def test_download_raises_on_size_mismatch_and_cleans_partial_file(tmp_path, mocker):
    service = MagicMock()
    service.files.return_value.get_media.return_value = MagicMock()

    def make_downloader(fh, _request):
        downloader = MagicMock()

        def next_chunk():
            fh.write(b"short")
            return None, True

        downloader.next_chunk.side_effect = next_chunk
        return downloader

    mocker.patch("src.drive.MediaIoBaseDownload", side_effect=make_downloader)

    with pytest.raises(RuntimeError, match="size mismatch"):
        drive.download(
            service,
            "fid",
            tmp_path,
            "video.mp4",
            expected_size_bytes=10,
        )

    assert not (tmp_path / "video.mp4").exists()


def test_upload_calls_create_with_metadata_and_media(tmp_path, mocker):
    local = tmp_path / "audio.mp3"
    local.write_bytes(b"id3-data")

    service = MagicMock()
    create_request = MagicMock()
    create_request.execute.return_value = {"id": "new123", "name": "audio.mp3", "parents": ["fld"]}
    service.files.return_value.create.return_value = create_request

    media_cls = mocker.patch("src.drive.MediaFileUpload", return_value="media-obj")

    result = drive.upload(service, local, "fld")

    assert result == {"id": "new123", "name": "audio.mp3", "parents": ["fld"]}
    media_cls.assert_called_once_with(str(local), mimetype="audio/mpeg", resumable=True)
    service.files.return_value.create.assert_called_once_with(
        body={"name": "audio.mp3", "parents": ["fld"]},
        media_body="media-obj",
        fields="id, name, parents",
        supportsAllDrives=True,
    )


def test_upload_accepts_app_properties(tmp_path, mocker):
    local = tmp_path / "audio.mp3"
    local.write_bytes(b"id3-data")

    service = MagicMock()
    service.files.return_value.create.return_value.execute.return_value = {"id": "x"}
    mocker.patch("src.drive.MediaFileUpload", return_value="media")

    drive.upload(
        service,
        local,
        "fld",
        app_properties={"source_video_id": "v1", "artifact_type": "mp3"},
    )

    body = service.files.return_value.create.call_args.kwargs["body"]
    assert body["appProperties"] == {
        "source_video_id": "v1",
        "artifact_type": "mp3",
    }


def test_upload_accepts_custom_mime_type(tmp_path, mocker):
    local = tmp_path / "video.mp4"
    local.write_bytes(b"mp4-data")

    service = MagicMock()
    service.files.return_value.create.return_value.execute.return_value = {"id": "x"}

    media_cls = mocker.patch("src.drive.MediaFileUpload", return_value="media")

    drive.upload(service, local, "fld", mime_type="video/mp4")

    media_cls.assert_called_once_with(str(local), mimetype="video/mp4", resumable=True)


def test_update_file_overwrites_in_place(tmp_path, mocker):
    local = tmp_path / "video.txt"
    local.write_text("final transcript", encoding="utf-8")

    service = MagicMock()
    update_request = MagicMock()
    update_request.execute.return_value = {"id": "t1", "name": "video.txt"}
    service.files.return_value.update.return_value = update_request

    media_cls = mocker.patch("src.drive.MediaFileUpload", return_value="media-obj")

    result = drive.update_file(service, "t1", local)

    assert result == {"id": "t1", "name": "video.txt"}
    media_cls.assert_called_once_with(str(local), mimetype="text/plain", resumable=True)
    service.files.return_value.update.assert_called_once_with(
        fileId="t1",
        media_body="media-obj",
        fields="id, name",
        supportsAllDrives=True,
    )


def test_update_file_accepts_app_properties(tmp_path, mocker):
    local = tmp_path / "video.txt"
    local.write_text("final transcript", encoding="utf-8")

    service = MagicMock()
    service.files.return_value.update.return_value.execute.return_value = {"id": "t1"}
    mocker.patch("src.drive.MediaFileUpload", return_value="media-obj")

    drive.update_file(
        service,
        "t1",
        local,
        app_properties={"source_video_id": "v1", "artifact_type": "txt"},
    )

    kwargs = service.files.return_value.update.call_args.kwargs
    assert kwargs["body"]["appProperties"] == {
        "source_video_id": "v1",
        "artifact_type": "txt",
    }


def test_set_file_app_properties_updates_metadata_only():
    service = MagicMock()
    service.files.return_value.get.return_value.execute.return_value = {
        "modifiedTime": "2026-01-02T03:04:05.678Z"
    }
    service.files.return_value.update.return_value.execute.return_value = {"id": "v1"}

    drive.set_file_app_properties(service, "v1", {"speaker_names": "[\"A\", \"B\"]"})

    service.files.return_value.update.assert_called_once_with(
        fileId="v1",
        body={
            "appProperties": {"speaker_names": "[\"A\", \"B\"]"},
            "modifiedTime": "2026-01-02T03:04:05.678Z",
        },
        fields="id, name, appProperties",
        supportsAllDrives=True,
    )


def test_set_file_app_properties_preserves_the_modified_time():
    """Writing a property must not move the file's date.

    Drive counts any files.update as an edit. People sort these shared folders by
    "Last modified", so a bookkeeping write that bumps the date corrupts the column.
    """
    service = MagicMock()
    service.files.return_value.get.return_value.execute.return_value = {
        "modifiedTime": "2025-03-14T18:24:53.633Z"
    }
    service.files.return_value.update.return_value.execute.return_value = {"id": "v1"}

    drive.set_file_app_properties(service, "v1", {"booking_match": "none"})

    service.files.return_value.get.assert_called_once_with(
        fileId="v1",
        fields="modifiedTime",
        supportsAllDrives=True,
    )
    service.files.return_value.update.assert_called_once_with(
        fileId="v1",
        body={
            "appProperties": {"booking_match": "none"},
            "modifiedTime": "2025-03-14T18:24:53.633Z",
        },
        fields="id, name, appProperties",
        supportsAllDrives=True,
    )


def test_set_file_app_properties_preserves_the_date_when_deleting_a_property():
    """The rematch path sends a null value to delete a property; same rule applies."""
    service = MagicMock()
    service.files.return_value.get.return_value.execute.return_value = {
        "modifiedTime": "2025-03-14T18:24:53.633Z"
    }
    service.files.return_value.update.return_value.execute.return_value = {"id": "v1"}

    drive.set_file_app_properties(service, "v1", {"booking_match": None})

    body = service.files.return_value.update.call_args.kwargs["body"]
    assert body["appProperties"] == {"booking_match": None}
    assert body["modifiedTime"] == "2025-03-14T18:24:53.633Z"


def test_set_file_app_properties_omits_modified_time_when_drive_returns_none():
    """A response without modifiedTime must not send a null date and 400 the request."""
    service = MagicMock()
    service.files.return_value.get.return_value.execute.return_value = {}
    service.files.return_value.update.return_value.execute.return_value = {"id": "v1"}

    drive.set_file_app_properties(service, "v1", {"booking_match": "none"})

    body = service.files.return_value.update.call_args.kwargs["body"]
    assert "modifiedTime" not in body


def test_list_folder_state_includes_txt_id():
    mp4 = [
        {"id": "v1", "name": "a.mp4", "mimeType": "video/mp4"},
        {"id": "v2", "name": "b.mp4", "mimeType": "video/mp4"},
    ]
    txt = [{"id": "t1", "name": "a.txt", "mimeType": "text/plain"}]
    service = _make_list_service({"mp4": mp4, "mp3": [], "txt": txt})

    items = drive.list_folder_state(service, "folder1")

    by_id = {it["file"]["id"]: it for it in items}
    assert by_id["v1"]["txt_id"] == "t1"
    assert by_id["v2"]["txt_id"] is None


def test_list_folder_state_txt_id_ignores_a_sibling_stt():
    """A `.stt` shares mimeType text/plain with the transcript and, before drive_stem
    strips the extension, its stem collides with the `.txt`'s ("a.stt" -> "a", same as
    "a.txt" -> "a"). It must never be mistaken for the transcript: the preset stage
    would be fed the assembled .stt document instead of the real transcript, and a
    fresh `.txt` write would overwrite the .stt's content.
    """
    mp4 = [{"id": "v1", "name": "a.mp4", "mimeType": "video/mp4"}]
    # `.stt` listed after `.txt` so a naive last-write-wins dict would pick it.
    txt = [
        {"id": "t1", "name": "a.txt", "mimeType": "text/plain"},
        {"id": "s1", "name": "a.stt", "mimeType": "text/plain"},
    ]
    service = _make_list_service({"mp4": mp4, "mp3": [], "txt": txt})

    items = drive.list_folder_state(service, "folder1")

    assert items[0]["txt_id"] == "t1"
    assert items[0]["has_txt"] is True


def test_list_folder_state_txt_id_ignores_an_stt_carrying_source_video_id():
    """Even if a `.stt` carried `source_video_id` (it should not, per the .stt
    upload, but this is the belt-and-suspenders side of that fix), the transcript
    lookup must still resolve to the real `.txt`, not the `.stt`.
    """
    mp4 = [{"id": "v1", "name": "a.mp4", "mimeType": "video/mp4"}]
    txt = [
        {
            "id": "s1",
            "name": "a.stt",
            "mimeType": "text/plain",
            "appProperties": {"source_video_id": "v1", "artifact_type": "stt"},
        },
        {"id": "t1", "name": "a.txt", "mimeType": "text/plain"},
    ]
    service = _make_list_service({"mp4": mp4, "mp3": [], "txt": txt})

    items = drive.list_folder_state(service, "folder1")

    assert items[0]["txt_id"] == "t1"


def test_list_folder_state_includes_stt_id_and_meta_yml_id_by_stem():
    mp4 = [{"id": "v1", "name": "a.mp4", "mimeType": "video/mp4"}]
    txt = [
        {"id": "s1", "name": "a.stt", "mimeType": "text/plain"},
        {"id": "y1", "name": "a.meta.yml", "mimeType": "text/plain"},
    ]
    service = _make_list_service({"mp4": mp4, "mp3": [], "txt": txt})

    items = drive.list_folder_state(service, "folder1")

    assert items[0]["stt_id"] == "s1"
    assert items[0]["meta_yml_id"] == "y1"
    # No .txt present: the .stt/.meta.yml siblings must not be mistaken for one.
    assert items[0]["txt_id"] is None
    assert items[0]["has_txt"] is False


def test_list_folder_state_stt_id_and_meta_yml_id_default_to_none():
    mp4 = [{"id": "v1", "name": "a.mp4", "mimeType": "video/mp4"}]
    txt = [{"id": "t1", "name": "a.txt", "mimeType": "text/plain"}]
    service = _make_list_service({"mp4": mp4, "mp3": [], "txt": txt})

    items = drive.list_folder_state(service, "folder1")

    assert items[0]["stt_id"] is None
    assert items[0]["meta_yml_id"] is None


def test_list_folder_state_includes_keypoints_id_by_stem():
    mp4 = [
        {"id": "v1", "name": "a.mp4", "mimeType": "video/mp4"},
        {"id": "v2", "name": "b.mp4", "mimeType": "video/mp4"},
    ]
    md = [{"id": "k1", "name": "a.keypoints.md", "mimeType": "text/markdown"}]
    service = _make_list_service({"mp4": mp4, "mp3": [], "txt": [], "md": md})

    items = drive.list_folder_state(service, "folder1")

    by_id = {it["file"]["id"]: it for it in items}
    assert by_id["v1"]["artifact_ids"] == {"keypoints": "k1"}
    assert by_id["v2"]["artifact_ids"] == {}


def test_list_folder_state_matches_keypoints_by_source_video_id_after_rename():
    mp4 = [{"id": "v1", "name": "new name.mp4", "mimeType": "video/mp4"}]
    md = [{
        "id": "k1",
        "name": "old name.keypoints.md",
        "mimeType": "text/markdown",
        "appProperties": {"source_video_id": "v1", "artifact_type": "keypoints"},
    }]
    service = _make_list_service({"mp4": mp4, "mp3": [], "txt": [], "md": md})

    items = drive.list_folder_state(service, "folder1")

    assert items[0]["artifact_ids"] == {"keypoints": "k1"}


def test_list_folder_state_keys_artifact_ids_by_artifact_type():
    # Multiple presets produce multiple sibling .md artifacts; each is keyed by its
    # own artifact_type appProperty so the OpenAI stage can skip ones already made.
    mp4 = [{"id": "v1", "name": "a.mp4", "mimeType": "video/mp4"}]
    md = [
        {
            "id": "k1",
            "name": "a.keypoints.md",
            "mimeType": "text/markdown",
            "appProperties": {"source_video_id": "v1", "artifact_type": "keypoints"},
        },
        {
            "id": "c1",
            "name": "a.transcript-cleanup.md",
            "mimeType": "text/markdown",
            "appProperties": {
                "source_video_id": "v1",
                "artifact_type": "transcript-cleanup",
            },
        },
    ]
    service = _make_list_service({"mp4": mp4, "mp3": [], "txt": [], "md": md})

    items = drive.list_folder_state(service, "folder1")

    assert items[0]["artifact_ids"] == {
        "keypoints": "k1",
        "transcript-cleanup": "c1",
    }


def test_list_folder_state_plain_user_md_not_matched_by_stem():
    # A plain "talk.md" (no ".keypoints" convention) must never match "talk.mp4".
    mp4 = [{"id": "v1", "name": "talk.mp4", "mimeType": "video/mp4"}]
    md = [{"id": "u1", "name": "talk.md", "mimeType": "text/markdown"}]
    service = _make_list_service({"mp4": mp4, "mp3": [], "txt": [], "md": md})

    items = drive.list_folder_state(service, "folder1")

    assert items[0]["artifact_ids"] == {}


def test_list_folder_state_keypoints_md_linked_to_other_video_not_stem_matched():
    # "notes.keypoints.md" is an artifact of a DIFFERENT video (source_video_id
    # points elsewhere); it must not be folded into the stem index and falsely
    # attached to "notes.mp4".
    mp4 = [{"id": "v1", "name": "notes.mp4", "mimeType": "video/mp4"}]
    md = [{
        "id": "k1",
        "name": "notes.keypoints.md",
        "mimeType": "text/markdown",
        "appProperties": {"source_video_id": "other", "artifact_type": "keypoints"},
    }]
    service = _make_list_service({"mp4": mp4, "mp3": [], "txt": [], "md": md})

    items = drive.list_folder_state(service, "folder1")

    assert items[0]["artifact_ids"] == {}


def test_get_file_metadata_requests_expected_fields():
    service = MagicMock()
    get_request = MagicMock()
    get_request.execute.return_value = {
        "id": "fid",
        "name": "video.mp4",
        "mimeType": "video/mp4",
        "parents": ["parent1"],
    }
    service.files.return_value.get.return_value = get_request

    result = drive.get_file_metadata(service, "fid")

    assert result["id"] == "fid"
    assert result["parents"] == ["parent1"]
    service.files.return_value.get.assert_called_once_with(
        fileId="fid",
        fields="id, name, mimeType, parents, size, appProperties",
        supportsAllDrives=True,
    )


def test_list_folder_state_returns_flags():
    mp4 = [
        {"id": "v1", "name": "a.mp4", "mimeType": "video/mp4"},
        {"id": "v2", "name": "b.mp4", "mimeType": "video/mp4"},
        {"id": "v3", "name": "c.mp4", "mimeType": "video/mp4"},
    ]
    mp3 = [
        {"id": "m1", "name": "a.mp3", "mimeType": "audio/mpeg"},
        {"id": "m2", "name": "b.mp3", "mimeType": "audio/mpeg"},
    ]
    txt = [{"id": "t1", "name": "a.txt", "mimeType": "text/plain"}]
    service = _make_list_service({"mp4": mp4, "mp3": mp3, "txt": txt})

    items = drive.list_folder_state(service, "folder1")

    assert len(items) == 3
    by_id = {it["file"]["id"]: it for it in items}
    assert by_id["v1"]["has_mp3"] is True
    assert by_id["v1"]["has_txt"] is True
    assert by_id["v1"]["mp3_id"] == "m1"
    assert by_id["v1"]["mp3_name"] == "a.mp3"
    assert by_id["v2"]["has_mp3"] is True
    assert by_id["v2"]["has_txt"] is False
    assert by_id["v2"]["mp3_id"] == "m2"
    assert by_id["v3"]["has_mp3"] is False
    assert by_id["v3"]["has_txt"] is False
    assert by_id["v3"]["mp3_id"] is None


def test_list_folder_state_matches_siblings_by_source_video_id_after_rename():
    mp4 = [{"id": "v1", "name": "new name.mp4", "mimeType": "video/mp4"}]
    mp3 = [{
        "id": "m1",
        "name": "old name.mp3",
        "mimeType": "audio/mpeg",
        "appProperties": {"source_video_id": "v1", "artifact_type": "mp3"},
    }]
    txt = [{
        "id": "t1",
        "name": "old name.txt",
        "mimeType": "text/plain",
        "appProperties": {"source_video_id": "v1", "artifact_type": "txt"},
    }]
    service = _make_list_service({"mp4": mp4, "mp3": mp3, "txt": txt})

    items = drive.list_folder_state(service, "folder1")

    assert items[0]["has_mp3"] is True
    assert items[0]["mp3_id"] == "m1"
    assert items[0]["has_txt"] is True
    assert items[0]["txt_id"] == "t1"


def test_drive_stem_preserves_slashes():
    name = "Call - 2026/05/28 17:27 GMT+04:00 – Recording.mp4"
    assert drive.drive_stem(name) == "Call - 2026/05/28 17:27 GMT+04:00 – Recording"


def test_drive_stem_plain_name():
    assert drive.drive_stem("video.mp4") == "video"


def test_safe_local_name_replaces_separators():
    name = "Call - 2026/05/28 – Recording.mp4"
    safe = drive.safe_local_name(name)
    assert "/" not in safe
    assert safe == "Call - 2026_05_28 – Recording.mp4"


def test_safe_local_name_replaces_windows_reserved_characters():
    name = 'Call - 2026/05/28 17:27 GMT+04:00 <final>|draft?.mp4'
    safe = drive.safe_local_name(name)

    for char in '<>:"/\\|?*\0':
        assert char not in safe
    assert safe == "Call - 2026_05_28 17_27 GMT+04_00 _final__draft_.mp4"


def test_list_folder_state_matches_siblings_with_slashes():
    mp4 = [{"id": "v1", "name": "Call 2026/05/28 Rec.mp4", "mimeType": "video/mp4"}]
    mp3 = [{"id": "m1", "name": "Call 2026/05/28 Rec.mp3", "mimeType": "audio/mpeg"}]
    txt = [{"id": "t1", "name": "Call 2026/05/28 Rec.txt", "mimeType": "text/plain"}]
    service = _make_list_service({"mp4": mp4, "mp3": mp3, "txt": txt})

    items = drive.list_folder_state(service, "folder1")

    assert len(items) == 1
    assert items[0]["has_mp3"] is True
    assert items[0]["has_txt"] is True
    assert items[0]["mp3_id"] == "m1"


def test_download_sanitizes_slash_name(tmp_path, mocker):
    service = MagicMock()
    service.files.return_value.get_media.return_value = MagicMock()

    def make_downloader(fh, _request):
        downloader = MagicMock()
        downloader.next_chunk.return_value = (None, True)
        return downloader

    mocker.patch("src.drive.MediaIoBaseDownload", side_effect=make_downloader)

    path = drive.download(service, "fid", tmp_path, "Call 2026/05/28 Rec.mp4")

    # No characters dropped, no unintended subdirectory created from the "/".
    assert path.parent == tmp_path
    assert path.name == "Call 2026_05_28 Rec.mp4"
    assert path.exists()


def test_upload_explicit_name_overrides_local_name(tmp_path, mocker):
    local = tmp_path / "Call 2026_05_28 Rec.mp3"
    local.write_bytes(b"abc")

    service = MagicMock()
    service.files.return_value.create.return_value.execute.return_value = {"id": "x"}
    mocker.patch("src.drive.MediaFileUpload", return_value="media")

    drive.upload(service, local, "fld", drive.MP3_MIME, name="Call 2026/05/28 Rec.mp3")

    create_kwargs = service.files.return_value.create.call_args.kwargs
    assert create_kwargs["body"]["name"] == "Call 2026/05/28 Rec.mp3"


def test_find_newest_mp4_returns_first_file():
    service = MagicMock()
    newest = {"id": "v9", "name": "newest.mp4", "mimeType": "video/mp4"}
    service.files.return_value.list.return_value.execute.return_value = {
        "files": [newest]
    }

    result = drive.find_newest_mp4(service, "folder1")

    # The folder is now part of the answer: with subfolders the caller can no longer
    # infer where the file lives from the folder it asked about.
    assert result == {**newest, "container_id": "folder1"}
    list_kwargs = service.files.return_value.list.call_args.kwargs
    assert list_kwargs["orderBy"] == "createdTime desc"
    assert list_kwargs["pageSize"] == 1
    assert "video/mp4" in list_kwargs["q"]
    assert "trashed = false" in list_kwargs["q"]
    assert "folder1" in list_kwargs["q"]


def test_find_newest_mp4_empty_folder_returns_none():
    service = MagicMock()
    service.files.return_value.list.return_value.execute.return_value = {"files": []}

    assert drive.find_newest_mp4(service, "folder1") is None


def test_list_folder_state_surfaces_booking_properties():
    mp4 = [
        {
            "id": "v1",
            "name": "call - 2026/08/08 09:00 GMT+04:00 – Recording.mp4",
            "mimeType": drive.MP4_MIME,
            "appProperties": {
                "booking_match": "none",
                "planfix_comment_task_id": "851030",
                "telegram_sent_chat_id": "-1001234567890",
            },
        }
    ]
    service = _make_list_service({"mp4": mp4, "mp3": [], "txt": []})

    items = drive.list_folder_state(service, "f1")

    assert items[0]["booking_match"] == "none"
    assert items[0]["planfix_comment_task_id"] == "851030"
    assert items[0]["telegram_sent_chat_id"] == "-1001234567890"


def test_list_folder_state_defaults_booking_properties_to_blank():
    mp4 = [{"id": "v1", "name": "call.mp4", "mimeType": drive.MP4_MIME}]
    service = _make_list_service({"mp4": mp4, "mp3": [], "txt": []})

    items = drive.list_folder_state(service, "f1")

    assert items[0]["booking_match"] == ""
    assert items[0]["planfix_comment_task_id"] == ""
    assert items[0]["telegram_sent_chat_id"] == ""


def test_list_mp4_timestamps_requests_the_timestamp_fields_and_pages():
    """The polling loop's listing deliberately omits timestamps; this one needs them."""
    service = MagicMock()
    pages = [
        {"files": [{"id": "v1", "name": "a.mp4"}], "nextPageToken": "page2"},
        {"files": [{"id": "v2", "name": "b.mp4"}]},
    ]
    service.files.return_value.list.return_value.execute.side_effect = pages

    result = drive.list_mp4_timestamps(service, "f1")

    assert [f["id"] for f in result] == ["v1", "v2"]
    first_call = service.files.return_value.list.call_args_list[0].kwargs
    assert "createdTime" in first_call["fields"]
    assert "modifiedTime" in first_call["fields"]
    assert "appProperties" in first_call["fields"]
    assert "'f1' in parents" in first_call["q"]
    assert "video/mp4" in first_call["q"]
    assert service.files.return_value.list.call_args_list[1].kwargs["pageToken"] == "page2"


def test_set_file_modified_time_sends_only_the_date():
    """The marks must survive the repair, or the backlog would be re-transcribed."""
    service = MagicMock()
    service.files.return_value.update.return_value.execute.return_value = {"id": "v1"}

    drive.set_file_modified_time(service, "v1", "2025-03-14T18:24:52.949Z")

    service.files.return_value.update.assert_called_once_with(
        fileId="v1",
        body={"modifiedTime": "2025-03-14T18:24:52.949Z"},
        fields="id, name, modifiedTime",
        supportsAllDrives=True,
    )


def test_fake_drive_filters_by_parent_and_not_just_mime():
    """Pins the fake itself: the old one ignored `in parents` and answered any query it
    did not recognise with an empty list, so a test that walked into subfolders passed
    while finding none of them. Both halves of that are checked here."""
    service = _make_drive_service([
        {"id": "v1", "name": "root.mp4", "mimeType": drive.MP4_MIME, "parents": ["root"]},
        {"id": "v2", "name": "child.mp4", "mimeType": drive.MP4_MIME, "parents": ["sub"]},
        {"id": "d1", "name": "sub", "mimeType": drive.FOLDER_MIME, "parents": ["root"]},
    ])

    root_mp4 = drive._list_files_by_mime(service, "root", drive.MP4_MIME)
    assert [f["id"] for f in root_mp4] == ["v1"]

    sub_mp4 = drive._list_files_by_mime(service, "sub", drive.MP4_MIME)
    assert [f["id"] for f in sub_mp4] == ["v2"]

    # The query the old fake fell through on.
    folders = drive._list_files_by_mime(service, "root", drive.FOLDER_MIME)
    assert [f["id"] for f in folders] == ["d1"]


def test_fake_drive_paginates():
    service = _make_drive_service(
        [
            {"id": f"v{i}", "name": f"{i}.mp4", "mimeType": drive.MP4_MIME, "parents": ["root"]}
            for i in range(5)
        ]
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(drive, "PAGE_SIZE", 2)
        found = drive._list_files_by_mime(service, "root", drive.MP4_MIME)

    assert [f["id"] for f in found] == ["v0", "v1", "v2", "v3", "v4"]


def _meet_root_service() -> MagicMock:
    """A Google Meet root: no mp4 of its own, one subfolder per meeting."""
    return _make_drive_service([
        {"id": "d1", "name": "may-doqs-end - 2026/09/09 18:53 CEST",
         "mimeType": drive.FOLDER_MIME, "parents": ["root"]},
        {"id": "d2", "name": "Планёрка продаж (recurring)",
         "mimeType": drive.FOLDER_MIME, "parents": ["root"]},
        {"id": "v1", "name": "may-doqs-end (2026-09-09 18:53 GMT+2).mp4",
         "mimeType": drive.MP4_MIME, "parents": ["d1"]},
        {"id": "t1", "name": "may-doqs-end (2026-09-09 18:53 GMT+2).txt",
         "mimeType": drive.TXT_MIME, "parents": ["d1"]},
        {"id": "v2", "name": "Планёрка продаж - 2026/09/01 17:00 GMT+04:00.mp4",
         "mimeType": drive.MP4_MIME, "parents": ["d2"]},
    ])


def test_list_subfolders_returns_direct_children_only():
    service = _make_drive_service([
        {"id": "d1", "name": "sub", "mimeType": drive.FOLDER_MIME, "parents": ["root"]},
        {"id": "d2", "name": "deeper", "mimeType": drive.FOLDER_MIME, "parents": ["d1"]},
        {"id": "v1", "name": "a.mp4", "mimeType": drive.MP4_MIME, "parents": ["root"]},
    ])

    assert [f["id"] for f in drive.list_subfolders(service, "root")] == ["d1"]


def test_list_subfolders_skips_trashed():
    service = _make_drive_service([
        {"id": "d1", "name": "live", "mimeType": drive.FOLDER_MIME, "parents": ["root"]},
        {"id": "d2", "name": "gone", "mimeType": drive.FOLDER_MIME, "parents": ["root"],
         "trashed": True},
    ])

    assert [f["id"] for f in drive.list_subfolders(service, "root")] == ["d1"]


def test_list_subfolders_of_a_flat_folder_is_empty():
    service = _make_drive_service([
        {"id": "v1", "name": "a.mp4", "mimeType": drive.MP4_MIME, "parents": ["flat"]},
    ])

    assert drive.list_subfolders(service, "flat") == []


def test_tree_state_finds_videos_in_subfolders():
    items = drive.list_folder_tree_state(_meet_root_service(), "root")

    assert sorted(it["file"]["id"] for it in items) == ["v1", "v2"]


def test_tree_state_carries_the_container_each_file_lives_in():
    """The caller can no longer assume the container is the folder it asked about:
    artifacts must be written next to the video, in its own meeting subfolder."""
    items = drive.list_folder_tree_state(_meet_root_service(), "root")

    by_id = {it["file"]["id"]: it for it in items}
    assert by_id["v1"]["container_id"] == "d1"
    assert by_id["v2"]["container_id"] == "d2"


def test_tree_state_keeps_sibling_state_scoped_to_its_own_subfolder():
    """A .txt in one meeting folder must not mark another meeting's video as done."""
    items = drive.list_folder_tree_state(_meet_root_service(), "root")

    by_id = {it["file"]["id"]: it for it in items}
    assert by_id["v1"]["has_txt"] is True
    assert by_id["v2"]["has_txt"] is False


def test_tree_state_still_reads_a_flat_folder_the_old_way():
    """Legacy Meet Recordings and hand-made folders keep working: no subfolders, and
    the container is the configured folder itself."""
    service = _make_drive_service([
        {"id": "v1", "name": "a.mp4", "mimeType": drive.MP4_MIME, "parents": ["flat"]},
        {"id": "t1", "name": "a.txt", "mimeType": drive.TXT_MIME, "parents": ["flat"]},
    ])

    items = drive.list_folder_tree_state(service, "flat")

    assert len(items) == 1
    assert items[0]["container_id"] == "flat"
    assert items[0]["has_txt"] is True


def test_tree_state_reads_a_folder_holding_both_videos_and_subfolders():
    service = _make_drive_service([
        {"id": "v0", "name": "loose.mp4", "mimeType": drive.MP4_MIME, "parents": ["root"]},
        {"id": "d1", "name": "sub", "mimeType": drive.FOLDER_MIME, "parents": ["root"]},
        {"id": "v1", "name": "nested.mp4", "mimeType": drive.MP4_MIME, "parents": ["d1"]},
    ])

    items = drive.list_folder_tree_state(service, "root")

    containers = {it["file"]["id"]: it["container_id"] for it in items}
    assert containers == {"v0": "root", "v1": "d1"}


def test_list_folder_state_reports_its_own_folder_as_the_container():
    service = _make_drive_service([
        {"id": "v1", "name": "a.mp4", "mimeType": drive.MP4_MIME, "parents": ["f1"]},
    ])

    assert drive.list_folder_state(service, "f1")[0]["container_id"] == "f1"


def test_list_folder_state_asks_drive_once_per_folder():
    """Four listings per folder was fine flat; with a subfolder per meeting it becomes
    four per meeting, every cycle, forever. One query carrying all four mime types
    costs the same round trip as one of them."""
    service = _meet_root_service()

    drive.list_folder_state(service, "d1")

    assert service.files.return_value.list.call_count == 1


def test_list_folder_state_still_separates_the_mime_types_it_asked_for_together():
    service = _make_drive_service([
        {"id": "v1", "name": "a.mp4", "mimeType": drive.MP4_MIME, "parents": ["f1"]},
        {"id": "m1", "name": "a.mp3", "mimeType": drive.MP3_MIME, "parents": ["f1"]},
        {"id": "t1", "name": "a.txt", "mimeType": drive.TXT_MIME, "parents": ["f1"]},
        {"id": "k1", "name": "a.keypoints.md", "mimeType": drive.MD_MIME, "parents": ["f1"]},
    ])

    item = drive.list_folder_state(service, "f1")[0]

    assert item["has_mp3"] is True
    assert item["has_txt"] is True
    assert item["txt_id"] == "t1"
    assert item["artifact_ids"] == {"keypoints": "k1"}


def test_list_folder_state_ignores_mime_types_it_did_not_ask_for():
    """The Google-made transcript sits in the same subfolder as a Google Doc; it must
    not be mistaken for our own text/plain transcript."""
    service = _make_drive_service([
        {"id": "v1", "name": "a.mp4", "mimeType": drive.MP4_MIME, "parents": ["f1"]},
        {"id": "g1", "name": "a - Transcript",
         "mimeType": "application/vnd.google-apps.document", "parents": ["f1"]},
    ])

    item = drive.list_folder_state(service, "f1")[0]

    assert item["has_txt"] is False


def _nested_service() -> MagicMock:
    return _make_drive_service([
        {"id": "sub", "name": "meeting", "mimeType": drive.FOLDER_MIME, "parents": ["root"]},
        {"id": "root", "name": "Google Meet", "mimeType": drive.FOLDER_MIME, "parents": ["mydrive"]},
        {"id": "other", "name": "Someone else", "mimeType": drive.FOLDER_MIME, "parents": ["elsewhere"]},
    ])


def test_configured_ancestor_of_a_configured_folder_is_itself():
    service = _nested_service()

    assert drive.find_configured_ancestor(service, "root", {"root"}) == "root"
    assert service.files.return_value.get.call_count == 0


def test_configured_ancestor_of_a_meeting_subfolder_is_the_configured_root():
    assert drive.find_configured_ancestor(_nested_service(), "sub", {"root"}) == "root"


def test_configured_ancestor_is_none_for_a_folder_outside_the_configured_set():
    """A shared-with-us folder nobody configured must not be processed just because
    it turned up: the changes feed reports everything the token can see."""
    assert drive.find_configured_ancestor(_nested_service(), "other", {"root"}) is None


def test_configured_ancestor_caches_across_calls():
    """Meeting folders do not move, and the feed reports many files from the same one."""
    service = _nested_service()
    cache: dict[str, str | None] = {}

    drive.find_configured_ancestor(service, "sub", {"root"}, cache=cache)
    calls_after_first = service.files.return_value.get.call_count
    drive.find_configured_ancestor(service, "sub", {"root"}, cache=cache)

    assert calls_after_first == 1
    assert service.files.return_value.get.call_count == 1


def test_configured_ancestor_caches_a_miss_too():
    service = _nested_service()
    cache: dict[str, str | None] = {}

    drive.find_configured_ancestor(service, "other", {"root"}, cache=cache)
    calls_after_first = service.files.return_value.get.call_count
    drive.find_configured_ancestor(service, "other", {"root"}, cache=cache)

    assert calls_after_first > 0
    assert service.files.return_value.get.call_count == calls_after_first
    assert cache["other"] is None


def test_configured_ancestor_gives_up_instead_of_looping_forever():
    """A malformed or circular parent chain must cost a bounded number of requests."""
    service = _make_drive_service([
        {"id": "a", "name": "a", "mimeType": drive.FOLDER_MIME, "parents": ["b"]},
        {"id": "b", "name": "b", "mimeType": drive.FOLDER_MIME, "parents": ["a"]},
    ])

    assert drive.find_configured_ancestor(service, "a", {"root"}) is None
    assert service.files.return_value.get.call_count <= drive.MAX_ANCESTOR_DEPTH


def test_configured_ancestor_handles_a_folder_with_no_parents():
    service = _make_drive_service([
        {"id": "orphan", "name": "orphan", "mimeType": drive.FOLDER_MIME},
    ])

    assert drive.find_configured_ancestor(service, "orphan", {"root"}) is None


def _dated_tree_service() -> MagicMock:
    return _make_drive_service([
        {"id": "d1", "name": "older meeting", "mimeType": drive.FOLDER_MIME, "parents": ["root"]},
        {"id": "d2", "name": "newer meeting", "mimeType": drive.FOLDER_MIME, "parents": ["root"]},
        {"id": "v1", "name": "old.mp4", "mimeType": drive.MP4_MIME, "parents": ["d1"],
         "createdTime": "2026-09-03T23:00:00Z"},
        {"id": "v2", "name": "new.mp4", "mimeType": drive.MP4_MIME, "parents": ["d2"],
         "createdTime": "2026-09-09T18:53:00Z"},
    ])


def test_newest_in_tree_looks_inside_subfolders():
    """`gdstt latest` pointed at a Meet root would otherwise report no mp4 at all:
    the root holds only subfolders."""
    newest = drive.find_newest_mp4_in_tree(_dated_tree_service(), "root")

    assert newest is not None
    assert newest["id"] == "v2"


def test_newest_in_tree_compares_across_subfolders_not_within_one():
    service = _make_drive_service([
        {"id": "d1", "name": "a", "mimeType": drive.FOLDER_MIME, "parents": ["root"]},
        {"id": "d2", "name": "b", "mimeType": drive.FOLDER_MIME, "parents": ["root"]},
        {"id": "v1", "name": "x.mp4", "mimeType": drive.MP4_MIME, "parents": ["d1"],
         "createdTime": "2026-09-10T10:00:00Z"},
        {"id": "v2", "name": "y.mp4", "mimeType": drive.MP4_MIME, "parents": ["d2"],
         "createdTime": "2026-09-09T10:00:00Z"},
    ])

    assert drive.find_newest_mp4_in_tree(service, "root")["id"] == "v1"


def test_newest_in_tree_includes_a_video_loose_in_the_folder_itself():
    service = _make_drive_service([
        {"id": "d1", "name": "a", "mimeType": drive.FOLDER_MIME, "parents": ["root"]},
        {"id": "v0", "name": "loose.mp4", "mimeType": drive.MP4_MIME, "parents": ["root"],
         "createdTime": "2026-09-11T10:00:00Z"},
        {"id": "v1", "name": "nested.mp4", "mimeType": drive.MP4_MIME, "parents": ["d1"],
         "createdTime": "2026-09-09T10:00:00Z"},
    ])

    assert drive.find_newest_mp4_in_tree(service, "root")["id"] == "v0"


def test_newest_in_tree_on_a_flat_folder_matches_the_single_folder_lookup():
    service = _make_drive_service([
        {"id": "v1", "name": "a.mp4", "mimeType": drive.MP4_MIME, "parents": ["flat"],
         "createdTime": "2026-09-01T10:00:00Z"},
    ])

    assert drive.find_newest_mp4_in_tree(service, "flat")["id"] == "v1"


def test_newest_in_tree_is_none_when_nothing_is_there():
    service = _make_drive_service([
        {"id": "d1", "name": "empty meeting", "mimeType": drive.FOLDER_MIME, "parents": ["root"]},
    ])

    assert drive.find_newest_mp4_in_tree(service, "root") is None


def test_newest_mp4_reports_the_folder_it_was_found_in():
    """The caller needs the container to write artifacts next to the video."""
    newest = drive.find_newest_mp4_in_tree(_dated_tree_service(), "root")

    assert newest["container_id"] == "d2"


def test_folder_state_reports_whether_drive_finished_with_the_video():
    """Drive fills videoMediaMetadata once it has processed an upload. Its absence is
    the cheapest signal that a video is still settling."""
    service = _make_drive_service([
        {"id": "v1", "name": "ready.mp4", "mimeType": drive.MP4_MIME, "parents": ["f1"],
         "videoMediaMetadata": {"width": 1920, "height": 1080, "durationMillis": "3619000"}},
        {"id": "v2", "name": "settling.mp4", "mimeType": drive.MP4_MIME, "parents": ["f1"]},
    ])

    by_id = {it["file"]["id"]: it for it in drive.list_folder_state(service, "f1")}

    assert by_id["v1"]["has_media_metadata"] is True
    assert by_id["v2"]["has_media_metadata"] is False


def test_folder_state_keeps_created_time_for_the_readiness_decision():
    service = _make_drive_service([
        {"id": "v1", "name": "a.mp4", "mimeType": drive.MP4_MIME, "parents": ["f1"],
         "createdTime": "2026-09-09T18:53:00Z"},
    ])

    item = drive.list_folder_state(service, "f1")[0]

    assert item["file"]["createdTime"] == "2026-09-09T18:53:00Z"


def test_start_page_token_is_asked_for_before_a_sweep():
    service = _make_drive_service([], start_page_token="tok-42")

    assert drive.get_start_page_token(service) == "tok-42"


def test_list_changes_returns_the_entries_and_the_next_cursor():
    service = _make_drive_service(
        [],
        changes=[{"fileId": "v1", "file": {"id": "v1", "mimeType": drive.MP4_MIME}}],
        new_start_page_token="tok-99",
    )

    entries, cursor = drive.list_changes(service, "tok-1")

    assert [e["fileId"] for e in entries] == ["v1"]
    assert cursor == "tok-99"


def test_list_changes_follows_every_page_before_reporting_the_cursor():
    """Saving the cursor after a partial read would skip whatever was on the pages we
    never asked for, and nothing would ever come back for them."""
    service = _make_drive_service(
        [],
        changes=[
            [{"fileId": "v1"}, {"fileId": "v2"}],
            [{"fileId": "v3"}],
        ],
        new_start_page_token="tok-end",
    )

    entries, cursor = drive.list_changes(service, "tok-1")

    assert [e["fileId"] for e in entries] == ["v1", "v2", "v3"]
    assert cursor == "tok-end"


def test_list_changes_asks_for_the_fields_the_caller_decides_on():
    """Deciding from the change entry alone is what keeps the feed cheap: without
    mimeType and parents every entry would cost a files.get."""
    service = _make_drive_service([], changes=[])

    drive.list_changes(service, "tok-1")

    fields = service.changes.return_value.list.call_args.kwargs["fields"]
    for needed in ("fileId", "removed", "mimeType", "parents", "trashed"):
        assert needed in fields



def test_transcript_name_for_a_calendar_recording():
    assert drive.meet_transcript_name(
        "Weekly - 2026/09/09 16:56 CEST - Recording.mp4"
    ) == "Weekly - 2026/09/09 16:56 CEST - Transcript"


def test_transcript_name_for_a_room_code_recording():
    assert drive.meet_transcript_name(
        "may-doqs-end (2026-09-09 18_53 GMT+2).mp4"
    ) == "may-doqs-end (2026-09-09 18_53 GMT+2) - Transcript"


def test_the_transcript_is_found_by_name_not_by_being_the_only_document():
    """A recurring meeting keeps every instance in one subfolder, so "the transcript
    here" has no single answer."""
    service = _make_drive_service([
        {"id": "d1", "name": "Weekly - 2026/09/02 10:00 CEST - Transcript",
         "mimeType": drive.GOOGLE_DOC_MIME, "parents": ["series"]},
        {"id": "d2", "name": "Weekly - 2026/09/09 10:00 CEST - Transcript",
         "mimeType": drive.GOOGLE_DOC_MIME, "parents": ["series"]},
    ])

    found = drive.find_meet_transcript(
        service, "series", "Weekly - 2026/09/09 10:00 CEST - Recording.mp4"
    )

    assert found["id"] == "d2"


def test_no_transcript_beside_the_recording_is_not_an_error():
    service = _make_drive_service([
        {"id": "v1", "name": "a.mp4", "mimeType": drive.MP4_MIME, "parents": ["f1"]},
    ])

    assert drive.find_meet_transcript(service, "f1", "a.mp4") is None


def test_a_google_doc_is_exported_rather_than_downloaded():
    """A Google Doc has no bytes to download, which is also why a listing asking for
    text/plain never saw Meet's transcripts."""
    service = MagicMock()
    service.files.return_value.export.return_value.execute.return_value = b"Attendees\n"

    text = drive.export_document_text(service, "d1")

    assert text == "Attendees\n"
    kwargs = service.files.return_value.export.call_args.kwargs
    assert kwargs["mimeType"] == "text/plain"



def test_configured_ancestor_raises_a_drive_failure_instead_of_saying_not_ours():
    """A failed lookup and "belongs to nobody" are different answers. Folding one into
    the other makes an expired token look like a decision, and the caller skips a real
    recording believing it decided something."""
    service = MagicMock()
    service.files.return_value.get.side_effect = RuntimeError("token expired")

    with pytest.raises(RuntimeError):
        drive.find_configured_ancestor(service, "sub", {"root"})


def test_transcript_name_survives_a_dot_in_the_meeting_title():
    """Drive stores a Meet recording under its meeting title with no extension, so
    splitting at the last dot truncates the title rather than an extension."""
    assert drive.meet_transcript_name(
        "Sync re: v2.0 - 2026/09/09 10:00 CEST - Recording"
    ) == "Sync re: v2.0 - 2026/09/09 10:00 CEST - Transcript"


def test_transcript_name_still_drops_a_real_mp4_extension():
    assert drive.meet_transcript_name(
        "may-doqs-end (2026-09-09 18_53 GMT+2).mp4"
    ) == "may-doqs-end (2026-09-09 18_53 GMT+2) - Transcript"


def test_listings_ask_only_whether_the_video_metadata_exists():
    """Its presence is all that is read, and this listing runs for every folder on
    every cycle."""
    service = _make_drive_service([])

    drive.list_folder_state(service, "f1")

    fields = service.files.return_value.list.call_args.kwargs["fields"]
    assert "videoMediaMetadata(durationMillis)" in fields


def test_start_page_token_is_asked_for_with_arguments_drive_accepts():
    """`getStartPageToken` takes `supportsAllDrives` but not
    `includeItemsFromAllDrives`; the client raises TypeError on the latter, which a
    MagicMock accepts without complaint. Losing the token is not loud -- the service
    just sweeps every folder forever."""
    service = _make_drive_service([])

    drive.get_start_page_token(service)

    kwargs = service.changes.return_value.getStartPageToken.call_args.kwargs
    assert kwargs == {"supportsAllDrives": True}


def test_mp4_timestamps_in_tree_sees_meeting_subfolders():
    """`planfix sent` and `bookings restore-dates` read through this. One level would
    answer "no recordings" on a Meet root -- a confident, wrong report."""
    service = _make_drive_service([
        {"id": "d1", "name": "meeting", "mimeType": drive.FOLDER_MIME, "parents": ["root"]},
        {"id": "v1", "name": "nested.mp4", "mimeType": drive.MP4_MIME, "parents": ["d1"],
         "createdTime": "2026-09-09T18:53:00Z"},
        {"id": "v0", "name": "loose.mp4", "mimeType": drive.MP4_MIME, "parents": ["root"],
         "createdTime": "2026-09-08T10:00:00Z"},
    ])

    found = drive.list_mp4_timestamps_in_tree(service, "root")

    assert sorted(f["id"] for f in found) == ["v0", "v1"]


def test_mp4_timestamps_in_tree_on_a_flat_folder_is_unchanged():
    service = _make_drive_service([
        {"id": "v1", "name": "a.mp4", "mimeType": drive.MP4_MIME, "parents": ["flat"],
         "createdTime": "2026-09-08T10:00:00Z"},
    ])

    assert [f["id"] for f in drive.list_mp4_timestamps_in_tree(service, "flat")] == ["v1"]


# --- Shortcuts and depth: the shape of a real employee folder ----------------------
#
# Checked read-only against a real employee's Google Meet folder: eleven meeting
# subfolders, eight recordings, and five shortcuts -- three meetings the employee only
# attended held nothing but shortcuts to the organizer's recording and transcript,
# none of which that account could open. Another account may open them, so a
# shortcut is followed whenever its target opens.


_ORGANIZERS_VIDEO = {
    "id": "organizers-video",
    "name": "someone-elses-call (2026-09-04 17:57 GMT+2).mp4",
    "mimeType": drive.MP4_MIME,
    "parents": ["organizers-meeting"],
    "size": "5000",
    "createdTime": "2026-09-04T15:10:00.000Z",
    "videoMediaMetadata": {"durationMillis": "60000"},
    # Someone else's bookkeeping on their own file must never be read as ours.
    "appProperties": {"telegram_sent_chat_id": "-100organizer"},
}


def _attended_meeting_service(*, target_readable=False, extra=()):
    files = [
        {"id": "own", "name": "exf-wxzm-uzk - 2026/09/09 17:42 CEST",
         "mimeType": drive.FOLDER_MIME, "parents": ["root"]},
        {"id": "v1", "name": "exf-wxzm-uzk (2026-09-09 17:42 GMT+2).mp4",
         "mimeType": drive.MP4_MIME, "parents": ["own"]},
        {"id": "attended", "name": "someone-elses-call - 2026/09/04 17:57 CEST",
         "mimeType": drive.FOLDER_MIME, "parents": ["root"]},
        {"id": "sc-video", "name": "someone-elses-call (2026-09-04 17:57 GMT+2).mp4",
         "mimeType": drive.SHORTCUT_MIME, "parents": ["attended"],
         "createdTime": "2026-09-04T15:11:00.000Z",
         "shortcutDetails": {"targetId": "organizers-video",
                             "targetMimeType": drive.MP4_MIME}},
        {"id": "sc-doc", "name": "someone-elses-call - Transcript",
         "mimeType": drive.SHORTCUT_MIME, "parents": ["attended"],
         "shortcutDetails": {"targetId": "organizers-doc",
                             "targetMimeType": drive.GOOGLE_DOC_MIME}},
        *extra,
    ]
    if target_readable:
        files.append(_ORGANIZERS_VIDEO)
    return _make_drive_service(
        files, unreadable=() if target_readable else ("organizers-video", "organizers-doc")
    )


def test_a_shortcut_to_a_recording_this_account_cannot_open_is_not_a_recording():
    """Drive reports the shortcut's own mime type; the real one is only in
    `shortcutDetails`. A target that does not open cannot be downloaded, so there is
    nothing to process -- `doctor --drive` is where that gets said."""
    items = drive.list_folder_tree_state(_attended_meeting_service(), "root")

    assert [it["file"]["id"] for it in items] == ["v1"]


def test_a_shortcut_to_a_recording_that_opens_is_listed_where_the_shortcut_sits():
    """Meet gives an attendee a shortcut, never a copy, whatever the access. When the
    recording opens, the call is this folder's to process: from the target's bytes,
    with artifacts beside the shortcut in the attendee's meeting folder."""
    items = drive.list_folder_tree_state(
        _attended_meeting_service(target_readable=True), "root"
    )

    followed = next(it for it in items if it["file"]["id"] == "sc-video")
    assert followed["container_id"] == "attended"
    assert followed["media_id"] == "organizers-video"
    assert followed["target_parents"] == ["organizers-meeting"]
    assert followed["file"]["name"] == "someone-elses-call (2026-09-04 17:57 GMT+2).mp4"
    # Size, readiness and age describe the recording, not the shortcut.
    assert followed["file"]["size"] == "5000"
    assert followed["file"]["createdTime"] == "2026-09-04T15:10:00.000Z"
    assert followed["has_media_metadata"] is True


def test_a_followed_shortcut_keeps_its_own_bookkeeping_not_the_organizers():
    """Markers are written onto the shortcut, which sits in the attendee's folder. The
    organizer's file carries the organizer's markers, and reading those would skip a
    Telegram summary this attendee never got."""
    service = _attended_meeting_service(target_readable=True)
    items = drive.list_folder_tree_state(service, "root")

    followed = next(it for it in items if it["file"]["id"] == "sc-video")
    assert followed["telegram_sent_chat_id"] == ""
    assert followed["file"]["appProperties"] == {}


def test_a_real_recording_carries_no_shortcut_fields():
    items = drive.list_folder_tree_state(
        _attended_meeting_service(target_readable=True), "root"
    )

    own = next(it for it in items if it["file"]["id"] == "v1")
    assert "media_id" not in own
    assert "target_parents" not in own


def test_a_shortcut_already_transcribed_here_is_not_looked_up_again():
    """Walking re-lists every folder every cycle. A finished call must not cost a
    request per attended meeting for good, so a transcript beside the shortcut is taken
    as the answer and the target is not asked about."""
    transcript = {
        "id": "t1", "name": "someone-elses-call (2026-09-04 17:57 GMT+2).txt",
        "mimeType": drive.TXT_MIME, "parents": ["attended"],
        "appProperties": {"source_video_id": "sc-video"},
    }
    service = _attended_meeting_service(extra=(transcript,))

    items = drive.list_folder_state(service, "attended")

    assert [it["file"]["id"] for it in items] == ["sc-video"]
    assert items[0]["media_id"] == "organizers-video"
    assert items[0]["has_txt"] is True
    service.files.return_value.get.assert_not_called()


def test_a_shortcut_to_a_trashed_recording_is_not_a_recording():
    service = _attended_meeting_service(
        extra=({**_ORGANIZERS_VIDEO, "trashed": True},)
    )
    service_files = service.files.return_value
    by_id = {"organizers-video": {**_ORGANIZERS_VIDEO, "trashed": True}}

    def get(**kwargs):
        request = MagicMock()
        request.execute.return_value = by_id.get(kwargs["fileId"], {})
        return request

    service_files.get.side_effect = get

    items = drive.list_folder_state(service, "attended")

    assert items == []


def test_a_drive_outage_resolving_a_target_is_not_mistaken_for_no_access():
    """A 503 folded into "cannot open" would drop a real call from the listing as if
    by decision; raised, it counts as a listing failure and holds the cursor."""
    from googleapiclient.errors import HttpError

    service = _attended_meeting_service()
    request = MagicMock()
    request.execute.side_effect = HttpError(MagicMock(status=503), b"")
    service.files.return_value.get.side_effect = None
    service.files.return_value.get.return_value = request

    with pytest.raises(HttpError):
        drive.list_folder_state(service, "attended")


def test_shortcuts_come_back_in_the_same_listing_request():
    service = _attended_meeting_service(target_readable=True)

    drive.list_folder_state(service, "attended")

    assert service.files.return_value.list.call_count == 1


def test_names_a_recording_accepts_a_shortcut_to_one():
    assert drive.names_a_recording({"mimeType": drive.MP4_MIME})
    assert drive.names_a_recording({
        "mimeType": drive.SHORTCUT_MIME,
        "shortcutDetails": {"targetMimeType": drive.MP4_MIME},
    })
    assert not drive.names_a_recording({
        "mimeType": drive.SHORTCUT_MIME,
        "shortcutDetails": {"targetMimeType": drive.GOOGLE_DOC_MIME},
    })
    assert not drive.names_a_recording({"mimeType": drive.MP3_MIME})


def test_list_changes_asks_whether_a_shortcut_points_at_a_recording():
    """Without the target's type on the entry, a new attended call in the feed would
    look like any other shortcut and be dropped."""
    service = _make_drive_service([], changes=[])

    drive.list_changes(service, "tok-1")

    fields = service.changes.return_value.list.call_args.kwargs["fields"]
    assert "shortcutDetails" in fields


def test_meet_transcript_is_found_through_a_shortcut_and_read_from_its_target():
    """An attendee's meeting folder holds a shortcut to the transcript too. A shortcut
    cannot be exported, so the target's id is what comes back."""
    service = _attended_meeting_service(target_readable=True)

    doc = drive.find_meet_transcript(service, "attended", "someone-elses-call - Recording")

    assert doc is not None
    assert doc["id"] == "organizers-doc"


def test_mp4_timestamps_include_shortcuts_to_recordings_only():
    """The reports read markers from here, and a followed call's markers live on its
    shortcut."""
    service = _attended_meeting_service()

    ids = [f["id"] for f in drive.list_mp4_timestamps(service, "attended")]

    assert ids == ["sc-video"]


def test_a_shortcut_to_a_folder_is_not_a_subfolder():
    """What Drive creates when a shared folder is added to someone's own Drive. The
    walk does not step into it, and cannot usefully: a shortcut is not a parent, so
    nothing inside would ever resolve back to the configured folder either."""
    service = _make_drive_service([
        {"id": "real", "name": "meeting", "mimeType": drive.FOLDER_MIME,
         "parents": ["root"]},
        {"id": "link", "name": "a colleague's folder", "mimeType": drive.SHORTCUT_MIME,
         "parents": ["root"],
         "shortcutDetails": {"targetId": "elsewhere", "targetMimeType": drive.FOLDER_MIME}},
    ])

    assert [f["id"] for f in drive.list_subfolders(service, "root")] == ["real"]


def test_the_walk_goes_exactly_one_level_into_meeting_folders():
    """A boundary, pinned so that moving it is a decision rather than an accident.
    Meet never nests deeper, and every real folder checked agreed; a project folder
    holding people's folders holding meetings would be one level too deep, which is
    why each person is configured separately."""
    service = _make_drive_service([
        {"id": "person", "name": "employee", "mimeType": drive.FOLDER_MIME,
         "parents": ["project"]},
        {"id": "meeting", "name": "a call", "mimeType": drive.FOLDER_MIME,
         "parents": ["person"]},
        {"id": "v1", "name": "a call.mp4", "mimeType": drive.MP4_MIME,
         "parents": ["meeting"]},
    ])

    assert drive.list_folder_tree_state(service, "project") == []
    assert [it["file"]["id"] for it in drive.list_folder_tree_state(service, "person")] == ["v1"]


def test_recording_shortcuts_are_listed_with_the_meeting_they_sit_in():
    shortcuts = drive.list_recording_shortcuts(_attended_meeting_service(), "root")

    assert shortcuts == [{
        "id": "sc-video",
        "name": "someone-elses-call (2026-09-04 17:57 GMT+2).mp4",
        "container_id": "attended",
        "target_id": "organizers-video",
    }]


def test_a_shortcut_to_a_transcript_is_not_reported_as_a_recording():
    shortcuts = drive.list_recording_shortcuts(_attended_meeting_service(), "root")

    assert all(s["id"] != "sc-doc" for s in shortcuts)


def test_a_folder_without_shortcuts_reports_none():
    assert drive.list_recording_shortcuts(_meet_root_service(), "root") == []


def test_shortcut_target_is_none_for_a_file_this_account_cannot_open():
    """Drive answers "you may not see this" with 404, the same as "no such file"."""
    from googleapiclient.errors import HttpError

    service = MagicMock()
    service.files.return_value.get.return_value.execute.side_effect = HttpError(
        MagicMock(status=404), b""
    )

    assert drive.get_shortcut_target(service, "organizers-video") is None


def test_shortcut_target_comes_back_when_the_file_opens():
    service = MagicMock()
    service.files.return_value.get.return_value.execute.return_value = {"id": "v1"}

    assert drive.get_shortcut_target(service, "v1") == {"id": "v1"}


def test_shortcut_target_does_not_hide_a_drive_outage_as_a_permission_answer():
    from googleapiclient.errors import HttpError

    service = MagicMock()
    service.files.return_value.get.return_value.execute.side_effect = HttpError(
        MagicMock(status=503), b""
    )

    with pytest.raises(HttpError):
        drive.get_shortcut_target(service, "v1")
