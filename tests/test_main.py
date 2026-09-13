from __future__ import annotations

import logging
import ssl
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError

from src import change_cursor, main, meta_entity
from src.auth import AuthError
from src.booking_gate import BookingDecision
from src.call_booking import CallBooking
from src.call_booking import append as append_booking
from src.config import Config, EmployeeFolder, resolve_config_file_path
from src.presets import BUILTIN_PRESETS, Preset
from src.preset_pipeline import PresetResult
from src.stt.base import STTError

_KEYPOINTS_BUILTIN = next(p for p in BUILTIN_PRESETS if p.name == "keypoints")


@pytest.fixture(autouse=True)
def _flat_folders_and_a_scratch_cursor(mocker, tmp_path):
    """Every folder here is flat, and the changes cursor lives in a scratch file.

    `run_once` and `process <folder>` now read a folder together with its meeting
    subfolders, a cycle now saves where the changes feed got to, and a recording may
    have a Meet transcript beside it. These fixtures describe none of that: the folder
    is flat, the cursor is nobody's business here, and there is no transcript.

    Both halves have teeth. Left alone, the real `list_subfolders` runs against a
    MagicMock whose `nextPageToken` is truthy and the paging loop never ends; and the
    cursor would be written under the default `data/` directory, inside the checkout.
    Redirecting `path_for` keeps production code and these tests agreeing on one
    throwaway path, so a test that does care about the cursor still reads what the
    cycle wrote.
    """
    mocker.patch("src.drive.list_subfolders", return_value=[])
    mocker.patch("src.drive.get_start_page_token", return_value="tok-sweep")
    mocker.patch("src.drive.find_meet_transcript", return_value=None)
    mocker.patch(
        "src.change_cursor.path_for",
        return_value=tmp_path / "cursor" / "changes_cursor.txt",
    )
    mocker.patch(
        "src.change_cursor.folders_path_for",
        return_value=tmp_path / "cursor" / "changes_folders.txt",
    )


def test_the_suite_never_resolves_the_repos_real_config():
    """No test may read the operator's live data/config.yml.

    `main()` calls `is_run_enabled()` on every loop iteration, and that reads the
    effective config from disk. On a checkout where the operator has run `gdstt
    stop`, the loop takes its paused branch -- whose only exit is `time.sleep`,
    which these tests mock away. Every loop test then spins forever instead of
    failing, which is how this suite once went from 12 seconds to a hang.

    tests/conftest.py points GDSTT_HOME at a per-test throwaway directory to make
    that impossible. This asserts the guard is actually in place.
    """
    assert resolve_config_file_path() != Path("data") / "config.yml"


def _as_folders(entries) -> tuple[EmployeeFolder, ...]:
    """Accept ids or EmployeeFolders, so folder-agnostic tests stay short."""
    return tuple(
        entry if isinstance(entry, EmployeeFolder) else EmployeeFolder(entry)
        for entry in entries
    )


def make_config(
    folders=None,
    bitrate="96k",
    poll_interval=600,
    data_dir=Path("data"),
    stt_provider="",
    openai_api_key="",
    deepgram_api_key="",
    stt_language="",
    deepgram_audio_source="m4a_copy",
    drive_mp3_artifact=True,
    stt_postprocess=False,
    output_target="drive",
    output_dir=None,
    output_also_drive=False,
    stt_presets=("keypoints",),
    openai_keypoints=False,
    presets=None,
    tags_allowed=(),
    referrals_allowed=(),
    meta_entities=None,
    webhook_url="",
    webhook_token="",
    proxy_url="",
    planfix_meta_fields=(
        "subject", "tags", "referral", "referral_note", "duration", "video_url",
    ),
) -> Config:
    if presets is None:
        # Mirror the legacy keypoints gate: the built-in keypoints pass is the only
        # enabled preset when requested, and none otherwise. Deliberately not the
        # whole BUILTIN_PRESETS tuple — `meta` is a built-in too, and pulling it in
        # here would silently widen every openai_keypoints=True test to two passes.
        presets = (_KEYPOINTS_BUILTIN,) if openai_keypoints else ()
    if meta_entities is None:
        # Mirrors how the real loader fills `meta.entities` when a config leaves it
        # unset: the built-in four, wired to the deprecated top-level allow-lists.
        meta_entities = meta_entity.default_entities(
            tuple(tags_allowed), tuple(referrals_allowed)
        )
    return Config(
        folders=_as_folders(folders if folders is not None else ["folderA"]),
        poll_interval=poll_interval,
        bitrate=bitrate,
        data_dir=data_dir,
        proxy_url=proxy_url,
        stt_provider=stt_provider,
        openai_api_key=openai_api_key,
        deepgram_api_key=deepgram_api_key,
        stt_language=stt_language,
        stt_postprocess=stt_postprocess,
        output_target=output_target,
        output_dir=output_dir,
        output_also_drive=output_also_drive,
        stt_presets=tuple(stt_presets),
        openai_keypoints=openai_keypoints,
        deepgram_audio_source=deepgram_audio_source,
        drive_mp3_artifact=drive_mp3_artifact,
        tags_allowed=tuple(tags_allowed),
        referrals_allowed=tuple(referrals_allowed),
        meta_entities=tuple(meta_entities),
        webhook_url=webhook_url,
        webhook_token=webhook_token,
        planfix_meta_fields=tuple(planfix_meta_fields),
        presets=tuple(presets),
    )


def _item(
    file_id="fid", name="video.mp4", *, has_mp3=False, has_txt=False,
    mp3_id=None, mp3_name=None, txt_id=None, keypoints_id=None,
    artifact_ids=None, size=None, stt_id=None, meta_yml_id=None,
):
    file_info = {"id": file_id, "name": name}
    if size is not None:
        file_info["size"] = str(size)
    ids = dict(artifact_ids or {})
    if keypoints_id is not None:
        ids.setdefault("keypoints", keypoints_id)
    return {
        "file": file_info,
        "has_mp3": has_mp3,
        "has_txt": has_txt,
        "mp3_id": mp3_id,
        "mp3_name": mp3_name,
        "txt_id": txt_id,
        "stt_id": stt_id,
        "meta_yml_id": meta_yml_id,
        "artifact_ids": ids,
    }


def test_process_item_downloads_extracts_uploads(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    download_mock = mocker.patch("src.main.drive.download", return_value=mp4_path)
    extract_mock = mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    upload_mock = mocker.patch("src.main.drive.upload", return_value={"id": "uploaded"})

    cfg = make_config(bitrate="128k")
    main.process_item(service, _item("fid1", "video.mp4"), "folderA", cfg)

    download_mock.assert_called_once()
    args, _ = download_mock.call_args
    assert args[0] is service
    assert args[1] == "fid1"
    assert isinstance(args[2], Path)
    assert args[3] == "video.mp4"

    extract_mock.assert_called_once_with(mp4_path, bitrate="128k")
    upload_mock.assert_called_once_with(
        service,
        mp3_path,
        "folderA",
        mime_type="audio/mpeg",
        name="video.mp3",
        app_properties={"source_video_id": "fid1", "artifact_type": "mp3"},
    )


def test_process_item_skips_when_already_done(mocker):
    service = MagicMock()
    download = mocker.patch("src.main.drive.download")
    extract = mocker.patch("src.main.extract_mp3")
    upload = mocker.patch("src.main.drive.upload")

    cfg = make_config()
    main.process_item(
        service, _item("fid", "v.mp4", has_mp3=True), "folder", cfg,
    )
    download.assert_not_called()
    extract.assert_not_called()
    upload.assert_not_called()


def test_process_item_temp_dir_is_cleaned_up(mocker):
    service = MagicMock()
    captured = {}

    def fake_download(service_arg, file_id, dest_dir, name, *, expected_size_bytes=None):
        captured["dest_dir"] = dest_dir
        path = dest_dir / name
        path.write_bytes(b"data")
        return path

    mocker.patch("src.main.drive.download", side_effect=fake_download)
    mocker.patch("src.main.extract_mp3", side_effect=lambda p, bitrate: p.with_suffix(".mp3"))
    mocker.patch("src.main.drive.upload")

    cfg = make_config()
    main.process_item(service, _item("fid", "v.mp4"), "f", cfg)

    assert "dest_dir" in captured
    assert not captured["dest_dir"].exists(), "temp dir should be cleaned up"


def test_process_item_runs_stt_when_enabled(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"
    m4a_path = tmp_path / "video.m4a"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    mocker.patch("src.main.extract_m4a_copy", return_value=m4a_path)
    upload_mock = mocker.patch("src.main.drive.upload")
    transcribe_mock = mocker.patch(
        "src.main.transcribe_file", return_value="hello world"
    )

    cfg = make_config(stt_provider="deepgram", deepgram_api_key="dg-x")
    main.process_item(service, _item("fid", "video.mp4"), "f", cfg)

    transcribe_mock.assert_called_once_with(m4a_path, cfg, cost_usd={})
    # Four uploads: mp3, txt, and the meta.yml/.stt written for every processed
    # recording.
    assert upload_mock.call_count == 4
    second_call = upload_mock.call_args_list[1]
    assert second_call.kwargs["mime_type"] == "text/plain"
    txt_path = second_call.args[1]
    assert txt_path.name == "video.txt"


def test_process_item_does_not_upload_blank_txt_when_transcript_is_empty(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    mocker.patch("src.main.extract_m4a_copy", return_value=tmp_path / "video.m4a")
    upload_mock = mocker.patch("src.main.drive.upload")
    mocker.patch(
        "src.main.transcribe_file",
        side_effect=STTError("deepgram returned an empty transcript for video.m4a"),
    )

    cfg = make_config(stt_provider="deepgram", deepgram_api_key="dg-x")

    with pytest.raises(STTError, match="empty transcript"):
        main.process_item(service, _item("fid", "video.mp4"), "f", cfg)

    assert upload_mock.call_count == 1
    assert upload_mock.call_args.kwargs["mime_type"] == "audio/mpeg"


def test_process_item_only_stt_when_mp3_already_exists(mocker, tmp_path):
    service = MagicMock()

    def fake_download(svc, file_id, dest_dir, name, *, expected_size_bytes=None):
        path = dest_dir / name
        path.write_bytes(b"x")
        return path

    download_mock = mocker.patch("src.main.drive.download", side_effect=fake_download)
    m4a_path = tmp_path / "video.m4a"
    m4a_mock = mocker.patch("src.main.extract_m4a_copy", return_value=m4a_path)
    extract_mock = mocker.patch("src.main.extract_mp3")
    upload_mock = mocker.patch("src.main.drive.upload")
    transcribe_mock = mocker.patch(
        "src.main.transcribe_file", return_value="text"
    )

    # mp3 artifact already exists, but Deepgram never reuses it: it re-derives
    # audio from the source mp4 and uploads only the new .txt (plus the
    # meta.yml/.stt written for every processed recording).
    cfg = make_config(stt_provider="deepgram", deepgram_api_key="dg-x")
    item = _item(
        "fid", "video.mp4", has_mp3=True, mp3_id="mp3id", mp3_name="video.mp3",
    )
    main.process_item(service, item, "folderX", cfg)

    extract_mock.assert_not_called()
    m4a_mock.assert_called_once()
    download_mock.assert_called_once()
    args, _ = download_mock.call_args
    assert args[1] == "fid"
    assert args[3] == "video.mp4"
    transcribe_mock.assert_called_once_with(m4a_path, cfg, cost_usd={})
    assert upload_mock.call_count == 3
    first_call = upload_mock.call_args_list[0]
    assert first_call.kwargs["mime_type"] == "text/plain"
    assert first_call.kwargs["name"] == "video.txt"


def test_process_item_passes_expected_source_size_to_download(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    m4a_path = tmp_path / "video.m4a"

    download_mock = mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_m4a_copy", return_value=m4a_path)
    mocker.patch("src.main.drive.upload")
    mocker.patch("src.main.transcribe_file", return_value="text")

    cfg = make_config(stt_provider="deepgram", deepgram_api_key="dg-x", drive_mp3_artifact=False)
    item = _item(
        "fid",
        "video.mp4",
        has_mp3=True,
        has_txt=False,
        mp3_id="mp3id",
        mp3_name="video.mp3",
        size=456,
    )

    main.process_item(service, item, "folderX", cfg)

    assert download_mock.call_args.args[:4] == (
        service,
        "fid",
        download_mock.call_args.args[2],
        "video.mp4",
    )
    assert download_mock.call_args.kwargs == {"expected_size_bytes": 456}


def test_process_item_extracts_temporary_audio_when_artifact_upload_is_disabled(
    mocker,
    tmp_path,
):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    download_mock = mocker.patch("src.main.drive.download", return_value=mp4_path)
    extract_mock = mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    upload_mock = mocker.patch("src.main.drive.upload")
    transcribe_mock = mocker.patch("src.main.transcribe_file", return_value="text")

    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        deepgram_audio_source="mp3_96k",
        drive_mp3_artifact=False,
    )

    telemetry = main.process_item(service, _item("fid", "video.mp4"), "folderX", cfg)

    download_mock.assert_called_once()
    assert download_mock.call_args.args[1] == "fid"
    extract_mock.assert_called_once_with(mp4_path, bitrate="96k")
    transcribe_mock.assert_called_once_with(mp3_path, cfg, cost_usd={})
    # txt, plus the meta.yml/.stt written for every processed recording.
    assert upload_mock.call_count == 3
    first_call = upload_mock.call_args_list[0]
    assert first_call.kwargs["mime_type"] == "text/plain"
    assert first_call.kwargs["name"] == "video.txt"
    assert telemetry.mp3_uploaded is False
    assert telemetry.txt_uploaded is True


def test_process_item_deepgram_m4a_downloads_mp4_even_when_mp3_exists(
    mocker,
    tmp_path,
):
    service = MagicMock()

    def fake_download(svc, file_id, dest_dir, name, *, expected_size_bytes=None):
        path = dest_dir / name
        path.write_bytes(b"x")
        return path

    download_mock = mocker.patch("src.main.drive.download", side_effect=fake_download)
    extract_mock = mocker.patch("src.main.extract_mp3")
    m4a_path = tmp_path / "video.m4a"
    m4a_mock = mocker.patch("src.main.extract_m4a_copy", return_value=m4a_path)
    upload_mock = mocker.patch("src.main.drive.upload")
    transcribe_mock = mocker.patch("src.main.transcribe_file", return_value="text")

    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        stt_language="ru",
        deepgram_audio_source="m4a_copy",
    )
    item = _item(
        "fid",
        "video.mp4",
        has_mp3=True,
        mp3_id="mp3id",
        mp3_name="video.mp3",
    )
    main.process_item(service, item, "folderX", cfg)

    assert download_mock.call_args.args[1] == "fid"
    assert download_mock.call_args.args[3] == "video.mp4"
    extract_mock.assert_not_called()
    m4a_mock.assert_called_once()
    transcribe_mock.assert_called_once_with(m4a_path, cfg, cost_usd={})
    # txt, plus the meta.yml/.stt written for every processed recording.
    assert upload_mock.call_count == 3
    assert upload_mock.call_args_list[0].args[1].name == "video.txt"


def test_process_item_deepgram_m4a_does_not_upload_mp3_by_default(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    m4a_path = tmp_path / "video.m4a"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    extract_mp3_mock = mocker.patch("src.main.extract_mp3")
    mocker.patch("src.main.extract_m4a_copy", return_value=m4a_path)
    upload_mock = mocker.patch("src.main.drive.upload")
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hello")

    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        stt_language="ru",
        deepgram_audio_source="m4a_copy",
        drive_mp3_artifact=False,
    )
    main.process_item(service, _item("fid", "video.mp4"), "folderX", cfg)

    extract_mp3_mock.assert_not_called()
    # txt, plus the meta.yml/.stt written for every processed recording.
    assert upload_mock.call_count == 3
    assert upload_mock.call_args_list[0].kwargs["mime_type"] == "text/plain"


def test_process_item_deepgram_m4a_uploads_mp3_when_artifact_enabled(
    mocker,
    tmp_path,
):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"
    m4a_path = tmp_path / "video.m4a"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    extract_mp3_mock = mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    mocker.patch("src.main.extract_m4a_copy", return_value=m4a_path)
    upload_mock = mocker.patch("src.main.drive.upload")
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hello")

    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        stt_language="ru",
        deepgram_audio_source="m4a_copy",
        drive_mp3_artifact=True,
    )
    main.process_item(service, _item("fid", "video.mp4"), "folderX", cfg)

    extract_mp3_mock.assert_called_once_with(mp4_path, bitrate="96k")
    # mp3, txt, plus the meta.yml/.stt written for every processed recording.
    assert upload_mock.call_count == 4
    assert upload_mock.call_args_list[0].kwargs["mime_type"] == "audio/mpeg"
    assert upload_mock.call_args_list[1].kwargs["mime_type"] == "text/plain"


def test_process_item_deepgram_mp3_96k_extracts_mp4_for_stt(mocker, tmp_path):
    service = MagicMock()

    def fake_download(svc, file_id, dest_dir, name, *, expected_size_bytes=None):
        path = dest_dir / name
        path.write_bytes(b"x")
        return path

    download_mock = mocker.patch("src.main.drive.download", side_effect=fake_download)
    mp3_path = tmp_path / "video.mp3"
    extract_mock = mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    upload_mock = mocker.patch("src.main.drive.upload")
    transcribe_mock = mocker.patch("src.main.transcribe_file", return_value="text")

    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        stt_language="ru",
        deepgram_audio_source="mp3_96k",
    )
    item = _item(
        "fid",
        "video.mp4",
        has_mp3=True,
        mp3_id="mp3id",
        mp3_name="video.mp3",
    )
    main.process_item(service, item, "folderX", cfg)

    assert download_mock.call_args.args[1] == "fid"
    extract_mock.assert_called_once()
    assert extract_mock.call_args.kwargs["bitrate"] == "96k"
    transcribe_mock.assert_called_once_with(mp3_path, cfg, cost_usd={})
    # txt, plus the meta.yml/.stt written for every processed recording.
    assert upload_mock.call_count == 3


def test_process_item_deepgram_mp3_192k_extracts_mp4_for_stt(mocker, tmp_path):
    service = MagicMock()

    def fake_download(svc, file_id, dest_dir, name, *, expected_size_bytes=None):
        path = dest_dir / name
        path.write_bytes(b"x")
        return path

    download_mock = mocker.patch("src.main.drive.download", side_effect=fake_download)
    mp3_path = tmp_path / "video.mp3"
    extract_mock = mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    upload_mock = mocker.patch("src.main.drive.upload")
    transcribe_mock = mocker.patch("src.main.transcribe_file", return_value="text")

    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        stt_language="ru",
        deepgram_audio_source="mp3_192k",
    )
    item = _item(
        "fid",
        "video.mp4",
        has_mp3=True,
        mp3_id="mp3id",
        mp3_name="video.mp3",
    )
    main.process_item(service, item, "folderX", cfg)

    assert download_mock.call_args.args[1] == "fid"
    extract_mock.assert_called_once()
    assert extract_mock.call_args.kwargs["bitrate"] == "192k"
    transcribe_mock.assert_called_once_with(mp3_path, cfg, cost_usd={})
    # txt, plus the meta.yml/.stt written for every processed recording.
    assert upload_mock.call_count == 3


def test_process_item_skips_completely_when_mp3_and_txt_present(mocker):
    service = MagicMock()
    download = mocker.patch("src.main.drive.download")
    extract = mocker.patch("src.main.extract_mp3")
    upload = mocker.patch("src.main.drive.upload")
    transcribe = mocker.patch("src.main.transcribe_file")

    cfg = make_config(stt_provider="deepgram", deepgram_api_key="dg-x")
    item = _item("fid", "v.mp4", has_mp3=True, has_txt=True,
                 mp3_id="m", mp3_name="v.mp3")
    main.process_item(service, item, "f", cfg)

    download.assert_not_called()
    extract.assert_not_called()
    upload.assert_not_called()
    transcribe.assert_not_called()


def test_process_item_reprocess_txt_overwrites_existing_txt(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    m4a_path = tmp_path / "video.m4a"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3")
    mocker.patch("src.main.extract_m4a_copy", return_value=m4a_path)
    mocker.patch("src.main.drive.upload")
    update_mock = mocker.patch("src.main.drive.update_file")
    transcribe_mock = mocker.patch("src.main.transcribe_file", return_value="fresh")

    cfg = make_config(stt_provider="deepgram", deepgram_api_key="dg-x")
    item = _item(
        "fid",
        "video.mp4",
        has_mp3=True,
        has_txt=True,
        mp3_id="m1",
        mp3_name="video.mp3",
        txt_id="t1",
    )
    main.process_item(service, item, "folderX", cfg, reprocess_txt=True)

    transcribe_mock.assert_called_once()
    update_mock.assert_called_once()
    assert update_mock.call_args.args[1] == "t1"


def test_process_target_single_file_resolves_parent(mocker):
    service = MagicMock()
    cfg = make_config()

    mocker.patch(
        "src.main.drive.get_file_metadata",
        return_value={
            "id": "v1",
            "name": "a.mp4",
            "mimeType": "video/mp4",
            "parents": ["folderA"],
        },
    )
    items = [_item("v1", "a.mp4"), _item("v2", "b.mp4")]
    list_mock = mocker.patch("src.main.drive.list_folder_state", return_value=items)
    process_mock = mocker.patch("src.main.process_item")

    main.process_target(service, "v1", cfg)

    list_mock.assert_called_once_with(service, "folderA")
    process_mock.assert_called_once()
    assert process_mock.call_args.args[1]["file"]["id"] == "v1"
    assert process_mock.call_args.args[2] == "folderA"


def test_process_target_retries_transient_metadata_error(mocker):
    service = MagicMock()
    cfg = make_config()

    meta_mock = mocker.patch(
        "src.main.drive.get_file_metadata",
        side_effect=[
            TimeoutError("temporary metadata timeout"),
            {
                "id": "v1",
                "name": "a.mp4",
                "mimeType": "video/mp4",
                "parents": ["folderA"],
            },
        ],
    )
    mocker.patch("src.main.time.sleep")
    items = [_item("v1", "a.mp4")]
    mocker.patch("src.main.drive.list_folder_state", return_value=items)
    process_mock = mocker.patch("src.main.process_item")

    main.process_target(service, "v1", cfg)

    assert meta_mock.call_count == 2
    process_mock.assert_called_once()


def test_process_target_folder_dry_run_does_not_process_items(mocker, caplog):
    service = MagicMock()
    cfg = make_config(stt_provider="deepgram", deepgram_api_key="dg-x")

    mocker.patch(
        "src.main.drive.get_file_metadata",
        return_value={"id": "folderA", "mimeType": main.drive.FOLDER_MIME},
    )
    items = [_item("v1", "pending.mp4", size=5_000_000)]
    mocker.patch("src.main.drive.list_folder_state", return_value=items)
    process_mock = mocker.patch("src.main.process_item")

    with caplog.at_level("INFO"):
        main.process_target(service, "folderA", cfg, is_folder=True, dry_run=True)

    process_mock.assert_not_called()
    assert "DRY RUN" in caplog.text
    assert "pending.mp4" in caplog.text


def test_dry_run_surfaces_preset_only_work(mocker, caplog):
    service = MagicMock()
    cfg = _two_preset_config()

    mocker.patch(
        "src.main.drive.get_file_metadata",
        return_value={"id": "folderA", "mimeType": main.drive.FOLDER_MIME},
    )
    # mp3+txt already done; only the expertizeme-managers preset is missing, so a
    # real run would spend OpenAI credits even though mp3/txt are skipped.
    item = _item(
        "v1",
        "done.mp4",
        has_mp3=True,
        has_txt=True,
        txt_id="t1",
        artifact_ids={"transcript-cleanup": "c1", "keypoints": "k1"},
    )
    mocker.patch("src.main.drive.list_folder_state", return_value=[item])
    mocker.patch("src.main.process_item")

    with caplog.at_level("INFO"):
        main.process_target(service, "folderA", cfg, is_folder=True, dry_run=True)

    assert "DRY RUN" in caplog.text
    assert "mp3=skip, txt=skip" in caplog.text
    assert "presets=expertizeme-managers" in caplog.text


def test_process_target_skips_large_file_without_confirmation(mocker, caplog):
    service = MagicMock()
    cfg = make_config(stt_provider="deepgram", deepgram_api_key="dg-x")

    mocker.patch(
        "src.main.drive.get_file_metadata",
        return_value={
            "id": "v1",
            "name": "large.mp4",
            "mimeType": "video/mp4",
            "parents": ["folderA"],
            "size": "200000000",
        },
    )
    items = [_item("v1", "large.mp4", size=200_000_000)]
    mocker.patch("src.main.drive.list_folder_state", return_value=items)
    process_mock = mocker.patch("src.main.process_item")

    with caplog.at_level("WARNING"):
        main.process_target(
            service,
            "v1",
            cfg,
            max_size_bytes=50_000_000,
            confirm_large=False,
        )

    process_mock.assert_not_called()
    assert "exceeds --max-size" in caplog.text


def test_process_target_processes_large_file_with_confirmation(mocker):
    service = MagicMock()
    cfg = make_config(stt_provider="deepgram", deepgram_api_key="dg-x")

    mocker.patch(
        "src.main.drive.get_file_metadata",
        return_value={
            "id": "v1",
            "name": "large.mp4",
            "mimeType": "video/mp4",
            "parents": ["folderA"],
            "size": "200000000",
        },
    )
    item = _item("v1", "large.mp4", size=200_000_000)
    mocker.patch("src.main.drive.list_folder_state", return_value=[item])
    process_mock = mocker.patch("src.main.process_item")

    main.process_target(
        service,
        "v1",
        cfg,
        max_size_bytes=50_000_000,
        confirm_large=True,
    )

    process_mock.assert_called_once()
    assert process_mock.call_args.args[1]["file"]["id"] == "v1"


def test_process_target_autodetects_folder(mocker):
    service = MagicMock()
    cfg = make_config()

    mocker.patch(
        "src.main.drive.get_file_metadata",
        return_value={
            "id": "folderX",
            "name": "My Folder",
            "mimeType": "application/vnd.google-apps.folder",
        },
    )
    items = [_item("v1", "a.mp4"), _item("v2", "b.mp4", has_mp3=True)]
    mocker.patch("src.main.drive.list_folder_state", return_value=items)
    process_mock = mocker.patch("src.main.process_item")

    main.process_target(service, "folderX", cfg)

    assert process_mock.call_count == 1
    assert process_mock.call_args.args[1]["file"]["id"] == "v1"
    assert process_mock.call_args.args[2] == "folderX"


def test_process_target_force_folder_flag(mocker):
    service = MagicMock()
    cfg = make_config()

    meta_mock = mocker.patch(
        "src.main.drive.get_file_metadata",
        return_value={"id": "folderX", "name": "f", "mimeType": "video/mp4"},
    )
    items = [_item("v1", "a.mp4")]
    mocker.patch("src.main.drive.list_folder_state", return_value=items)
    process_mock = mocker.patch("src.main.process_item")

    main.process_target(service, "folderX", cfg, is_folder=True)

    meta_mock.assert_called_once()
    process_mock.assert_called_once()
    assert process_mock.call_args.args[2] == "folderX"


def test_process_target_file_not_found_raises(mocker):
    service = MagicMock()
    cfg = make_config()

    mocker.patch(
        "src.main.drive.get_file_metadata",
        return_value={
            "id": "v9",
            "name": "missing.mp4",
            "mimeType": "video/mp4",
            "parents": ["folderA"],
        },
    )
    mocker.patch("src.main.drive.list_folder_state", return_value=[_item("v1", "a.mp4")])
    mocker.patch("src.main.process_item")

    with pytest.raises(RuntimeError):
        main.process_target(service, "v9", cfg)


def test_process_target_file_without_parent_raises(mocker):
    service = MagicMock()
    cfg = make_config()

    mocker.patch(
        "src.main.drive.get_file_metadata",
        return_value={"id": "v1", "name": "a.mp4", "mimeType": "video/mp4"},
    )

    with pytest.raises(RuntimeError):
        main.process_target(service, "v1", cfg)


def test_run_once_iterates_all_folders_and_files(mocker):
    service = MagicMock()
    cfg = make_config(folders=["f1", "f2"])

    listings = {
        "f1": [_item("v1", "a.mp4")],
        "f2": [_item("v2", "b.mp4"), _item("v3", "c.mp4")],
    }
    mocker.patch(
        "src.main.drive.list_folder_state",
        side_effect=lambda svc, fid: listings[fid],
    )
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(service, cfg)

    assert process_mock.call_count == 3
    calls = [(c.args[2], c.args[1]["file"]["id"]) for c in process_mock.call_args_list]
    assert ("f1", "v1") in calls
    assert ("f2", "v2") in calls
    assert ("f2", "v3") in calls


def test_run_once_retries_transient_listing_error(mocker, caplog):
    service = MagicMock()
    cfg = make_config(folders=["f1"])

    list_mock = mocker.patch(
        "src.main.drive.list_folder_state",
        side_effect=[TimeoutError("temporary api timeout"), [_item("v1", "a.mp4")]],
    )
    sleep_mock = mocker.patch("src.main.time.sleep")
    process_mock = mocker.patch("src.main.process_item")
    notify_mock = mocker.patch("src.main.notify.notify_error")

    with caplog.at_level("INFO"):
        main.run_once(service, cfg)

    assert list_mock.call_count == 2
    sleep_mock.assert_called_once()
    process_mock.assert_called_once()
    notify_mock.assert_not_called()
    assert "retry_total=1" in caplog.text


@pytest.mark.parametrize(
    "exc",
    [
        # httplib2 (what the Google API client uses) surfaces a dropped connection as a
        # builtin BrokenPipeError, not as one of requests' exceptions. Observed in
        # production: Drive closed a reused keep-alive socket and the whole cycle failed.
        BrokenPipeError(32, "Broken pipe"),
        ConnectionResetError(104, "Connection reset by peer"),
        ConnectionAbortedError(103, "Software caused connection abort"),
        ssl.SSLError("record layer failure"),
    ],
)
def test_transient_classifier_accepts_socket_level_errors(exc):
    assert main._is_transient_runtime_error(exc) is True


@pytest.mark.parametrize("exc", [RefreshError("token gone"), AuthError("token gone")])
def test_transient_classifier_still_rejects_auth_errors(exc):
    assert main._is_transient_runtime_error(exc) is False


def test_run_once_retries_broken_pipe_from_listing(mocker, caplog):
    service = MagicMock()
    cfg = make_config(folders=["f1"])

    list_mock = mocker.patch(
        "src.main.drive.list_folder_state",
        side_effect=[BrokenPipeError(32, "Broken pipe"), [_item("v1", "a.mp4")]],
    )
    sleep_mock = mocker.patch("src.main.time.sleep")
    process_mock = mocker.patch("src.main.process_item")
    notify_mock = mocker.patch("src.main.notify.notify_error")

    with caplog.at_level("INFO"):
        main.run_once(service, cfg)

    assert list_mock.call_count == 2
    sleep_mock.assert_called_once()
    process_mock.assert_called_once()
    # A retried socket drop must stay silent: alerting on it trains the operator to
    # ignore the channel that carries the real failures.
    notify_mock.assert_not_called()
    assert "retry_total=1" in caplog.text


def test_run_once_propagates_auth_error_from_listing(mocker):
    service = MagicMock()
    cfg = make_config(folders=["f1"])

    mocker.patch(
        "src.main.drive.list_folder_state",
        side_effect=AuthError("token gone"),
    )

    with pytest.raises(AuthError, match="token gone"):
        main.run_once(service, cfg)


def test_run_once_propagates_auth_error_from_processing(mocker):
    service = MagicMock()
    cfg = make_config(folders=["f1"])

    mocker.patch(
        "src.main.drive.list_folder_state",
        return_value=[_item("v1", "a.mp4")],
    )
    mocker.patch(
        "src.main.process_item",
        side_effect=AuthError("token gone"),
    )

    with pytest.raises(AuthError, match="token gone"):
        main.run_once(service, cfg)


def test_run_once_dry_run_does_not_process_items(mocker, caplog):
    service = MagicMock()
    cfg = make_config(
        folders=["folderA"],
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
    )
    mocker.patch(
        "src.main.drive.list_folder_state",
        return_value=[_item("v1", "pending.mp4", size=5_000_000)],
    )
    process_mock = mocker.patch("src.main.process_item")

    with caplog.at_level("INFO"):
        main.run_once(service, cfg, dry_run=True)

    process_mock.assert_not_called()
    assert "DRY RUN" in caplog.text
    assert "pending.mp4" in caplog.text
    assert "Cycle summary" in caplog.text


def test_run_once_skips_large_pending_items_without_confirmation(mocker, caplog):
    service = MagicMock()
    cfg = make_config(
        folders=["folderA"],
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
    )
    mocker.patch(
        "src.main.drive.list_folder_state",
        return_value=[_item("v1", "large.mp4", size=200_000_000)],
    )
    process_mock = mocker.patch("src.main.process_item")

    with caplog.at_level("WARNING"):
        main.run_once(service, cfg, max_size_bytes=50_000_000)

    process_mock.assert_not_called()
    assert "exceeds --max-size" in caplog.text


def test_run_once_filters_already_processed(mocker):
    service = MagicMock()
    cfg = make_config(folders=["f1"])

    items = [
        _item("v1", "a.mp4", has_mp3=False),
        _item("v2", "b.mp4", has_mp3=True),
    ]
    mocker.patch("src.main.drive.list_folder_state", return_value=items)
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(service, cfg)

    assert process_mock.call_count == 1
    assert process_mock.call_args.args[1]["file"]["id"] == "v1"


def test_run_once_with_stt_includes_files_missing_txt(mocker):
    service = MagicMock()
    cfg = make_config(folders=["f1"], stt_provider="deepgram", deepgram_api_key="dg-x")

    items = [
        _item("v1", "a.mp4", has_mp3=True, has_txt=True, mp3_id="m1", mp3_name="a.mp3"),
        _item("v2", "b.mp4", has_mp3=True, has_txt=False, mp3_id="m2", mp3_name="b.mp3"),
    ]
    mocker.patch("src.main.drive.list_folder_state", return_value=items)
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(service, cfg)

    assert process_mock.call_count == 1
    assert process_mock.call_args.args[1]["file"]["id"] == "v2"


def test_run_once_continues_on_per_file_error(mocker):
    service = MagicMock()
    cfg = make_config(folders=["f1"])

    items = [
        _item("good1", "ok1.mp4"),
        _item("bad", "fail.mp4"),
        _item("good2", "ok2.mp4"),
    ]
    mocker.patch("src.main.drive.list_folder_state", return_value=items)

    processed_ids = []

    def fake_process(svc, item, folder, c, *, booking_decision=None):
        if item["file"]["id"] == "bad":
            raise RuntimeError("ffmpeg failed")
        processed_ids.append(item["file"]["id"])

    mocker.patch("src.main.process_item", side_effect=fake_process)
    notify_mock = mocker.patch("src.main.notify.notify_error")

    main.run_once(service, cfg)

    assert processed_ids == ["good1", "good2"]
    notify_mock.assert_called_once()
    assert "fail.mp4" in notify_mock.call_args.args[0]


def test_process_item_retries_transient_download_error(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    download_mock = mocker.patch(
        "src.main.drive.download",
        side_effect=[TimeoutError("temporary download timeout"), mp4_path],
    )
    sleep_mock = mocker.patch("src.main.time.sleep")
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    upload_mock = mocker.patch("src.main.drive.upload", return_value={"id": "uploaded"})

    cfg = make_config(bitrate="128k")
    telemetry = main.process_item(service, _item("fid1", "video.mp4"), "folderA", cfg)

    assert download_mock.call_count == 2
    sleep_mock.assert_called_once()
    upload_mock.assert_called_once()
    assert telemetry.processing_mode == "artifact-only"
    assert telemetry.retry_count == 1


def test_process_item_retries_download_size_mismatch(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    download_mock = mocker.patch(
        "src.main.drive.download",
        side_effect=[
            main.drive.DownloadIntegrityError("Downloaded file size mismatch"),
            mp4_path,
        ],
    )
    sleep_mock = mocker.patch("src.main.time.sleep")
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    upload_mock = mocker.patch("src.main.drive.upload", return_value={"id": "uploaded"})

    cfg = make_config(bitrate="128k")
    main.process_item(service, _item("fid1", "video.mp4", size=123), "folderA", cfg)

    assert download_mock.call_count == 2
    assert download_mock.call_args_list[0].kwargs == {"expected_size_bytes": 123}
    sleep_mock.assert_called_once()
    upload_mock.assert_called_once()


def test_process_item_logs_process_summary(mocker, tmp_path, caplog):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    mocker.patch("src.main.drive.upload", return_value={"id": "uploaded"})
    mocker.patch("src.main.time.monotonic", side_effect=[10.0, 11.5])

    cfg = make_config(stt_provider="", drive_mp3_artifact=True)

    with caplog.at_level("INFO"):
        main.process_item(service, _item("fid1", "video.mp4"), "folderA", cfg)

    assert (
        "Process summary [file=video.mp4, file_id=fid1, folder=folderA, "
        "provider=artifact-only, processing_mode=artifact-only, outcome=success, "
        "retry_count=0, duration_s=1.500, cost_usd={}, usage={}]"
    ) in caplog.text


def test_process_item_logs_failed_summary(mocker, tmp_path, caplog):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    mocker.patch("src.main.extract_m4a_copy", return_value=tmp_path / "video.m4a")
    mocker.patch("src.main.drive.upload")
    mocker.patch("src.main.transcribe_file", side_effect=STTError("provider failed"))
    mocker.patch("src.main.time.monotonic", side_effect=[20.0, 21.0])

    cfg = make_config(stt_provider="deepgram", deepgram_api_key="dg-x")

    with caplog.at_level("INFO"):
        with pytest.raises(STTError, match="provider failed"):
            main.process_item(service, _item("fid1", "video.mp4"), "folderA", cfg)

    assert (
        "Process summary [file=video.mp4, file_id=fid1, folder=folderA, "
        "provider=deepgram, processing_mode=artifact-and-txt, outcome=failed, "
        "retry_count=0, duration_s=1.000, cost_usd={}, usage={}]"
    ) in caplog.text


def test_process_item_logs_txt_only_processing_mode(mocker, tmp_path, caplog):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    m4a_path = tmp_path / "video.m4a"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_m4a_copy", return_value=m4a_path)
    mocker.patch("src.main.transcribe_file", return_value="hello")
    mocker.patch("src.main.drive.upload", return_value={"id": "uploaded"})
    mocker.patch("src.main.time.monotonic", side_effect=[30.0, 31.25])

    cfg = make_config(stt_provider="deepgram", deepgram_api_key="dg-x", drive_mp3_artifact=True)

    with caplog.at_level("INFO"):
        telemetry = main.process_item(
            service,
            _item("fid1", "video.mp4", has_mp3=True, has_txt=False, mp3_id="m1", mp3_name="video.mp3"),
            "folderA",
            cfg,
        )

    assert telemetry.processing_mode == "txt-only"
    assert (
        "Process summary [file=video.mp4, file_id=fid1, folder=folderA, "
        "provider=deepgram, processing_mode=txt-only, outcome=success, "
        "retry_count=0, duration_s=1.250, cost_usd={}, usage={}]"
    ) in caplog.text


def test_process_item_summary_surfaces_cost_and_keypoints_usage(mocker, tmp_path, caplog):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    mocker.patch("src.main.drive.upload")

    def fake_transcribe(path, config, *, cost_usd):
        cost_usd["deepgram"] = 0.0123
        return "Speaker 1: hi"

    mocker.patch("src.main.transcribe_file", side_effect=fake_transcribe)
    mocker.patch(
        "src.main.preset_pipeline.run_presets",
        return_value={
            "keypoints": PresetResult(
                name="keypoints",
                text="## Задачи\n- [ ] do it",
                usage={"input_tokens": 100, "output_tokens": 40, "total_tokens": 140},
            )
        },
    )

    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        deepgram_audio_source="mp3_96k",
        openai_api_key="sk-x",
        openai_keypoints=True,
        drive_mp3_artifact=False,
    )

    with caplog.at_level("INFO"):
        telemetry = main.process_item(
            service, _item("fid1", "video.mp4"), "folderA", cfg
        )

    # Telemetry carries the computed Deepgram spend and OpenAI keypoints usage.
    assert telemetry.cost_usd == {"deepgram": 0.0123}
    assert telemetry.usage == {
        "openai_keypoints": {
            "input_tokens": 100,
            "output_tokens": 40,
            "total_tokens": 140,
        }
    }
    # The summary log line surfaces them instead of discarding the spend.
    assert "cost_usd={'deepgram': 0.0123}" in caplog.text
    assert "'openai_keypoints': {'input_tokens': 100" in caplog.text


def test_run_once_continues_on_listing_error(mocker):
    service = MagicMock()
    cfg = make_config(folders=["bad_folder", "good_folder"])

    def fake_list(svc, folder_id):
        if folder_id == "bad_folder":
            raise RuntimeError("api error")
        return [_item("v1", "a.mp4")]

    mocker.patch("src.main.drive.list_folder_state", side_effect=fake_list)
    process_mock = mocker.patch("src.main.process_item")
    notify_mock = mocker.patch("src.main.notify.notify_error")

    main.run_once(service, cfg)

    assert process_mock.call_count == 1
    assert process_mock.call_args.args[2] == "good_folder"
    notify_mock.assert_called_once()
    assert "bad_folder" in notify_mock.call_args.args[0]


def test_run_once_no_folders_does_nothing(mocker):
    service = MagicMock()
    cfg = make_config(folders=[])

    list_mock = mocker.patch("src.main.drive.list_folder_state")
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(service, cfg)

    list_mock.assert_not_called()
    process_mock.assert_not_called()


def test_run_once_passes_config_to_process(mocker):
    service = MagicMock()
    cfg = make_config(folders=["f1"], bitrate="192k")

    mocker.patch(
        "src.main.drive.list_folder_state",
        return_value=[_item("v1", "a.mp4")],
    )
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(service, cfg)

    assert process_mock.call_args.args[3] is cfg


def test_run_once_logs_folder_and_cycle_summary(mocker, caplog):
    service = MagicMock()
    cfg = make_config(folders=["f1"], stt_provider="deepgram", deepgram_api_key="dg-x")

    items = [
        _item("v1", "a.mp4", has_mp3=True, has_txt=False, mp3_id="m1", mp3_name="a.mp3"),
        _item("v2", "b.mp4", has_mp3=True, has_txt=True, mp3_id="m2", mp3_name="b.mp3"),
    ]
    mocker.patch("src.main.drive.list_folder_state", return_value=items)
    mocker.patch("src.main.process_item")
    mocker.patch("src.main.time.monotonic", side_effect=[100.0, 101.25])

    with caplog.at_level("INFO"):
        main.run_once(service, cfg)

    assert (
        "Folder f1 summary [total=2, pending=1, skipped_size=0, skipped_old=0, "
        "dry_run=False]" in caplog.text
    )
    assert (
        "Cycle summary [provider=deepgram, outcome=success, folders=1, pending=1, "
        "processed=1, failed=0, retry_total=0, skipped_size=0, skipped_unmatched=0, "
        "skipped_old=0, folder_errors=0, deferred=0, cursor_moved=True, dry_run=False, "
        "duration_s=1.250]"
    ) in caplog.text


def test_run_once_aggregates_retry_total_from_process_telemetry(mocker, caplog):
    service = MagicMock()
    cfg = make_config(folders=["f1"], stt_provider="deepgram", deepgram_api_key="dg-x")

    mocker.patch(
        "src.main.drive.list_folder_state",
        return_value=[_item("v1", "a.mp4", has_mp3=True, has_txt=False, mp3_id="m1", mp3_name="a.mp3")],
    )
    mocker.patch(
        "src.main.process_item",
        return_value=main._ProcessTelemetry(
            provider="openai",
            processing_mode="txt-only",
            retry_count=2,
            duration_s=0.5,
        ),
    )
    mocker.patch("src.main.time.monotonic", side_effect=[200.0, 201.0])

    with caplog.at_level("INFO"):
        main.run_once(service, cfg)

    assert "retry_total=2" in caplog.text


def test_main_runs_loop_and_sleeps(mocker):
    cfg = make_config(folders=["f1"], poll_interval=42)
    mocker.patch("src.main.load_config", return_value=cfg)
    service = MagicMock()
    mocker.patch("src.main.build_drive_service", return_value=service)

    run_calls = {"n": 0}

    def fake_run_once(svc, c, **kwargs):
        run_calls["n"] += 1
        if run_calls["n"] >= 2:
            raise KeyboardInterrupt

    mocker.patch("src.main.run_once", side_effect=fake_run_once)
    sleep_mock = mocker.patch("src.main.time.sleep")

    with pytest.raises(KeyboardInterrupt):
        main.main()

    assert run_calls["n"] == 2
    sleep_mock.assert_called_with(42)


def test_run_once_propagates_refresh_error(mocker):
    service = MagicMock()
    cfg = make_config(folders=["f1"])

    mocker.patch(
        "src.main.drive.list_folder_state",
        side_effect=RefreshError("token revoked"),
    )
    notify_mock = mocker.patch("src.main.notify.notify_error")

    with pytest.raises(RefreshError):
        main.run_once(service, cfg)

    notify_mock.assert_not_called()


def test_run_once_propagates_refresh_error_from_process(mocker):
    service = MagicMock()
    cfg = make_config(folders=["f1"])

    mocker.patch(
        "src.main.drive.list_folder_state",
        return_value=[_item("v1", "a.mp4")],
    )
    mocker.patch("src.main.process_item", side_effect=RefreshError("token revoked"))
    notify_mock = mocker.patch("src.main.notify.notify_error")

    with pytest.raises(RefreshError):
        main.run_once(service, cfg)

    notify_mock.assert_not_called()


def test_main_exits_on_bootstrap_auth_error(mocker):
    cfg = make_config(folders=["f1"], poll_interval=1)
    mocker.patch("src.main.load_config", return_value=cfg)
    mocker.patch(
        "src.main.build_drive_service", side_effect=AuthError("malformed token")
    )
    notify_mock = mocker.patch("src.main.notify.notify_error")
    sleep_mock = mocker.patch("src.main.time.sleep")

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 1
    notify_mock.assert_called_once()
    assert "bootstrap" in notify_mock.call_args.args[0]
    sleep_mock.assert_not_called()


def test_main_exits_on_bootstrap_refresh_error(mocker):
    cfg = make_config(folders=["f1"], poll_interval=1)
    mocker.patch("src.main.load_config", return_value=cfg)
    mocker.patch(
        "src.main.build_drive_service", side_effect=RefreshError("revoked")
    )
    notify_mock = mocker.patch("src.main.notify.notify_error")

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 1
    notify_mock.assert_called_once()


def test_main_exits_on_refresh_error(mocker):
    cfg = make_config(folders=["f1"], poll_interval=1)
    mocker.patch("src.main.load_config", return_value=cfg)
    mocker.patch("src.main.build_drive_service", return_value=MagicMock())
    mocker.patch("src.main.run_once", side_effect=RefreshError("revoked"))
    notify_mock = mocker.patch("src.main.notify.notify_error")
    sleep_mock = mocker.patch("src.main.time.sleep")

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 1
    notify_mock.assert_called_once()
    assert "OAuth" in notify_mock.call_args.args[0]
    sleep_mock.assert_not_called()


def test_main_exits_on_auth_error(mocker):
    cfg = make_config(folders=["f1"], poll_interval=1)
    mocker.patch("src.main.load_config", return_value=cfg)
    mocker.patch("src.main.build_drive_service", return_value=MagicMock())
    mocker.patch("src.main.run_once", side_effect=AuthError("token gone"))
    notify_mock = mocker.patch("src.main.notify.notify_error")

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 1
    notify_mock.assert_called_once()


def test_main_notifies_on_cycle_exception(mocker):
    cfg = make_config(folders=["f1"], poll_interval=1)
    mocker.patch("src.main.load_config", return_value=cfg)
    mocker.patch("src.main.build_drive_service", return_value=MagicMock())

    call_count = {"n": 0}

    def fake_run_once(svc, c, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom")
        raise KeyboardInterrupt

    mocker.patch("src.main.run_once", side_effect=fake_run_once)
    notify_mock = mocker.patch("src.main.notify.notify_error")
    mocker.patch("src.main.time.sleep")

    with pytest.raises(KeyboardInterrupt):
        main.main()

    notify_mock.assert_called_once()
    assert "boom" in notify_mock.call_args.args[0]


def test_process_item_preserves_slash_name_on_upload(mocker, tmp_path):
    service = MagicMock()
    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        deepgram_audio_source="mp3_96k",
    )

    captured = {}

    def fake_download(svc, file_id, dest_dir, name, *, expected_size_bytes=None):
        # Drive name with "/" must become a filesystem-safe local temp file.
        path = dest_dir / main.drive.safe_local_name(name)
        path.write_bytes(b"x")
        return path

    def fake_extract(mp4_path, bitrate):
        return mp4_path.with_suffix(".mp3")

    def fake_upload(svc, local_path, folder, mime_type, name=None, app_properties=None):
        captured.setdefault("uploads", []).append((name, local_path.name, mime_type))

    mocker.patch("src.main.drive.download", side_effect=fake_download)
    mocker.patch("src.main.extract_mp3", side_effect=fake_extract)
    mocker.patch("src.main.drive.upload", side_effect=fake_upload)
    mocker.patch("src.main.transcribe_file", return_value="text")

    item = _item("fid", "Call 2026/05/28 Rec.mp4")
    main.process_item(service, item, "folderX", cfg)

    uploads = dict((name, local) for name, local, _ in captured["uploads"])
    # Drive upload names keep the original "/"; local temp names are sanitized.
    assert "Call 2026/05/28 Rec.mp3" in uploads
    assert "Call 2026/05/28 Rec.txt" in uploads
    for drive_name, local_name in uploads.items():
        assert "/" not in local_name


def test_save_and_upload_txt_creates_when_no_txt_id(mocker, tmp_path):
    service = MagicMock()
    upload_mock = mocker.patch("src.main.drive.upload")
    update_mock = mocker.patch("src.main.drive.update_file")

    main._save_and_upload_txt(
        service, "fid", "video.mp4", "hello", "folderA", tmp_path, make_config(),
    )

    update_mock.assert_not_called()
    upload_mock.assert_called_once()
    assert upload_mock.call_args.kwargs["name"] == "video.txt"


def test_save_and_upload_txt_overwrites_existing(mocker, tmp_path):
    service = MagicMock()
    upload_mock = mocker.patch("src.main.drive.upload")
    update_mock = mocker.patch("src.main.drive.update_file")

    main._save_and_upload_txt(
        service, "fid", "video.mp4", "final text", "folderA", tmp_path, make_config(),
        txt_id="t1",
    )

    upload_mock.assert_not_called()
    update_mock.assert_called_once()
    args = update_mock.call_args.args
    assert args[1] == "t1"
    assert args[2].read_text(encoding="utf-8") == "final text"


def test_process_item_writes_txt_to_local_folder_when_output_target_folder(
    mocker,
    tmp_path,
):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"
    out_dir = tmp_path / "transcripts"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    upload_mock = mocker.patch("src.main.drive.upload")
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")

    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        deepgram_audio_source="mp3_96k",
        drive_mp3_artifact=False,
        output_target="folder",
        output_dir=out_dir,
    )
    main.process_item(service, _item("fid", "video.mp4"), "folderX", cfg)

    # No txt upload to Drive; the transcript landed in the local folder instead.
    upload_mock.assert_not_called()
    assert (out_dir / "video.txt").read_text(encoding="utf-8") == "Speaker 1: hi"


def test_process_item_postprocesses_transcript_before_upload(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    captured = {}

    def fake_upload(svc, local_path, folder, mime_type, name=None, app_properties=None):
        # `.meta.yml`/`.stt` share this mime type too, so match on the `.txt`
        # suffix specifically rather than on mime_type alone.
        if name and name.endswith(".txt"):
            captured["txt"] = local_path.read_text(encoding="utf-8")

    mocker.patch("src.main.drive.upload", side_effect=fake_upload)
    mocker.patch(
        "src.main.transcribe_file",
        return_value="Speaker 1: hi there\nSpeaker 2: hello back",
    )

    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        deepgram_audio_source="mp3_96k",
        stt_postprocess=True,
    )
    main.process_item(service, _item("fid", "Alice and Bob.mp4"), "f", cfg)

    assert captured["txt"] == "Alice: hi there\nBob: hello back"


def test_process_item_generates_keypoints_when_enabled(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    captured: dict = {}

    def fake_upload(svc, local_path, folder, mime_type, name=None, app_properties=None):
        captured.setdefault("uploads", []).append(
            (name, local_path.read_text(encoding="utf-8"), mime_type, app_properties)
        )

    mocker.patch("src.main.drive.upload", side_effect=fake_upload)
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")
    kp_mock = mocker.patch(
        "src.main.preset_pipeline.run_presets",
        return_value={
            "keypoints": PresetResult(
                name="keypoints",
                text="## Задачи\n\n## Тезисы\n- point\n\n## Открытые вопросы",
            )
        },
    )

    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        deepgram_audio_source="mp3_96k",
        openai_api_key="sk-x",
        openai_keypoints=True,
        drive_mp3_artifact=False,
    )
    main.process_item(service, _item("fid", "video.mp4"), "folderX", cfg)

    kp_mock.assert_called_once()
    # The preset DAG runs on the produced transcript, restricted to the missing
    # keypoints preset.
    assert kp_mock.call_args.args[0] == "Speaker 1: hi"
    assert kp_mock.call_args.kwargs["only"] == ["keypoints"]
    uploads = {
        name: (text, mime, props) for name, text, mime, props in captured["uploads"]
    }
    assert "video.keypoints.md" in uploads
    text, mime, props = uploads["video.keypoints.md"]
    assert "## Задачи" in text
    assert mime == "text/markdown"
    assert props == {"source_video_id": "fid", "artifact_type": "keypoints"}


def test_process_item_overwrites_existing_keypoints_on_reprocess(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    upload_mock = mocker.patch("src.main.drive.upload")
    update_mock = mocker.patch("src.main.drive.update_file")
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")
    mocker.patch(
        "src.main.preset_pipeline.run_presets",
        return_value={
            "keypoints": PresetResult(
                name="keypoints", text="## Задачи\n- [ ] do it"
            )
        },
    )

    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        deepgram_audio_source="mp3_96k",
        openai_api_key="sk-x",
        openai_keypoints=True,
        drive_mp3_artifact=False,
    )
    item = _item(
        "fid", "video.mp4", has_txt=True, txt_id="t1", keypoints_id="k1",
        stt_id="s1", meta_yml_id="y1",
    )
    main.process_item(service, item, "folderX", cfg, reprocess_txt=True)

    # The .txt, the .keypoints.md, the .stt, and the .meta.yml siblings are all
    # overwritten in place, not re-uploaded as duplicates.
    update_ids = [call.args[1] for call in update_mock.call_args_list]
    assert "t1" in update_ids
    assert "k1" in update_ids
    assert "s1" in update_ids
    assert "y1" in update_ids
    upload_mock.assert_not_called()


def test_process_item_skips_keypoints_when_disabled(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    mocker.patch("src.main.drive.upload")
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")
    kp_mock = mocker.patch("src.main.preset_pipeline.run_presets")

    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        deepgram_audio_source="mp3_96k",
        openai_api_key="sk-x",
        openai_keypoints=False,
        drive_mp3_artifact=False,
    )
    main.process_item(service, _item("fid", "video.mp4"), "folderX", cfg)

    kp_mock.assert_not_called()


def test_process_item_writes_keypoints_to_local_folder(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"
    out_dir = tmp_path / "transcripts"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    upload_mock = mocker.patch("src.main.drive.upload")
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")
    mocker.patch(
        "src.main.preset_pipeline.run_presets",
        return_value={
            "keypoints": PresetResult(
                name="keypoints", text="## Задачи\n- [ ] do it"
            )
        },
    )

    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        deepgram_audio_source="mp3_96k",
        openai_api_key="sk-x",
        openai_keypoints=True,
        drive_mp3_artifact=False,
        output_target="folder",
        output_dir=out_dir,
    )
    main.process_item(service, _item("fid", "video.mp4"), "folderX", cfg)

    upload_mock.assert_not_called()
    assert (out_dir / "video.txt").read_text(encoding="utf-8") == "Speaker 1: hi"
    assert (out_dir / "video.keypoints.md").read_text(encoding="utf-8") == (
        "## Задачи\n- [ ] do it"
    )


def test_apply_local_output_state_marks_has_txt_from_local_file(tmp_path):
    out_dir = tmp_path / "transcripts"
    out_dir.mkdir()
    (out_dir / "video.txt").write_text("done", encoding="utf-8")
    cfg = make_config(output_target="folder", output_dir=out_dir)

    items = [_item("fid", "video.mp4"), _item("gid", "other.mp4")]
    main._apply_local_output_state(items, cfg)

    assert items[0]["has_txt"] is True
    assert items[1]["has_txt"] is False


def test_apply_local_output_state_noop_for_drive_target(tmp_path):
    out_dir = tmp_path / "transcripts"
    out_dir.mkdir()
    (out_dir / "video.txt").write_text("done", encoding="utf-8")
    cfg = make_config(output_target="drive", output_dir=out_dir)

    items = [_item("fid", "video.mp4")]
    main._apply_local_output_state(items, cfg)

    assert items[0]["has_txt"] is False


def test_run_once_skips_already_transcribed_local_file_in_folder_mode(
    mocker, tmp_path
):
    out_dir = tmp_path / "transcripts"
    out_dir.mkdir()
    # A prior run already wrote the transcript locally.
    (out_dir / "video.txt").write_text("Speaker 1: hi", encoding="utf-8")

    service = MagicMock()
    # Drive has no .txt sibling because folder mode wrote it locally.
    mocker.patch(
        "src.main.drive.list_folder_state",
        return_value=[_item("fid", "video.mp4")],
    )
    process_mock = mocker.patch("src.main.process_item")

    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg",
        deepgram_audio_source="m4a_copy",
        drive_mp3_artifact=False,
        output_target="folder",
        output_dir=out_dir,
    )
    main.run_once(service, cfg)

    process_mock.assert_not_called()


def test_process_item_does_not_write_empty_keypoints(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    captured: dict = {}

    def fake_upload(svc, local_path, folder, mime_type, name=None, app_properties=None):
        captured.setdefault("names", []).append(name)

    mocker.patch("src.main.drive.upload", side_effect=fake_upload)
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")
    mocker.patch(
        "src.main.preset_pipeline.run_presets",
        return_value={"keypoints": PresetResult(name="keypoints", text="   \n  ")},
    )

    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        deepgram_audio_source="mp3_96k",
        openai_api_key="sk-x",
        openai_keypoints=True,
        drive_mp3_artifact=False,
    )
    main.process_item(service, _item("fid", "video.mp4"), "folderX", cfg)

    # A blank keypoints doc is not uploaded, but the .txt and the meta.yml/.stt
    # written for every processed recording still are.
    assert captured["names"] == ["video.txt", "video.meta.yml", "video.stt"]


def test_process_item_uses_speaker_names_from_drive_properties(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    captured = {}

    def fake_upload(svc, local_path, folder, mime_type, name=None, app_properties=None):
        # `.meta.yml`/`.stt` share this mime type too, so match on the `.txt`
        # suffix specifically rather than on mime_type alone.
        if name and name.endswith(".txt"):
            captured["txt"] = local_path.read_text(encoding="utf-8")

    mocker.patch("src.main.drive.upload", side_effect=fake_upload)
    mocker.patch(
        "src.main.transcribe_file",
        return_value="Speaker 1: hi there\nSpeaker 2: hello back",
    )

    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        deepgram_audio_source="mp3_96k",
        stt_postprocess=True,
    )
    item = _item("fid", "Unhelpful file name.mp4")
    item["file"]["appProperties"] = {"speaker_names": "[\"Alice\", \"Bob\"]"}
    main.process_item(service, item, "f", cfg)

    assert captured["txt"] == "Alice: hi there\nBob: hello back"


def _two_preset_config(**overrides):
    presets = (
        Preset(name="transcript-cleanup", instructions="clean it"),
        Preset(
            name="keypoints",
            instructions="summarize",
            artifact_suffix=".keypoints.md",
            depends_on=("transcript-cleanup",),
        ),
        Preset(
            name="expertizeme-managers",
            instructions="managers",
            depends_on=("transcript-cleanup",),
        ),
    )
    base = dict(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        deepgram_audio_source="mp3_96k",
        openai_api_key="sk-x",
        openai_keypoints=True,
        drive_mp3_artifact=False,
        presets=presets,
    )
    base.update(overrides)
    return make_config(**base)


def test_process_item_writes_one_artifact_per_produced_preset(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    captured: dict = {}

    def fake_upload(svc, local_path, folder, mime_type, name=None, app_properties=None):
        captured.setdefault("uploads", []).append(
            (name, local_path.read_text(encoding="utf-8"), mime_type, app_properties)
        )

    mocker.patch("src.main.drive.upload", side_effect=fake_upload)
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")
    mocker.patch(
        "src.main.preset_pipeline.run_presets",
        return_value={
            "transcript-cleanup": PresetResult(
                name="transcript-cleanup", text="cleaned transcript"
            ),
            "keypoints": PresetResult(name="keypoints", text="## Задачи\n- [ ] do it"),
            "expertizeme-managers": PresetResult(
                name="expertizeme-managers", text="manager notes"
            ),
        },
    )

    main.process_item(service, _item("fid", "video.mp4"), "folderX", _two_preset_config())

    uploads = {
        name: (text, mime, props) for name, text, mime, props in captured["uploads"]
    }
    # The .txt plus one sibling per produced preset, each tagged with its own
    # artifact_type and using its own suffix.
    assert "video.txt" in uploads
    assert uploads["video.transcript-cleanup.md"][2] == {
        "source_video_id": "fid",
        "artifact_type": "transcript-cleanup",
    }
    assert uploads["video.keypoints.md"][2] == {
        "source_video_id": "fid",
        "artifact_type": "keypoints",
    }
    assert uploads["video.expertizeme-managers.md"][2] == {
        "source_video_id": "fid",
        "artifact_type": "expertizeme-managers",
    }


def test_process_item_skips_presets_with_existing_artifacts(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    mocker.patch("src.main.drive.upload")
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")
    download_text = mocker.patch(
        "src.main.drive.download_text", return_value="cleaned"
    )
    run_mock = mocker.patch(
        "src.main.preset_pipeline.run_presets",
        return_value={
            "expertizeme-managers": PresetResult(
                name="expertizeme-managers", text="notes"
            ),
        },
    )

    # transcript-cleanup and keypoints already have artifacts; only
    # expertizeme-managers is missing and must be requested. Its dependency
    # transcript-cleanup is reused from its persisted artifact (download_text)
    # instead of being re-run.
    item = _item(
        "fid",
        "video.mp4",
        artifact_ids={"transcript-cleanup": "c1", "keypoints": "k1"},
    )
    config = _two_preset_config(webhook_url="https://hook.example/x")
    main.process_item(service, item, "folderX", config)

    run_mock.assert_called_once()
    assert run_mock.call_args.kwargs["only"] == ["expertizeme-managers"]
    assert run_mock.call_args.kwargs["precomputed"] == {"transcript-cleanup": "cleaned"}
    # c1 feeds the dependency; k1 is read back only so the webhook payload carries
    # the keypoints produced on an earlier cycle. Neither preset is re-run.
    assert download_text.call_count == 2
    download_text.assert_any_call(service, "c1")
    download_text.assert_any_call(service, "k1")


def test_process_item_skips_webhook_backfill_when_no_webhook_configured(
    mocker, tmp_path
):
    """Without a receiver, don't pay a Drive read per earlier-cycle artifact.

    The backfill's only consumer is the completion webhook, which no-ops on a blank
    URL — so reading ``k1`` back would be a round-trip whose result is discarded.
    The dependency read (``c1``) still happens: it feeds the preset that is re-run.
    """
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    mocker.patch("src.main.drive.upload")
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")
    download_text = mocker.patch(
        "src.main.drive.download_text", return_value="cleaned"
    )
    mocker.patch(
        "src.main.preset_pipeline.run_presets",
        return_value={
            "expertizeme-managers": PresetResult(
                name="expertizeme-managers", text="notes"
            ),
        },
    )

    item = _item(
        "fid",
        "video.mp4",
        artifact_ids={"transcript-cleanup": "c1", "keypoints": "k1"},
    )
    main.process_item(service, item, "folderX", _two_preset_config(webhook_url=""))

    download_text.assert_called_once_with(service, "c1")


def test_process_item_skips_preset_stage_when_all_present(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    mocker.patch("src.main.drive.upload")
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")
    run_mock = mocker.patch("src.main.preset_pipeline.run_presets")

    item = _item(
        "fid",
        "video.mp4",
        artifact_ids={
            "transcript-cleanup": "c1",
            "keypoints": "k1",
            "expertizeme-managers": "e1",
        },
    )
    main.process_item(service, item, "folderX", _two_preset_config())

    run_mock.assert_not_called()


def test_process_item_backfills_webhook_when_every_preset_already_present(
    mocker, tmp_path
):
    """A regenerated `.txt` still ships the earlier cycle's artifacts to the receiver.

    When a file's `.txt` sibling is deleted but its preset artifacts survive, the
    file is re-selected and re-transcribed, yet no preset is missing — so the stage
    runs nothing. The webhook fires regardless (the `.txt` was uploaded), so it must
    still carry the artifacts sitting on Drive rather than an empty map.
    """
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    mocker.patch("src.main.drive.upload")
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")
    mocker.patch("src.main.drive.download_text", side_effect=lambda svc, fid: fid)
    run_mock = mocker.patch("src.main.preset_pipeline.run_presets")

    item = _item(
        "fid",
        "video.mp4",
        artifact_ids={
            "transcript-cleanup": "c1",
            "keypoints": "k1",
            "expertizeme-managers": "e1",
        },
    )
    config = _two_preset_config(webhook_url="https://hook.example/x")
    telemetry = main.process_item(service, item, "folderX", config)

    run_mock.assert_not_called()
    assert telemetry is not None
    assert telemetry.artifacts == {
        "transcript-cleanup": "c1",
        "keypoints": "k1",
        "expertizeme-managers": "e1",
    }


def test_process_item_survives_backfill_read_failure(mocker, tmp_path):
    """A failed backfill read degrades the payload instead of failing the file.

    The backfill's reads exist only to enrich the webhook, and they run after every
    artifact is already persisted. If a Drive read raised out of the stage, a file
    that fully succeeded would be counted failed and alerted on — and it would never
    reach the webhook at all, since the next cycle finds no preset missing.
    """
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    mocker.patch("src.main.drive.upload")
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")

    def flaky_download(svc, fid):
        if fid == "k1":
            raise RuntimeError("drive 404")
        return fid

    mocker.patch("src.main.drive.download_text", side_effect=flaky_download)
    notify = mocker.patch("src.main.webhook.notify_complete")

    item = _item(
        "fid",
        "video.mp4",
        artifact_ids={
            "transcript-cleanup": "c1",
            "keypoints": "k1",
            "expertizeme-managers": "e1",
        },
    )
    config = _two_preset_config(webhook_url="https://hook.example/x")
    telemetry = main.process_item(service, item, "folderX", config)

    # The unreadable preset drops out; the rest still reach the receiver.
    assert telemetry is not None
    assert telemetry.artifacts == {
        "transcript-cleanup": "c1",
        "expertizeme-managers": "e1",
    }
    notify.assert_called_once()


def test_process_item_raises_aggregated_error_but_persists_successes(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    captured: dict = {}

    def fake_upload(svc, local_path, folder, mime_type, name=None, app_properties=None):
        captured.setdefault("names", []).append(name)

    mocker.patch("src.main.drive.upload", side_effect=fake_upload)
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")
    mocker.patch(
        "src.main.preset_pipeline.run_presets",
        return_value={
            "transcript-cleanup": PresetResult(
                name="transcript-cleanup", text="cleaned"
            ),
            "keypoints": PresetResult(name="keypoints", text="## Задачи"),
            "expertizeme-managers": PresetResult(
                name="expertizeme-managers", error="boom"
            ),
        },
    )

    with pytest.raises(RuntimeError, match="preset DAG had failures"):
        main.process_item(service, _item("fid", "video.mp4"), "folderX", _two_preset_config())

    # The successful presets' artifacts were written before the error surfaced.
    assert "video.transcript-cleanup.md" in captured["names"]
    assert "video.keypoints.md" in captured["names"]
    assert "video.expertizeme-managers.md" not in captured["names"]


def test_process_item_reprocesses_missing_presets_from_existing_drive_txt(mocker, tmp_path):
    service = MagicMock()
    download_text = mocker.patch(
        "src.main.drive.download_text", return_value="existing transcript"
    )
    transcribe = mocker.patch("src.main.transcribe_file")
    mocker.patch("src.main.drive.upload")
    run_mock = mocker.patch(
        "src.main.preset_pipeline.run_presets",
        return_value={
            "expertizeme-managers": PresetResult(
                name="expertizeme-managers", text="notes"
            ),
        },
    )

    # The .txt plus transcript-cleanup and keypoints already exist on Drive; only
    # expertizeme-managers is missing. It must be regenerated by re-feeding the
    # existing transcript, without re-running STT, and its dependency
    # transcript-cleanup is reused from its artifact rather than re-run.
    item = _item(
        "fid",
        "video.mp4",
        has_txt=True,
        txt_id="t1",
        artifact_ids={"transcript-cleanup": "c1", "keypoints": "k1"},
    )
    config = _two_preset_config(webhook_url="https://hook.example/x")
    main.process_item(service, item, "folderX", config)

    transcribe.assert_not_called()
    # t1 = the existing transcript; c1 = the reused transcript-cleanup artifact;
    # k1 = the earlier keypoints, read back so the webhook payload is complete
    # (the k1 read is why this config carries a webhook.url).
    assert download_text.call_count == 3
    download_text.assert_any_call(service, "t1")
    download_text.assert_any_call(service, "c1")
    download_text.assert_any_call(service, "k1")
    run_mock.assert_called_once()
    assert run_mock.call_args.args[0] == "existing transcript"
    assert run_mock.call_args.kwargs["only"] == ["expertizeme-managers"]
    assert run_mock.call_args.kwargs["precomputed"] == {
        "transcript-cleanup": "existing transcript"
    }


def test_process_item_skips_preset_reprocess_without_drive_txt(mocker):
    service = MagicMock()
    download_text = mocker.patch("src.main.drive.download_text")
    run_mock = mocker.patch("src.main.preset_pipeline.run_presets")
    transcribe = mocker.patch("src.main.transcribe_file")

    # Folder-mode style: the transcript exists locally (has_txt) but there is no
    # Drive .txt sibling (txt_id is None), so the preset stage must not reprocess
    # (artifact_ids is not tracked for local files and would loop forever).
    item = _item("fid", "video.mp4", has_txt=True, txt_id=None, artifact_ids={})
    result = main.process_item(service, item, "folderX", _two_preset_config())

    assert result is None
    download_text.assert_not_called()
    run_mock.assert_not_called()
    transcribe.assert_not_called()


def test_preset_only_reprocess_without_transcript_does_not_run_stt(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"
    mp3_path = tmp_path / "video.mp3"
    download = mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    transcribe = mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")
    run_mock = mocker.patch("src.main.preset_pipeline.run_presets")

    result = main.process_item(
        service,
        _item("fid", "video.mp4", has_txt=False, txt_id=None),
        "folderX",
        _two_preset_config(drive_mp3_artifact=False),
        reprocess_presets=["keypoints"],
    )

    assert result is None
    download.assert_not_called()
    transcribe.assert_not_called()
    run_mock.assert_not_called()


def test_apply_local_output_state_tracks_local_txt_and_preset_artifacts(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "video.txt").write_text("local transcript", encoding="utf-8")
    (out / "video.transcript-cleanup.md").write_text("cleaned", encoding="utf-8")
    cfg = _two_preset_config(output_target="folder", output_dir=out)
    item = _item("fid", "video.mp4", has_txt=False, txt_id=None, artifact_ids={})

    main._apply_local_output_state([item], cfg)

    assert item["has_txt"] is True
    assert item["local_txt_path"] == out / "video.txt"
    assert item["local_artifact_paths"] == {
        "transcript-cleanup": out / "video.transcript-cleanup.md"
    }


def test_process_item_reprocesses_presets_from_local_folder_transcript(mocker, tmp_path):
    service = MagicMock()
    out = tmp_path / "out"
    out.mkdir()
    local_txt = out / "video.txt"
    local_txt.write_text("existing local transcript", encoding="utf-8")
    mocker.patch("src.main.transcribe_file")
    mocker.patch("src.main.drive.download_text")
    run_mock = mocker.patch(
        "src.main.preset_pipeline.run_presets",
        return_value={
            "transcript-cleanup": PresetResult(
                name="transcript-cleanup", text="cleaned"
            ),
            "keypoints": PresetResult(name="keypoints", text="notes"),
        },
    )

    item = _item("fid", "video.mp4", has_txt=True, txt_id=None, artifact_ids={})
    item["local_txt_path"] = local_txt
    cfg = _two_preset_config(output_target="folder", output_dir=out)
    result = main.process_item(
        service,
        item,
        "folderX",
        cfg,
        reprocess_presets=["keypoints"],
    )

    assert result is not None
    assert run_mock.call_args.args[0] == "existing local transcript"
    assert (out / "video.transcript-cleanup.md").read_text(encoding="utf-8") == "cleaned"
    assert (out / "video.keypoints.md").read_text(encoding="utf-8") == "notes"


def test_dry_run_preset_names_expands_missing_dependencies_for_reprocess():
    cfg = _two_preset_config()
    item = _item("fid", "video.mp4", has_txt=True, txt_id="txt-1", artifact_ids={})

    names = main._dry_run_preset_names(
        item,
        cfg,
        needs_txt=False,
        reprocess_txt=False,
        reprocess_presets=["keypoints"],
    )

    assert names == ["transcript-cleanup", "keypoints"]


def test_pending_items_includes_drive_txt_with_missing_preset():
    cfg = _two_preset_config()
    done = _item(
        "v1", "a.mp4", has_txt=True, txt_id="t1",
        artifact_ids={
            "transcript-cleanup": "c1", "keypoints": "k1", "expertizeme-managers": "e1",
        },
    )
    missing = _item(
        "v2", "b.mp4", has_txt=True, txt_id="t2",
        artifact_ids={"transcript-cleanup": "c2"},
    )
    folder_local = _item("v3", "c.mp4", has_txt=True, txt_id=None, artifact_ids={})

    pending = main._pending_items([done, missing, folder_local], cfg)

    assert [item["file"]["id"] for item in pending] == ["v2"]


# --- run loop stop flag (gdstt stop) ----------------------------------------

class _LoopStop(Exception):
    """Sentinel raised from a patched time.sleep to break the polling loop."""


def test_main_loop_idles_while_run_disabled_without_running(mocker):
    # `gdstt stop` keeps the loop alive but idle: run_once is never called while
    # run.enabled is false. The container stays up (no break/exit) so a Docker
    # `restart: unless-stopped` policy does not auto-resume processing.
    cfg = make_config(folders=["f1"], poll_interval=7)
    mocker.patch("src.main.load_config", return_value=cfg)
    mocker.patch("src.main.build_drive_service", return_value=MagicMock())
    mocker.patch("src.main.is_run_enabled", return_value=False)
    once = mocker.patch("src.main.run_once")
    # Break the otherwise-infinite idle loop after a couple of sleeps.
    sleep = mocker.patch("src.main.time.sleep", side_effect=[None, _LoopStop()])

    with pytest.raises(_LoopStop):
        main.main()

    once.assert_not_called()
    assert sleep.call_args_list == [mocker.call(7), mocker.call(7)]


def test_main_loop_runs_while_enabled_and_idles_when_disabled(mocker):
    # Enabled twice (two cycles), then disabled (idle). run_once is called only
    # while enabled; once disabled the loop idles instead of exiting.
    cfg = make_config(folders=["f1"], poll_interval=5)
    mocker.patch("src.main.load_config", return_value=cfg)
    mocker.patch("src.main.build_drive_service", return_value=MagicMock())
    mocker.patch("src.main.is_run_enabled", side_effect=[True, True, False])
    once = mocker.patch("src.main.run_once")
    sleep = mocker.patch("src.main.time.sleep", side_effect=[None, None, _LoopStop()])

    with pytest.raises(_LoopStop):
        main.main()

    assert once.call_count == 2
    assert sleep.call_count == 3


def test_main_does_not_enable_run_on_startup(mocker):
    # main() must not auto-enable run.enabled, so a sticky `gdstt stop` survives a
    # container restart instead of resuming on the next boot.
    import dataclasses

    cfg = dataclasses.replace(make_config(folders=["f1"]), run_enabled=False)
    mocker.patch("src.main.load_config", return_value=cfg)
    mocker.patch("src.main.build_drive_service", return_value=MagicMock())
    mocker.patch("src.main.is_run_enabled", return_value=False)
    mocker.patch("src.main.run_once")
    mocker.patch("src.main.time.sleep", side_effect=_LoopStop())
    set_enabled = mocker.patch(
        "src.config.set_run_enabled", side_effect=AssertionError("must not be called")
    )

    with pytest.raises(_LoopStop):
        main.main()

    set_enabled.assert_not_called()


def test_run_preset_stage_forces_only_selected(mocker):
    presets = (
        Preset(name="transcript-cleanup", instructions="c"),
        Preset(name="keypoints", instructions="k", depends_on=("transcript-cleanup",)),
        Preset(name="action-items", instructions="a", depends_on=("transcript-cleanup",)),
    )
    cfg = make_config(presets=presets)
    run_presets = mocker.patch(
        "src.main.preset_pipeline.run_presets",
        return_value={
            "action-items": PresetResult(name="action-items", text="out"),
        },
    )
    mocker.patch("src.main._save_and_upload_preset")
    # transcript-cleanup artifact already exists -> reused as a precomputed dependency.
    mocker.patch("src.main._call_with_transient_retries", return_value="cleanup text")

    main._run_preset_stage(
        MagicMock(),
        "file-1",
        "Alice and Bob.mp4",
        "Speaker 1: hi",
        "folderA",
        "folderA",
        Path("/tmp"),
        cfg,
        speaker_names=None,
        artifact_ids={"transcript-cleanup": "tc-id"},
        reprocess=False,
        only_presets=["action-items"],
        usage={},
        unproduced=set(),
    )

    kwargs = run_presets.call_args.kwargs
    assert kwargs["only"] == ["action-items"]
    assert kwargs["precomputed"] == {"transcript-cleanup": "cleanup text"}


def test_process_item_telemetry_carries_preset_artifacts(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=tmp_path / "video.mp3")
    mocker.patch("src.main.drive.upload")
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")
    mocker.patch(
        "src.main.preset_pipeline.run_presets",
        return_value={
            "transcript-cleanup": PresetResult(
                name="transcript-cleanup", text="Alice: hi"
            ),
            "keypoints": PresetResult(name="keypoints", text="## Задачи"),
            "expertizeme-managers": PresetResult(
                name="expertizeme-managers", text="notes"
            ),
        },
    )

    telemetry = main.process_item(
        service, _item("fid", "video.mp4"), "folderX", _two_preset_config()
    )

    assert telemetry is not None
    assert telemetry.transcript == "Speaker 1: hi"
    assert telemetry.artifacts == {
        "transcript-cleanup": "Alice: hi",
        "keypoints": "## Задачи",
        "expertizeme-managers": "notes",
    }


def test_process_item_telemetry_omits_empty_preset_artifacts(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.drive.upload")
    mocker.patch("src.main.extract_mp3", return_value=tmp_path / "video.mp3")
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")
    mocker.patch(
        "src.main.preset_pipeline.run_presets",
        return_value={
            "transcript-cleanup": PresetResult(
                name="transcript-cleanup", text="Alice: hi"
            ),
            "keypoints": PresetResult(name="keypoints", text="   "),
        },
    )

    telemetry = main.process_item(
        service, _item("fid", "video.mp4"), "folderX", _two_preset_config()
    )

    assert telemetry is not None
    assert telemetry.artifacts == {"transcript-cleanup": "Alice: hi"}


def test_process_item_telemetry_artifacts_empty_without_presets(mocker, tmp_path):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=tmp_path / "video.mp3")
    mocker.patch("src.main.drive.upload")
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")

    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        deepgram_audio_source="mp3_96k",
        drive_mp3_artifact=False,
    )
    telemetry = main.process_item(service, _item("fid", "video.mp4"), "folderX", cfg)

    assert telemetry is not None
    assert telemetry.artifacts == {}
    assert telemetry.transcript == "Speaker 1: hi"


def test_process_item_telemetry_carries_artifacts_on_preset_refeed(mocker, tmp_path):
    """Only ``expertizeme-managers`` is missing, so it alone is re-run — but the
    webhook fires once per file, so the presets that succeeded on an earlier cycle
    must be read back from their artifacts and reach the receiver too."""
    service = MagicMock()
    mocker.patch(
        "src.main.drive.download_text",
        side_effect=lambda svc, file_id: {
            "t1": "existing transcript",
            "c1": "Alice: hi",
            "k1": "earlier keypoints",
        }[file_id],
    )
    mocker.patch("src.main.transcribe_file")
    mocker.patch("src.main.drive.upload")
    mocker.patch(
        "src.main.preset_pipeline.run_presets",
        return_value={
            "transcript-cleanup": PresetResult(
                name="transcript-cleanup", text="Alice: hi"
            ),
            "expertizeme-managers": PresetResult(
                name="expertizeme-managers", text="notes"
            ),
        },
    )

    item = _item(
        "fid",
        "video.mp4",
        has_txt=True,
        txt_id="t1",
        artifact_ids={"transcript-cleanup": "c1", "keypoints": "k1"},
    )
    # The backfill exists only to feed the receiver, so it is gated on a configured
    # webhook — this test asserts the backfill, hence the URL.
    config = _two_preset_config(webhook_url="https://hook.example/x")
    telemetry = main.process_item(service, item, "folderX", config)

    assert telemetry is not None
    # The re-fed transcript, not a fresh STT pass.
    assert telemetry.transcript == "existing transcript"
    assert telemetry.artifacts == {
        "transcript-cleanup": "Alice: hi",
        "expertizeme-managers": "notes",
        "keypoints": "earlier keypoints",
    }


# --- completion webhook ------------------------------------------------------


_META_ARTIFACT = (
    "---\n"
    "subject: Консультация по визе O-1\n"
    "tags: [O-1, клиентская-консультация, invented-tag]\n"
    "referral: рекомендация\n"
    "referral_note: Посоветовала знакомая\n"
    "---\n"
)


def _webhook_config(**overrides):
    """A two-preset config whose DAG also produces a `meta` artifact."""
    presets = (
        Preset(name="transcript-cleanup", instructions="clean it"),
        Preset(
            name="keypoints",
            instructions="summarize",
            artifact_suffix=".keypoints.md",
            depends_on=("transcript-cleanup",),
        ),
        Preset(
            name="meta",
            instructions="subject, tags, and referral",
            artifact_suffix=".meta.md",
            depends_on=("transcript-cleanup",),
        ),
    )
    base = dict(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        deepgram_audio_source="mp3_96k",
        openai_api_key="sk-x",
        openai_keypoints=True,
        drive_mp3_artifact=False,
        presets=presets,
        folders=[
            EmployeeFolder("folderX", name="Олег Иванов", email="oleg@expertizeme.org")
        ],
        tags_allowed=("O-1", "клиентская-консультация"),
        referrals_allowed=("рекомендация",),
        webhook_url="https://example.com/hooks/gdstt",
        webhook_token="secret",
    )
    base.update(overrides)
    return make_config(**base)


def _mock_successful_run(mocker, tmp_path, *, meta_text=_META_ARTIFACT):
    mocker.patch("src.main.drive.download", return_value=tmp_path / "video.mp4")
    mocker.patch("src.main.extract_mp3", return_value=tmp_path / "video.mp3")
    mocker.patch("src.main.drive.upload")
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")
    mocker.patch(
        "src.main.preset_pipeline.run_presets",
        return_value={
            "transcript-cleanup": PresetResult(
                name="transcript-cleanup", text="Ольга: привет"
            ),
            "keypoints": PresetResult(name="keypoints", text="## Задачи"),
            "meta": PresetResult(name="meta", text=meta_text),
        },
    )
    return mocker.patch("src.main.webhook.notify_complete")


def test_webhook_fired_once_with_employee_and_artifacts(mocker, tmp_path):
    notify = _mock_successful_run(mocker, tmp_path)

    main.process_item(
        MagicMock(), _item("fid", "video.mp4"), "folderX", _webhook_config()
    )

    notify.assert_called_once()
    kwargs = notify.call_args.kwargs
    assert kwargs["url"] == "https://example.com/hooks/gdstt"
    assert kwargs["token"] == "secret"
    assert kwargs["payload"] == {
        "file": {"id": "fid", "name": "video.mp4", "folder_id": "folderX"},
        "employee": {"name": "Олег Иванов", "email": "oleg@expertizeme.org"},
        "transcript": "Speaker 1: hi",
        "artifacts": {
            "transcript-cleanup": "Ольга: привет",
            "keypoints": "## Задачи",
            # `meta` is parsed into structured fields; `invented-tag` is outside the
            # configured allow-list and must not reach the receiver.
            "meta": {
                "subject": "Консультация по визе O-1",
                "tags": ["O-1", "клиентская-консультация"],
                "referral": "рекомендация",
                "referral_note": "Посоветовала знакомая",
            },
        },
    }


def test_webhook_withheld_while_a_preset_produced_no_artifact(mocker, tmp_path, caplog):
    """A blank `ok` preset writes no artifact, so the file stays pending and is
    re-selected every cycle. Firing here would re-POST the transcript forever — the
    receiver gets no retry and has no dedupe key — so the webhook waits."""
    notify = _mock_successful_run(mocker, tmp_path, meta_text="   ")

    with caplog.at_level(logging.WARNING, logger="src.main"):
        main.process_item(
            MagicMock(), _item("fid", "video.mp4"), "folderX", _webhook_config()
        )

    notify.assert_not_called()
    assert "Completion webhook withheld" in caplog.text


def test_webhook_unknown_employee_sends_empty_strings(mocker, tmp_path):
    notify = _mock_successful_run(mocker, tmp_path)

    # The file's folder isn't in `folders` at all — the payload keeps the key.
    main.process_item(
        MagicMock(), _item("fid", "video.mp4"), "otherFolder", _webhook_config()
    )

    payload = notify.call_args.kwargs["payload"]
    assert payload["employee"] == {"name": "", "email": ""}
    assert payload["file"]["folder_id"] == "otherFolder"


def test_webhook_malformed_meta_degrades_to_empty_fields(mocker, tmp_path):
    notify = _mock_successful_run(mocker, tmp_path, meta_text="not frontmatter at all")

    main.process_item(
        MagicMock(), _item("fid", "video.mp4"), "folderX", _webhook_config()
    )

    artifacts = notify.call_args.kwargs["payload"]["artifacts"]
    assert artifacts["meta"] == {
        "subject": "",
        "tags": [],
        "referral": "",
        "referral_note": "",
    }


def test_webhook_meta_payload_carries_one_key_per_configured_entity():
    entities = meta_entity.parse_entities(
        [
            {"name": "subject", "prompt": "Тема.", "label": ""},
            {"name": "target_filing", "prompt": "Подача."},
        ]
    )
    config = make_config(meta_entities=entities)
    payload = main._webhook_payload(
        "f1",
        "запись.mp4",
        "folder1",
        config,
        "[00:00:01] Менеджер: привет",
        {"meta": "---\nsubject: Обсудили визу\ntarget_filing: O-1 осенью\n---\n"},
    )
    assert payload["artifacts"]["meta"] == {
        "subject": "Обсудили визу",
        "target_filing": "O-1 осенью",
    }


def test_webhook_not_fired_when_file_skipped(mocker, tmp_path):
    notify = mocker.patch("src.main.webhook.notify_complete")

    # Nothing to do: mp3 and txt exist and no preset is configured.
    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        webhook_url="https://example.com/hooks/gdstt",
    )
    telemetry = main.process_item(
        MagicMock(),
        _item("fid", "video.mp4", has_mp3=True, has_txt=True),
        "folderA",
        cfg,
    )

    assert telemetry is None
    notify.assert_not_called()


def test_webhook_receives_the_configured_proxy(mocker, tmp_path):
    """The proxy is honoured inside notify_complete; this pins the wiring. Without
    it, a proxied deployment silently stops delivering — notify_complete swallows the
    connection error and the file still processes."""
    notify = _mock_successful_run(mocker, tmp_path)

    main.process_item(
        MagicMock(),
        _item("fid", "video.mp4"),
        "folderX",
        _webhook_config(proxy_url="http://proxy:3128"),
    )

    assert notify.call_args.kwargs["proxy_url"] == "http://proxy:3128"


def test_webhook_not_fired_for_an_mp3_only_pass(mocker, tmp_path):
    """STT disabled and only the mp3 artifact wanted: there is no transcript and no
    preset output, so POSTing blanks would overwrite a good record on the receiver."""
    notify = mocker.patch("src.main.webhook.notify_complete")
    mocker.patch("src.main.drive.download", return_value=tmp_path / "video.mp4")
    mocker.patch("src.main.extract_mp3", return_value=tmp_path / "video.mp3")
    mocker.patch("src.main.drive.upload")

    cfg = _webhook_config(stt_provider="", presets=(), drive_mp3_artifact=True)
    telemetry = main.process_item(
        MagicMock(), _item("fid", "video.mp4"), "folderX", cfg
    )

    assert telemetry is not None
    assert telemetry.mp3_uploaded is True
    notify.assert_not_called()


def test_webhook_not_refired_when_only_a_late_mp3_is_added(mocker, tmp_path):
    """Enabling drive_mp3_artifact after transcripts already exist backfills the mp3
    only; the file's webhook already fired on the cycle that produced the transcript."""
    notify = mocker.patch("src.main.webhook.notify_complete")
    mocker.patch("src.main.drive.download", return_value=tmp_path / "video.mp4")
    mocker.patch("src.main.extract_mp3", return_value=tmp_path / "video.mp3")
    mocker.patch("src.main.drive.upload")

    cfg = _webhook_config(presets=(), drive_mp3_artifact=True)
    telemetry = main.process_item(
        MagicMock(),
        _item("fid", "video.mp4", has_txt=True, txt_id="t1", has_mp3=False),
        "folderX",
        cfg,
    )

    assert telemetry is not None
    notify.assert_not_called()


def test_webhook_not_fired_on_failure(mocker, tmp_path):
    notify = _mock_successful_run(mocker, tmp_path)
    mocker.patch("src.main.transcribe_file", side_effect=STTError("deepgram down"))

    with pytest.raises(STTError):
        main.process_item(
            MagicMock(), _item("fid", "video.mp4"), "folderX", _webhook_config()
        )

    notify.assert_not_called()


def test_webhook_exception_does_not_fail_the_file(mocker, tmp_path):
    notify = _mock_successful_run(mocker, tmp_path)
    notify.side_effect = RuntimeError("receiver exploded")

    # notify_complete swallows its own errors, but a bug there must not undo a file
    # that already transcribed and uploaded every artifact.
    telemetry = main.process_item(
        MagicMock(), _item("fid", "video.mp4"), "folderX", _webhook_config()
    )

    assert telemetry is not None
    assert telemetry.txt_uploaded is True


def test_process_summary_log_omits_artifact_text(mocker, tmp_path, caplog):
    service = MagicMock()
    mp4_path = tmp_path / "video.mp4"

    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.drive.upload")
    mocker.patch("src.main.extract_mp3", return_value=tmp_path / "video.mp3")
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: secret words")
    mocker.patch(
        "src.main.preset_pipeline.run_presets",
        return_value={
            "transcript-cleanup": PresetResult(
                name="transcript-cleanup", text="Alice: confidential"
            ),
        },
    )

    with caplog.at_level("INFO"):
        main.process_item(
            service, _item("fid", "video.mp4"), "folderX", _two_preset_config()
        )

    summary = [r.getMessage() for r in caplog.records if "Process summary" in r.msg]
    assert len(summary) == 1
    assert "confidential" not in summary[0]
    assert "secret words" not in summary[0]


# --- call-booking gate helpers -------------------------------------------------

GATE_FOLDER_ID = "f1"


@pytest.fixture
def gate_config(tmp_path):
    """A config with the receiver enabled and the gate armed."""
    config_file = tmp_path / "config.yml"
    config_file.write_text("", encoding="utf-8")
    return Config(
        folders=(
            EmployeeFolder(
                folder_id=GATE_FOLDER_ID, name="Kate", email="kate@example.com"
            ),
        ),
        poll_interval=600,
        bitrate="96k",
        data_dir=tmp_path,
        proxy_url="",
        stt_provider="deepgram",
        openai_api_key="sk",
        deepgram_api_key="dg",
        stt_language="ru",
        call_booking_enabled=True,
        call_booking_disable_recognition=True,
        call_booking_threshold_minutes=15,
        meta_entities=meta_entity.default_entities(),
        config_file=config_file,
    )


def gate_item(
    file_id="v1",
    *,
    booking_match="",
    planfix_comment_task_id="",
    telegram_sent_chat_id="",
):
    """One `list_folder_state` item for an mp4 that still needs a transcript."""
    return {
        "file": {"id": file_id, "name": f"{file_id}.mp4", "mimeType": "video/mp4"},
        "has_mp3": True,
        "has_txt": False,
        "mp3_id": None,
        "mp3_name": None,
        "txt_id": None,
        "artifact_ids": {},
        "booking_match": booking_match,
        "planfix_comment_task_id": planfix_comment_task_id,
        "telegram_sent_chat_id": telegram_sent_chat_id,
    }


def patch_folder_items(monkeypatch, items):
    monkeypatch.setattr(main.drive, "list_folder_state", lambda service, fid: items)


def patch_decision(monkeypatch, decision):
    monkeypatch.setattr(
        main.booking_gate, "resolve", lambda file_info, folder_id, config: decision
    )


UNMATCHED_DECISION = BookingDecision(state="unmatched", reason="no-booking")
MATCHED_DECISION = BookingDecision(state="matched", task_id="851030")


def test_run_once_skips_and_marks_an_unmatched_recording(monkeypatch, gate_config):
    monkeypatch.setattr(main.booking_server, "is_running", lambda: True)
    patch_decision(monkeypatch, UNMATCHED_DECISION)
    patch_folder_items(monkeypatch, [gate_item("v1")])
    marked = []
    monkeypatch.setattr(
        main.booking_gate, "mark_unmatched", lambda svc, fid: marked.append(fid)
    )
    process_item = MagicMock()
    monkeypatch.setattr(main, "process_item", process_item)

    main.run_once(MagicMock(), gate_config)

    process_item.assert_not_called()
    assert marked == ["v1"]


def test_run_once_survives_a_drive_failure_while_marking(monkeypatch, gate_config, caplog):
    """A transient Drive error from mark_unmatched must not kill the polling loop.

    Carried from the Task 6 review: mark_unmatched/clear_mark do not catch Drive
    API exceptions themselves, so run_once must contain the failure.
    """
    monkeypatch.setattr(main.booking_server, "is_running", lambda: True)
    patch_decision(monkeypatch, UNMATCHED_DECISION)
    patch_folder_items(monkeypatch, [gate_item("v1")])

    def raise_http_error(svc, fid):
        raise HttpError(MagicMock(status=503), b"unavailable")

    monkeypatch.setattr(main.booking_gate, "mark_unmatched", raise_http_error)
    process_item = MagicMock()
    monkeypatch.setattr(main, "process_item", process_item)

    with caplog.at_level(logging.INFO):
        main.run_once(MagicMock(), gate_config)

    process_item.assert_not_called()
    assert "Failed to mark" in caplog.text
    assert "skipped_unmatched=1" in caplog.text


def test_run_once_does_not_mark_when_the_receiver_is_down(
    monkeypatch, gate_config, caplog
):
    monkeypatch.setattr(main.booking_server, "is_running", lambda: False)
    patch_decision(monkeypatch, UNMATCHED_DECISION)
    patch_folder_items(monkeypatch, [gate_item("v1")])
    marked = []
    monkeypatch.setattr(
        main.booking_gate, "mark_unmatched", lambda svc, fid: marked.append(fid)
    )
    process_item = MagicMock()
    monkeypatch.setattr(main, "process_item", process_item)

    with caplog.at_level(logging.WARNING):
        main.run_once(MagicMock(), gate_config)

    process_item.assert_not_called()
    assert marked == []
    assert "not listening" in caplog.text


def test_run_once_counts_skipped_unmatched_separately(monkeypatch, gate_config, caplog):
    monkeypatch.setattr(main.booking_server, "is_running", lambda: True)
    patch_decision(monkeypatch, UNMATCHED_DECISION)
    patch_folder_items(monkeypatch, [gate_item("v1")])
    monkeypatch.setattr(main.booking_gate, "mark_unmatched", lambda svc, fid: None)

    with caplog.at_level(logging.INFO):
        main.run_once(MagicMock(), gate_config)

    assert "skipped_unmatched=1" in caplog.text
    assert "processed=0" in caplog.text


def test_run_once_never_revisits_an_already_marked_recording(monkeypatch, gate_config):
    monkeypatch.setattr(main.booking_server, "is_running", lambda: True)
    resolve = MagicMock()
    monkeypatch.setattr(main.booking_gate, "resolve", resolve)
    patch_folder_items(monkeypatch, [gate_item("v1", booking_match="none")])
    process_item = MagicMock()
    monkeypatch.setattr(main, "process_item", process_item)

    main.run_once(MagicMock(), gate_config)

    process_item.assert_not_called()
    resolve.assert_not_called()


def test_run_once_processes_a_matched_recording(monkeypatch, gate_config):
    monkeypatch.setattr(main.booking_server, "is_running", lambda: True)
    patch_decision(monkeypatch, MATCHED_DECISION)
    patch_folder_items(monkeypatch, [gate_item("v1")])
    process_item = MagicMock(return_value=None)
    monkeypatch.setattr(main, "process_item", process_item)

    main.run_once(MagicMock(), gate_config)

    process_item.assert_called_once()
    assert process_item.call_args.kwargs["booking_decision"] == MATCHED_DECISION


def test_run_once_processes_when_disable_recognition_is_off(monkeypatch, gate_config):
    permissive = replace(gate_config, call_booking_disable_recognition=False)
    monkeypatch.setattr(main.booking_server, "is_running", lambda: True)
    patch_decision(monkeypatch, UNMATCHED_DECISION)
    patch_folder_items(monkeypatch, [gate_item("v1")])
    marked = []
    monkeypatch.setattr(
        main.booking_gate, "mark_unmatched", lambda svc, fid: marked.append(fid)
    )
    process_item = MagicMock(return_value=None)
    monkeypatch.setattr(main, "process_item", process_item)

    main.run_once(MagicMock(), permissive)

    process_item.assert_called_once()
    assert marked == []


def test_process_target_ignores_the_mark_and_the_gate(monkeypatch, gate_config):
    """Manual processing is the supported way to undo a mark."""
    monkeypatch.setattr(main.booking_server, "is_running", lambda: True)
    patch_folder_items(monkeypatch, [gate_item("v1", booking_match="none")])
    monkeypatch.setattr(
        main.drive,
        "get_file_metadata",
        lambda service, fid: {
            "id": "v1", "name": "v1.mp4", "mimeType": "video/mp4",
            "parents": [GATE_FOLDER_ID],
        },
    )
    marked = []
    monkeypatch.setattr(
        main.booking_gate, "mark_unmatched", lambda svc, fid: marked.append(fid)
    )
    process_item = MagicMock(return_value=None)
    monkeypatch.setattr(main, "process_item", process_item)

    main.process_target(MagicMock(), "v1", gate_config, is_folder=False)

    process_item.assert_called_once()
    assert marked == []
    assert process_item.call_args.kwargs.get("booking_decision") is None


def test_process_target_folder_branch_ignores_the_mark_and_the_gate(
    monkeypatch, gate_config
):
    """The folder branch shares ``_pending_items`` with ``run_once`` -- the one place
    a regression could leak the gate into manual processing. `process_target` must
    still process a marked item without ever consulting the gate.
    """
    monkeypatch.setattr(main.booking_server, "is_running", lambda: True)
    patch_folder_items(monkeypatch, [gate_item("v1", booking_match="none")])
    resolve = MagicMock()
    monkeypatch.setattr(main.booking_gate, "resolve", resolve)
    marked = []
    monkeypatch.setattr(
        main.booking_gate, "mark_unmatched", lambda svc, fid: marked.append(fid)
    )
    process_item = MagicMock(return_value=None)
    monkeypatch.setattr(main, "process_item", process_item)

    main.process_target(MagicMock(), GATE_FOLDER_ID, gate_config, is_folder=True)

    process_item.assert_called_once()
    resolve.assert_not_called()
    assert marked == []


def test_run_once_matches_a_real_booking_through_the_real_gate(monkeypatch, gate_config):
    """Integration coverage for the seam every other gate test mocks around: Drive
    file name -> meeting_time.parse_meeting_start -> journal load -> match ->
    decision. Only the Drive listing and process_item are mocked; a real Config and
    a real ``booking_gate.resolve`` run against a journal seeded with
    ``call_booking.append``.
    """
    # "YYYY/MM/DD HH:MM GMT+04:00" is one of the formats parse_meeting_start accepts
    # (see src/meeting_time.py). The date is yesterday's rather than a fixed one: the
    # journal drops bookings older than call_booking.RETENTION_DAYS, so a pinned date
    # makes this test start failing on its own once that many days have passed.
    video_start_utc = (datetime.now(timezone.utc) - timedelta(days=1)).replace(
        hour=5, minute=0, second=0, microsecond=0
    )
    file_name = (
        "Call with Kate - "
        + (video_start_utc + timedelta(hours=4)).strftime("%Y/%m/%d %H:%M")
        + " GMT+04:00 – Recording.mp4"
    )

    append_booking(
        gate_config.call_bookings_file,
        CallBooking(
            task_id="851030",
            manager_email="kate@example.com",
            start_time=video_start_utc + timedelta(minutes=5),
        ),
    )
    patch_folder_items(
        monkeypatch,
        [
            {
                "file": {"id": "v1", "name": file_name, "mimeType": "video/mp4"},
                "has_mp3": True,
                "has_txt": False,
                "mp3_id": None,
                "mp3_name": None,
                "txt_id": None,
                "artifact_ids": {},
                "booking_match": "",
                "planfix_comment_task_id": "",
            }
        ],
    )
    process_item = MagicMock(return_value=None)
    monkeypatch.setattr(main, "process_item", process_item)

    main.run_once(MagicMock(), gate_config)

    process_item.assert_called_once()
    decision = process_item.call_args.kwargs["booking_decision"]
    assert decision.is_matched is True
    assert decision.task_id == "851030"


# --- Planfix comment ------------------------------------------------------------


@pytest.fixture
def planfix_config(gate_config):
    return replace(
        gate_config,
        planfix_create_comment_url="https://crm.example.com/planfix_create_comment",
        planfix_token="planfix-token",
        planfix_presets=("keypoints",),
    )


def test_resolve_speaker_names_hands_the_folder_owner_over_as_the_manager(monkeypatch):
    """The one identity we know for certain is the folder's owner."""
    seen = {}

    class FakePipeline:
        last_usage = {"input_tokens": 11, "output_tokens": 3}

        def __init__(self, **kwargs):
            seen["model"] = kwargs.get("model")

        def run(self, instructions, input_text):
            seen["input"] = input_text
            return '{"1": "Mels", "2": "Angelica Munkueva"}', {}

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(main, "OpenAIPipeline", FakePipeline)
    config = make_config(
        folders=[EmployeeFolder("f1", name="Анжелика Мункуева", email="a@e.org")],
        openai_api_key="key",
    )
    transcript = (
        "[00:00:01] Speaker 1: Здравствуйте, я по заявке.\n"
        "[00:00:04] Speaker 2: Добрый день, я из ExpertizeMe.\n"
    )
    usage: dict[str, dict[str, int]] = {}

    names = main._resolve_speaker_names(
        transcript,
        "Angelica Munkueva(ExpertizeMe) и Mels - 2026/08/13 14:29 CEST - Recording",
        "f1",
        config,
        usage=usage,
    )

    assert names == ["Mels", "Angelica Munkueva"]
    assert "Анжелика Мункуева" in seen["input"]
    assert seen["closed"] is True
    assert usage["openai_speaker_roles"] == {"input_tokens": 11, "output_tokens": 3}


def test_resolve_speaker_names_is_skipped_without_an_openai_key(monkeypatch):
    def explode(**kwargs):
        raise AssertionError("must not build a pipeline without a key")

    monkeypatch.setattr(main, "OpenAIPipeline", explode)
    config = make_config(folders=[EmployeeFolder("f1", name="Анжелика")], openai_api_key="")

    assert (
        main._resolve_speaker_names(
            "[00:00:01] Speaker 1: раз\n[00:00:04] Speaker 2: два",
            "Alice and Bob - 2026/08/13 14:29 CEST - Recording",
            "f1",
            config,
        )
        is None
    )


def test_resolve_speaker_names_returns_none_when_the_name_has_one_participant(monkeypatch):
    def explode(**kwargs):
        raise AssertionError("nothing to disambiguate, must not spend a call")

    monkeypatch.setattr(main, "OpenAIPipeline", explode)
    config = make_config(folders=[EmployeeFolder("f1", name="Анжелика")], openai_api_key="key")

    assert (
        main._resolve_speaker_names(
            "[00:00:01] Speaker 1: раз\n[00:00:04] Speaker 2: два",
            "Планёрка - 2026/08/13 14:29 CEST - Recording",
            "f1",
            config,
        )
        is None
    )


def test_planfix_header_labels_come_from_the_entities():
    entities = meta_entity.parse_entities(
        [
            {"name": "subject", "prompt": "Тема.", "label": ""},
            {
                "name": "deadlines",
                "prompt": "Сроки.",
                "multiple": True,
                "label": "Дедлайны",
            },
            {"name": "target_filing", "prompt": "Подача."},
        ]
    )
    document = {
        "subject": "Обсудили визу",
        "deadlines": ["виза до октября", "оффер к сентябрю"],
        "target_filing": "O-1 осенью",
        "duration": "00:31:02",
    }
    lines = main._planfix_meta_lines(
        document,
        ("subject", "deadlines", "target_filing", "duration"),
        entities,
    )
    assert lines[0] == "**Обсудили визу**"
    assert "**Дедлайны:** виза до октября, оффер к сентябрю" in lines
    # No label declared, so the name is the label.
    assert "**target_filing:** O-1 осенью" in lines
    # Code-known fields keep their built-in labels.
    assert "**Длительность:** 00:31:02" in lines


def test_planfix_header_skips_an_entity_with_no_value():
    entities = meta_entity.parse_entities(
        [
            {"name": "subject", "prompt": "Тема.", "label": ""},
            {"name": "case_deadline", "prompt": "Срок.", "label": "Срок сбора кейса"},
        ]
    )
    lines = main._planfix_meta_lines(
        {"subject": "Обсудили визу", "case_deadline": ""},
        ("subject", "case_deadline"),
        entities,
    )
    assert lines == ["**Обсудили визу**"]


def test_planfix_header_uses_the_first_empty_label_field_as_the_heading():
    entities = meta_entity.parse_entities(
        [
            {"name": "alt", "prompt": "Другое.", "label": ""},
            {"name": "subject", "prompt": "Тема.", "label": ""},
        ]
    )
    lines = main._planfix_meta_lines(
        {"subject": "Обсудили визу", "alt": "Второй заголовок"},
        ("subject", "alt"),
        entities,
    )
    assert lines[0] == "**Обсудили визу**"
    assert "**Второй заголовок**" in lines[1:]


def test_planfix_description_concatenates_presets_in_order():
    description = main._planfix_description(
        {"keypoints": "Задачи: раз", "action-items": "Сделать два", "meta": "topic: x"},
        ("keypoints", "action-items"),
        None,
        (),
    )

    assert description == "<p>Задачи: раз</p><p><br></p><p>Сделать два</p>"


def test_planfix_description_skips_presets_without_an_artifact():
    description = main._planfix_description(
        {"keypoints": "Задачи: раз"}, ("keypoints", "action-items"), None, ()
    )

    assert description == "<p>Задачи: раз</p>"


def test_planfix_description_omits_the_preset_name():
    """The preset name is pipeline vocabulary, not something a manager should read."""
    description = main._planfix_description(
        {"keypoints": "## Тезисы\n\n- раз"}, ("keypoints",), None, ()
    )

    assert "keypoints" not in description


def test_planfix_description_marks_the_keypoints_sections():
    description = main._planfix_description(
        {"keypoints": "## Задачи\n\n### Mels\n\n- раз\n\n## Тезисы\n\n- два\n\n"
         "## Открытые вопросы\n\n- три"},
        ("keypoints",),
        None,
        (),
    )

    assert "<p><b>☑️ Задачи</b></p>" in description
    assert "<p><b>📝 Тезисы</b></p>" in description
    assert "<p><b>❓ Открытые вопросы</b></p>" in description
    # An assignee sub-heading is a person's name, not a section, and keeps no marker.
    assert "<p><b>Mels</b></p>" in description


def test_planfix_description_puts_a_blank_line_around_a_marked_heading():
    """The heading needs air on both sides, but only one blank line where two meet."""
    description = main._planfix_description(
        {"keypoints": "## Задачи\n\n### Mels\n\n- раз\n\n## Тезисы\n\n- два"},
        ("keypoints",),
        {"subject": "Созвон"},
        ("subject",),
        meta_entity.default_entities(),
    )

    assert description == (
        "<p><b>Созвон</b></p><p><br></p>"
        "<p><b>☑️ Задачи</b></p><p><br></p>"
        "<p><b>Mels</b></p><ul><li>раз</li></ul><p><br></p>"
        "<p><b>📝 Тезисы</b></p><p><br></p><ul><li>два</li></ul>"
    )
    assert "<p><br></p><p><br></p>" not in description


def test_planfix_description_does_not_open_on_a_blank_line():
    """The first section's leading gap has nothing to separate it from."""
    description = main._planfix_description(
        {"keypoints": "## Задачи\n\n- раз"}, ("keypoints",), None, ()
    )

    assert description.startswith("<p><b>☑️ Задачи</b></p>")


def test_planfix_description_leaves_an_unknown_heading_alone():
    description = main._planfix_description(
        {"keypoints": "## Прочее\n\n- раз"}, ("keypoints",), None, ()
    )

    assert "<p><b>Прочее</b></p>" in description


def test_planfix_description_renders_markdown_as_html():
    """Planfix shows Markdown as literal characters, so the body must arrive as HTML."""
    description = main._planfix_description(
        {"keypoints": "## Задачи\n\n- [ ] позвонить\n- написать"}, ("keypoints",), None, ()
    )

    assert description == (
        "<p><b>☑️ Задачи</b></p><p><br></p><ul><li>позвонить</li><li>написать</li></ul>"
    )
    assert "\n" not in description


def test_planfix_description_is_blank_when_nothing_matched():
    assert main._planfix_description({"meta": "topic: x"}, ("keypoints",), None, ()) == ""


def test_planfix_description_puts_the_subject_and_tags_on_top():
    html = main._planfix_description(
        {"keypoints": "## Задачи\n- [ ] x"},
        ("keypoints",),
        {"subject": "Обсудили кейс", "tags": ["O-1"], "referral": "рекомендация"},
        ("subject", "tags", "referral"),
        meta_entity.default_entities(),
    )
    assert html.index("Обсудили кейс") < html.index("Задачи")
    assert "O-1" in html and "рекомендация" in html
    assert "\n" not in html


def test_planfix_description_skips_empty_and_unknown_fields():
    html = main._planfix_description(
        {"keypoints": "## Задачи"},
        ("keypoints",),
        {"subject": "s", "referral": "", "tags": []},
        ("subject", "referral", "tags", "not_a_field"),
        meta_entity.default_entities(),
    )
    # The brief's original assertion checked "Реферал" not in html, but no label
    # `_planfix_labels()` returns spells "Реферал" (referral's label is "Откуда
    # узнал", supplied by the entity, not the deleted `_PLANFIX_META_LABELS`
    # constant -- see `_PLANFIX_CODE_LABELS` for what's left of it, the code-known
    # fields only), so that half could never fail and proved nothing. Asserting on
    # the label the code actually emits for the empty `referral` field makes this
    # cover the skip.
    assert "Откуда узнал" not in html and "Теги" not in html


def test_planfix_description_normalises_a_multiline_meta_value():
    """referral_note is free LLM text and can carry embedded newlines. Left
    unnormalised, markdown_to_html would split the header on them -- fracturing it
    into extra label-less paragraphs, or worse, a stray bullet/heading if the
    continuation happened to start with "- " or "#". The header must stay one
    logical Markdown line before conversion.

    A ``keypoints`` section is included alongside the header under test: a
    header-only call now (correctly) renders empty -- see
    ``test_planfix_description_is_blank_when_only_the_header_would_render`` -- so
    this needs a section present to exercise the header-formatting behaviour."""
    html = main._planfix_description(
        {"keypoints": "х"}, ("keypoints",),
        {"referral_note": "Позвонила\nв понедельник,\n- уточнила детали"},
        ("referral_note",),
        meta_entity.default_entities(),
    )
    assert "<p><b>Подробности:</b> Позвонила в понедельник, - уточнила детали</p>" in html
    assert "\n" not in html


def test_planfix_description_renders_the_video_url_as_a_link():
    """source_name is excluded from the default field list *because* it becomes this
    anchor's text instead of a line of its own (per the design spec); a fixed
    "Запись" label would defeat that purpose, so the anchor text itself must be
    asserted, not just the href. A ``keypoints`` section rides along so the header
    (which alone now renders empty) is actually exercised."""
    html = main._planfix_description(
        {"keypoints": "х"}, ("keypoints",),
        {"video_url": "https://drive.google.com/file/d/X/view", "source_name": "Созвон.mp4"},
        ("video_url",),
    )
    assert '<a href="https://drive.google.com/file/d/X/view">Созвон.mp4</a>' in html


def test_planfix_description_video_url_falls_back_to_the_fixed_label():
    html = main._planfix_description(
        {"keypoints": "х"}, ("keypoints",),
        {"video_url": "https://drive.google.com/file/d/X/view"},
        ("video_url",),
    )
    assert '<a href="https://drive.google.com/file/d/X/view">Запись</a>' in html


def test_planfix_description_without_a_meta_document_is_unchanged():
    html = main._planfix_description({"keypoints": "## Задачи"}, ("keypoints",), None, ())
    assert "Задачи" in html


def test_planfix_description_is_blank_when_only_the_header_would_render():
    """A backfill cycle can find every configured preset already artifacted (so
    ``artifacts`` carries none of them this cycle) while ``meta_document`` still
    carries a header. Without a check that at least one preset section rendered,
    ``_planfix_description`` would return a header-only body -- which
    ``_send_planfix_comment`` would then post and permanently mark as sent, so the
    real preset comment could never go out. Nothing to comment must mean an empty
    description."""
    html = main._planfix_description(
        {},
        ("keypoints",),
        {
            "duration": "00:31:42",
            "video_url": "https://drive.google.com/file/d/X/view",
            "source_name": "rec.mp4",
        },
        ("duration", "video_url"),
    )
    assert html == ""


def test_comment_is_sent_for_a_matched_recording(monkeypatch, planfix_config):
    send = MagicMock(return_value=True)
    monkeypatch.setattr(main.planfix, "send_comment", send)
    marked = MagicMock()
    monkeypatch.setattr(main.drive, "set_file_app_properties", marked)

    main._send_planfix_comment(
        MagicMock(), gate_item("v1"), "v1", planfix_config,
        {"keypoints": "Задачи: раз"}, MATCHED_DECISION,
    )

    send.assert_called_once()
    assert send.call_args.kwargs["task_id"] == "851030"
    assert send.call_args.kwargs["description"] == "<p>Задачи: раз</p>"
    marked.assert_called_once()
    assert marked.call_args[0][2] == {"planfix_comment_task_id": "851030"}


def test_comment_includes_the_meta_header_when_a_document_is_passed(
    monkeypatch, planfix_config
):
    """Covers the wiring `process_item` relies on: `_send_planfix_comment` must
    forward `meta_document` and `config.planfix_meta_fields` into
    `_planfix_description`, and the header must land before the preset sections.
    Without this test, deleting the `meta_document=meta_document` argument at the
    `process_item` call site would leave the whole suite green."""
    send = MagicMock(return_value=True)
    monkeypatch.setattr(main.planfix, "send_comment", send)
    monkeypatch.setattr(main.drive, "set_file_app_properties", MagicMock())

    main._send_planfix_comment(
        MagicMock(), gate_item("v1"), "v1", planfix_config,
        {"keypoints": "Задачи: раз"}, MATCHED_DECISION,
        meta_document={"subject": "Обсудили визу O-1"},
    )

    send.assert_called_once()
    description = send.call_args.kwargs["description"]
    assert "Обсудили визу O-1" in description
    assert description.index("Обсудили визу O-1") < description.index("Задачи: раз")


def test_comment_is_not_sent_twice(monkeypatch, planfix_config):
    send = MagicMock(return_value=True)
    monkeypatch.setattr(main.planfix, "send_comment", send)

    main._send_planfix_comment(
        MagicMock(),
        gate_item("v1", planfix_comment_task_id="851030"),
        "v1", planfix_config, {"keypoints": "Задачи: раз"}, MATCHED_DECISION,
    )

    send.assert_not_called()


def test_comment_is_not_sent_for_an_unmatched_recording(monkeypatch, planfix_config):
    send = MagicMock(return_value=True)
    monkeypatch.setattr(main.planfix, "send_comment", send)

    main._send_planfix_comment(
        MagicMock(), gate_item("v1"), "v1", planfix_config,
        {"keypoints": "Задачи: раз"}, UNMATCHED_DECISION,
    )

    send.assert_not_called()


def test_comment_is_not_sent_when_the_url_is_blank(monkeypatch, planfix_config):
    send = MagicMock(return_value=True)
    monkeypatch.setattr(main.planfix, "send_comment", send)
    unconfigured = replace(planfix_config, planfix_create_comment_url="")

    main._send_planfix_comment(
        MagicMock(), gate_item("v1"), "v1", unconfigured,
        {"keypoints": "Задачи: раз"}, MATCHED_DECISION,
    )

    send.assert_not_called()


def test_failed_comment_notifies_telegram_and_leaves_no_marker(
    monkeypatch, planfix_config
):
    send = MagicMock(return_value=False)
    monkeypatch.setattr(main.planfix, "send_comment", send)
    marked = MagicMock()
    monkeypatch.setattr(main.drive, "set_file_app_properties", marked)
    notified = MagicMock()
    monkeypatch.setattr(main.notify, "notify_error", notified)

    main._send_planfix_comment(
        MagicMock(), gate_item("v1"), "v1", planfix_config,
        {"keypoints": "Задачи: раз"}, MATCHED_DECISION,
    )

    send.assert_called_once()
    # No marker means `gdstt reprocess` can resend it.
    marked.assert_not_called()
    notified.assert_called_once()


def test_send_planfix_comment_posts_nothing_for_a_header_only_document(
    monkeypatch, planfix_config
):
    """A preset-backfill cycle can find every configured preset already artifacted
    (none of them appear in this cycle's ``artifacts``) while ``meta_document`` still
    carries a header (duration, video link). Regression for the bug where
    `_planfix_description` returned a non-empty, header-only body in that case:
    `_send_planfix_comment`'s `if not description` guard exists precisely so a
    contentless comment is neither posted nor marked sent -- a marker written here
    would permanently block the real preset comment from ever going out."""
    send = MagicMock(return_value=True)
    monkeypatch.setattr(main.planfix, "send_comment", send)
    marked = MagicMock()
    monkeypatch.setattr(main.drive, "set_file_app_properties", marked)

    main._send_planfix_comment(
        MagicMock(), gate_item("v1"), "v1", planfix_config,
        {},  # no preset produced this cycle -- already-artifacted, so not re-run
        MATCHED_DECISION,
        meta_document={
            "duration": "00:31:42",
            "video_url": "https://drive.google.com/file/d/X/view",
            "source_name": "rec.mp4",
        },
    )

    send.assert_not_called()
    marked.assert_not_called()


def test_process_item_sends_the_comment_on_the_success_path(mocker, tmp_path):
    """Copy of ``test_webhook_fired_once_with_employee_and_artifacts``: same arrange,
    with `_send_planfix_comment` mocked and `booking_decision` passed through.

    Also asserts on the `meta_document` kwarg `process_item` hands to
    `_send_planfix_comment`: `_send_planfix_comment.assert_called_once()` alone would
    stay green even if `process_item` dropped the `meta_document=meta_document`
    argument at its call site (the header would then silently vanish from every
    production comment), so this pins down the actual production wiring, not just
    that the function got called."""
    _mock_successful_run(mocker, tmp_path)
    sent = MagicMock()
    mocker.patch.object(main, "_send_planfix_comment", sent)

    main.process_item(
        MagicMock(),
        _item("fid", "video.mp4"),
        "folderX",
        _webhook_config(),
        booking_decision=MATCHED_DECISION,
    )

    sent.assert_called_once()
    meta_document = sent.call_args.kwargs["meta_document"]
    assert meta_document is not None
    assert meta_document["subject"] == "Консультация по визе O-1"


def test_process_item_withholds_the_comment_when_a_preset_produced_nothing(
    mocker, tmp_path
):
    """Copy of ``test_webhook_withheld_while_a_preset_produced_no_artifact``: same
    arrange (a blank ``meta`` preset), with `_send_planfix_comment` mocked and
    `booking_decision` passed through."""
    _mock_successful_run(mocker, tmp_path, meta_text="   ")
    sent = MagicMock()
    mocker.patch.object(main, "_send_planfix_comment", sent)

    main.process_item(
        MagicMock(),
        _item("fid", "video.mp4"),
        "folderX",
        _webhook_config(),
        booking_decision=MATCHED_DECISION,
    )

    sent.assert_not_called()


_CONFIGURED_ENTITY_META_ARTIFACT = (
    "---\n"
    "subject: Консультация по визе O-1\n"
    "tags: [O-1, клиентская-консультация]\n"
    "referral: рекомендация\n"
    "referral_note: Посоветовала знакомая\n"
    "target_filing: O-1 осенью\n"
    "---\n"
)


def test_process_item_carries_a_configured_entity_into_meta_yml_webhook_and_planfix(
    mocker, tmp_path
):
    """The composition the spec asked a test for: a config-declared entity that is
    none of the built-in four (`target_filing`) must reach all three outputs of one
    `process_item` call -- the written `.meta.yml`, the webhook payload's
    `artifacts.meta`, and the Planfix comment -- not just whichever one its own
    unit test happens to cover. Each of those three is unit-tested alone elsewhere;
    nothing before this exercised the composition.
    """
    entities = meta_entity.parse_entities(
        [
            {"name": "subject", "prompt": "Тема.", "label": ""},
            {
                "name": "tags",
                "prompt": "Теги.",
                "type": "enum",
                "multiple": True,
                "allowed": ["O-1", "клиентская-консультация"],
            },
            {
                "name": "referral",
                "prompt": "Откуда.",
                "type": "enum",
                "allowed": ["рекомендация"],
            },
            {"name": "referral_note", "prompt": "Детали.", "requires": "referral"},
            # The non-default entity: it is in none of the built-in four
            # (subject/tags/referral/referral_note).
            {"name": "target_filing", "prompt": "Куда подача."},
        ]
    )
    out_dir = tmp_path / "out"
    config = _webhook_config(
        meta_entities=entities,
        planfix_meta_fields=(
            "subject", "tags", "referral", "referral_note", "target_filing",
        ),
        output_target="folder",
        output_dir=out_dir,
    )
    # `_webhook_config`/`make_config` do not expose the Planfix settings, so wire
    # them on afterwards -- mirrors how `planfix_config` extends `gate_config`.
    config = replace(
        config,
        planfix_create_comment_url="https://crm.example.com/planfix_create_comment",
        planfix_token="planfix-token",
    )

    notify = _mock_successful_run(
        mocker, tmp_path, meta_text=_CONFIGURED_ENTITY_META_ARTIFACT
    )
    send_comment = mocker.patch(
        "src.main.planfix.send_comment", return_value=True
    )
    mocker.patch("src.main.drive.set_file_app_properties")

    main.process_item(
        MagicMock(),
        _item("fid", "video.mp4"),
        "folderX",
        config,
        booking_decision=MATCHED_DECISION,
    )

    # 1. the written `.meta.yml`
    meta_yml = (out_dir / "video.meta.yml").read_text(encoding="utf-8")
    assert "target_filing: O-1 осенью" in meta_yml

    # 2. the webhook payload's `artifacts.meta`
    notify.assert_called_once()
    payload_meta = notify.call_args.kwargs["payload"]["artifacts"]["meta"]
    assert payload_meta["target_filing"] == "O-1 осенью"

    # 3. the Planfix comment
    send_comment.assert_called_once()
    description = send_comment.call_args.kwargs["description"]
    assert "target_filing" in description
    assert "O-1 осенью" in description


# --- start the receiver from the polling loop ----------------------------------


class _StopLoop(Exception):
    """Raised from a patched time.sleep to break main()'s infinite loop."""


def stop_after_one_cycle(monkeypatch, config):
    """Patch main() down to one cycle. Returns the run_once mock."""
    monkeypatch.setattr(main, "load_config", lambda **kwargs: config)
    monkeypatch.setattr(main, "build_drive_service", lambda **kwargs: MagicMock())
    monkeypatch.setattr(main, "is_run_enabled", lambda **kwargs: True)
    run_once = MagicMock()
    monkeypatch.setattr(main, "run_once", run_once)

    def _sleep(_seconds):
        raise _StopLoop

    monkeypatch.setattr(main.time, "sleep", _sleep)
    return run_once


def test_main_starts_the_receiver_when_enabled(monkeypatch, gate_config):
    start = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(main.booking_server, "start", start)
    stop_after_one_cycle(monkeypatch, gate_config)

    with pytest.raises(_StopLoop):
        main.main()

    start.assert_called_once_with(gate_config)


def test_main_survives_a_receiver_bind_failure(monkeypatch, gate_config, caplog):
    monkeypatch.setattr(
        main.booking_server, "start", MagicMock(side_effect=OSError("port in use"))
    )
    notified = MagicMock()
    monkeypatch.setattr(main.notify, "notify_error", notified)
    run_once = stop_after_one_cycle(monkeypatch, gate_config)

    with caplog.at_level(logging.ERROR), pytest.raises(_StopLoop):
        main.main()

    # The polling loop is the primary job; a dead receiver degrades the gate but must
    # not stop transcription.
    run_once.assert_called_once()
    notified.assert_called_once()
    assert "booking receiver" in caplog.text.lower()


# --- write .meta.yml and .stt for every processed recording -------------------


_STT_NAME = "Angelica Munkueva(ExpertizeMe) и Mels - 2026/08/13 14:29 CEST.mp4"
_STT_STEM = "Angelica Munkueva(ExpertizeMe) и Mels - 2026_08_13 14_29 CEST"
_STT_TRANSCRIPT = "[00:00:05] Angelica Munkueva: Здравствуйте\n[00:31:42] Mels: Спасибо"
# Named distinctly from the webhook tests' own `_META_ARTIFACT` (line 2537) so this
# block does not shadow that constant's binding for readers scanning the file.
_STT_META_ARTIFACT = "---\nsubject: Обсудили кейс\ntags: []\n---"


def _stt_config(tmp_path, **overrides):
    out = tmp_path / "results"
    out.mkdir(exist_ok=True)
    return make_config(
        # `_as_folders` (used by `make_config`) only accepts bare ids or
        # `EmployeeFolder` instances, not the {"folder_id": ...} mapping form the
        # brief predicted -- that form silently binds the whole dict as `folder_id`,
        # so `config.folder_by_id("folderA")` would never match and the meta
        # document's manager/manager_email would go empty.
        folders=[EmployeeFolder(
            "folderA", name="Анжелика Мункуева", email="angelica@expertizeme.org"
        )],
        presets=(_KEYPOINTS_BUILTIN,),
        output_target="folder",
        output_dir=out,
        **overrides,
    )


def _write_documents(
    cfg, tmp_path, artifacts, transcript=_STT_TRANSCRIPT, container_id="folderA"
):
    return main._write_call_documents(
        MagicMock(),
        "fid1",
        _STT_NAME,
        "folderA",
        container_id,
        transcript,
        artifacts,
        cfg,
        tmp_path,
        item={},
        booking_decision=MATCHED_DECISION,
    )


def test_write_call_documents_writes_the_stt_and_the_meta_yml(tmp_path):
    cfg = _stt_config(tmp_path)
    document = _write_documents(
        cfg, tmp_path, {"keypoints": "## Задачи\n- [ ] x", "meta": _STT_META_ARTIFACT}
    )
    stt = (cfg.output_dir / f"{_STT_STEM}.stt").read_text(encoding="utf-8")
    assert stt.index("## Задачи") < stt.index("## Мета") < stt.index("## Расшифровка")
    assert (cfg.output_dir / f"{_STT_STEM}.meta.yml").exists()
    assert document["subject"] == "Обсудили кейс"
    assert document["planfix_task_id"] == "851030"


def test_write_call_documents_reads_a_preset_from_its_local_artifact(tmp_path):
    """A cycle that re-ran only `meta` must still get keypoints into the .stt."""
    cfg = _stt_config(tmp_path)
    (cfg.output_dir / f"{_STT_STEM}.keypoints.md").write_text(
        "## Задачи\n- [ ] x", encoding="utf-8"
    )
    _write_documents(cfg, tmp_path, {"meta": _STT_META_ARTIFACT})
    stt = (cfg.output_dir / f"{_STT_STEM}.stt").read_text(encoding="utf-8")
    assert "## Задачи" in stt


def test_write_call_documents_falls_back_to_the_raw_transcript(tmp_path):
    """No transcript-cleanup artifact means the .stt carries the raw text."""
    cfg = _stt_config(tmp_path)
    _write_documents(cfg, tmp_path, {})
    stt = (cfg.output_dir / f"{_STT_STEM}.stt").read_text(encoding="utf-8")
    assert "[00:00:05] Angelica Munkueva: Здравствуйте" in stt


def test_write_call_documents_writes_the_meta_yml_when_the_preset_produced_nothing(tmp_path):
    cfg = _stt_config(tmp_path)
    document = _write_documents(cfg, tmp_path, {"keypoints": "## Задачи"})
    assert (cfg.output_dir / f"{_STT_STEM}.meta.yml").exists()
    assert document["subject"] == ""
    assert document["client"] == "Mels"


def test_process_item_survives_a_failed_stt_meta_write(mocker, tmp_path, caplog):
    """A `.stt`/`.meta.yml` write failure must not cost the recording its webhook or
    Planfix comment: by the time `_write_call_documents` runs, the `.txt` and every
    preset artifact are already persisted, so a re-raise here would leave the next
    cycle seeing a fully-processed recording (has_txt, no missing presets) that
    never got notified and never will be retried.
    """
    notify = _mock_successful_run(mocker, tmp_path)
    mocker.patch(
        "src.main._write_call_documents", side_effect=RuntimeError("disk full")
    )

    with caplog.at_level(logging.WARNING):
        telemetry = main.process_item(
            MagicMock(), _item("fid", "video.mp4"), "folderX", _webhook_config()
        )

    assert telemetry is not None
    assert telemetry.meta_document is None
    notify.assert_called_once()
    assert any(
        "stt" in record.message.lower() and "video.mp4" in record.message
        for record in caplog.records
    )


def test_apply_local_output_state_stt_and_meta_yml_do_not_mark_a_recording_processed(
    tmp_path,
):
    """The load-bearing invariant: a `.stt`/`.meta.yml` sitting in `output_dir` with
    no `.txt` sibling must not make `_apply_local_output_state`/`_pending_items`
    think the recording is done. Only `.txt` and the preset artifacts may do that
    (see `_write_call_documents`'s docstring) -- getting this wrong would silently
    stop re-selecting recordings that were never actually transcribed.
    """
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    (out_dir / "video.stt").write_text("assembled", encoding="utf-8")
    (out_dir / "video.meta.yml").write_text("subject: ''\n", encoding="utf-8")

    cfg = make_config(
        stt_provider="deepgram",
        drive_mp3_artifact=False,
        output_target="folder",
        output_dir=out_dir,
    )
    items = [_item("fid", "video.mp4")]
    main._apply_local_output_state(items, cfg)

    assert items[0]["has_txt"] is False
    assert main._pending_items(items, cfg) == items


def test_apply_local_output_state_stt_and_meta_yml_do_not_block_a_processed_recording(
    tmp_path,
):
    """The converse of the test above: once the real `.txt` is present (and every
    preset artifact, none configured here), a `.stt`/`.meta.yml` sitting alongside it
    must not make the recording look pending again.
    """
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    (out_dir / "video.txt").write_text("Speaker 1: hi", encoding="utf-8")
    (out_dir / "video.stt").write_text("assembled", encoding="utf-8")
    (out_dir / "video.meta.yml").write_text("subject: ''\n", encoding="utf-8")

    cfg = make_config(
        stt_provider="deepgram",
        drive_mp3_artifact=False,
        output_target="folder",
        output_dir=out_dir,
    )
    items = [_item("fid", "video.mp4")]
    main._apply_local_output_state(items, cfg)

    assert items[0]["has_txt"] is True
    assert main._pending_items(items, cfg) == []


# --- Telegram summary -----------------------------------------------------------


TELEGRAM_CHAT_ID = "-1001234567890"


@pytest.fixture
def telegram_config(gate_config):
    """A gate config whose folder delivers summaries to a Telegram chat."""
    return replace(
        gate_config,
        folders=(
            EmployeeFolder(
                folder_id=GATE_FOLDER_ID,
                name="Kate",
                email="kate@example.com",
                telegram=TELEGRAM_CHAT_ID,
            ),
        ),
        telegram_bot_token="bot-token",
        planfix_presets=("keypoints",),
    )


def test_telegram_summary_strips_markdown_to_plain_text():
    """Telegram is sent without a parse mode, so `##`/`**`/`[](...)` would show up
    literally. The content must be the same as the Planfix comment's -- only the
    markup comes off."""
    text = main._telegram_summary(
        {"keypoints": "## Задачи\n\n- Собрать документы"},
        ("keypoints",),
        {
            "subject": "Виза O-1",
            "duration": "00:31:42",
            "video_url": "https://drive.google.com/file/d/X/view",
            "source_name": "rec.mp4",
        },
        ("subject", "duration", "video_url"),
        meta_entities=meta_entity.default_entities(),
    )

    assert "**" not in text
    assert "##" not in text
    assert "](" not in text
    assert "Задачи" in text
    assert "- Собрать документы" in text
    assert "rec.mp4: https://drive.google.com/file/d/X/view" in text
    # Header before the preset sections, same order as the CRM comment.
    assert text.index("Виза O-1") < text.index("Задачи")


def test_telegram_summary_is_blank_when_only_the_header_would_render():
    """Same guard as `_planfix_description`: a duration and a link are not a summary,
    and sending one would write the `telegram_sent_chat_id` marker and permanently
    block the real one."""
    assert main._telegram_summary(
        {},
        ("keypoints",),
        {"duration": "00:31:42"},
        ("duration",),
        meta_entities=meta_entity.default_entities(),
    ) == ""


def test_telegram_summary_is_sent_and_marked(monkeypatch, telegram_config):
    send = MagicMock(return_value=True)
    monkeypatch.setattr(main.notify, "send_message", send)
    marked = MagicMock()
    monkeypatch.setattr(main.drive, "set_file_app_properties", marked)

    main._send_telegram_summary(
        MagicMock(), gate_item("v1"), "v1", GATE_FOLDER_ID, telegram_config,
        {"keypoints": "Задачи: раз"}, UNMATCHED_DECISION,
    )

    send.assert_called_once()
    assert send.call_args.kwargs["chat_id"] == TELEGRAM_CHAT_ID
    assert send.call_args.kwargs["bot_token"] == "bot-token"
    assert send.call_args[0][0] == "Задачи: раз"
    marked.assert_called_once()
    assert marked.call_args[0][2] == {"telegram_sent_chat_id": TELEGRAM_CHAT_ID}


def test_telegram_summary_is_not_sent_for_a_folder_without_a_chat(
    monkeypatch, gate_config
):
    send = MagicMock(return_value=True)
    monkeypatch.setattr(main.notify, "send_message", send)

    main._send_telegram_summary(
        MagicMock(), gate_item("v1"), "v1", GATE_FOLDER_ID, gate_config,
        {"keypoints": "Задачи: раз"}, MATCHED_DECISION,
    )

    send.assert_not_called()


def test_telegram_summary_is_not_sent_twice(monkeypatch, telegram_config):
    send = MagicMock(return_value=True)
    monkeypatch.setattr(main.notify, "send_message", send)

    main._send_telegram_summary(
        MagicMock(),
        gate_item("v1", telegram_sent_chat_id=TELEGRAM_CHAT_ID),
        "v1", GATE_FOLDER_ID, telegram_config,
        {"keypoints": "Задачи: раз"}, UNMATCHED_DECISION,
    )

    send.assert_not_called()


def test_telegram_summary_also_goes_out_for_a_matched_recording(
    monkeypatch, telegram_config
):
    """Default: the chat is an independent channel, so a call that reached Planfix
    reaches the chat too."""
    send = MagicMock(return_value=True)
    monkeypatch.setattr(main.notify, "send_message", send)
    monkeypatch.setattr(main.drive, "set_file_app_properties", MagicMock())
    configured = replace(
        telegram_config,
        planfix_create_comment_url="https://crm.example.com/planfix_create_comment",
    )

    main._send_telegram_summary(
        MagicMock(), gate_item("v1"), "v1", GATE_FOLDER_ID, configured,
        {"keypoints": "Задачи: раз"}, MATCHED_DECISION,
    )

    send.assert_called_once()


def test_ignore_telegram_when_planfix_keeps_a_matched_recording_out_of_the_chat(
    monkeypatch, telegram_config
):
    send = MagicMock(return_value=True)
    monkeypatch.setattr(main.notify, "send_message", send)
    configured = replace(
        telegram_config,
        planfix_create_comment_url="https://crm.example.com/planfix_create_comment",
        planfix_ignore_telegram_when_planfix=True,
    )

    main._send_telegram_summary(
        MagicMock(), gate_item("v1"), "v1", GATE_FOLDER_ID, configured,
        {"keypoints": "Задачи: раз"}, MATCHED_DECISION,
    )

    send.assert_not_called()


def test_ignore_telegram_when_planfix_still_sends_an_unmatched_recording(
    monkeypatch, telegram_config
):
    """The option is a de-duplication rule, not an off switch: a call Planfix never
    saw is exactly the one the chat exists for."""
    send = MagicMock(return_value=True)
    monkeypatch.setattr(main.notify, "send_message", send)
    monkeypatch.setattr(main.drive, "set_file_app_properties", MagicMock())
    configured = replace(
        telegram_config,
        planfix_create_comment_url="https://crm.example.com/planfix_create_comment",
        planfix_ignore_telegram_when_planfix=True,
    )

    main._send_telegram_summary(
        MagicMock(), gate_item("v1"), "v1", GATE_FOLDER_ID, configured,
        {"keypoints": "Задачи: раз"}, UNMATCHED_DECISION,
    )

    send.assert_called_once()


def test_a_failed_telegram_send_leaves_no_marker(monkeypatch, telegram_config):
    """No marker means `gdstt reprocess` can resend it."""
    monkeypatch.setattr(main.notify, "send_message", MagicMock(return_value=False))
    marked = MagicMock()
    monkeypatch.setattr(main.drive, "set_file_app_properties", marked)

    main._send_telegram_summary(
        MagicMock(), gate_item("v1"), "v1", GATE_FOLDER_ID, telegram_config,
        {"keypoints": "Задачи: раз"}, UNMATCHED_DECISION,
    )

    marked.assert_not_called()


def test_run_once_processes_an_unmatched_recording_in_a_telegram_folder(
    monkeypatch, telegram_config
):
    """`telegram` on a folder means "recognize always". Skipping here -- and worse,
    writing the permanent `booking_match=none` mark -- would park every recording in a
    folder that has no bookings by design."""
    monkeypatch.setattr(main.booking_server, "is_running", lambda: True)
    patch_decision(monkeypatch, UNMATCHED_DECISION)
    patch_folder_items(monkeypatch, [gate_item("v1")])
    marked = MagicMock()
    monkeypatch.setattr(main.booking_gate, "mark_unmatched", marked)
    process_item = MagicMock(return_value=None)
    monkeypatch.setattr(main, "process_item", process_item)

    main.run_once(MagicMock(), telegram_config)

    process_item.assert_called_once()
    marked.assert_not_called()


# --- Meeting subfolders -----------------------------------------------------------
#
# Google Meet files every call into its own subfolder, so the folder a video lives in
# is no longer the folder the configuration names. Both ids matter, and they must not
# be swapped: the configured one says whose recording this is, the container says
# where the artifacts go.


def _subfolder_item(file_id, name, container_id, **kwargs):
    item = _item(file_id, name, **kwargs)
    item["container_id"] = container_id
    return item


def test_run_once_reads_a_folder_together_with_its_meeting_subfolders(mocker):
    """Pointed at a Google Meet root, the old single-level listing found subfolders and
    zero videos -- the silent shape of this whole outage."""
    cfg = make_config(folders=["root"], stt_provider="")
    tree_mock = mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])
    mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    tree_mock.assert_called_once_with(mocker.ANY, "root")


def test_run_once_still_names_the_employee_from_the_configured_folder(mocker):
    """The video sits in a subfolder nobody configured; the employee is the folder
    above it. Passing the subfolder here is what would silently blank the employee,
    the Planfix routing and the folder's Telegram chat."""
    cfg = make_config(folders=["root"])
    item = _subfolder_item("v1", "a.mp4", "meeting-1")
    mocker.patch("src.main.drive.list_folder_tree_state", return_value=[item])
    mocker.patch("src.main.booking_gate.resolve", return_value=MATCHED_DECISION)
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    assert process_mock.call_args.args[2] == "root"


def test_process_item_writes_the_transcript_into_the_meeting_subfolder(mocker, tmp_path):
    """The .txt belongs beside its video, not in the employee's root."""
    cfg = make_config(folders=["root"], stt_provider="deepgram", output_dir=tmp_path)
    upload_mock = mocker.patch("src.main._save_and_upload_txt")
    mocker.patch("src.main._run_preset_stage", return_value={})
    mocker.patch("src.main._try_write_call_documents", return_value=None)
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")
    mp4_path = tmp_path / "a.mp4"
    mp4_path.write_bytes(b"video")
    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=tmp_path / "a.mp3")
    mocker.patch("src.main.extract_m4a_copy", return_value=tmp_path / "a.m4a")
    mocker.patch("src.main.drive.upload")

    main.process_item(
        MagicMock(),
        _subfolder_item("v1", "a.mp4", "meeting-1"),
        "root",
        cfg,
        booking_decision=MATCHED_DECISION,
    )

    assert upload_mock.call_args.args[4] == "meeting-1"


def test_process_item_falls_back_to_the_configured_folder_for_a_flat_item(mocker, tmp_path):
    """An item with no container -- a flat folder, or one a caller built by hand --
    keeps writing where it always did."""
    cfg = make_config(folders=["root"], stt_provider="deepgram", output_dir=tmp_path)
    upload_mock = mocker.patch("src.main._save_and_upload_txt")
    mocker.patch("src.main._run_preset_stage", return_value={})
    mocker.patch("src.main._try_write_call_documents", return_value=None)
    mocker.patch("src.main.transcribe_file", return_value="Speaker 1: hi")
    mp4_path = tmp_path / "a.mp4"
    mp4_path.write_bytes(b"video")
    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=tmp_path / "a.mp3")
    mocker.patch("src.main.extract_m4a_copy", return_value=tmp_path / "a.m4a")
    mocker.patch("src.main.drive.upload")

    main.process_item(
        MagicMock(), _item("v1", "a.mp4"), "root", cfg,
        booking_decision=MATCHED_DECISION,
    )

    assert upload_mock.call_args.args[4] == "root"


def test_webhook_payload_reports_the_configured_folder_not_the_subfolder():
    """The payload's folder_id is documented to consumers, who key it to the employee.
    Sending the meeting subfolder would change that contract to a value that means
    nothing outside this service."""
    cfg = make_config(folders=[EmployeeFolder("root", name="Анжелика", email="a@b.c")])

    payload = main._webhook_payload(
        "v1", "a.mp4", "root", cfg, "transcript", {},
    )

    assert payload["file"]["folder_id"] == "root"
    assert payload["employee"]["name"] == "Анжелика"


def test_process_one_file_in_a_subfolder_resolves_the_employee_above_it(mocker):
    service = MagicMock()
    cfg = make_config(folders=["root"])
    mocker.patch(
        "src.main.drive.get_file_metadata",
        return_value={
            "id": "v1", "name": "a.mp4", "mimeType": "video/mp4",
            "parents": ["meeting-1"],
        },
    )
    ancestor_mock = mocker.patch(
        "src.main.drive.find_configured_ancestor", return_value="root"
    )
    list_mock = mocker.patch(
        "src.main.drive.list_folder_state",
        return_value=[_subfolder_item("v1", "a.mp4", "meeting-1")],
    )
    process_mock = mocker.patch("src.main.process_item")

    main.process_target(service, "v1", cfg)

    # Listed where the file is, attributed to the folder above it.
    list_mock.assert_called_once_with(service, "meeting-1")
    ancestor_mock.assert_called_once_with(service, "meeting-1", {"root"})
    assert process_mock.call_args.args[2] == "root"


def test_process_one_file_keeps_its_own_folder_when_nothing_is_configured_above(mocker):
    """A hand-made folder outside the configuration is still processed, just without
    an employee -- the behaviour that existed before subfolders."""
    service = MagicMock()
    cfg = make_config(folders=["root"])
    mocker.patch(
        "src.main.drive.get_file_metadata",
        return_value={
            "id": "v1", "name": "a.mp4", "mimeType": "video/mp4",
            "parents": ["stt-test"],
        },
    )
    mocker.patch("src.main.drive.find_configured_ancestor", return_value=None)
    mocker.patch(
        "src.main.drive.list_folder_state",
        return_value=[_subfolder_item("v1", "a.mp4", "stt-test")],
    )
    process_mock = mocker.patch("src.main.process_item")

    main.process_target(service, "v1", cfg)

    assert process_mock.call_args.args[2] == "stt-test"


def test_process_a_folder_walks_its_subfolders(mocker):
    service = MagicMock()
    cfg = make_config(folders=["root"], stt_provider="")
    mocker.patch(
        "src.main.drive.get_file_metadata",
        return_value={"id": "root", "name": "Google Meet",
                      "mimeType": "application/vnd.google-apps.folder"},
    )
    mocker.patch("src.main.drive.find_configured_ancestor", return_value="root")
    tree_mock = mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])

    main.process_target(service, "root", cfg, is_folder=True)

    tree_mock.assert_called_once_with(service, "root")


def _settling_item(file_id, name, created_at, *, has_media_metadata):
    item = _item(file_id, name)
    item["file"]["createdTime"] = created_at
    item["has_media_metadata"] = has_media_metadata
    return item


def _now():
    return datetime(2026, 9, 9, 20, 0, tzinfo=timezone.utc)


def test_a_video_drive_has_not_finished_with_is_left_for_the_next_cycle(mocker):
    """Meet's upload lands minutes to an hour after the meeting folder appears. Taking
    a video Drive is still processing buys a wasted download and a wasted STT run."""
    cfg = make_config(folders=["root"], stt_provider="deepgram")
    item = _settling_item("v1", "a.mp4", "2026-09-09T19:58:00Z", has_media_metadata=False)
    mocker.patch("src.main._utcnow", return_value=_now())

    assert main._pending_items([item], cfg) == []


def test_a_finished_video_is_picked_up_at_once(mocker):
    cfg = make_config(folders=["root"], stt_provider="deepgram")
    item = _settling_item("v1", "a.mp4", "2026-09-09T19:58:00Z", has_media_metadata=True)
    mocker.patch("src.main._utcnow", return_value=_now())

    assert len(main._pending_items([item], cfg)) == 1


def test_waiting_for_metadata_gives_up_rather_than_stalling_forever(mocker):
    """A video that never gets metadata -- an odd encode, a Drive that simply never
    fills it in -- must still be transcribed. Waiting without a limit would lose it
    silently, which is the failure mode this whole change exists to remove."""
    cfg = make_config(folders=["root"], stt_provider="deepgram")
    item = _settling_item("v1", "a.mp4", "2026-09-08T06:00:00Z", has_media_metadata=False)
    mocker.patch("src.main._utcnow", return_value=_now())

    assert len(main._pending_items([item], cfg)) == 1


def test_a_video_of_unknown_age_is_not_held_back(mocker):
    """No createdTime means no way to tell young from stuck; processing is the safe
    side of that guess."""
    cfg = make_config(folders=["root"], stt_provider="deepgram")
    item = _item("v1", "a.mp4")
    item["has_media_metadata"] = False
    mocker.patch("src.main._utcnow", return_value=_now())

    assert len(main._pending_items([item], cfg)) == 1


def test_items_from_before_this_change_are_not_held_back(mocker):
    """An item built by a caller that knows nothing of media metadata -- every existing
    test, and `reprocess` -- must behave as it always did."""
    cfg = make_config(folders=["root"], stt_provider="deepgram")
    mocker.patch("src.main._utcnow", return_value=_now())

    assert len(main._pending_items([_item("v1", "a.mp4")], cfg)) == 1


# --- The changes feed -------------------------------------------------------------
#
# Drive keeps a journal of what changed; one request reads it, whatever the number of
# folders. Walking every folder stays as the fallback, which is what makes the cursor
# safe to lose.


def _http_error(status):
    return HttpError(MagicMock(status=status), b"")


def _change(file_id, container, *, mime="video/mp4", removed=False, trashed=False):
    return {
        "fileId": file_id,
        "removed": removed,
        "file": {
            "id": file_id,
            "name": f"{file_id}.mp4",
            "mimeType": mime,
            "parents": [container],
            "trashed": trashed,
        },
    }


def _cursor_file(cfg):
    return change_cursor.path_for(cfg.data_dir)


def _save_cursor(cfg, token):
    """Save a cursor the way a real cycle does: together with the folders it covers.

    A cursor on its own cannot be vouched for, and an unvouched cursor makes the next
    cycle sweep -- that is the whole point of `changes_folders.txt`. So a test that
    wants the feed to be read has to set up both, exactly like `run_once` does.
    """
    change_cursor.write(change_cursor.path_for(cfg.data_dir), token)
    change_cursor.write_folders(
        change_cursor.folders_path_for(cfg.data_dir),
        change_cursor.fingerprint(folder.folder_id for folder in cfg.folders),
    )


def test_the_first_cycle_sweeps_and_remembers_where_it_got_to(mocker, tmp_path):
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    tree_mock = mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])
    mocker.patch("src.main.drive.get_start_page_token", return_value="tok-1")
    changes_mock = mocker.patch("src.main.drive.list_changes")

    main.run_once(MagicMock(), cfg)

    tree_mock.assert_called_once()
    changes_mock.assert_not_called()
    assert change_cursor.read(_cursor_file(cfg)) == "tok-1"


def test_the_cursor_is_taken_before_the_sweep_not_after(mocker, tmp_path):
    """A recording that lands while the sweep is running has to turn up in the next
    feed read. A cursor taken afterwards would step straight over it."""
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    order = []
    mocker.patch(
        "src.main.drive.get_start_page_token",
        side_effect=lambda *a, **k: (order.append("cursor"), "tok-1")[1],
    )
    mocker.patch(
        "src.main.drive.list_folder_tree_state",
        side_effect=lambda *a, **k: (order.append("sweep"), [])[1],
    )

    main.run_once(MagicMock(), cfg)

    assert order == ["cursor", "sweep"]


def test_a_later_cycle_reads_the_feed_instead_of_sweeping(mocker, tmp_path):
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    tree_mock = mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])
    changes_mock = mocker.patch("src.main.drive.list_changes", return_value=([], "tok-2"))

    main.run_once(MagicMock(), cfg)

    changes_mock.assert_called_once_with(mocker.ANY, "tok-1")
    tree_mock.assert_not_called()
    assert change_cursor.read(_cursor_file(cfg)) == "tok-2"


def test_a_changed_video_has_only_its_own_folder_listed(mocker, tmp_path):
    """The point of the feed: look where something happened, not everywhere."""
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    mocker.patch(
        "src.main.drive.list_changes",
        return_value=([_change("v1", "meeting-1")], "tok-2"),
    )
    mocker.patch("src.main.drive.find_configured_ancestor", return_value="root")
    list_mock = mocker.patch("src.main.drive.list_folder_state", return_value=[])

    main.run_once(MagicMock(), cfg)

    list_mock.assert_called_once_with(mocker.ANY, "meeting-1")


def test_two_videos_in_one_meeting_folder_cost_one_listing(mocker, tmp_path):
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    mocker.patch(
        "src.main.drive.list_changes",
        return_value=([_change("v1", "meeting-1"), _change("v2", "meeting-1")], "tok-2"),
    )
    mocker.patch("src.main.drive.find_configured_ancestor", return_value="root")
    list_mock = mocker.patch("src.main.drive.list_folder_state", return_value=[])

    main.run_once(MagicMock(), cfg)

    assert list_mock.call_count == 1


def test_a_video_found_through_the_feed_is_attributed_to_its_configured_folder(
    mocker, tmp_path
):
    cfg = make_config(folders=["root"], data_dir=tmp_path)
    _save_cursor(cfg, "tok-1")
    mocker.patch(
        "src.main.drive.list_changes",
        return_value=([_change("v1", "meeting-1")], "tok-2"),
    )
    mocker.patch("src.main.drive.find_configured_ancestor", return_value="root")
    mocker.patch(
        "src.main.drive.list_folder_state",
        return_value=[_subfolder_item("v1", "a.mp4", "meeting-1")],
    )
    mocker.patch("src.main.booking_gate.resolve", return_value=MATCHED_DECISION)
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    assert process_mock.call_args.args[2] == "root"


def test_our_own_uploads_in_the_feed_are_ignored(mocker, tmp_path):
    """Every artifact this service writes comes back through the feed. Deciding from
    the entry alone is what keeps that free."""
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    mocker.patch(
        "src.main.drive.list_changes",
        return_value=([_change("t1", "meeting-1", mime="text/plain")], "tok-2"),
    )
    list_mock = mocker.patch("src.main.drive.list_folder_state", return_value=[])

    main.run_once(MagicMock(), cfg)

    list_mock.assert_not_called()


def test_deleted_and_trashed_entries_are_ignored(mocker, tmp_path):
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    mocker.patch(
        "src.main.drive.list_changes",
        return_value=(
            [
                _change("v1", "meeting-1", removed=True),
                _change("v2", "meeting-2", trashed=True),
            ],
            "tok-2",
        ),
    )
    list_mock = mocker.patch("src.main.drive.list_folder_state", return_value=[])

    main.run_once(MagicMock(), cfg)

    list_mock.assert_not_called()


def test_a_video_in_a_folder_nobody_configured_is_ignored(mocker, tmp_path):
    """The feed reports everything the account can see, not only what we watch."""
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    mocker.patch(
        "src.main.drive.list_changes",
        return_value=([_change("v1", "someone-elses")], "tok-2"),
    )
    mocker.patch("src.main.drive.find_configured_ancestor", return_value=None)
    list_mock = mocker.patch("src.main.drive.list_folder_state", return_value=[])

    main.run_once(MagicMock(), cfg)

    list_mock.assert_not_called()


def test_a_cursor_drive_no_longer_knows_falls_back_to_a_sweep(mocker, tmp_path):
    """Aging out of the journal is documented, not exceptional: sweep, take a fresh
    cursor, carry on."""
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-stale")
    mocker.patch("src.main.drive.list_changes", side_effect=_http_error(410))
    tree_mock = mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])
    mocker.patch("src.main.drive.get_start_page_token", return_value="tok-fresh")
    notify_mock = mocker.patch("src.main.notify.notify_error")

    main.run_once(MagicMock(), cfg)

    tree_mock.assert_called_once()
    assert change_cursor.read(_cursor_file(cfg)) == "tok-fresh"
    notify_mock.assert_not_called()


def test_a_deleted_cursor_file_makes_the_next_cycle_sweep(mocker, tmp_path):
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    change_cursor.clear(_cursor_file(cfg))
    tree_mock = mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])
    mocker.patch("src.main.drive.get_start_page_token", return_value="tok-2")
    mocker.patch("src.main.drive.list_changes")

    main.run_once(MagicMock(), cfg)

    tree_mock.assert_called_once()


def test_a_feed_that_fails_for_another_reason_keeps_the_cursor(mocker, tmp_path):
    """A network blip must not throw away the cursor: that would turn a retry into a
    full sweep of every folder."""
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    mocker.patch("src.main.drive.list_changes", side_effect=_http_error(500))
    mocker.patch("src.main.time.sleep")
    tree_mock = mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])
    notify_mock = mocker.patch("src.main.notify.notify_error")

    main.run_once(MagicMock(), cfg)

    tree_mock.assert_not_called()
    notify_mock.assert_called_once()
    assert change_cursor.read(_cursor_file(cfg)) == "tok-1"


def test_a_dry_run_never_moves_the_cursor(mocker, tmp_path):
    """Otherwise a real run after a dry one starts past everything the dry run saw,
    and those recordings are never processed."""
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    mocker.patch("src.main.drive.list_changes", return_value=([], "tok-2"))

    main.run_once(MagicMock(), cfg, dry_run=True)

    assert change_cursor.read(_cursor_file(cfg)) == "tok-1"


def test_the_cursor_moves_only_after_the_work_is_done(mocker, tmp_path):
    """A cycle that dies half way through must see the same changes again. Re-reading
    them is free, because the folder listing decides what still needs doing."""
    cfg = make_config(folders=["root"], data_dir=tmp_path)
    _save_cursor(cfg, "tok-1")
    mocker.patch(
        "src.main.drive.list_changes",
        return_value=([_change("v1", "meeting-1")], "tok-2"),
    )
    mocker.patch("src.main.drive.find_configured_ancestor", return_value="root")
    mocker.patch(
        "src.main.drive.list_folder_state",
        return_value=[_subfolder_item("v1", "a.mp4", "meeting-1")],
    )
    mocker.patch("src.main.booking_gate.resolve", return_value=MATCHED_DECISION)

    seen = {}

    def explode(*args, **kwargs):
        seen["cursor_during_work"] = change_cursor.read(_cursor_file(cfg))
        raise RuntimeError("processing blew up")

    mocker.patch("src.main.process_item", side_effect=explode)
    mocker.patch("src.main.notify.notify_error")

    main.run_once(MagicMock(), cfg)

    assert seen["cursor_during_work"] == "tok-1"



# --- Names from Meet's own transcript ---------------------------------------------


_MEET_DOC = """may-doqs-end (2026-09-09 18:53 GMT+2) - Transcript
Attendees
Oksana Ciciarelli, Oksana Ciciarelli's Presentation, Roman Starodubtsev
Transcript
Oksana Ciciarelli: one
Roman Starodubtsev: two
"""


def test_meet_transcript_names_a_room_code_call_the_file_name_cannot(mocker):
    """The gap this closes: a call started outside the calendar is named after the
    meeting room, so there is nothing in the name to read."""
    mocker.patch("src.main.drive.find_meet_transcript", return_value={"id": "d1"})
    mocker.patch("src.main.drive.export_document_text", return_value=_MEET_DOC)

    names, text = main._read_meet_transcript(
        MagicMock(), "meeting-1", "may-doqs-end (2026-09-09 18_53 GMT+2).mp4"
    )

    assert names == ["Oksana Ciciarelli", "Roman Starodubtsev"]
    # The turns travel with the names: they are the model's evidence of who is who.
    assert text == _MEET_DOC


def test_no_transcript_leaves_the_file_name_in_charge(mocker):
    mocker.patch("src.main.drive.find_meet_transcript", return_value=None)

    assert main._read_meet_transcript(MagicMock(), "meeting-1", "a.mp4") is None


def test_an_unreadable_transcript_does_not_fail_the_recording(mocker):
    """Losing the names is a worse transcript; losing the recording is an outage."""
    mocker.patch(
        "src.main.drive.find_meet_transcript", side_effect=RuntimeError("no access")
    )

    assert main._read_meet_transcript(MagicMock(), "meeting-1", "a.mp4") is None


def test_a_transcript_naming_fewer_than_two_people_is_not_used(mocker):
    """One name cannot tell two diarized speakers apart, and the caller already has a
    better-tested path for that."""
    mocker.patch("src.main.drive.find_meet_transcript", return_value={"id": "d1"})
    mocker.patch(
        "src.main.drive.export_document_text",
        return_value="Call - Transcript\nAttendees\nAlice\nTranscript\nAlice: one\n",
    )

    assert main._read_meet_transcript(MagicMock(), "meeting-1", "a.mp4") is None


def test_meet_names_and_turns_are_handed_to_the_model(mocker):
    """The model still decides who is who; Meet gives it the people and, in its turns,
    the evidence of which voice is whose."""
    cfg = make_config(folders=["root"], openai_api_key="sk-test")
    resolve_mock = mocker.patch(
        "src.main.speaker_roles.resolve", return_value=["Roman", "Oksana"]
    )
    mocker.patch("src.main.OpenAIPipeline")

    main._resolve_speaker_names(
        "Speaker 1: hi", "may-doqs-end (2026-09-09 18_53 GMT+2).mp4", "root", cfg,
        candidates=["Oksana Ciciarelli", "Roman Starodubtsev"],
        meet_text=_MEET_DOC,
    )

    assert resolve_mock.call_args.kwargs["candidates"] == [
        "Oksana Ciciarelli",
        "Roman Starodubtsev",
    ]
    assert resolve_mock.call_args.kwargs["meet_text"] == _MEET_DOC
    assert resolve_mock.call_args.kwargs["calendar_manager"] == ""


def test_the_calendar_titles_marked_manager_is_handed_to_the_model(mocker):
    cfg = make_config(folders=["root"], openai_api_key="sk-test")
    resolve_mock = mocker.patch("src.main.speaker_roles.resolve", return_value=None)
    mocker.patch("src.main.OpenAIPipeline")

    main._resolve_speaker_names(
        "Speaker 1: hi",
        "Angelica Munkueva(ExpertizeMe) и Mels - 2026/08/13 14:29 CEST - Recording.mp4",
        "root",
        cfg,
    )

    assert resolve_mock.call_args.kwargs["calendar_manager"] == "Angelica Munkueva"


def test_without_candidates_the_file_name_is_still_the_source(mocker):
    cfg = make_config(folders=["root"], openai_api_key="sk-test")
    resolve_mock = mocker.patch(
        "src.main.speaker_roles.resolve", return_value=["Alice", "Bob"]
    )
    mocker.patch("src.main.OpenAIPipeline")

    main._resolve_speaker_names(
        "Speaker 1: hi", "Alice and Bob - 2026/09/09 10:00 CEST.mp4", "root", cfg,
    )

    assert resolve_mock.call_args.kwargs["candidates"] == ["Alice", "Bob"]


def _transcript_written_with_meet_beside(
    mocker, tmp_path, file_name, *, resolved, key, meet_doc=_MEET_DOC
):
    """Run a recording through STT with Meet's transcript beside it."""
    mocker.patch("src.main.drive.download", return_value=tmp_path / "video.mp4")
    mocker.patch("src.main.extract_mp3", return_value=tmp_path / "video.mp3")
    captured = {}

    def fake_upload(svc, local_path, folder, mime_type, name=None, app_properties=None):
        if name and name.endswith(".txt"):
            captured["txt"] = local_path.read_text(encoding="utf-8")

    mocker.patch("src.main.drive.upload", side_effect=fake_upload)
    mocker.patch(
        "src.main.transcribe_file",
        return_value="Speaker 1: hi there\nSpeaker 2: hello back",
    )
    find_mock = mocker.patch(
        "src.main.drive.find_meet_transcript",
        return_value={"id": "d1"} if meet_doc else None,
    )
    mocker.patch("src.main.drive.export_document_text", return_value=meet_doc)
    mocker.patch("src.main.OpenAIPipeline")
    resolve_mock = mocker.patch("src.main.speaker_roles.resolve", return_value=resolved)
    preset_spy = mocker.spy(main, "_run_preset_stage")
    cfg = make_config(
        stt_provider="deepgram",
        deepgram_api_key="dg-x",
        deepgram_audio_source="mp3_96k",
        stt_postprocess=True,
        openai_api_key=key,
    )

    main.process_item(MagicMock(), _item("fid", file_name), "f", cfg)

    return SimpleNamespace(
        txt=captured["txt"],
        resolve=resolve_mock,
        find=find_mock,
        preset_names=preset_spy.call_args.kwargs["speaker_names"],
    )


_ROOM_CODE_CALL = "may-doqs-end (2026-09-09 18_53 GMT+2).mp4"


def test_meets_names_unconfirmed_by_the_model_are_not_bound_by_order(mocker, tmp_path):
    """The regression this pins: Meet listed the people in the order it heard them,
    diarization heard someone else first, and binding the two by position put the
    manager's words under the client's name. Numbered speakers are less, not wrong."""
    run = _transcript_written_with_meet_beside(
        mocker, tmp_path, _ROOM_CODE_CALL, resolved=None, key="sk-test"
    )

    run.resolve.assert_called_once()
    assert run.txt == "Speaker 1: hi there\nSpeaker 2: hello back"


def test_presets_still_hear_who_was_on_a_call_nobody_could_place(mocker, tmp_path):
    """The presets' hint says "in no particular order", so Meet's names carry no swap
    there -- and on a room-code call they are the only names there are."""
    run = _transcript_written_with_meet_beside(
        mocker, tmp_path, _ROOM_CODE_CALL, resolved=None, key="sk-test"
    )

    assert run.preset_names == ["Oksana Ciciarelli", "Roman Starodubtsev"]


def test_without_a_model_meets_transcript_is_not_read(mocker, tmp_path):
    """Nothing could place its names on speakers, so reading it would only cost two
    Drive requests and log names nobody uses."""
    run = _transcript_written_with_meet_beside(
        mocker, tmp_path, _ROOM_CODE_CALL, resolved=["x", "y"], key=""
    )

    run.find.assert_not_called()
    run.resolve.assert_not_called()
    assert run.txt == "Speaker 1: hi there\nSpeaker 2: hello back"


def test_meets_names_label_the_speakers_the_model_placed_them_on(mocker, tmp_path):
    run = _transcript_written_with_meet_beside(
        mocker,
        tmp_path,
        _ROOM_CODE_CALL,
        resolved=["Roman Starodubtsev", "Oksana Ciciarelli"],
        key="sk-test",
    )

    assert run.resolve.call_args.kwargs["meet_text"] == _MEET_DOC
    assert run.txt == "Roman Starodubtsev: hi there\nOksana Ciciarelli: hello back"
    assert run.preset_names == ["Roman Starodubtsev", "Oksana Ciciarelli"]


def test_a_calendar_call_the_model_could_not_place_stays_numbered(mocker, tmp_path):
    """The file name lists the organizer first; binding that by position is right only
    when the manager happens to speak first, and wrong without a trace otherwise. Once
    a model was asked, an unanswered call keeps numbered speakers."""
    run = _transcript_written_with_meet_beside(
        mocker, tmp_path, "Alice and Bob - 2026/09/09 10:00 CEST.mp4",
        resolved=None, key="sk-test", meet_doc=None,
    )

    run.resolve.assert_called_once()
    assert run.txt == "Speaker 1: hi there\nSpeaker 2: hello back"
    # ``None`` lets the presets read the names from the file name, unordered.
    assert run.preset_names is None


def test_without_a_model_a_calendar_call_keeps_the_file_names_order(mocker, tmp_path):
    """No model, no presets: the file name's order is the only naming there is, as it
    always was."""
    run = _transcript_written_with_meet_beside(
        mocker, tmp_path, "Alice and Bob - 2026/09/09 10:00 CEST.mp4",
        resolved=None, key="",
    )

    run.resolve.assert_not_called()
    assert run.txt == "Alice: hi there\nBob: hello back"


def test_resolve_speaker_names_is_empty_when_the_model_was_asked_and_did_not_answer(
    mocker,
):
    """``[]`` and ``None`` mean different things to the caller: numbered speakers, or
    the file name's order because no model was ever asked."""
    cfg = make_config(folders=["root"], openai_api_key="sk-test")
    mocker.patch("src.main.speaker_roles.resolve", return_value=None)
    mocker.patch("src.main.OpenAIPipeline")

    assert (
        main._resolve_speaker_names(
            "Speaker 1: hi", "Alice and Bob - 2026/09/09 10:00 CEST.mp4", "root", cfg
        )
        == []
    )


# --- The cursor may only move past work that is actually finished -----------------


def _one_change_cycle(mocker, tmp_path, **overrides):
    cfg = make_config(folders=["root"], data_dir=tmp_path, **overrides)
    _save_cursor(cfg, "tok-1")
    mocker.patch(
        "src.main.drive.list_changes",
        return_value=([_change("v1", "meeting-1")], "tok-2"),
    )
    mocker.patch("src.main.drive.find_configured_ancestor", return_value="root")
    return cfg


def test_a_failed_recording_holds_the_cursor_so_it_is_seen_again(mocker, tmp_path):
    """The feed names a folder once, when something happens in it. A recording that
    failed writes no artifact, so nothing there will ever change again -- stepping
    over it loses it for good."""
    cfg = _one_change_cycle(mocker, tmp_path)
    mocker.patch(
        "src.main.drive.list_folder_state",
        return_value=[_subfolder_item("v1", "a.mp4", "meeting-1")],
    )
    mocker.patch("src.main.booking_gate.resolve", return_value=MATCHED_DECISION)
    mocker.patch("src.main.process_item", side_effect=RuntimeError("stt timed out"))
    mocker.patch("src.main.notify.notify_error")

    main.run_once(MagicMock(), cfg)

    assert change_cursor.read(change_cursor.path_for(cfg.data_dir)) == "tok-1"


def test_a_folder_that_could_not_be_listed_holds_the_cursor(mocker, tmp_path):
    cfg = _one_change_cycle(mocker, tmp_path, stt_provider="")
    mocker.patch(
        "src.main.drive.list_folder_state", side_effect=RuntimeError("drive 500")
    )
    mocker.patch("src.main.time.sleep")
    mocker.patch("src.main.notify.notify_error")

    main.run_once(MagicMock(), cfg)

    assert change_cursor.read(change_cursor.path_for(cfg.data_dir)) == "tok-1"


def test_a_video_left_to_settle_holds_the_cursor(mocker, tmp_path):
    """Otherwise the change that revealed the video is consumed while the video is
    deliberately skipped, and it depends on Drive emitting a second one later."""
    cfg = _one_change_cycle(mocker, tmp_path, stt_provider="deepgram")
    item = _subfolder_item("v1", "a.mp4", "meeting-1")
    item["file"]["createdTime"] = "2026-09-09T19:58:00Z"
    item["has_media_metadata"] = False
    mocker.patch("src.main.drive.list_folder_state", return_value=[item])
    mocker.patch("src.main._utcnow", return_value=_now())

    main.run_once(MagicMock(), cfg)

    assert change_cursor.read(change_cursor.path_for(cfg.data_dir)) == "tok-1"


def test_a_clean_cycle_still_moves_the_cursor(mocker, tmp_path):
    cfg = _one_change_cycle(mocker, tmp_path)
    mocker.patch(
        "src.main.drive.list_folder_state",
        return_value=[_subfolder_item("v1", "a.mp4", "meeting-1")],
    )
    mocker.patch("src.main.booking_gate.resolve", return_value=MATCHED_DECISION)
    mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    assert change_cursor.read(change_cursor.path_for(cfg.data_dir)) == "tok-2"


def test_an_unresolvable_folder_is_not_treated_as_someone_elses(mocker, tmp_path):
    """An expired token during the ancestor lookup used to read as "belongs to
    nobody", and the change was consumed on the strength of that."""
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    mocker.patch(
        "src.main.drive.list_changes",
        return_value=([_change("v1", "meeting-1")], "tok-2"),
    )
    mocker.patch(
        "src.main.drive.find_configured_ancestor",
        side_effect=RuntimeError("drive 502"),
    )
    notify_mock = mocker.patch("src.main.notify.notify_error")

    main.run_once(MagicMock(), cfg)

    assert change_cursor.read(change_cursor.path_for(cfg.data_dir)) == "tok-1"
    notify_mock.assert_called_once()


def test_an_auth_failure_during_the_ancestor_lookup_still_stops_the_cycle(
    mocker, tmp_path
):
    """Auth errors are re-raised everywhere else so the container restarts after
    re-auth; being swallowed here would let a whole cycle report success."""
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    mocker.patch(
        "src.main.drive.list_changes",
        return_value=([_change("v1", "meeting-1")], "tok-2"),
    )
    mocker.patch(
        "src.main.drive.find_configured_ancestor",
        side_effect=RefreshError("token expired"),
    )

    with pytest.raises(RefreshError):
        main.run_once(MagicMock(), cfg)


def test_the_mp3_is_written_into_the_meeting_subfolder(mocker, tmp_path):
    """The requirement the whole migration started from: an artifact belongs beside
    its video. The mp3 upload is its own call site and was the one left pointing at
    the configured folder -- so every artifact landed a level above the recording."""
    cfg = make_config(folders=["root"], output_dir=tmp_path)
    upload_mock = mocker.patch("src.main.drive.upload")
    mp4_path = tmp_path / "a.mp4"
    mp4_path.write_bytes(b"video")
    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=tmp_path / "a.mp3")
    mocker.patch("src.main._run_preset_stage", return_value={})
    mocker.patch("src.main._try_write_call_documents", return_value=None)

    main.process_item(
        MagicMock(),
        _subfolder_item("v1", "a.mp4", "meeting-1"),
        "root",
        cfg,
        booking_decision=MATCHED_DECISION,
    )

    assert upload_mock.call_args.args[2] == "meeting-1"


def test_the_mp3_of_a_flat_folder_still_goes_where_it_always_did(mocker, tmp_path):
    cfg = make_config(folders=["root"], output_dir=tmp_path)
    upload_mock = mocker.patch("src.main.drive.upload")
    mp4_path = tmp_path / "a.mp4"
    mp4_path.write_bytes(b"video")
    mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=tmp_path / "a.mp3")
    mocker.patch("src.main._run_preset_stage", return_value={})
    mocker.patch("src.main._try_write_call_documents", return_value=None)

    main.process_item(
        MagicMock(), _item("v1", "a.mp4"), "root", cfg,
        booking_decision=MATCHED_DECISION,
    )

    assert upload_mock.call_args.args[2] == "root"



# --- Everything that keys off the configured folder must survive a subfolder -------
#
# Telegram, Planfix routing, the employee name and the meta document all resolve
# through `config.folder_by_id`. Hand any of them a meeting subfolder and they get
# None back -- no chat, no forced recognition, no manager -- without raising.


def _telegram_config(tmp_path, chat="-1001234567890"):
    return make_config(
        folders=[EmployeeFolder("root", name="Анжелика", email="a@b.c", telegram=chat)],
        data_dir=tmp_path,
        stt_provider="",
    )


def test_a_folders_telegram_chat_is_found_for_a_video_in_a_subfolder(tmp_path):
    """The chat lives on the configured folder. Looking it up by the meeting
    subfolder returns "" -- which also silently turns off the unconditional
    recognition that having a chat is supposed to mean."""
    cfg = _telegram_config(tmp_path)

    assert main.folder_telegram_chat(cfg, "root") == "-1001234567890"
    assert main.folder_telegram_chat(cfg, "meeting-1") == ""


def test_a_telegram_folder_still_recognises_a_subfolder_recording_without_a_booking(
    mocker, tmp_path
):
    """A folder with a chat is watched for its own sake, so "no booking" must not
    skip it -- and must not mark it unmatched, which would park it for good."""
    cfg = replace(_telegram_config(tmp_path), call_booking_disable_recognition=True)
    mocker.patch(
        "src.main.drive.list_folder_tree_state",
        return_value=[_subfolder_item("v1", "a.mp4", "meeting-1")],
    )
    mocker.patch(
        "src.main.booking_gate.resolve",
        return_value=BookingDecision(state="unmatched", reason="no-booking"),
    )
    mark_mock = mocker.patch("src.main.booking_gate.mark_unmatched")
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    process_mock.assert_called_once()
    mark_mock.assert_not_called()


def test_the_telegram_summary_is_sent_for_a_video_in_a_subfolder(mocker, tmp_path):
    cfg = _telegram_config(tmp_path)
    send_mock = mocker.patch("src.main.notify.send_message", return_value=True)
    mocker.patch("src.main.drive.set_file_app_properties")

    main._send_telegram_summary(
        MagicMock(),
        _subfolder_item("v1", "a.mp4", "meeting-1"),
        "v1",
        "root",
        replace(cfg, telegram_bot_token="bot-token"),
        {"keypoints": "## Задачи"},
        MATCHED_DECISION,
    )

    assert send_mock.call_args.kwargs["chat_id"] == "-1001234567890"


def test_the_planfix_comment_is_sent_for_a_video_in_a_subfolder(mocker, tmp_path):
    """Planfix is addressed by the booking's task id, not by a folder -- this pins
    that the subfolder did not disturb the path to it."""
    cfg = make_config(folders=["root"], data_dir=tmp_path)
    cfg = replace(cfg, planfix_create_comment_url="https://planfix.example/api",
                  planfix_token="t", planfix_presets=("keypoints",))
    send_mock = mocker.patch("src.main.planfix.send_comment", return_value=True)
    mocker.patch("src.main.drive.set_file_app_properties")

    main._send_planfix_comment(
        MagicMock(),
        _subfolder_item("v1", "a.mp4", "meeting-1"),
        "v1",
        cfg,
        {"keypoints": "## Задачи"},
        MATCHED_DECISION,
    )

    assert send_mock.call_args.kwargs["task_id"] == "851030"


def test_name_rules_still_route_a_subfolder_recording(mocker, tmp_path):
    """`name_rules` resolve through `folder_by_id` in the gate. A subfolder id there
    would drop the rule and send the comment to the wrong task -- or to none."""
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    resolve_mock = mocker.patch(
        "src.main.booking_gate.resolve", return_value=MATCHED_DECISION
    )
    mocker.patch(
        "src.main.drive.list_folder_tree_state",
        return_value=[_subfolder_item("v1", "a.mp4", "meeting-1")],
    )
    mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    assert resolve_mock.call_args.args[1] == "root"


def test_the_meta_document_names_the_employee_for_a_subfolder_recording(
    mocker, tmp_path
):
    cfg = make_config(
        folders=[EmployeeFolder("root", name="Анжелика", email="a@b.c")],
        data_dir=tmp_path,
    )
    build_mock = mocker.patch("src.main.meta_doc.build", return_value={})
    mocker.patch("src.main.meta_doc.to_yaml", return_value="")
    mocker.patch("src.main.stt_document.assemble", return_value="")
    mocker.patch("src.main.output.write_artifact")

    main._write_call_documents(
        MagicMock(), "v1", "a.mp4", "root", "meeting-1", "transcript", {},
        cfg, tmp_path, item={}, booking_decision=MATCHED_DECISION,
    )

    assert build_mock.call_args.kwargs["folder_id"] == "root"


# --- A cursor vouches only for the folders it was taken against -------------------
#
# Adding a folder is how an employee gets onboarded, not a one-off migration. The
# recordings already sitting in that folder were never a change after the saved
# cursor, so the feed will never name it: without noticing the config changed, the
# backlog stays invisible until someone resets the cursor by hand.


def test_a_cursor_saved_with_its_folders_reads_the_feed(mocker, tmp_path):
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    tree_mock = mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])
    changes_mock = mocker.patch(
        "src.main.drive.list_changes", return_value=([], "tok-2")
    )

    main.run_once(MagicMock(), cfg)

    changes_mock.assert_called_once_with(mocker.ANY, "tok-1")
    tree_mock.assert_not_called()


def test_adding_a_folder_sweeps_once_instead_of_trusting_the_feed(mocker, tmp_path):
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    grown = make_config(folders=["root", "new"], data_dir=tmp_path, stt_provider="")
    tree_mock = mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])
    changes_mock = mocker.patch("src.main.drive.list_changes")
    token_mock = mocker.patch(
        "src.main.drive.get_start_page_token", return_value="tok-fresh"
    )

    main.run_once(MagicMock(), grown)

    changes_mock.assert_not_called()
    assert tree_mock.call_count == 2
    token_mock.assert_called_once()
    assert change_cursor.read(_cursor_file(grown)) == "tok-fresh"


def test_dropping_a_folder_also_sweeps_once(mocker, tmp_path):
    """Not because shrinking misses anything, but because the pair is one identity:
    treating a subset as covered would be a second rule, with its own way to be
    wrong."""
    cfg = make_config(folders=["root", "second"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    shrunk = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    tree_mock = mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])
    changes_mock = mocker.patch("src.main.drive.list_changes")
    mocker.patch("src.main.drive.get_start_page_token", return_value="tok-fresh")

    main.run_once(MagicMock(), shrunk)

    changes_mock.assert_not_called()
    tree_mock.assert_called_once()


def test_a_cursor_with_no_recorded_folders_sweeps_once(mocker, tmp_path):
    """An instance that predates the folder file, or one whose file is unreadable.
    The cursor cannot be vouched for, and one sweep is the whole cost of finding
    out -- after which the pair is written and the feed takes over again."""
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    change_cursor.write(change_cursor.path_for(cfg.data_dir), "tok-1")
    tree_mock = mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])
    changes_mock = mocker.patch("src.main.drive.list_changes")
    mocker.patch("src.main.drive.get_start_page_token", return_value="tok-fresh")

    main.run_once(MagicMock(), cfg)

    changes_mock.assert_not_called()
    tree_mock.assert_called_once()
    assert change_cursor.read_folders(
        change_cursor.folders_path_for(cfg.data_dir)
    ) == change_cursor.fingerprint(["root"])


def test_the_sweep_records_the_folders_it_covered(mocker, tmp_path):
    cfg = make_config(folders=["root", "second"], data_dir=tmp_path, stt_provider="")
    mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])
    mocker.patch("src.main.drive.get_start_page_token", return_value="tok-1")

    main.run_once(MagicMock(), cfg)

    assert change_cursor.read_folders(
        change_cursor.folders_path_for(cfg.data_dir)
    ) == change_cursor.fingerprint(["root", "second"])


def test_a_cycle_that_could_not_list_records_no_folder_set(mocker, tmp_path):
    """The same cycle_drained guard the cursor has, and it is what makes editing the
    config before the folder is actually shared safe: that listing fails, which
    counts as a folder error, which leaves both files alone until the share lands."""
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    mocker.patch("src.main.drive.get_start_page_token", return_value="tok-1")
    mocker.patch(
        "src.main.drive.list_folder_tree_state",
        side_effect=RuntimeError("not shared yet"),
    )
    mocker.patch("src.main.time.sleep")
    mocker.patch("src.main.notify.notify_error")

    main.run_once(MagicMock(), cfg)

    assert change_cursor.read(_cursor_file(cfg)) is None
    assert (
        change_cursor.read_folders(change_cursor.folders_path_for(cfg.data_dir))
        is None
    )


def test_changes_mode_refuses_when_the_folder_set_changed(mocker, tmp_path):
    """Reading the feed anyway would be worse than useless. The cycle would find
    nothing pending, drain, and record the new folder set as vouched for without it
    ever having been swept -- so the backlog it cannot see would stay invisible for
    good, and the operator would have been told to run a cycle that no longer helps."""
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    grown = make_config(folders=["root", "new"], data_dir=tmp_path, stt_provider="")
    changes_mock = mocker.patch(
        "src.main.drive.list_changes", return_value=([], "tok-2")
    )
    tree_mock = mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])

    with pytest.raises(SystemExit, match="watched folders changed"):
        main.run_once(MagicMock(), grown, mode="changes")

    changes_mock.assert_not_called()
    tree_mock.assert_not_called()
    assert change_cursor.read(_cursor_file(grown)) == "tok-1"
    assert change_cursor.read_folders(
        change_cursor.folders_path_for(grown.data_dir)
    ) == change_cursor.fingerprint(["root"])


def test_a_walk_cycle_leaves_the_folder_set_alone(mocker, tmp_path):
    """Walk is a look, not a new starting point: it must not become one by quietly
    vouching for a config the feed was never asked about."""
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    grown = make_config(folders=["root", "new"], data_dir=tmp_path, stt_provider="")
    mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])

    main.run_once(MagicMock(), grown, mode="walk")

    assert change_cursor.read(_cursor_file(grown)) == "tok-1"
    assert change_cursor.read_folders(
        change_cursor.folders_path_for(grown.data_dir)
    ) == change_cursor.fingerprint(["root"])


# --- The polling loop must be able to run without the feed ------------------------


def test_the_service_loop_takes_the_configured_discovery_path(mocker):
    """Mode is a CLI flag. Without passing the config through, the daemon could only
    ever run auto -- and the one assumption still unproven about the feed (folders
    shared *to* the service rather than owned by it) would have no switch."""
    cfg = replace(make_config(folders=["f1"], poll_interval=1), run_discovery="walk")
    mocker.patch("src.main.load_config", return_value=cfg)
    mocker.patch("src.main.build_drive_service", return_value=MagicMock())
    modes = []

    def fake_run_once(svc, c, **kwargs):
        modes.append(kwargs.get("mode"))
        raise KeyboardInterrupt

    mocker.patch("src.main.run_once", side_effect=fake_run_once)
    mocker.patch("src.main.time.sleep")

    with pytest.raises(KeyboardInterrupt):
        main.main()

    assert modes == ["walk"]


# --- `since`: leaving a backlog alone without leaving it half-remembered ----------
#
# A folder shared to the service arrives with everything the person ever recorded.
# The cutoff is a scope rule, not a record of work: nothing is written to Drive, so
# moving the date back brings the backlog straight back into scope.


def _dated_item(file_id, name, *, created=None, media_metadata=True):
    item = _item(file_id, name)
    if created is not None:
        item["file"]["createdTime"] = created
    item["has_media_metadata"] = media_metadata
    return item


# A room-code recording: Meet puts the meeting time in the name.
OLD_CALL = "exf-wxzm-uzk (2026-09-09 17_42 GMT+2).mp4"
NEW_CALL = "exf-wxzm-uzk (2026-11-20 17_42 GMT+2).mp4"


def test_a_recording_older_than_since_is_left_alone(mocker, tmp_path):
    cfg = replace(
        make_config(folders=["root"], data_dir=tmp_path, stt_provider=""),
        run_since="2026-10-01",
    )
    mocker.patch(
        "src.main.drive.list_folder_tree_state",
        return_value=[_dated_item("v1", OLD_CALL), _dated_item("v2", NEW_CALL)],
    )
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    assert process_mock.call_count == 1
    assert process_mock.call_args.args[1]["file"]["name"] == NEW_CALL


def test_the_meeting_time_in_the_name_beats_when_drive_received_it(mocker, tmp_path):
    """The two answer different questions. These examples were recorded on the 9th
    and re-uploaded on the 12th, which reset `createdTime` by three days; real Meet
    lag is a couple of hours, which still carries a late-evening call into the next
    day. Either way "calls from the 10th" has to mean the call."""
    cfg = replace(
        make_config(folders=["root"], data_dir=tmp_path, stt_provider=""),
        run_since="2026-09-10",
    )
    mocker.patch(
        "src.main.drive.list_folder_tree_state",
        return_value=[
            _dated_item("v1", OLD_CALL, created="2026-09-12T05:54:48.536Z")
        ],
    )
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    process_mock.assert_not_called()


def test_a_name_without_a_time_falls_back_to_when_drive_received_it(mocker, tmp_path):
    cfg = replace(
        make_config(folders=["root"], data_dir=tmp_path, stt_provider=""),
        run_since="2026-10-01",
    )
    mocker.patch(
        "src.main.drive.list_folder_tree_state",
        return_value=[
            _dated_item("v1", "hand-renamed.mp4", created="2026-08-01T10:00:00Z")
        ],
    )
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    process_mock.assert_not_called()


def test_a_recording_nobody_can_date_stays_in_scope(mocker, tmp_path):
    """Fail open. Dropping a recording because its date is unreadable would be a
    silent loss, which is the failure this whole area exists to remove."""
    cfg = replace(
        make_config(folders=["root"], data_dir=tmp_path, stt_provider=""),
        run_since="2026-10-01",
    )
    mocker.patch(
        "src.main.drive.list_folder_tree_state",
        return_value=[_dated_item("v1", "hand-renamed.mp4")],
    )
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    process_mock.assert_called_once()


def test_an_out_of_scope_recording_does_not_hold_the_changes_cursor(mocker, tmp_path):
    """The one that makes this safe. An old recording Drive never finished
    processing would otherwise count as deferred, and deferred holds the cursor -- so
    the backlog an operator asked to ignore would freeze the feed instead of being
    ignored. It is a permanent skip by design, like one over `--max-size`."""
    cfg = replace(
        make_config(folders=["root"], data_dir=tmp_path, stt_provider=""),
        run_since="2026-10-01",
    )
    mocker.patch(
        "src.main.drive.list_folder_tree_state",
        return_value=[_dated_item("v1", OLD_CALL, media_metadata=False)],
    )
    mocker.patch("src.main.drive.get_start_page_token", return_value="tok-1")
    mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    assert change_cursor.read(_cursor_file(cfg)) == "tok-1"


def test_a_folders_own_since_overrides_the_global_one(mocker, tmp_path):
    """Onboarding is an event about a person: whoever joins in three months brings a
    backlog of their own, and one global date cannot be right for both."""
    cfg = replace(
        make_config(
            folders=[
                EmployeeFolder("early", since=""),
                EmployeeFolder("late", since="2026-12-01"),
            ],
            data_dir=tmp_path,
            stt_provider="",
        ),
        run_since="2026-09-01",
    )
    mocker.patch(
        "src.main.drive.list_folder_tree_state",
        return_value=[_dated_item("v1", NEW_CALL)],
    )
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    # The November call is in scope for the folder on the September default and out
    # of scope for the one that only starts in December.
    assert [c.args[2] for c in process_mock.call_args_list] == ["early"]


def test_the_since_flag_overrides_every_configured_cutoff(mocker, tmp_path):
    cfg = replace(
        make_config(
            folders=[EmployeeFolder("root", since="2026-12-01")],
            data_dir=tmp_path,
            stt_provider="",
        ),
        run_since="2026-12-01",
    )
    mocker.patch(
        "src.main.drive.list_folder_tree_state",
        return_value=[_dated_item("v1", OLD_CALL)],
    )
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg, since="2026-01-01")

    process_mock.assert_called_once()


def test_a_dry_run_names_what_the_cutoff_leaves_out(mocker, tmp_path, caplog):
    """Counted per folder in a real cycle -- a year of history would print itself
    every ten minutes -- but named here, because this is where an operator looks to
    find out what a date is about to do."""
    cfg = replace(
        make_config(folders=["root"], data_dir=tmp_path, stt_provider=""),
        run_since="2026-10-01",
    )
    mocker.patch(
        "src.main.drive.list_folder_tree_state",
        return_value=[_dated_item("v1", OLD_CALL)],
    )

    with caplog.at_level(logging.INFO):
        main.run_once(MagicMock(), cfg, dry_run=True)

    assert "not in scope" in caplog.text
    assert OLD_CALL in caplog.text
    assert "skipped_old=1" in caplog.text


# --- Found by the live emulation, not by reading the code ------------------------


def _drive_400(location):
    """Drive's own error body, as captured live for a malformed page token."""
    body = (
        '{"error": {"code": 400, "message": "Invalid Value", "errors": '
        '[{"reason": "invalid", "location": "%s", "locationType": "parameter"}]}}'
    ) % location
    return HttpError(MagicMock(status=400), body.encode("utf-8"))


def test_a_malformed_cursor_is_swept_over_instead_of_failing_forever(mocker, tmp_path):
    """Drive answers a corrupt page token with 400, not 404/410. Read as an ordinary
    feed failure it held the cursor -- and a held cursor is the same bad token next
    cycle, so the service failed every cycle for good while the cursor module said a
    corrupt cursor costs one sweep. Reproduced live with `not-a-token` and `0`."""
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "not-a-token")
    mocker.patch(
        "src.main.drive.list_changes", side_effect=_drive_400("pageToken")
    )
    tree_mock = mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])
    mocker.patch("src.main.drive.get_start_page_token", return_value="tok-fresh")
    notify_mock = mocker.patch("src.main.notify.notify_error")

    main.run_once(MagicMock(), cfg)

    tree_mock.assert_called_once()
    assert change_cursor.read(_cursor_file(cfg)) == "tok-fresh"
    notify_mock.assert_not_called()


def test_a_400_about_anything_but_the_cursor_is_still_a_failure(mocker, tmp_path):
    """The match is on the parameter, not on 400: a request broken some other way
    must surface, not be quietly swept over every ten minutes."""
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    mocker.patch("src.main.drive.list_changes", side_effect=_drive_400("fields"))
    tree_mock = mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])
    notify_mock = mocker.patch("src.main.notify.notify_error")

    main.run_once(MagicMock(), cfg)

    tree_mock.assert_not_called()
    notify_mock.assert_called_once()
    assert change_cursor.read(_cursor_file(cfg)) == "tok-1"


def test_a_400_without_a_readable_body_is_not_mistaken_for_a_bad_cursor(
    mocker, tmp_path
):
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    mocker.patch("src.main.drive.list_changes", side_effect=_http_error(400))
    tree_mock = mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])
    mocker.patch("src.main.notify.notify_error")

    main.run_once(MagicMock(), cfg)

    tree_mock.assert_not_called()
    assert change_cursor.read(_cursor_file(cfg)) == "tok-1"


def test_one_configured_folder_is_one_listing_however_many_meetings_changed(
    mocker, tmp_path
):
    """Found live: an old cursor made the feed name three meetings under one
    employee, and the log reported that employee's folder three times over."""
    cfg = make_config(folders=["root"], data_dir=tmp_path, stt_provider="")
    _save_cursor(cfg, "tok-1")
    mocker.patch(
        "src.main.drive.list_changes",
        return_value=(
            [
                _change("v1", "meeting-1"),
                _change("v2", "meeting-2"),
                _change("v3", "meeting-3"),
            ],
            "tok-2",
        ),
    )
    mocker.patch("src.main.drive.find_configured_ancestor", return_value="root")
    mocker.patch(
        "src.main.drive.list_folder_state",
        side_effect=lambda _service, container: [
            _subfolder_item(f"v-{container}", f"{container}.mp4", container)
        ],
    )

    found = main._discover(MagicMock(), cfg)

    assert [folder_id for folder_id, _ in found.listings] == ["root"]
    items = found.listings[0][1]
    assert sorted(item["container_id"] for item in items) == [
        "meeting-1", "meeting-2", "meeting-3",
    ]


def test_a_recording_that_always_fails_holds_the_cursor_every_cycle(mocker, tmp_path):
    """A trade-off, pinned so that it is visible rather than rediscovered.

    Holding the cursor is what keeps a failed recording from being lost; the price is
    that a recording which can never succeed -- a corrupt upload, say -- holds it on
    every cycle and is retried on every cycle, while the feed re-reads a tail that
    grows until Drive expires the token and a sweep takes a fresh one. New recordings
    still flow, because each cycle reads the changes after the held point too. Nothing
    caps the retries today; if that ever changes, this test is where it shows."""
    cfg = _one_change_cycle(mocker, tmp_path)
    mocker.patch(
        "src.main.drive.list_folder_state",
        return_value=[_subfolder_item("v1", "a.mp4", "meeting-1")],
    )
    mocker.patch("src.main.booking_gate.resolve", return_value=MATCHED_DECISION)
    process_mock = mocker.patch(
        "src.main.process_item", side_effect=RuntimeError("not a video")
    )
    mocker.patch("src.main.notify.notify_error")

    main.run_once(MagicMock(), cfg)
    main.run_once(MagicMock(), cfg)

    assert process_mock.call_count == 2
    assert change_cursor.read(change_cursor.path_for(cfg.data_dir)) == "tok-1"


def test_a_recording_drive_never_processes_is_held_only_within_the_grace(
    mocker, tmp_path
):
    """Reproduced live with an upload that is not a video, so Drive never fills
    `videoMediaMetadata`: inside the grace it is deferred and holds the cursor; past
    it, it is treated as ready rather than waited on for ever."""
    fresh = _item("v1", "a.mp4")
    fresh["has_media_metadata"] = False
    stale = _item("v2", "b.mp4")
    stale["has_media_metadata"] = False
    now = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)
    fresh["file"]["createdTime"] = (now - timedelta(minutes=5)).isoformat()
    stale["file"]["createdTime"] = (now - timedelta(hours=3)).isoformat()

    assert main._is_still_settling(fresh, now) is True
    assert main._is_still_settling(stale, now) is False


# --- Attended calls: a shortcut to a recording -------------------------------------
#
# Meet gives the organizer the recording and every attendee a shortcut to it, whatever
# anyone's access. A shortcut whose recording opens is followed -- unless the
# organizer's folder is configured too, in which case that folder does the call and
# the shortcut would only do it a second time.

SHORTCUT_MIME = "application/vnd.google-apps.shortcut"


def _shortcut_item(file_id, name, container_id, *, media_id, target_parents, **kwargs):
    item = _subfolder_item(file_id, name, container_id, **kwargs)
    item["file"]["mimeType"] = SHORTCUT_MIME
    item["media_id"] = media_id
    item["target_parents"] = target_parents
    return item


def _organizer_is(configured_id, *, below):
    """``find_configured_ancestor`` for a Drive where only ``below`` sits under a
    configured folder."""
    def resolve(service, container_id, configured_ids, cache=None):
        return configured_id if container_id == below else None
    return resolve


def test_a_followed_shortcut_downloads_the_recording_but_keeps_artifacts_beside_itself(
    mocker, tmp_path
):
    """The bytes are the organizer's; the artifacts and their pairing belong to the
    attendee's meeting folder, where the shortcut is -- the organizer's folder may not
    even be writable."""
    service = MagicMock()
    mp4_path = tmp_path / "call.mp4"
    mp3_path = tmp_path / "call.mp3"
    download_mock = mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_mp3", return_value=mp3_path)
    upload_mock = mocker.patch("src.main.drive.upload", return_value={"id": "u1"})
    item = _shortcut_item(
        "sc1", "call.mp4", "attended",
        media_id="organizers-video", target_parents=["org-meeting"],
    )

    main.process_item(service, item, "attendee", make_config(bitrate="128k"))

    assert download_mock.call_args.args[1] == "organizers-video"
    assert upload_mock.call_args.args[2] == "attended"
    assert upload_mock.call_args.kwargs["app_properties"]["source_video_id"] == "sc1"


def test_a_call_whose_organizer_is_configured_is_processed_once_from_that_folder(
    mocker, tmp_path
):
    """Both employees are watched: the organizer holds the recording, the attendee a
    shortcut to it. Following the shortcut too would pay for the call twice and post
    its summary twice."""
    cfg = make_config(folders=["attendee", "organizer"], data_dir=tmp_path)
    shortcut = _shortcut_item(
        "sc1", "call.mp4", "attended",
        media_id="v-org", target_parents=["org-meeting"],
    )
    real = _subfolder_item("v-org", "call.mp4", "org-meeting")
    mocker.patch(
        "src.main.drive.list_folder_tree_state",
        side_effect=lambda service, folder_id: {
            "attendee": [shortcut], "organizer": [real],
        }[folder_id],
    )
    mocker.patch(
        "src.main.drive.find_configured_ancestor",
        side_effect=_organizer_is("organizer", below="org-meeting"),
    )
    mocker.patch("src.main.booking_gate.resolve", return_value=MATCHED_DECISION)
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    processed = [call.args[1]["file"]["id"] for call in process_mock.call_args_list]
    assert processed == ["v-org"]


def test_a_call_organized_outside_the_watched_folders_is_processed_from_the_shortcut(
    mocker, tmp_path
):
    """A client's call, or a colleague nobody configured: the shortcut is the only way
    in, and the call belongs to the attendee whose folder it is in."""
    cfg = make_config(folders=["attendee"], data_dir=tmp_path)
    shortcut = _shortcut_item(
        "sc1", "call.mp4", "attended",
        media_id="clients-video", target_parents=["clients-meeting"],
    )
    mocker.patch("src.main.drive.list_folder_tree_state", return_value=[shortcut])
    mocker.patch("src.main.drive.find_configured_ancestor", return_value=None)
    mocker.patch("src.main.booking_gate.resolve", return_value=MATCHED_DECISION)
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    process_mock.assert_called_once()
    assert process_mock.call_args.args[1] is shortcut
    assert process_mock.call_args.args[2] == "attendee"


def test_a_recording_folder_this_account_cannot_climb_is_not_a_configured_one(
    mocker, tmp_path
):
    """Every configured folder is readable, and so is everything inside it. A 404 on
    the way up therefore means "not ours", and failing toward processing keeps the call
    from vanishing."""
    cfg = make_config(folders=["attendee"], data_dir=tmp_path)
    shortcut = _shortcut_item(
        "sc1", "call.mp4", "attended",
        media_id="clients-video", target_parents=["clients-meeting"],
    )
    mocker.patch("src.main.drive.list_folder_tree_state", return_value=[shortcut])
    mocker.patch(
        "src.main.drive.find_configured_ancestor", side_effect=_http_error(404)
    )
    mocker.patch("src.main.booking_gate.resolve", return_value=MATCHED_DECISION)
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    process_mock.assert_called_once()


def test_not_knowing_whose_call_it_is_counts_as_a_listing_failure(mocker, tmp_path):
    """An outage must not read as "the organizer is not configured" -- that would
    process a call its organizer's folder is about to process too."""
    cfg = make_config(folders=["attendee"], data_dir=tmp_path)
    shortcut = _shortcut_item(
        "sc1", "call.mp4", "attended",
        media_id="v-org", target_parents=["org-meeting"],
    )
    mocker.patch("src.main.drive.list_folder_tree_state", return_value=[shortcut])
    mocker.patch(
        "src.main.drive.find_configured_ancestor",
        side_effect=RuntimeError("drive is down"),
    )
    notify_mock = mocker.patch("src.main.notify.notify_error")
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    process_mock.assert_not_called()
    notify_mock.assert_called_once()


def test_the_feed_names_the_meeting_folder_of_a_new_shortcut_to_a_recording(
    mocker, tmp_path
):
    """An attended call reaches the attendee's folder as a shortcut, never as an mp4;
    a feed that only let `video/mp4` through would never look at it."""
    cfg = make_config(folders=["root"], data_dir=tmp_path)
    _save_cursor(cfg, "tok-1")
    entry = _change("sc1", "meeting-1", mime=SHORTCUT_MIME)
    entry["file"]["shortcutDetails"] = {"targetMimeType": "video/mp4"}
    mocker.patch("src.main.drive.list_changes", return_value=([entry], "tok-2"))
    mocker.patch("src.main.drive.find_configured_ancestor", return_value="root")
    listing = mocker.patch("src.main.drive.list_folder_state", return_value=[])

    main.run_once(MagicMock(), cfg)

    listing.assert_called_once_with(mocker.ANY, "meeting-1")


def test_the_feed_ignores_a_shortcut_to_anything_but_a_recording(mocker, tmp_path):
    cfg = make_config(folders=["root"], data_dir=tmp_path)
    _save_cursor(cfg, "tok-1")
    entry = _change("sc-doc", "meeting-1", mime=SHORTCUT_MIME)
    entry["file"]["shortcutDetails"] = {
        "targetMimeType": "application/vnd.google-apps.document"
    }
    mocker.patch("src.main.drive.list_changes", return_value=([entry], "tok-2"))
    mocker.patch("src.main.drive.find_configured_ancestor", return_value="root")
    listing = mocker.patch("src.main.drive.list_folder_state", return_value=[])

    main.run_once(MagicMock(), cfg)

    listing.assert_not_called()


def test_an_attended_recording_drive_is_still_processing_holds_the_cursor(
    mocker, tmp_path
):
    """The recording finishes in the organizer's Drive, so the feed never reports that
    under this folder. Holding the cursor is what brings the shortcut's change back
    next cycle; stepping past it would lose the call for good."""
    cfg = _one_change_cycle(mocker, tmp_path)
    shortcut = _shortcut_item(
        "sc1", "call.mp4", "meeting-1", media_id="v-org", target_parents=[],
    )
    shortcut["has_media_metadata"] = False
    shortcut["file"]["createdTime"] = (
        datetime.now(timezone.utc) - timedelta(minutes=5)
    ).isoformat()
    mocker.patch("src.main.drive.list_folder_state", return_value=[shortcut])
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    process_mock.assert_not_called()
    assert change_cursor.read(change_cursor.path_for(cfg.data_dir)) == "tok-1"


def test_processing_a_shortcut_by_id_follows_it(mocker, tmp_path):
    cfg = make_config(folders=["root"], data_dir=tmp_path)
    mocker.patch(
        "src.main.drive.get_file_metadata",
        return_value={"id": "sc1", "mimeType": SHORTCUT_MIME, "parents": ["attended"]},
    )
    mocker.patch(
        "src.main.drive.find_configured_ancestor",
        side_effect=_organizer_is("root", below="attended"),
    )
    shortcut = _shortcut_item(
        "sc1", "call.mp4", "attended",
        media_id="clients-video", target_parents=["clients-meeting"],
    )
    mocker.patch("src.main.drive.list_folder_state", return_value=[shortcut])
    process_mock = mocker.patch("src.main.process_item")

    main.process_target(MagicMock(), "sc1", cfg)

    process_mock.assert_called_once()
    assert process_mock.call_args.args[2] == "root"


def test_processing_a_shortcut_the_organizers_folder_covers_says_so(mocker, tmp_path):
    """Being told "not an MP4" would send an operator looking for a broken file."""
    cfg = make_config(folders=["attendee", "organizer"], data_dir=tmp_path)
    mocker.patch(
        "src.main.drive.get_file_metadata",
        return_value={"id": "sc1", "mimeType": SHORTCUT_MIME, "parents": ["attended"]},
    )

    def resolve(service, container_id, configured_ids, cache=None):
        return {"attended": "attendee", "org-meeting": "organizer"}.get(container_id)

    mocker.patch("src.main.drive.find_configured_ancestor", side_effect=resolve)
    shortcut = _shortcut_item(
        "sc1", "call.mp4", "attended",
        media_id="v-org", target_parents=["org-meeting"],
    )
    mocker.patch("src.main.drive.list_folder_state", return_value=[shortcut])
    process_mock = mocker.patch("src.main.process_item")

    with pytest.raises(RuntimeError, match="configured folder"):
        main.process_target(MagicMock(), "sc1", cfg)

    process_mock.assert_not_called()


def test_the_call_document_names_the_shortcut_like_every_other_id_of_the_call(
    mocker, tmp_path
):
    """One id per call wherever an operator can copy it from: the meta document, the
    logs, `gdstt reprocess`. Drive's own view link for a shortcut is
    `/file/d/<shortcut id>/view`, and the organizer's id would send `reprocess` to a
    folder with no employee and, likely, no write access."""
    build_mock = mocker.patch("src.main.meta_doc.build", return_value={})
    mocker.patch("src.main.meta_doc.to_yaml", return_value="")
    mocker.patch("src.main.stt_document.assemble", return_value="")
    mocker.patch("src.main.output.write_artifact")
    item = _shortcut_item(
        "sc1", "call.mp4", "attended",
        media_id="organizers-video", target_parents=["org-meeting"],
    )

    main._write_call_documents(
        MagicMock(), "sc1", "call.mp4", "attendee", "attended", "text", {},
        make_config(), tmp_path, item=item, booking_decision=MATCHED_DECISION,
    )

    assert build_mock.call_args.kwargs["file_id"] == "sc1"


def test_a_recording_folder_drive_refuses_with_403_is_not_a_configured_one(
    mocker, tmp_path
):
    cfg = make_config(folders=["attendee"], data_dir=tmp_path)
    shortcut = _shortcut_item(
        "sc1", "call.mp4", "attended",
        media_id="clients-video", target_parents=["clients-meeting"],
    )
    mocker.patch("src.main.drive.list_folder_tree_state", return_value=[shortcut])
    mocker.patch(
        "src.main.drive.find_configured_ancestor", side_effect=_http_error(403)
    )
    mocker.patch("src.main.booking_gate.resolve", return_value=MATCHED_DECISION)
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    process_mock.assert_called_once()


def test_a_drive_error_deciding_whose_call_it_is_is_retried_then_held(
    mocker, tmp_path
):
    """A 500 is not "no access": it is retried like any listing, and if it persists the
    folder counts as unlisted -- the call is neither processed nor forgotten."""
    cfg = make_config(folders=["attendee"], data_dir=tmp_path)
    shortcut = _shortcut_item(
        "sc1", "call.mp4", "attended",
        media_id="v-org", target_parents=["org-meeting"],
    )
    tree_mock = mocker.patch(
        "src.main.drive.list_folder_tree_state", return_value=[shortcut]
    )
    mocker.patch(
        "src.main.drive.find_configured_ancestor", side_effect=_http_error(500)
    )
    mocker.patch("src.main.time.sleep")
    notify_mock = mocker.patch("src.main.notify.notify_error")
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    process_mock.assert_not_called()
    notify_mock.assert_called_once()
    assert tree_mock.call_count > 1


def test_a_shortcut_to_the_employees_own_recording_is_not_processed_twice(
    mocker, tmp_path
):
    """Someone adding a shortcut to their own call inside their own Meet folder: the
    recording is under the same configured folder, so only the recording counts."""
    cfg = make_config(folders=["root"], data_dir=tmp_path)
    real = _subfolder_item("v1", "call.mp4", "meeting-1")
    shortcut = _shortcut_item(
        "sc1", "call.mp4", "meeting-2", media_id="v1", target_parents=["meeting-1"],
    )
    mocker.patch(
        "src.main.drive.list_folder_tree_state", return_value=[real, shortcut]
    )
    mocker.patch(
        "src.main.drive.find_configured_ancestor",
        side_effect=_organizer_is("root", below="meeting-1"),
    )
    mocker.patch("src.main.booking_gate.resolve", return_value=MATCHED_DECISION)
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    processed = [call.args[1]["file"]["id"] for call in process_mock.call_args_list]
    assert processed == ["v1"]


def test_a_shortcut_done_here_is_not_asked_whose_call_it_is(mocker, tmp_path):
    """No parents were looked up, so there is nothing to climb -- and a call already
    processed here must not start costing ancestor lookups."""
    cfg = make_config(folders=["attendee"], data_dir=tmp_path)
    shortcut = _shortcut_item(
        "sc1", "call.mp4", "attended",
        media_id="v-org", target_parents=None, has_mp3=True,
    )
    mocker.patch("src.main.drive.list_folder_tree_state", return_value=[shortcut])
    ancestor_mock = mocker.patch("src.main.drive.find_configured_ancestor")
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    ancestor_mock.assert_not_called()
    process_mock.assert_not_called()


def test_the_feed_leaves_a_call_to_the_organizers_configured_folder(mocker, tmp_path):
    cfg = make_config(folders=["attendee", "organizer"], data_dir=tmp_path)
    _save_cursor(cfg, "tok-1")
    entry = _change("sc1", "attended", mime=SHORTCUT_MIME)
    entry["file"]["shortcutDetails"] = {"targetMimeType": "video/mp4"}
    mocker.patch("src.main.drive.list_changes", return_value=([entry], "tok-2"))

    def resolve(service, container_id, configured_ids, cache=None):
        return {"attended": "attendee", "org-meeting": "organizer"}.get(container_id)

    mocker.patch("src.main.drive.find_configured_ancestor", side_effect=resolve)
    shortcut = _shortcut_item(
        "sc1", "call.mp4", "attended",
        media_id="v-org", target_parents=["org-meeting"],
    )
    mocker.patch("src.main.drive.list_folder_state", return_value=[shortcut])
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    process_mock.assert_not_called()
    # A permanent skip by design, so it must not hold the cursor.
    assert change_cursor.read(change_cursor.path_for(cfg.data_dir)) == "tok-2"


def test_processing_a_folder_by_id_leaves_calls_to_the_organizers_folder(
    mocker, tmp_path
):
    cfg = make_config(folders=["attendee", "organizer"], data_dir=tmp_path)
    mocker.patch(
        "src.main.drive.get_file_metadata",
        return_value={"id": "attendee", "mimeType": "application/vnd.google-apps.folder"},
    )

    def resolve(service, container_id, configured_ids, cache=None):
        return {"attendee": "attendee", "org-meeting": "organizer"}.get(container_id)

    mocker.patch("src.main.drive.find_configured_ancestor", side_effect=resolve)
    internal = _shortcut_item(
        "sc1", "internal.mp4", "attended",
        media_id="v-org", target_parents=["org-meeting"],
    )
    external = _shortcut_item(
        "sc2", "client.mp4", "attended",
        media_id="v-client", target_parents=["clients-meeting"],
    )
    mocker.patch(
        "src.main.drive.list_folder_tree_state", return_value=[internal, external]
    )
    process_mock = mocker.patch("src.main.process_item")

    main.process_target(MagicMock(), "attendee", cfg)

    processed = [call.args[1]["file"]["id"] for call in process_mock.call_args_list]
    assert processed == ["sc2"]


def test_a_followed_shortcut_is_transcribed_from_the_recording_and_delivered_as_itself(
    mocker, tmp_path
):
    """The path production takes -- no mp3 artifact, straight to the transcript --
    downloads too, and every marker written after delivery must land on the shortcut:
    the organizer's file is not ours to mark, and may not even be writable."""
    service = MagicMock()
    mp4_path = tmp_path / "call.mp4"
    download_mock = mocker.patch("src.main.drive.download", return_value=mp4_path)
    mocker.patch("src.main.extract_m4a_copy", return_value=tmp_path / "call.m4a")
    mocker.patch("src.main.transcribe_file", return_value="hello world")
    mocker.patch("src.main.drive.upload", return_value={"id": "u1"})
    planfix_mock = mocker.patch("src.main._send_planfix_comment")
    telegram_mock = mocker.patch("src.main._send_telegram_summary")
    item = _shortcut_item(
        "sc1", "call.mp4", "attended",
        media_id="organizers-video", target_parents=["org-meeting"],
    )
    cfg = make_config(
        stt_provider="deepgram", deepgram_api_key="dg-x", drive_mp3_artifact=False
    )

    main.process_item(service, item, "attendee", cfg, booking_decision=MATCHED_DECISION)

    download_mock.assert_called_once()
    assert download_mock.call_args.args[1] == "organizers-video"
    assert planfix_mock.call_args.args[2] == "sc1"
    assert telegram_mock.call_args.args[2] == "sc1"


def test_an_unmatched_attended_call_is_marked_on_its_shortcut(monkeypatch, gate_config):
    monkeypatch.setattr(main.booking_server, "is_running", lambda: True)
    patch_decision(monkeypatch, UNMATCHED_DECISION)
    shortcut = gate_item("sc1")
    shortcut["file"]["mimeType"] = SHORTCUT_MIME
    shortcut["media_id"] = "organizers-video"
    shortcut["target_parents"] = []
    patch_folder_items(monkeypatch, [shortcut])
    marked = []
    monkeypatch.setattr(
        main.booking_gate, "mark_unmatched", lambda svc, fid: marked.append(fid)
    )
    monkeypatch.setattr(main, "process_item", MagicMock())

    main.run_once(MagicMock(), gate_config)

    assert marked == ["sc1"]


def test_a_recording_shared_without_its_folder_is_processed_from_the_shortcut(
    mocker, tmp_path
):
    """A client can share just the recording. There is then no folder above it to
    climb, and "no parents" must read as "not ours", not as an error."""
    cfg = make_config(folders=["attendee"], data_dir=tmp_path)
    shortcut = _shortcut_item(
        "sc1", "call.mp4", "attended", media_id="clients-video", target_parents=[],
    )
    mocker.patch("src.main.drive.list_folder_tree_state", return_value=[shortcut])
    ancestor_mock = mocker.patch("src.main.drive.find_configured_ancestor")
    mocker.patch("src.main.booking_gate.resolve", return_value=MATCHED_DECISION)
    process_mock = mocker.patch("src.main.process_item")

    main.run_once(MagicMock(), cfg)

    process_mock.assert_called_once()
    ancestor_mock.assert_not_called()
