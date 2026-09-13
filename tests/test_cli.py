from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
import yaml

from src import change_cursor, cli
from src.call_booking import CallBooking, append
from src.config import EmployeeFolder
from src.presets import Preset
from tests.test_main import make_config


@dataclass
class _Telemetry:
    cost_usd: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)


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



def _normalized_help(text: str) -> str:
    return " ".join(text.split())


def write_cli_config(tmp_path, **sections):
    """Write ``sections`` as ``<tmp_path>/config.yml`` and return its path.

    Bare ``{}`` already loads (``validate_providers=False``), so a caller only
    supplies the top-level keys (e.g. ``call_booking={...}``) it needs for the
    scenario under test.
    """
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(dict(sections), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def test_build_parser_requires_subcommand():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        cli.main(["bogus"])


def test_configure_console_encoding_uses_utf8_for_text_streams():
    stdout = MagicMock()
    stderr = MagicMock()

    cli._configure_console_encoding(stdout=stdout, stderr=stderr)

    stdout.reconfigure.assert_called_once_with(encoding="utf-8", errors="replace")
    stderr.reconfigure.assert_called_once_with(encoding="utf-8", errors="replace")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("100", 100),
        ("50MB", 50_000_000),
        ("1.5GiB", 1_610_612_736),
    ],
)
def test_parse_size(raw, expected):
    assert cli._parse_size(raw) == expected


def test_parse_size_rejects_unknown_unit():
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_size("50xb")


def test_parse_size_rejects_zero():
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_size("0")


def test_parse_size_rejects_zero_with_unit():
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_size("0MB")


def test_process_max_size_zero_rejected(mocker, tmp_path, capsys):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    target_mock = mocker.patch("src.cli.main_module.process_target")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["process", "file123", "--max-size", "0"])

    assert excinfo.value.code == 2
    target_mock.assert_not_called()
    err = capsys.readouterr().err
    assert "greater than zero" in err


def test_auth_dispatch(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    flow_mock = mocker.patch("src.cli.auth.run_interactive_flow")

    cli.main(["auth"])

    flow_mock.assert_called_once_with(config=cfg, manual=False, response_url=None)


def test_auth_skips_provider_validation(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    load_mock = mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.run_interactive_flow")

    cli.main(["auth"])

    load_mock.assert_called_once_with(validate_providers=False, config_path=None)


def test_auth_passes_response_url(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    flow_mock = mocker.patch("src.cli.auth.run_interactive_flow")

    cli.main(["auth", "http://localhost/?code=abc"])

    flow_mock.assert_called_once_with(
        config=cfg,
        manual=False,
        response_url="http://localhost/?code=abc",
    )


def test_auth_manual_dispatch(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    flow_mock = mocker.patch("src.cli.auth.run_interactive_flow")

    cli.main(["auth", "--manual"])

    flow_mock.assert_called_once_with(config=cfg, manual=True, response_url=None)


def test_auth_import_credentials_dispatch(mocker, tmp_path):
    import_mock = mocker.patch(
        "src.cli.import_google_credentials", return_value=tmp_path / "config.yml"
    )

    cli.main(["auth", "import-credentials", "/some/creds.json"])

    import_mock.assert_called_once_with("/some/creds.json", config_path=None)


def test_auth_use_files_dispatch(mocker, tmp_path):
    use_mock = mocker.patch(
        "src.cli.use_google_files", return_value=tmp_path / "config.yml"
    )

    cli.main(
        [
            "auth",
            "use-files",
            "--credentials-file",
            "/c/creds.json",
            "--token-file",
            "/c/tok.json",
        ]
    )

    use_mock.assert_called_once_with(
        "/c/creds.json", token_file="/c/tok.json", config_path=None
    )


def test_auth_use_files_defaults_token_file(mocker, tmp_path):
    use_mock = mocker.patch(
        "src.cli.use_google_files", return_value=tmp_path / "config.yml"
    )

    cli.main(["auth", "use-files", "--credentials-file", "/c/creds.json"])

    use_mock.assert_called_once_with("/c/creds.json", token_file=None, config_path=None)


def test_doctor_reports_google_source_without_secrets(mocker, capsys, tmp_path):
    from dataclasses import replace

    cfg = replace(
        make_config(folders=["f1"], data_dir=tmp_path),
        google_credentials={"installed": {"client_secret": "sup3rsecret"}},
        google_token={"refresh_token": "rt-secret"},
    )
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service")

    cli.main(["doctor"])

    out = capsys.readouterr().out
    assert "Google credentials: inline" in out
    assert "Google token: inline" in out
    assert "sup3rsecret" not in out
    assert "rt-secret" not in out


def test_doctor_uses_drive_only_config_and_skips_auth_by_default(
    mocker,
    capsys,
    tmp_path,
):
    cfg = make_config(folders=["f1"], data_dir=tmp_path, stt_provider="deepgram")
    (tmp_path / "credentials.json").write_text("{}", encoding="utf-8")
    load_mock = mocker.patch("src.cli.load_config", return_value=cfg)
    build_mock = mocker.patch("src.cli.auth.build_drive_service")

    cli.main(["doctor"])

    load_mock.assert_called_once_with(validate_providers=False, config_path=None)
    build_mock.assert_not_called()
    out = capsys.readouterr().out
    assert "credentials.json: OK" in out
    assert "token.json: missing" in out
    assert "folders: 1 configured" in out


def test_doctor_lists_each_folder_with_employee_name_and_email(mocker, capsys, tmp_path):
    cfg = make_config(
        folders=[
            EmployeeFolder("f1", name="Олег Иванов", email="oleg@expertizeme.org"),
            EmployeeFolder("f2"),
        ],
        data_dir=tmp_path,
    )
    mocker.patch("src.cli.load_config", return_value=cfg)

    cli.main(["doctor"])

    out = capsys.readouterr().out
    assert "folders: 2 configured" in out
    assert "f1: Олег Иванов <oleg@expertizeme.org>" in out
    assert "f2: (no employee configured)" in out


def test_doctor_drive_check_lists_configured_folders(mocker, capsys, tmp_path):
    cfg = make_config(folders=["f1"], data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    list_mock = mocker.patch("src.cli.drive.list_folder_tree_state", return_value=[])
    mocker.patch("src.cli.drive.list_subfolders", return_value=[])
    mocker.patch(
        "src.cli.drive.describe_folder",
        return_value={"id": "f1", "name": "Meet Recordings", "parents": ["mydrive"]},
    )

    cli.main(["doctor", "--drive"])

    list_mock.assert_called_once_with(service, "f1")
    out = capsys.readouterr().out
    assert "Drive auth: OK" in out
    # Reachability alone is what let this read healthy for two months; the name and
    # the date of the last file are the part that gives a moved folder away.
    assert "Folder f1: 'Meet Recordings'" in out
    assert "0 mp4 file(s)" in out


def test_doctor_reports_stt_provider_without_pipeline_readiness(mocker, capsys, tmp_path):
    cfg = make_config(folders=["f1"], data_dir=tmp_path, stt_provider="deepgram")
    mocker.patch("src.cli.load_config", return_value=cfg)
    build_mock = mocker.patch("src.cli.auth.build_drive_service")

    cli.main(["doctor"])

    build_mock.assert_not_called()
    out = capsys.readouterr().out
    assert "stt.provider: deepgram" in out


def test_run_dispatch_validates_enables_then_calls_main(mocker):
    calls = []
    mocker.patch("src.cli.load_config", side_effect=lambda *a, **k: calls.append("load"))
    mocker.patch("src.cli.set_run_enabled", side_effect=lambda *a, **k: calls.append("set"))
    mocker.patch("src.cli.main_module.main", side_effect=lambda *a, **k: calls.append("main"))

    cli.main(["run"])

    # `gdstt run` must validate the config (load_config) BEFORE touching run.enabled,
    # so a broken/missing config fails fast instead of leaving a half-enabled state.
    assert calls == ["load", "set", "main"]


def test_top_level_help_recommends_safe_operator_flow(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])

    out = _normalized_help(capsys.readouterr().out)
    assert "doctor -> list -> process <file-id> --dry-run -> process <file-id>" in out
    assert "run and folder-wide processing can spend STT credits across pending files" in out
    assert "Manage gdstt configuration (active config.yml)" in out


def test_run_help_warns_about_continuous_processing(capsys):
    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])

    out = _normalized_help(capsys.readouterr().out)
    assert "Run the polling loop continuously." in out
    assert "can process every pending configured folder and spend STT credits repeatedly" in out


def test_run_once_help_warns_and_points_to_dry_run(capsys):
    with pytest.raises(SystemExit):
        cli.main(["run-once", "--help"])

    out = _normalized_help(capsys.readouterr().out)
    assert "Run a single polling cycle across the configured folders." in out
    assert "can spend STT credits across multiple pending files" in out
    assert "Use --dry-run first" in out


def test_process_help_warns_about_folder_scope_and_reprocess(capsys):
    with pytest.raises(SystemExit):
        cli.main(["process", "--help"])

    out = _normalized_help(capsys.readouterr().out)
    assert "Process one Drive file or folder on demand." in out
    assert "use --dry-run first" in out
    assert "can process many files and spend STT credits" in out
    assert "--reprocess-txt intentionally reruns STT and overwrites the linked .txt" in out


def test_run_once_dispatch(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    load_mock = mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    build_mock = mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    run_once_mock = mocker.patch("src.cli.main_module.run_once")

    cli.main(["run-once"])

    load_mock.assert_called_once_with(config_path=None)
    build_mock.assert_called_once_with(config=cfg)
    run_once_mock.assert_called_once_with(
        service,
        cfg,
        mode="auto",
        dry_run=False,
        max_size_bytes=None,
        confirm_large=False,
        since="",
    )


def test_run_once_dispatches_safety_flags(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    run_once_mock = mocker.patch("src.cli.main_module.run_once")

    cli.main(["run-once", "--dry-run", "--max-size", "50MB", "--confirm-large"])

    run_once_mock.assert_called_once_with(
        service,
        cfg,
        mode="auto",
        dry_run=True,
        max_size_bytes=50_000_000,
        confirm_large=True,
            since="",
    )


def test_process_dispatch_autodetect(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    load_mock = mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    target_mock = mocker.patch("src.cli.main_module.process_target")

    cli.main(["process", "file123"])

    load_mock.assert_called_once_with(config_path=None)
    target_mock.assert_called_once_with(
        service,
        "file123",
        cfg,
        is_folder=None,
        reprocess_txt=False,
        dry_run=False,
        max_size_bytes=None,
        confirm_large=False,
    )


def test_process_dispatch_folder_flag(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    target_mock = mocker.patch("src.cli.main_module.process_target")

    cli.main(["process", "folder123", "--folder"])

    target_mock.assert_called_once_with(
        service,
        "folder123",
        cfg,
        is_folder=True,
        reprocess_txt=False,
        dry_run=False,
        max_size_bytes=None,
        confirm_large=False,
    )


def test_process_dispatch_reprocess_txt_flag(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    target_mock = mocker.patch("src.cli.main_module.process_target")

    cli.main(["process", "file123", "--reprocess-txt"])

    target_mock.assert_called_once_with(
        service,
        "file123",
        cfg,
        is_folder=None,
        reprocess_txt=True,
        dry_run=False,
        max_size_bytes=None,
        confirm_large=False,
    )


def test_process_dispatches_safety_flags(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    target_mock = mocker.patch("src.cli.main_module.process_target")

    cli.main(["process", "folder123", "--folder", "--dry-run", "--max-size", "1.5GiB"])

    target_mock.assert_called_once_with(
        service,
        "folder123",
        cfg,
        is_folder=True,
        reprocess_txt=False,
        dry_run=True,
        max_size_bytes=1_610_612_736,
        confirm_large=False,
    )


def test_latest_dispatch_uses_first_folder(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path, folders=["folderA", "folderB"])
    load_mock = mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    build_mock = mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    newest = {"id": "v9", "name": "newest.mp4"}
    find_mock = mocker.patch("src.cli.drive.find_newest_mp4", return_value=newest)
    target_mock = mocker.patch("src.cli.main_module.process_target")

    cli.main(["latest"])

    load_mock.assert_called_once_with(config_path=None)
    build_mock.assert_called_once_with(config=cfg)
    find_mock.assert_called_once_with(service, "folderA")
    target_mock.assert_called_once_with(
        service,
        "v9",
        cfg,
        is_folder=False,
        dry_run=False,
        max_size_bytes=None,
        confirm_large=False,
    )


def test_latest_dispatch_honors_folder_and_dry_run(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    newest = {"id": "v1", "name": "x.mp4"}
    find_mock = mocker.patch("src.cli.drive.find_newest_mp4", return_value=newest)
    target_mock = mocker.patch("src.cli.main_module.process_target")

    cli.main(["latest", "--folder", "folderZ", "--dry-run"])

    find_mock.assert_called_once_with(service, "folderZ")
    target_mock.assert_called_once_with(
        service,
        "v1",
        cfg,
        is_folder=False,
        dry_run=True,
        max_size_bytes=None,
        confirm_large=False,
    )


def test_latest_dispatch_forwards_size_guards(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path, folders=["folderA"])
    mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    newest = {"id": "v1", "name": "x.mp4"}
    mocker.patch("src.cli.drive.find_newest_mp4", return_value=newest)
    target_mock = mocker.patch("src.cli.main_module.process_target")

    cli.main(["latest", "--max-size", "1GB", "--confirm-large"])

    target_mock.assert_called_once_with(
        service,
        "v1",
        cfg,
        is_folder=False,
        dry_run=False,
        max_size_bytes=1_000_000_000,
        confirm_large=True,
    )


def test_latest_no_mp4_skips_processing(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    mocker.patch("src.cli.drive.find_newest_mp4", return_value=None)
    target_mock = mocker.patch("src.cli.main_module.process_target")

    cli.main(["latest"])

    target_mock.assert_not_called()


def test_latest_without_folder_config_errors(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path, folders=[])
    mocker.patch("src.cli.load_config", return_value=cfg)

    with pytest.raises(SystemExit):
        cli.main(["latest"])


def test_speakers_set_writes_drive_app_property(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    load_mock = mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    set_mock = mocker.patch("src.cli.drive.set_file_app_properties")

    cli.main(["speakers", "set", "file123", "Alice", "Bob"])

    load_mock.assert_called_once_with(validate_providers=False, config_path=None)
    set_mock.assert_called_once_with(
        service,
        "file123",
        {"speaker_names": "[\"Alice\", \"Bob\"]"},
    )


def test_transcribe_prints_to_stdout(mocker, capsys, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"\x00")
    load_mock = mocker.patch("src.cli.load_config", return_value=cfg)
    transcribe_mock = mocker.patch(
        "src.cli.transcribe_file", return_value="hello world"
    )

    cli.main(["transcribe", str(audio_path)])

    load_mock.assert_called_once_with(config_path=None)
    transcribe_mock.assert_called_once()
    args, _ = transcribe_mock.call_args
    assert args[0] == audio_path
    assert args[1] is cfg
    out = capsys.readouterr().out
    assert "hello world" in out


def test_transcribe_writes_to_output_file(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"\x00")
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.transcribe_file", return_value="the transcript")
    out_path = tmp_path / "out.txt"

    cli.main(["transcribe", str(audio_path), "-o", str(out_path)])

    assert out_path.read_text(encoding="utf-8") == "the transcript"


def test_transcribe_passes_cost_dict_and_prints_cost(mocker, capsys, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"\x00")
    mocker.patch("src.cli.load_config", return_value=cfg)

    def fake_transcribe(path, config, *, cost_usd=None):
        if cost_usd is not None:
            cost_usd["deepgram"] = 0.1234
        return "hello world"

    mocker.patch("src.cli.transcribe_file", side_effect=fake_transcribe)

    cli.main(["transcribe", str(audio_path)])

    out = capsys.readouterr().out
    assert "Deepgram cost: $0.1234" in out


def test_transcribe_reports_pending_cost_when_unavailable(mocker, capsys, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"\x00")
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.transcribe_file", return_value="hi")

    cli.main(["transcribe", str(audio_path)])

    out = capsys.readouterr().out
    assert "Deepgram cost: pending" in out


def test_transcribe_missing_path_exits_cleanly(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    transcribe_mock = mocker.patch("src.cli.transcribe_file")
    missing = tmp_path / "nope.mp3"

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["transcribe", str(missing)])

    assert excinfo.value.code == 1
    transcribe_mock.assert_not_called()


def test_relabel_dispatch_reads_map_and_writes_output(mocker, tmp_path):
    src_path = tmp_path / "src.md"
    src_path.write_text("[00:00:01] Speaker 1: hi\n", encoding="utf-8")
    map_path = tmp_path / "map.json"
    map_path.write_text('{"default": {"Speaker 1": "Alice"}}', encoding="utf-8")
    out_path = tmp_path / "out.md"
    relabel_mock = mocker.patch(
        "src.cli.relabel_transcript.relabel", return_value="rendered"
    )

    cli.main(["relabel", "--in", str(src_path), "--out", str(out_path), "--map", str(map_path)])

    relabel_mock.assert_called_once_with(
        "[00:00:01] Speaker 1: hi\n",
        {"default": {"Speaker 1": "Alice"}},
        include_header=True,
    )
    assert out_path.read_text(encoding="utf-8") == "rendered"


def test_relabel_dispatch_no_header_flag(mocker, tmp_path):
    src_path = tmp_path / "src.md"
    src_path.write_text("[00:00:01] Speaker 1: hi\n", encoding="utf-8")
    map_path = tmp_path / "map.json"
    map_path.write_text('{"default": {"Speaker 1": "Alice"}}', encoding="utf-8")
    out_path = tmp_path / "out.md"
    relabel_mock = mocker.patch(
        "src.cli.relabel_transcript.relabel", return_value="rendered"
    )

    cli.main(
        [
            "relabel",
            "--in",
            str(src_path),
            "--out",
            str(out_path),
            "--map",
            str(map_path),
            "--no-header",
        ]
    )

    _, kwargs = relabel_mock.call_args
    assert kwargs["include_header"] is False


def test_list_dispatch_uses_configured_folders(mocker, capsys, tmp_path):
    cfg = make_config(folders=["f1"], data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    items = [
        {"file": {"id": "v1", "name": "a.mp4"}, "has_mp3": True, "has_txt": False},
        {"file": {"id": "v2", "name": "b.mp4"}, "has_mp3": False, "has_txt": False},
    ]
    list_mock = mocker.patch(
        "src.cli.drive.list_folder_state", return_value=items
    )

    cli.main(["list"])

    list_mock.assert_called_once_with(service, "f1")
    out = capsys.readouterr().out
    assert "a.mp4" in out
    assert "b.mp4" in out
    assert "[mp3]" in out


def test_status_alias_with_explicit_folder(mocker, capsys, tmp_path):
    cfg = make_config(folders=[], data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    list_mock = mocker.patch("src.cli.drive.list_folder_state", return_value=[])

    cli.main(["status", "--folder", "explicit"])

    list_mock.assert_called_once_with(service, "explicit")


def test_list_no_folders_exits(mocker, tmp_path):
    cfg = make_config(folders=[], data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch("src.cli.drive.list_folder_state")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["list"])

    assert excinfo.value.code == 1


def test_list_no_folders_skips_authentication(mocker, tmp_path):
    # The empty-folder check must short-circuit before authenticating, so a
    # missing/expired token can't mask the intended local error.
    cfg = make_config(folders=[], data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    build_mock = mocker.patch("src.cli.auth.build_drive_service")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["list"])

    assert excinfo.value.code == 1
    build_mock.assert_not_called()


def test_list_skips_provider_validation(mocker, tmp_path):
    cfg = make_config(folders=["f1"], data_dir=tmp_path)
    load_mock = mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch("src.cli.drive.list_folder_state", return_value=[])

    cli.main(["list"])

    load_mock.assert_called_once_with(validate_providers=False, config_path=None)


def test_process_prints_spend_summary_from_telemetry(mocker, capsys, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    telemetry = [
        _Telemetry(
            cost_usd={"deepgram": 0.2500},
            usage={"openai_keypoints": {
                "total_tokens": 300, "input_tokens": 200, "output_tokens": 100,
            }},
        )
    ]
    mocker.patch("src.cli.main_module.process_target", return_value=telemetry)

    cli.main(["process", "file123"])

    out = capsys.readouterr().out
    assert "Spend summary:" in out
    assert "Deepgram cost $0.2500" in out
    assert "OpenAI keypoints tokens" in out
    assert "total=300" in out
    assert "input=200" in out
    assert "output=100" in out


def test_process_spend_summary_reports_all_presets(mocker, capsys, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    telemetry = [
        _Telemetry(
            cost_usd={"deepgram": 0.10},
            usage={
                "openai_keypoints": {
                    "total_tokens": 300, "input_tokens": 200, "output_tokens": 100,
                },
                "openai_expertizeme-managers": {
                    "total_tokens": 50, "input_tokens": 40, "output_tokens": 10,
                },
            },
        )
    ]
    mocker.patch("src.cli.main_module.process_target", return_value=telemetry)

    cli.main(["process", "file123"])

    out = capsys.readouterr().out
    assert "OpenAI keypoints tokens: total=300, input=200, output=100" in out
    assert "OpenAI expertizeme-managers tokens: total=50, input=40, output=10" in out


def test_process_spend_summary_reports_pending_cost(mocker, capsys, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    telemetry = [_Telemetry(cost_usd={"deepgram": None})]
    mocker.patch("src.cli.main_module.process_target", return_value=telemetry)

    cli.main(["process", "file123"])

    out = capsys.readouterr().out
    assert "Deepgram cost pending" in out
    assert "OpenAI keypoints tokens" not in out


def test_process_spend_summary_omits_deepgram_when_stt_not_run(mocker, capsys, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    telemetry = [
        _Telemetry(cost_usd={}, usage={"openai_keypoints": {"total_tokens": 42}}),
    ]
    mocker.patch("src.cli.main_module.process_target", return_value=telemetry)

    cli.main(["process", "file123"])

    out = capsys.readouterr().out
    assert "pending" not in out
    assert "Deepgram cost" not in out
    assert "OpenAI keypoints tokens: total=42" in out


def test_process_spend_summary_mixed_deepgram_presence(mocker, capsys, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    telemetry = [
        _Telemetry(cost_usd={}),
        _Telemetry(cost_usd={"deepgram": None}),
        _Telemetry(cost_usd={"deepgram": 0.1234}),
    ]
    mocker.patch("src.cli.main_module.process_target", return_value=telemetry)

    cli.main(["process", "folder123", "--folder"])

    out = capsys.readouterr().out
    assert "file 1: Deepgram cost" not in out
    assert "file 2: Deepgram cost pending" in out
    assert "file 3: Deepgram cost $0.1234" in out
    assert "combined Deepgram cost: $0.1234" in out


def test_process_spend_summary_combined_total(mocker, capsys, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    telemetry = [
        _Telemetry(cost_usd={"deepgram": 0.10}),
        _Telemetry(cost_usd={"deepgram": 0.20}),
    ]
    mocker.patch("src.cli.main_module.process_target", return_value=telemetry)

    cli.main(["process", "folder123", "--folder"])

    out = capsys.readouterr().out
    assert "combined Deepgram cost: $0.3000" in out


def test_process_spend_summary_nothing_processed(mocker, capsys, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch("src.cli.main_module.process_target", return_value=[])

    cli.main(["process", "file123"])

    out = capsys.readouterr().out
    assert "nothing processed" in out


def test_process_spend_summary_dry_run(mocker, capsys, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch("src.cli.main_module.process_target", return_value=[])

    cli.main(["process", "file123", "--dry-run"])

    out = capsys.readouterr().out
    assert "dry-run" in out


def test_latest_prints_spend_summary(mocker, capsys, tmp_path):
    cfg = make_config(data_dir=tmp_path, folders=["folderA"])
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch(
        "src.cli.drive.find_newest_mp4",
        return_value={"id": "v1", "name": "x.mp4"},
    )
    telemetry = [_Telemetry(cost_usd={"deepgram": 0.5000})]
    mocker.patch("src.cli.main_module.process_target", return_value=telemetry)

    cli.main(["latest"])

    out = capsys.readouterr().out
    assert "Spend summary:" in out
    assert "Deepgram cost $0.5000" in out


def test_config_flag_is_passed_to_doctor(mocker, capsys, tmp_path):
    # --config is threaded straight into load_config (and resolve_config_file_path)
    # as a one-shot override; it is never routed through process env.
    target = tmp_path / "custom.yml"
    cfg = make_config(folders=["f1"], data_dir=tmp_path)
    load = mocker.patch("src.cli.load_config", return_value=cfg)

    cli.main(["--config", str(target), "doctor"])

    load.assert_called_once_with(config_path=str(target), validate_providers=False)
    out = capsys.readouterr().out
    assert f"config: {target} (missing)" in out


def test_doctor_without_config_flag_passes_none(mocker, tmp_path):
    # Without --config, load_config receives config_path=None so the resolver falls
    # back to GDSTT_HOME/config.yml (or ./data/config.yml).
    cfg = make_config(folders=["f1"], data_dir=tmp_path)
    load = mocker.patch("src.cli.load_config", return_value=cfg)

    cli.main(["doctor"])

    load.assert_called_once_with(config_path=None, validate_providers=False)


def test_doctor_reports_preset_dag(mocker, capsys, tmp_path):
    presets = (
        Preset(name="transcript-cleanup", instructions="clean"),
        Preset(
            name="keypoints",
            instructions="kp",
            depends_on=("transcript-cleanup",),
        ),
    )
    cfg = make_config(folders=["f1"], data_dir=tmp_path, presets=presets)
    mocker.patch("src.cli.load_config", return_value=cfg)

    cli.main(["doctor"])

    out = capsys.readouterr().out
    assert "Presets: 2 enabled" in out
    assert "transcript-cleanup <- transcript" in out
    assert "keypoints <- transcript-cleanup" in out


def test_doctor_reports_no_presets_when_none_enabled(mocker, capsys, tmp_path):
    cfg = make_config(folders=["f1"], data_dir=tmp_path, presets=())
    mocker.patch("src.cli.load_config", return_value=cfg)

    cli.main(["doctor"])

    out = capsys.readouterr().out
    assert "Presets: none enabled" in out


def test_config_init_command_dispatch(mocker, capsys, tmp_path):
    config_file = tmp_path / "config.yml"
    init = mocker.patch("src.cli.init_config", return_value=config_file)

    cli.main(["config", "init"])

    init.assert_called_once_with(
        config_path=None,
        data_dir=None,
        output_dir=None,
        prompt_dir=None,
        force=False,
    )
    out = capsys.readouterr().out
    assert str(config_file) in out


def test_config_init_command_passes_flags(mocker, tmp_path):
    config_file = tmp_path / "config.yml"
    init = mocker.patch("src.cli.init_config", return_value=config_file)

    cli.main(
        [
            "config",
            "init",
            "--data-dir",
            "store",
            "--output-dir",
            "out",
            "--prompt-dir",
            "pr",
            "--force",
        ]
    )

    init.assert_called_once_with(
        config_path=None,
        data_dir="store",
        output_dir="out",
        prompt_dir="pr",
        force=True,
    )


def test_config_init_command_reports_error(mocker):
    mocker.patch("src.cli.init_config", side_effect=ValueError("already exists"))

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["config", "init"])

    assert excinfo.value.code == 1


def test_config_init_creates_real_config_without_secrets(monkeypatch, capsys, tmp_path):
    home = tmp_path / "instance"
    monkeypatch.setenv("GDSTT_HOME", str(home))

    cli.main(["config", "init"])

    config_file = home / "config.yml"
    assert config_file.is_file()
    assert (home / "prompts" / "keypoints.md").is_file()
    out = capsys.readouterr().out
    assert str(config_file) in out


def test_config_path_prints_resolved_without_secrets(monkeypatch, capsys, tmp_path):
    home = tmp_path / "instance"
    monkeypatch.setenv("GDSTT_HOME", str(home))

    cli.main(["config", "path"])

    out = capsys.readouterr().out.strip()
    assert out == str(home / "config.yml")


def test_config_get_command_dispatch(mocker, capsys):
    get = mocker.patch("src.cli.config_get", return_value="model: gpt")

    cli.main(["config", "get"])

    get.assert_called_once_with(None, config_path=None, show_secrets=False)
    assert "model: gpt" in capsys.readouterr().out


def test_config_get_command_passes_key(mocker, capsys):
    get = mocker.patch("src.cli.config_get", return_value="gpt-5.4")

    cli.main(["config", "get", "openai.model"])

    get.assert_called_once_with("openai.model", config_path=None, show_secrets=False)
    assert "gpt-5.4" in capsys.readouterr().out


def test_config_get_command_show_secrets(mocker, capsys):
    get = mocker.patch("src.cli.config_get", return_value="sk-secret")

    cli.main(["config", "get", "openai.api_key", "--show-secrets"])

    get.assert_called_once_with("openai.api_key", config_path=None, show_secrets=True)
    assert "sk-secret" in capsys.readouterr().out


def test_config_get_command_reports_error(mocker):
    mocker.patch("src.cli.config_get", side_effect=ValueError("not set"))

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["config", "get", "nope"])

    assert excinfo.value.code == 1


def test_config_set_command_dispatch(mocker, capsys, tmp_path):
    config_file = tmp_path / "config.yml"
    setter = mocker.patch("src.cli.config_set", return_value=config_file)

    cli.main(["config", "set", "openai.model", "gpt-5.4"])

    setter.assert_called_once_with("openai.model", "gpt-5.4", config_path=None)
    assert "Set openai.model" in capsys.readouterr().out


def test_config_set_command_reports_error(mocker):
    mocker.patch("src.cli.config_set", side_effect=ValueError("bad"))

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["config", "set", "output.target", "s3"])

    assert excinfo.value.code == 1


def test_config_unset_command_dispatch(mocker, capsys, tmp_path):
    config_file = tmp_path / "config.yml"
    unset = mocker.patch("src.cli.config_unset", return_value=config_file)

    cli.main(["config", "unset", "proxy_url"])

    unset.assert_called_once_with("proxy_url", config_path=None)
    assert "Unset proxy_url" in capsys.readouterr().out


def test_config_unset_command_reports_error(mocker):
    mocker.patch("src.cli.config_unset", side_effect=ValueError("not set"))

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["config", "unset", "nope"])

    assert excinfo.value.code == 1


# --- reprocess stage spec + dispatch ----------------------------------------

_STAGES = ["transcript-cleanup", "keypoints", "action-items"]


def test_parse_stage_spec_all_and_empty():
    assert cli._parse_stage_spec(None, _STAGES) == (False, _STAGES)
    assert cli._parse_stage_spec("all", _STAGES) == (False, _STAGES)


def test_parse_stage_spec_single_and_range():
    assert cli._parse_stage_spec("2", _STAGES) == (False, ["keypoints"])
    assert cli._parse_stage_spec("2-3", _STAGES) == (False, ["keypoints", "action-items"])
    assert cli._parse_stage_spec("1,3", _STAGES) == (
        False,
        ["transcript-cleanup", "action-items"],
    )


def test_parse_stage_spec_zero_means_full_reprocess():
    assert cli._parse_stage_spec("0", _STAGES) == (True, [])
    # 0 with others still collapses to a full transcript reprocess.
    assert cli._parse_stage_spec("0,2", _STAGES) == (True, [])


def test_parse_stage_spec_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        cli._parse_stage_spec("4", _STAGES)


def test_parse_stage_spec_bad_token_raises():
    with pytest.raises(ValueError):
        cli._parse_stage_spec("x", _STAGES)


def _chain_config():
    presets = (
        Preset(name="transcript-cleanup", instructions="c"),
        Preset(name="keypoints", instructions="k", depends_on=("transcript-cleanup",)),
        Preset(name="action-items", instructions="a", depends_on=("transcript-cleanup",)),
    )
    return make_config(presets=presets)


def test_reprocess_command_passes_selected_presets(mocker, capsys):
    mocker.patch("src.cli.load_config", return_value=_chain_config())
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    proc = mocker.patch("src.cli.main_module.process_target", return_value=[])

    cli.main(["reprocess", "file-1", "2-3"])

    kwargs = proc.call_args.kwargs
    assert kwargs["reprocess_presets"] == ["keypoints", "action-items"]
    assert kwargs["reprocess_txt"] is False


def test_reprocess_command_stage_zero_is_full(mocker):
    mocker.patch("src.cli.load_config", return_value=_chain_config())
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    proc = mocker.patch("src.cli.main_module.process_target", return_value=[])

    cli.main(["reprocess", "file-1", "0"])

    kwargs = proc.call_args.kwargs
    assert kwargs["reprocess_txt"] is True
    assert kwargs["reprocess_presets"] is None


def test_reprocess_command_bad_spec_exits(mocker):
    mocker.patch("src.cli.load_config", return_value=_chain_config())

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["reprocess", "file-1", "9"])
    assert excinfo.value.code == 1


def test_stop_command_sets_run_disabled(mocker, capsys, tmp_path):
    setter = mocker.patch("src.cli.set_run_enabled", return_value=tmp_path / "config.yml")

    cli.main(["stop"])

    setter.assert_called_once_with(False, config_path=None)
    out = capsys.readouterr().out
    assert "run.enabled=false" in out
    assert "stays paused across restarts" in out


def test_start_command_sets_run_enabled(mocker, capsys, tmp_path):
    setter = mocker.patch("src.cli.set_run_enabled", return_value=tmp_path / "config.yml")

    cli.main(["start"])

    setter.assert_called_once_with(True, config_path=None)
    assert "run.enabled=true" in capsys.readouterr().out


def test_bookings_list_prints_the_journal(tmp_path, capsys, monkeypatch):
    config_path = write_cli_config(tmp_path)
    # Yesterday, not a fixed date: `bookings list` reads through the same retention
    # window as the gate, so a pinned date makes this test expire by itself.
    start_time = (datetime.now(timezone.utc) - timedelta(days=1)).replace(
        hour=7, minute=0, second=0, microsecond=0
    )
    append(
        tmp_path / "call_bookings.jsonl",
        CallBooking(
            task_id="851030",
            manager_email="kate@example.com",
            start_time=start_time,
        ),
    )

    cli.main(["--config", str(config_path), "bookings", "list"])

    out = capsys.readouterr().out
    assert "851030" in out
    assert "kate@example.com" in out
    assert start_time.isoformat() in out


def test_bookings_list_reports_an_empty_journal(tmp_path, capsys):
    config_path = write_cli_config(tmp_path)

    cli.main(["--config", str(config_path), "bookings", "list"])

    assert "no bookings" in capsys.readouterr().out.lower()


def test_bookings_rematch_clears_the_mark(tmp_path, monkeypatch):
    config_path = write_cli_config(tmp_path)
    service = MagicMock()
    monkeypatch.setattr(cli.auth, "build_drive_service", lambda **kwargs: service)
    cleared = []
    monkeypatch.setattr(
        cli.booking_gate, "clear_mark", lambda svc, fid: cleared.append(fid)
    )

    cli.main(["--config", str(config_path), "bookings", "rematch", "v1"])

    assert cleared == ["v1"]


def test_bookings_restore_dates_restores_selected_files(tmp_path, monkeypatch, capsys):
    config_path = write_cli_config(
        tmp_path, folders=[{"folder_id": "f1", "email": "a@example.com"}]
    )
    service = MagicMock()
    monkeypatch.setattr(cli.auth, "build_drive_service", lambda **kwargs: service)
    monkeypatch.setattr(
        cli.drive,
        "list_mp4_timestamps",
        lambda svc, folder_id: [
            {
                "id": "v1",
                "name": "old.mp4",
                "createdTime": "2025-03-14T18:24:52.949Z",
                "modifiedTime": "2026-08-11T22:13:27.539Z",
                "appProperties": {"booking_match": "none"},
            }
        ],
    )
    restored = []
    monkeypatch.setattr(
        cli.drive,
        "set_file_modified_time",
        lambda svc, fid, when: restored.append((fid, when)),
    )

    cli.main(["--config", str(config_path), "bookings", "restore-dates"])

    assert restored == [("v1", "2025-03-14T18:24:52.949Z")]
    assert "Restored modifiedTime on 1 file(s)" in capsys.readouterr().out


def test_bookings_restore_dates_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    """The whole point of the flag: see the list before touching production files."""
    config_path = write_cli_config(
        tmp_path, folders=[{"folder_id": "f1", "email": "a@example.com"}]
    )
    service = MagicMock()
    monkeypatch.setattr(cli.auth, "build_drive_service", lambda **kwargs: service)
    monkeypatch.setattr(
        cli.drive,
        "list_mp4_timestamps",
        lambda svc, folder_id: [
            {
                "id": "v1",
                "name": "old.mp4",
                "createdTime": "2025-03-14T18:24:52.949Z",
                "modifiedTime": "2026-08-11T22:13:27.539Z",
                "appProperties": {"booking_match": "none"},
            }
        ],
    )

    def fail(*args, **kwargs):
        raise AssertionError("--dry-run must not write")

    monkeypatch.setattr(cli.drive, "set_file_modified_time", fail)

    cli.main(["--config", str(config_path), "bookings", "restore-dates", "--dry-run"])

    out = capsys.readouterr().out
    assert "v1" in out
    assert "dry-run" in out or "would" in out


def test_bookings_restore_dates_walks_every_configured_folder(
    tmp_path, monkeypatch, capsys
):
    config_path = write_cli_config(
        tmp_path,
        folders=[
            {"folder_id": "f1", "email": "a@example.com"},
            {"folder_id": "f2", "email": "b@example.com"},
        ],
    )
    monkeypatch.setattr(cli.auth, "build_drive_service", lambda **kwargs: MagicMock())
    seen = []
    monkeypatch.setattr(
        cli.drive,
        "list_mp4_timestamps",
        lambda svc, folder_id: seen.append(folder_id) or [],
    )
    monkeypatch.setattr(cli.drive, "set_file_modified_time", lambda *a, **k: None)

    cli.main(["--config", str(config_path), "bookings", "restore-dates"])

    assert seen == ["f1", "f2"]


def test_doctor_reports_call_booking_without_leaking_the_token(tmp_path, capsys):
    config_path = write_cli_config(
        tmp_path,
        call_booking={
            "enabled": True,
            "authorization_token": "super-secret",
            "listen_port": 9100,
        },
        planfix={
            "create_comment_url": "https://crm.example.com/planfix_create_comment",
            "token": "another-super-secret",
        },
    )

    cli.main(["--config", str(config_path), "doctor"])

    out = capsys.readouterr().out
    assert "call_booking" in out
    assert "9100" in out
    assert "super-secret" not in out
    # planfix.token uses the identical set/unset expression as call_booking's token
    # and must be covered the same way -- a distinct secret so this assertion can't
    # pass by accident from the call_booking one above matching a substring of it.
    assert "planfix" in out
    assert "another-super-secret" not in out


def _sent_files(count):
    return [
        {
            "id": f"v{n}",
            "name": f"call-{n}.mp4",
            "createdTime": f"2026-08-{n:02d}T10:00:00.000Z",
            "appProperties": {"planfix_comment_task_id": f"90000{n}"},
        }
        for n in range(1, count + 1)
    ]


def _run_sent(tmp_path, monkeypatch, files, extra=(), **config_kwargs):
    config_path = write_cli_config(
        tmp_path, folders=[{"folder_id": "f1", "email": "a@example.com"}], **config_kwargs
    )
    monkeypatch.setattr(cli.auth, "build_drive_service", lambda **kwargs: MagicMock())
    monkeypatch.setattr(cli.drive, "list_mp4_timestamps", lambda svc, folder_id: files)
    cli.main(["--config", str(config_path), "planfix", "sent", *extra])


def test_planfix_sent_lists_only_recordings_with_a_marker(tmp_path, monkeypatch, capsys):
    files = _sent_files(1) + [
        {"id": "v9", "name": "never-sent.mp4", "createdTime": "2026-08-09T10:00:00.000Z"}
    ]
    _run_sent(tmp_path, monkeypatch, files)

    out = capsys.readouterr().out
    assert "call-1.mp4" in out
    assert "never-sent.mp4" not in out


def test_planfix_sent_includes_a_call_followed_through_a_shortcut(
    tmp_path, monkeypatch, capsys
):
    """Its marker lives on the shortcut, so that is what the report must read."""
    files = [{
        "id": "sc1",
        "name": "attended.mp4",
        "mimeType": "application/vnd.google-apps.shortcut",
        "createdTime": "2026-08-01T10:00:00.000Z",
        "appProperties": {"planfix_comment_task_id": "900001"},
        "shortcutDetails": {"targetId": "clients-video", "targetMimeType": "video/mp4"},
    }]
    _run_sent(tmp_path, monkeypatch, files)

    out = capsys.readouterr().out
    assert "attended.mp4" in out
    assert "/file/d/sc1/view" in out


def test_planfix_sent_puts_the_newest_first(tmp_path, monkeypatch, capsys):
    """The question this answers is 'what happened lately'."""
    _run_sent(tmp_path, monkeypatch, _sent_files(3))

    out = capsys.readouterr().out
    assert out.index("call-3.mp4") < out.index("call-2.mp4") < out.index("call-1.mp4")


def test_planfix_sent_caps_the_list_and_says_so(tmp_path, monkeypatch, capsys):
    """A truncated list that looks complete hides calls nobody will go looking for."""
    _run_sent(tmp_path, monkeypatch, _sent_files(3), extra=["--limit", "2"])

    out = capsys.readouterr().out
    assert "call-1.mp4" not in out
    assert "2 of 3 shown" in out


def test_planfix_sent_limit_zero_prints_everything(tmp_path, monkeypatch, capsys):
    _run_sent(tmp_path, monkeypatch, _sent_files(3), extra=["--limit", "0"])

    out = capsys.readouterr().out
    assert "call-1.mp4" in out
    assert "shown" not in out


def test_planfix_sent_falls_back_to_the_bare_task_id(tmp_path, monkeypatch, capsys):
    """Without planfix.task_url there is no link to build; don't print a broken one."""
    _run_sent(tmp_path, monkeypatch, _sent_files(1))

    out = capsys.readouterr().out
    assert "task 900001" in out
    assert "https://drive.google.com/file/d/v1/view" in out


def test_planfix_sent_reports_an_empty_log(tmp_path, monkeypatch, capsys):
    _run_sent(tmp_path, monkeypatch, [])

    assert "No recording carries a sent-comment marker." in capsys.readouterr().out


# --- Subfolders and the changes feed from the operator's side ---------------------


def _doctor_config(mocker, tmp_path, **overrides):
    cfg = make_config(**{"folders": ["folderA"], "data_dir": tmp_path, **overrides})
    mocker.patch("src.cli.load_config", return_value=cfg)
    return cfg


def _folder_meta(name="Google Meet", parents=("mydrive",), trashed=False):
    return {"id": "root", "name": name, "parents": list(parents), "trashed": trashed}


def test_doctor_names_the_folder_so_a_moved_one_is_obvious(mocker, capsys, tmp_path):
    """Counting files answered "can I reach it", and stayed yes for two months after
    Google moved the recordings. The name is the whole diagnosis."""
    _doctor_config(mocker, tmp_path)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch(
        "src.cli.drive.describe_folder",
        return_value=_folder_meta(name="Legacy Meet Recordings", parents=("gm",)),
    )
    mocker.patch("src.cli.drive.list_folder_tree_state", return_value=[])
    mocker.patch("src.cli.drive.list_subfolders", return_value=[])

    cli.main(["doctor", "--drive"])

    out = capsys.readouterr().out
    assert "Legacy Meet Recordings" in out
    assert "parent gm" in out


def test_doctor_reports_when_a_folder_last_received_anything(mocker, capsys, tmp_path):
    _doctor_config(mocker, tmp_path)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch("src.cli.drive.describe_folder", return_value=_folder_meta())
    mocker.patch(
        "src.cli.drive.list_folder_tree_state",
        return_value=[
            {"file": {"id": "v1", "name": "a.mp4", "createdTime": "2026-09-09T18:53:00Z"}},
        ],
    )
    mocker.patch("src.cli.drive.list_subfolders", return_value=[{"id": "d1"}])

    cli.main(["doctor", "--drive"])

    out = capsys.readouterr().out
    assert "2026-09-09T18:53:00Z" in out
    assert "1 subfolder(s)" in out


def test_doctor_says_a_folder_that_never_received_anything_never_did(
    mocker, capsys, tmp_path
):
    _doctor_config(mocker, tmp_path)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch("src.cli.drive.describe_folder", return_value=_folder_meta())
    mocker.patch("src.cli.drive.list_folder_tree_state", return_value=[])
    mocker.patch("src.cli.drive.list_subfolders", return_value=[])

    cli.main(["doctor", "--drive"])

    assert "newest never" in capsys.readouterr().out


def test_doctor_reports_an_unreachable_folder_instead_of_crashing(
    mocker, capsys, tmp_path
):
    """A diagnostic that raises is no diagnostic: the operator runs it precisely when
    something is wrong."""
    _doctor_config(mocker, tmp_path)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch(
        "src.cli.drive.describe_folder", side_effect=RuntimeError("no access")
    )

    cli.main(["doctor", "--drive"])

    assert "UNREACHABLE" in capsys.readouterr().out


def test_doctor_reports_the_cursor(mocker, capsys, tmp_path):
    _doctor_config(mocker, tmp_path)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch("src.cli.drive.describe_folder", return_value=_folder_meta())
    mocker.patch("src.cli.drive.list_folder_tree_state", return_value=[])
    mocker.patch("src.cli.drive.list_subfolders", return_value=[])

    cli.main(["doctor", "--drive"])

    assert "changes cursor" in capsys.readouterr().out


def test_list_walks_subfolders_and_says_where_each_file_lives(mocker, capsys, tmp_path):
    _doctor_config(mocker, tmp_path)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    tree_mock = mocker.patch(
        "src.cli.drive.list_folder_tree_state",
        return_value=[{
            "file": {"id": "v1", "name": "a.mp4"},
            "container_id": "meeting-1",
            "has_mp3": False,
            "has_txt": True,
        }],
    )

    cli.main(["list"])

    tree_mock.assert_called_once()
    out = capsys.readouterr().out
    assert "a.mp4" in out
    assert "meeting-1" in out


def test_list_marks_a_call_followed_through_a_shortcut(mocker, capsys, tmp_path):
    _doctor_config(mocker, tmp_path)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch(
        "src.cli.drive.list_folder_tree_state",
        return_value=[{
            "file": {"id": "sc1", "name": "attended.mp4"},
            "container_id": "meeting-1",
            "media_id": "clients-video",
            "target_parents": ["clients-meeting"],
            "has_mp3": False,
            "has_txt": False,
        }],
    )
    mocker.patch("src.main.drive.find_configured_ancestor", return_value=None)

    cli.main(["list"])

    out = capsys.readouterr().out.splitlines()
    line = next(text for text in out if "attended.mp4" in text)
    assert "via shortcut" in line


def test_list_leaves_out_a_shortcut_the_organizers_folder_processes(
    mocker, capsys, tmp_path
):
    """`list` must not contradict what a cycle does, and a cycle leaves that call to
    the organizer's folder."""
    _doctor_config(mocker, tmp_path, folders=["folderA", "organizer"])
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch(
        "src.cli.drive.list_folder_tree_state",
        side_effect=lambda service, folder_id: [{
            "file": {"id": "sc1", "name": "internal.mp4"},
            "container_id": "meeting-1",
            "media_id": "v-org",
            "target_parents": ["org-meeting"],
            "has_mp3": False,
            "has_txt": False,
        }] if folder_id == "folderA" else [],
    )
    mocker.patch("src.main.drive.find_configured_ancestor", return_value="organizer")

    cli.main(["list"])

    assert "internal.mp4" not in capsys.readouterr().out


def test_latest_looks_inside_subfolders(mocker, tmp_path):
    _doctor_config(mocker, tmp_path)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    newest_mock = mocker.patch(
        "src.cli.drive.find_newest_mp4_in_tree", return_value=None
    )

    cli.main(["latest"])

    newest_mock.assert_called_once()


def test_changes_without_a_cursor_says_the_next_cycle_sweeps(mocker, capsys, tmp_path):
    _doctor_config(mocker, tmp_path)

    cli.main(["changes"])

    assert "sweeps every folder" in capsys.readouterr().out


def test_changes_never_moves_the_cursor(mocker, capsys, tmp_path):
    """Looking into the feed must not consume it, or the cycle that follows finds
    nothing and the recording is skipped."""
    cfg = _doctor_config(mocker, tmp_path)
    _save_cursor(cfg, "tok-1")
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch("src.cli.drive.list_changes", return_value=([], "tok-2"))

    cli.main(["changes"])

    assert change_cursor.read(change_cursor.path_for(cfg.data_dir)) == "tok-1"
    assert "not saved" in capsys.readouterr().out


def test_changes_shows_our_videos_with_the_folder_they_belong_to(
    mocker, capsys, tmp_path
):
    cfg = _doctor_config(mocker, tmp_path)
    _save_cursor(cfg, "tok-1")
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch(
        "src.cli.drive.list_changes",
        return_value=(
            [{
                "fileId": "v1",
                "file": {
                    "id": "v1", "name": "call.mp4", "mimeType": "video/mp4",
                    "parents": ["meeting-1"], "trashed": False,
                },
            }],
            "tok-2",
        ),
    )
    mocker.patch("src.cli.drive.find_configured_ancestor", return_value="folderA")

    cli.main(["changes"])

    out = capsys.readouterr().out
    assert "call.mp4" in out
    assert "meeting-1" in out
    assert "folderA" in out


def test_changes_shows_an_attended_call_that_arrived_as_a_shortcut(
    mocker, capsys, tmp_path
):
    """The cycle takes it; a report that said "nothing of ours" would contradict it."""
    cfg = _doctor_config(mocker, tmp_path)
    _save_cursor(cfg, "tok-1")
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch(
        "src.cli.drive.list_changes",
        return_value=(
            [{
                "fileId": "sc1",
                "file": {
                    "id": "sc1", "name": "attended.mp4",
                    "mimeType": "application/vnd.google-apps.shortcut",
                    "shortcutDetails": {"targetMimeType": "video/mp4"},
                    "parents": ["meeting-1"], "trashed": False,
                },
            }],
            "tok-2",
        ),
    )
    mocker.patch("src.cli.drive.find_configured_ancestor", return_value="folderA")

    cli.main(["changes"])

    line = next(
        text for text in capsys.readouterr().out.splitlines() if "attended.mp4" in text
    )
    assert "shortcut" in line


def test_changes_raw_shows_entries_that_are_not_ours(mocker, capsys, tmp_path):
    cfg = _doctor_config(mocker, tmp_path)
    _save_cursor(cfg, "tok-1")
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch(
        "src.cli.drive.list_changes",
        return_value=(
            [{
                "fileId": "t1",
                "file": {
                    "id": "t1", "name": "call.txt", "mimeType": "text/plain",
                    "parents": ["meeting-1"], "trashed": False,
                },
            }],
            "tok-2",
        ),
    )

    cli.main(["changes", "--raw"])

    assert "call.txt" in capsys.readouterr().out


def test_cursor_show_reports_an_absent_cursor(mocker, capsys, tmp_path):
    _doctor_config(mocker, tmp_path)

    cli.main(["cursor", "show"])

    assert "absent" in capsys.readouterr().out


def test_cursor_reset_forgets_it_and_says_so(mocker, capsys, tmp_path):
    cfg = _doctor_config(mocker, tmp_path)
    _save_cursor(cfg, "tok-1")

    cli.main(["cursor", "reset"])

    assert change_cursor.read(change_cursor.path_for(cfg.data_dir)) is None
    assert "sweeps every folder" in capsys.readouterr().out


def test_cursor_reset_is_harmless_when_there_is_nothing_to_reset(
    mocker, capsys, tmp_path
):
    _doctor_config(mocker, tmp_path)

    cli.main(["cursor", "reset"])

    assert "already sweeps" in capsys.readouterr().out


def test_run_once_walk_mode_leaves_the_cursor_where_it_was(mocker, tmp_path):
    """A "check everything now" must not become a new starting point: the feed has to
    pick up exactly where it was."""
    cfg = _doctor_config(mocker, tmp_path)
    _save_cursor(cfg, "tok-1")
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch("src.main.drive.list_folder_tree_state", return_value=[])
    mocker.patch("src.main.drive.get_start_page_token", return_value="tok-9")

    cli.main(["run-once", "--mode", "walk"])

    assert change_cursor.read(change_cursor.path_for(cfg.data_dir)) == "tok-1"


def test_run_once_changes_mode_refuses_without_a_cursor(mocker, tmp_path):
    """Better a plain refusal than a silent full sweep when the operator asked to
    exercise the feed."""
    _doctor_config(mocker, tmp_path)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())

    with pytest.raises(SystemExit):
        cli.main(["run-once", "--mode", "changes"])


# --- run-once without --mode follows the service, not a hardcoded default ------


def test_run_once_without_a_mode_follows_the_configured_discovery(mocker, tmp_path):
    """A deployment pinned to run.discovery=walk must not be silently exercised on
    the other path just because the operator typed the command by hand."""
    cfg = dataclasses.replace(make_config(data_dir=tmp_path), run_discovery="walk")
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    run_once_mock = mocker.patch("src.cli.main_module.run_once")

    cli.main(["run-once"])

    assert run_once_mock.call_args.kwargs["mode"] == "walk"


def test_an_explicit_mode_still_wins_over_the_config(mocker, tmp_path):
    cfg = dataclasses.replace(make_config(data_dir=tmp_path), run_discovery="walk")
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    run_once_mock = mocker.patch("src.cli.main_module.run_once")

    cli.main(["run-once", "--mode", "changes"])

    assert run_once_mock.call_args.kwargs["mode"] == "changes"


def test_cursor_show_says_which_folders_the_cursor_covers(mocker, tmp_path, capsys):
    cfg = make_config(data_dir=tmp_path, folders=["root"])
    mocker.patch("src.cli.load_config", return_value=cfg)
    _save_cursor(cfg, "tok-1")

    cli.main(["cursor", "show"])

    out = capsys.readouterr().out
    assert "cursor: tok-1" in out
    assert "all covered" in out


def test_cursor_show_names_a_folder_the_cursor_cannot_vouch_for(
    mocker, tmp_path, capsys
):
    """The question an operator actually has after editing the config: does the
    saved cursor still mean anything for the folder I just added?"""
    cfg = make_config(data_dir=tmp_path, folders=["root"])
    _save_cursor(cfg, "tok-1")
    grown = make_config(data_dir=tmp_path, folders=["root", "new"])
    mocker.patch("src.cli.load_config", return_value=grown)

    cli.main(["cursor", "show"])

    out = capsys.readouterr().out
    assert "changed since the cursor was taken" in out
    assert "added:   new" in out


def test_cursor_reset_forgets_the_folder_set_too(mocker, tmp_path):
    """Left behind, it would vouch for a cursor that no longer exists."""
    cfg = make_config(data_dir=tmp_path, folders=["root"])
    mocker.patch("src.cli.load_config", return_value=cfg)
    _save_cursor(cfg, "tok-1")

    cli.main(["cursor", "reset"])

    assert change_cursor.read(change_cursor.path_for(cfg.data_dir)) is None
    assert (
        change_cursor.read_folders(change_cursor.folders_path_for(cfg.data_dir))
        is None
    )


# --- run-once --since ---------------------------------------------------------


def test_run_once_passes_the_since_flag_through(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    run_once_mock = mocker.patch("src.cli.main_module.run_once")

    cli.main(["run-once", "--since", "2026-09-12"])

    assert run_once_mock.call_args.kwargs["since"] == "2026-09-12T00:00:00+00:00"


def test_run_once_without_since_leaves_the_config_in_charge(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    run_once_mock = mocker.patch("src.cli.main_module.run_once")

    cli.main(["run-once"])

    assert run_once_mock.call_args.kwargs["since"] == ""


def test_an_unreadable_since_fails_before_drive_is_touched(mocker, tmp_path):
    """Parse-time, not cycle-time: a typo must not cost an authentication round trip
    and then a confusing traceback halfway through a folder."""
    build_mock = mocker.patch("src.cli.auth.build_drive_service")

    with pytest.raises(SystemExit):
        cli.main(["run-once", "--since", "last tuesday"])

    build_mock.assert_not_called()


def test_list_marks_recordings_a_cutoff_leaves_out(mocker, capsys, tmp_path):
    """Otherwise the report and the service disagree: `list` would show a folder
    full of recordings with no transcript while every cycle skipped all of them, and
    the operator would have no way to tell which one was lying."""
    cfg = dataclasses.replace(
        make_config(data_dir=tmp_path, folders=["root"]), run_since="2026-10-01"
    )
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch(
        "src.cli.drive.list_folder_tree_state",
        return_value=[
            {
                "file": {"id": "v1", "name": "exf-wxzm-uzk (2026-09-09 17_42 GMT+2).mp4"},
                "container_id": "meeting-1",
            },
            {
                "file": {"id": "v2", "name": "exf-wxzm-uzk (2026-11-20 17_42 GMT+2).mp4"},
                "container_id": "meeting-2",
            },
        ],
    )

    cli.main(["list"])

    out = capsys.readouterr().out
    old_line = next(line for line in out.splitlines() if "2026-09-09" in line)
    new_line = next(line for line in out.splitlines() if "2026-11-20" in line)
    assert "before since, not processed" in old_line
    assert "before since" not in new_line


def test_doctor_says_what_becomes_of_each_attended_call(mocker, capsys, tmp_path):
    """Found on a real employee folder: calls they attended existed only as shortcuts,
    and every other line of the diagnosis read as healthy. Each shortcut ends one of
    three ways, and only one of them needs an operator."""
    from googleapiclient.errors import HttpError

    _doctor_config(mocker, tmp_path, folders=["folderA", "organizer"])
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch("src.cli.drive.describe_folder", return_value=_folder_meta())
    mocker.patch("src.cli.drive.list_folder_tree_state", return_value=[])
    mocker.patch("src.cli.drive.list_subfolders", return_value=[])
    mocker.patch(
        "src.cli.drive.list_recording_shortcuts",
        side_effect=lambda service, folder_id: [
            {"id": "s1", "name": "a.mp4", "container_id": "m1", "target_id": "t1"},
            {"id": "s2", "name": "b.mp4", "container_id": "m2", "target_id": "t2"},
            {"id": "s3", "name": "c.mp4", "container_id": "m3", "target_id": "t3"},
        ] if folder_id == "folderA" else [],
    )
    mocker.patch(
        "src.cli.drive.get_shortcut_target",
        side_effect=lambda service, target_id: {
            "t1": None,
            "t2": {"id": "t2", "parents": ["org-meeting"]},
            "t3": {"id": "t3", "parents": ["clients-meeting"]},
        }[target_id],
    )

    def resolve(service, container_id, configured_ids, cache=None):
        if container_id == "clients-meeting":
            raise HttpError(MagicMock(status=404), b"")
        return "organizer" if container_id == "org-meeting" else None

    mocker.patch("src.main.drive.find_configured_ancestor", side_effect=resolve)

    cli.main(["doctor", "--drive"])

    out = capsys.readouterr().out
    assert (
        "3 shortcut(s) to recordings: 1 processed from this folder, "
        "1 left to the organizer's configured folder, 1 not readable by this account"
    ) in out


def _doctor_with_shortcuts(mocker, tmp_path, shortcuts, targets):
    _doctor_config(mocker, tmp_path)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch("src.cli.drive.describe_folder", return_value=_folder_meta())
    mocker.patch("src.cli.drive.list_folder_tree_state", return_value=[])
    mocker.patch("src.cli.drive.list_subfolders", return_value=[])
    mocker.patch("src.cli.drive.list_recording_shortcuts", return_value=shortcuts)
    target_mock = mocker.patch(
        "src.cli.drive.get_shortcut_target",
        side_effect=lambda service, target_id: targets[target_id],
    )
    mocker.patch("src.main.drive.find_configured_ancestor", return_value=None)
    return target_mock


def test_doctor_counts_a_shortcut_without_a_target_as_unreadable(
    mocker, capsys, tmp_path
):
    target_mock = _doctor_with_shortcuts(
        mocker, tmp_path,
        [{"id": "s1", "name": "a.mp4", "container_id": "m1", "target_id": None}],
        {},
    )

    cli.main(["doctor", "--drive"])

    assert "1 not readable by this account" in capsys.readouterr().out
    target_mock.assert_not_called()


def test_doctor_only_asks_for_action_when_a_call_reaches_no_folder(
    mocker, capsys, tmp_path
):
    _doctor_with_shortcuts(
        mocker, tmp_path,
        [{"id": "s1", "name": "a.mp4", "container_id": "m1", "target_id": "t1"}],
        {"t1": {"id": "t1", "parents": ["clients-meeting"]}},
    )

    cli.main(["doctor", "--drive"])

    out = capsys.readouterr().out
    assert "1 processed from this folder" in out
    assert "share those recordings" not in out


def test_doctor_reports_a_shortcut_check_it_could_not_finish(mocker, capsys, tmp_path):
    """A diagnostic reports; it does not crash on the one line that failed."""
    _doctor_config(mocker, tmp_path)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch("src.cli.drive.describe_folder", return_value=_folder_meta())
    mocker.patch("src.cli.drive.list_folder_tree_state", return_value=[])
    mocker.patch("src.cli.drive.list_subfolders", return_value=[])
    mocker.patch(
        "src.cli.drive.list_recording_shortcuts",
        return_value=[{"id": "s1", "name": "a.mp4", "container_id": "m1", "target_id": "t1"}],
    )
    mocker.patch(
        "src.cli.drive.get_shortcut_target", side_effect=RuntimeError("drive is down")
    )

    cli.main(["doctor", "--drive"])

    assert "shortcuts to recordings: could not check (drive is down)" in (
        capsys.readouterr().out
    )


def test_doctor_stays_quiet_about_shortcuts_when_there_are_none(
    mocker, capsys, tmp_path
):
    _doctor_config(mocker, tmp_path)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch("src.cli.drive.describe_folder", return_value=_folder_meta())
    mocker.patch("src.cli.drive.list_folder_tree_state", return_value=[])
    mocker.patch("src.cli.drive.list_subfolders", return_value=[])
    mocker.patch("src.cli.drive.list_recording_shortcuts", return_value=[])

    cli.main(["doctor", "--drive"])

    assert "shortcut" not in capsys.readouterr().out
