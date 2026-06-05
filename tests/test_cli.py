from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from src import cli
from src.presets import Preset
from tests.test_main import make_config


@dataclass
class _Telemetry:
    cost_usd: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)


def _normalized_help(text: str) -> str:
    return " ".join(text.split())


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

    load_mock.assert_called_once_with(validate_providers=False)


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

    import_mock.assert_called_once_with("/some/creds.json")


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

    use_mock.assert_called_once_with("/c/creds.json", token_file="/c/tok.json")


def test_auth_use_files_defaults_token_file(mocker, tmp_path):
    use_mock = mocker.patch(
        "src.cli.use_google_files", return_value=tmp_path / "config.yml"
    )

    cli.main(["auth", "use-files", "--credentials-file", "/c/creds.json"])

    use_mock.assert_called_once_with("/c/creds.json", token_file=None)


def test_doctor_reports_google_source_without_secrets(mocker, capsys, tmp_path):
    from dataclasses import replace

    cfg = replace(
        make_config(folder_ids=["f1"], data_dir=tmp_path),
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
    cfg = make_config(folder_ids=["f1"], data_dir=tmp_path, stt_provider="deepgram")
    (tmp_path / "credentials.json").write_text("{}", encoding="utf-8")
    load_mock = mocker.patch("src.cli.load_config", return_value=cfg)
    build_mock = mocker.patch("src.cli.auth.build_drive_service")

    cli.main(["doctor"])

    load_mock.assert_called_once_with(validate_providers=False)
    build_mock.assert_not_called()
    out = capsys.readouterr().out
    assert "credentials.json: OK" in out
    assert "token.json: missing" in out
    assert "FOLDER_IDS: 1 configured" in out


def test_doctor_drive_check_lists_configured_folders(mocker, capsys, tmp_path):
    cfg = make_config(folder_ids=["f1"], data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    list_mock = mocker.patch("src.cli.drive.list_folder_state", return_value=[])

    cli.main(["doctor", "--drive"])

    list_mock.assert_called_once_with(service, "f1")
    out = capsys.readouterr().out
    assert "Drive auth: OK" in out
    assert "Folder f1: OK, 0 mp4 file(s)" in out


def test_doctor_reports_stt_provider_without_pipeline_readiness(mocker, capsys, tmp_path):
    cfg = make_config(folder_ids=["f1"], data_dir=tmp_path, stt_provider="deepgram")
    mocker.patch("src.cli.load_config", return_value=cfg)
    build_mock = mocker.patch("src.cli.auth.build_drive_service")

    cli.main(["doctor"])

    build_mock.assert_not_called()
    out = capsys.readouterr().out
    assert "STT_PROVIDER: deepgram" in out


def test_run_dispatch_calls_main(mocker):
    main_mock = mocker.patch("src.cli.main_module.main")

    cli.main(["run"])

    main_mock.assert_called_once_with()


def test_top_level_help_recommends_safe_operator_flow(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])

    out = _normalized_help(capsys.readouterr().out)
    assert "doctor -> list -> process <file-id> --dry-run -> process <file-id>" in out
    assert "run and folder-wide processing can spend STT credits across pending files" in out


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

    load_mock.assert_called_once_with()
    build_mock.assert_called_once_with(config=cfg)
    run_once_mock.assert_called_once_with(
        service,
        cfg,
        dry_run=False,
        max_size_bytes=None,
        confirm_large=False,
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
        dry_run=True,
        max_size_bytes=50_000_000,
        confirm_large=True,
    )


def test_process_dispatch_autodetect(mocker, tmp_path):
    cfg = make_config(data_dir=tmp_path)
    load_mock = mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    target_mock = mocker.patch("src.cli.main_module.process_target")

    cli.main(["process", "file123"])

    load_mock.assert_called_once_with()
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
    cfg = make_config(data_dir=tmp_path, folder_ids=["folderA", "folderB"])
    load_mock = mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    build_mock = mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    newest = {"id": "v9", "name": "newest.mp4"}
    find_mock = mocker.patch("src.cli.drive.find_newest_mp4", return_value=newest)
    target_mock = mocker.patch("src.cli.main_module.process_target")

    cli.main(["latest"])

    load_mock.assert_called_once_with()
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
    cfg = make_config(data_dir=tmp_path, folder_ids=["folderA"])
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
    cfg = make_config(data_dir=tmp_path, folder_ids=[])
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

    load_mock.assert_called_once_with(validate_providers=False)
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

    load_mock.assert_called_once_with()
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
    cfg = make_config(folder_ids=["f1"], data_dir=tmp_path)
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
    cfg = make_config(folder_ids=[], data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    service = MagicMock()
    mocker.patch("src.cli.auth.build_drive_service", return_value=service)
    list_mock = mocker.patch("src.cli.drive.list_folder_state", return_value=[])

    cli.main(["status", "--folder", "explicit"])

    list_mock.assert_called_once_with(service, "explicit")


def test_list_no_folders_exits(mocker, tmp_path):
    cfg = make_config(folder_ids=[], data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch("src.cli.drive.list_folder_state")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["list"])

    assert excinfo.value.code == 1


def test_list_no_folders_skips_authentication(mocker, tmp_path):
    # The empty-folder check must short-circuit before authenticating, so a
    # missing/expired token can't mask the intended local error.
    cfg = make_config(folder_ids=[], data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    build_mock = mocker.patch("src.cli.auth.build_drive_service")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["list"])

    assert excinfo.value.code == 1
    build_mock.assert_not_called()


def test_list_skips_provider_validation(mocker, tmp_path):
    cfg = make_config(folder_ids=["f1"], data_dir=tmp_path)
    load_mock = mocker.patch("src.cli.load_config", return_value=cfg)
    mocker.patch("src.cli.auth.build_drive_service", return_value=MagicMock())
    mocker.patch("src.cli.drive.list_folder_state", return_value=[])

    cli.main(["list"])

    load_mock.assert_called_once_with(validate_providers=False)


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
    cfg = make_config(data_dir=tmp_path, folder_ids=["folderA"])
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


def test_config_migrate_command_writes_file(mocker, capsys, tmp_path):
    config_file = tmp_path / "config.yml"
    migrate = mocker.patch("src.cli.migrate_config", return_value=config_file)

    cli.main(["config", "migrate"])

    migrate.assert_called_once_with(force=False)
    out = capsys.readouterr().out
    assert str(config_file) in out
    assert "Wrote configuration" in out


def test_config_migrate_command_passes_force(mocker, tmp_path):
    config_file = tmp_path / "config.yml"
    migrate = mocker.patch("src.cli.migrate_config", return_value=config_file)

    cli.main(["config", "migrate", "--force"])

    migrate.assert_called_once_with(force=True)


def test_config_migrate_command_reports_error(mocker):
    mocker.patch("src.cli.migrate_config", side_effect=ValueError("already exists"))

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["config", "migrate"])

    assert excinfo.value.code == 1


def test_config_flag_exports_env_and_doctor_reports_path(
    mocker, capsys, tmp_path, monkeypatch
):
    # Seed the var so monkeypatch restores/clears it after the test even though the
    # CLI assigns os.environ directly.
    monkeypatch.setenv(cli.CONFIG_PATH_ENV_VAR, str(tmp_path / "placeholder.yml"))
    cfg = make_config(folder_ids=["f1"], data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)
    target = tmp_path / "custom.yml"

    cli.main(["--config", str(target), "doctor"])

    assert os.environ[cli.CONFIG_PATH_ENV_VAR] == str(target)
    out = capsys.readouterr().out
    assert f"config: {target} (missing)" in out


def test_doctor_without_config_flag_leaves_env_var_untouched(
    mocker, tmp_path, monkeypatch
):
    sentinel = str(tmp_path / "from-env.yml")
    monkeypatch.setenv(cli.CONFIG_PATH_ENV_VAR, sentinel)
    cfg = make_config(folder_ids=["f1"], data_dir=tmp_path)
    mocker.patch("src.cli.load_config", return_value=cfg)

    cli.main(["doctor"])

    assert os.environ[cli.CONFIG_PATH_ENV_VAR] == sentinel


def test_doctor_reports_preset_dag(mocker, capsys, tmp_path, monkeypatch):
    monkeypatch.delenv(cli.CONFIG_PATH_ENV_VAR, raising=False)
    presets = (
        Preset(name="transcript-cleanup", instructions="clean"),
        Preset(
            name="keypoints",
            instructions="kp",
            depends_on=("transcript-cleanup",),
        ),
    )
    cfg = make_config(folder_ids=["f1"], data_dir=tmp_path, presets=presets)
    mocker.patch("src.cli.load_config", return_value=cfg)

    cli.main(["doctor"])

    out = capsys.readouterr().out
    assert "Presets: 2 enabled" in out
    assert "transcript-cleanup <- transcript" in out
    assert "keypoints <- transcript-cleanup" in out


def test_doctor_reports_no_presets_when_none_enabled(mocker, capsys, tmp_path, monkeypatch):
    monkeypatch.delenv(cli.CONFIG_PATH_ENV_VAR, raising=False)
    cfg = make_config(folder_ids=["f1"], data_dir=tmp_path, presets=())
    mocker.patch("src.cli.load_config", return_value=cfg)

    cli.main(["doctor"])

    out = capsys.readouterr().out
    assert "Presets: none enabled" in out


def test_config_init_command_dispatch(mocker, capsys, tmp_path):
    config_file = tmp_path / "config.yml"
    init = mocker.patch("src.cli.init_config", return_value=config_file)

    cli.main(["config", "init"])

    init.assert_called_once_with(
        local=False,
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
            "--local",
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
        local=True,
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
    config_file = tmp_path / "config.yml"
    monkeypatch.setenv(cli.CONFIG_PATH_ENV_VAR, str(config_file))

    cli.main(["config", "init"])

    assert config_file.is_file()
    assert (tmp_path / "prompts" / "keypoints.md").is_file()
    out = capsys.readouterr().out
    assert str(config_file) in out


def test_config_path_prints_resolved_without_secrets(monkeypatch, capsys, tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        "folder_ids: [x]\nstt:\n  provider: deepgram\n", encoding="utf-8"
    )
    monkeypatch.setenv(cli.CONFIG_PATH_ENV_VAR, str(config_file))

    cli.main(["config", "path"])

    out = capsys.readouterr().out.strip()
    assert out == str(config_file)


def test_config_path_reports_pointer_both_ends(monkeypatch, capsys, tmp_path):
    real = tmp_path / "real" / "config.yml"
    real.parent.mkdir()
    real.write_text("folder_ids: [x]\n", encoding="utf-8")
    pointer = tmp_path / "pointer.yml"
    pointer.write_text(f"config_file: {real}\n", encoding="utf-8")
    monkeypatch.setenv(cli.CONFIG_PATH_ENV_VAR, str(pointer))

    cli.main(["config", "path"])

    out = capsys.readouterr().out
    assert f"bootstrap: {pointer}" in out
    assert f"effective: {real}" in out


def test_config_link_command_dispatch(mocker, capsys, tmp_path):
    dest = tmp_path / "linked" / "config.yml"
    link = mocker.patch("src.cli.link_config", return_value=dest)

    cli.main(["config", "link", str(tmp_path / "linked")])

    link.assert_called_once_with(
        str(tmp_path / "linked"),
        copy_prompts=False,
        force=False,
    )
    out = capsys.readouterr().out
    assert str(dest) in out


def test_config_link_command_passes_flags(mocker, tmp_path):
    dest = tmp_path / "linked" / "config.yml"
    link = mocker.patch("src.cli.link_config", return_value=dest)

    cli.main(
        ["config", "link", str(tmp_path / "linked"), "--copy-prompts", "--force"]
    )

    link.assert_called_once_with(
        str(tmp_path / "linked"),
        copy_prompts=True,
        force=True,
    )


def test_config_link_command_reports_error(mocker, tmp_path):
    mocker.patch("src.cli.link_config", side_effect=ValueError("already exists"))

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["config", "link", str(tmp_path / "linked")])

    assert excinfo.value.code == 1


def test_config_get_command_dispatch(mocker, capsys):
    get = mocker.patch("src.cli.config_get", return_value="model: gpt")

    cli.main(["config", "get"])

    get.assert_called_once_with(None, show_secrets=False)
    assert "model: gpt" in capsys.readouterr().out


def test_config_get_command_passes_key(mocker, capsys):
    get = mocker.patch("src.cli.config_get", return_value="gpt-5.4")

    cli.main(["config", "get", "openai.model"])

    get.assert_called_once_with("openai.model", show_secrets=False)
    assert "gpt-5.4" in capsys.readouterr().out


def test_config_get_command_show_secrets(mocker, capsys):
    get = mocker.patch("src.cli.config_get", return_value="sk-secret")

    cli.main(["config", "get", "openai.api_key", "--show-secrets"])

    get.assert_called_once_with("openai.api_key", show_secrets=True)
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

    setter.assert_called_once_with("openai.model", "gpt-5.4")
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

    unset.assert_called_once_with("proxy_url")
    assert "Unset proxy_url" in capsys.readouterr().out


def test_config_unset_command_reports_error(mocker):
    mocker.patch("src.cli.config_unset", side_effect=ValueError("not set"))

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["config", "unset", "nope"])

    assert excinfo.value.code == 1
