from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import TextIO

from src import auth, drive
from src import main as main_module
from src import relabel_transcript
from src.config import (
    CONFIG_PATH_ENV_VAR,
    config_get,
    config_set,
    config_unset,
    import_google_credentials,
    init_config,
    link_config,
    load_config,
    migrate_config,
    resolve_config_file_path,
    resolve_effective_config_path,
    use_google_files,
)
from src.stt.transcribe import transcribe_file

logger = logging.getLogger(__name__)

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?)?\s*$", re.IGNORECASE)
_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1000,
    "kb": 1000,
    "m": 1000**2,
    "mb": 1000**2,
    "g": 1000**3,
    "gb": 1000**3,
    "t": 1000**4,
    "tb": 1000**4,
    "ki": 1024,
    "kib": 1024,
    "mi": 1024**2,
    "mib": 1024**2,
    "gi": 1024**3,
    "gib": 1024**3,
    "ti": 1024**4,
    "tib": 1024**4,
}


def _parse_size(raw: str) -> int:
    match = _SIZE_RE.match(raw)
    if not match:
        raise argparse.ArgumentTypeError(
            "expected a byte size like 50000000, 50MB, or 1.5GiB"
        )
    amount = float(match.group(1))
    unit = (match.group(2) or "").lower()
    if unit not in _SIZE_UNITS:
        raise argparse.ArgumentTypeError(f"unknown size unit: {unit}")
    size = int(amount * _SIZE_UNITS[unit])
    if size <= 0:
        raise argparse.ArgumentTypeError(
            "size must be greater than zero; --max-size 0 would skip every file"
        )
    return size


def _format_deepgram_cost(cost_usd: dict[str, float | None]) -> str:
    cost = cost_usd.get("deepgram") if cost_usd else None
    if cost is None:
        return "pending"
    return f"${cost:.4f}"


def _format_preset_usage_lines(usage: dict[str, dict[str, int]]) -> list[str]:
    """Render one OpenAI token-usage line per preset (``openai_<preset>`` key)."""
    lines: list[str] = []
    for key in sorted(usage or {}):
        if not key.startswith("openai_"):
            continue
        stats = usage.get(key) or {}
        total = stats.get("total_tokens")
        prompt = stats.get("input_tokens")
        completion = stats.get("output_tokens")
        parts = []
        if total is not None:
            parts.append(f"total={total}")
        if prompt is not None:
            parts.append(f"input={prompt}")
        if completion is not None:
            parts.append(f"output={completion}")
        if not parts:
            continue
        preset_name = key[len("openai_"):]
        lines.append(f"OpenAI {preset_name} tokens: " + ", ".join(parts))
    return lines


def _print_spend_summary(telemetry: list, *, dry_run: bool = False) -> None:
    if not telemetry:
        if dry_run:
            print("Spend summary: dry-run, nothing processed.")
        else:
            print("Spend summary: nothing processed.")
        return

    lines = ["Spend summary:"]
    total_cost = 0.0
    have_any_cost = False
    for index, item in enumerate(telemetry, start=1):
        cost_usd = getattr(item, "cost_usd", {}) or {}
        usage = getattr(item, "usage", {}) or {}
        cost = cost_usd.get("deepgram")
        if isinstance(cost, (int, float)):
            total_cost += float(cost)
            have_any_cost = True
        lines.append(f"  file {index}: Deepgram cost {_format_deepgram_cost(cost_usd)}")
        for preset_line in _format_preset_usage_lines(usage):
            lines.append(f"    {preset_line}")
    if len(telemetry) > 1 and have_any_cost:
        lines.append(f"  combined Deepgram cost: ${total_cost:.4f}")
    print("\n".join(lines))


def _configure_console_encoding(
    *, stdout: TextIO | None = None, stderr: TextIO | None = None
) -> None:
    for stream in (stdout or sys.stdout, stderr or sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue


def cmd_auth(args: argparse.Namespace) -> None:
    config = load_config(validate_providers=False)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    auth.run_interactive_flow(
        config=config,
        manual=args.manual,
        response_url=args.response_url,
    )
    logger.info("Token saved")


def cmd_auth_import_credentials(args: argparse.Namespace) -> None:
    try:
        path = import_google_credentials(args.path)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    print(f"Imported OAuth client credentials into {path}")


def cmd_auth_use_files(args: argparse.Namespace) -> None:
    try:
        path = use_google_files(
            args.credentials_file,
            token_file=args.token_file,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    print(f"Switched Google auth to file mode in {path}")


def cmd_run(args: argparse.Namespace) -> None:
    main_module.main()


def cmd_run_once(args: argparse.Namespace) -> None:
    config = load_config()
    service = auth.build_drive_service(config=config)
    main_module.run_once(
        service,
        config,
        dry_run=args.dry_run,
        max_size_bytes=args.max_size,
        confirm_large=args.confirm_large,
    )


def cmd_process(args: argparse.Namespace) -> None:
    config = load_config()
    service = auth.build_drive_service(config=config)
    is_folder = True if args.folder else None
    telemetry = main_module.process_target(
        service,
        args.target,
        config,
        is_folder=is_folder,
        reprocess_txt=args.reprocess_txt,
        dry_run=args.dry_run,
        max_size_bytes=args.max_size,
        confirm_large=args.confirm_large,
    )
    _print_spend_summary(telemetry, dry_run=args.dry_run)


def cmd_latest(args: argparse.Namespace) -> None:
    config = load_config()
    folder_id = args.folder or (config.folder_ids[0] if config.folder_ids else None)
    if not folder_id:
        logger.error("No folder to inspect; set FOLDER_IDS or pass --folder")
        raise SystemExit(1)
    if not args.folder and len(config.folder_ids) > 1:
        logger.info(
            "%d folders configured; using the first (%s). Pass --folder to pick another.",
            len(config.folder_ids), folder_id,
        )
    service = auth.build_drive_service(config=config)
    newest = drive.find_newest_mp4(service, folder_id)
    if newest is None:
        logger.info("Folder %s has no mp4 files", folder_id)
        return
    logger.info("Latest mp4 in %s: %s (%s)", folder_id, newest["name"], newest["id"])
    telemetry = main_module.process_target(
        service,
        newest["id"],
        config,
        is_folder=False,
        dry_run=args.dry_run,
        max_size_bytes=args.max_size,
        confirm_large=args.confirm_large,
    )
    _print_spend_summary(telemetry, dry_run=args.dry_run)


def _print_preset_dag(config) -> None:
    """Report the resolved preset DAG (names, dependencies, enabled state)."""
    presets = config.presets
    if not presets:
        print("Presets: none enabled")
        return
    # config.presets only ever holds enabled presets (merge_presets drops disabled
    # ones), so there is no per-preset enabled/disabled state to annotate here.
    print(f"Presets: {len(presets)} enabled")
    for preset in presets:
        deps = ", ".join(preset.depends_on) if preset.depends_on else "transcript"
        print(f"  {preset.name} <- {deps}")


def _describe_google_credentials(config) -> str:
    """Describe the OAuth client source for doctor without leaking secrets."""
    if config.google_credentials is not None:
        return "inline (config.google.credentials)"
    if config.google_credentials_file is not None:
        present = "OK" if Path(config.google_credentials_file).exists() else "missing"
        return f"file {config.google_credentials_file} ({present})"
    fallback = config.data_dir / "credentials.json"
    return f"data_dir {fallback} ({'OK' if fallback.exists() else 'missing'})"


def _describe_google_token(config) -> str:
    """Describe the saved-token source for doctor without leaking secrets."""
    if config.google_token is not None:
        return "inline (config.google.token)"
    if config.google_token_file is not None:
        present = "OK" if Path(config.google_token_file).exists() else "missing"
        return f"file {config.google_token_file} ({present})"
    fallback = config.data_dir / "token.json"
    return f"data_dir {fallback} ({'OK' if fallback.exists() else 'missing'})"


def cmd_doctor(args: argparse.Namespace) -> None:
    config_path = resolve_config_file_path()
    try:
        config = load_config(validate_providers=False)
    except ValueError as exc:
        # doctor is the command an operator runs to diagnose a broken config, so a
        # config error (e.g. an unresolvable preset prompt_file) must be reported as
        # a diagnostic line rather than crashing with a traceback.
        print(f"config: {config_path} ({'OK' if config_path.exists() else 'missing'})")
        print(f"config error: {exc}")
        raise SystemExit(1) from exc
    credentials_path = config.data_dir / "credentials.json"
    token_path = config.data_dir / "token.json"

    print(f"config: {config_path} ({'OK' if config_path.exists() else 'missing'})")
    print(f"DATA_DIR: {config.data_dir}")
    print(f"credentials.json: {'OK' if credentials_path.exists() else 'missing'}")
    print(f"token.json: {'OK' if token_path.exists() else 'missing'}")
    # Report the Google auth source without ever printing secrets (client_secret /
    # token / refresh_token stay masked; only the source kind/location is shown).
    print(f"Google credentials: {_describe_google_credentials(config)}")
    print(f"Google token: {_describe_google_token(config)}")
    print(f"FOLDER_IDS: {len(config.folder_ids)} configured")
    print(f"STT_PROVIDER: {config.stt_provider or 'not configured'}")
    _print_preset_dag(config)

    if not args.drive:
        print("Drive auth: not checked (use --drive)")
        return

    service = auth.build_drive_service(config=config)
    print("Drive auth: OK")
    for folder_id in config.folder_ids:
        items = drive.list_folder_state(service, folder_id)
        print(f"Folder {folder_id}: OK, {len(items)} mp4 file(s)")


def cmd_config_migrate(args: argparse.Namespace) -> None:
    try:
        path = migrate_config(force=args.force)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    print(f"Wrote configuration to {path}")


def cmd_config_init(args: argparse.Namespace) -> None:
    try:
        path = init_config(
            local=args.local,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            prompt_dir=args.prompt_dir,
            force=args.force,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    print(f"Wrote configuration to {path}")


def cmd_config_path(args: argparse.Namespace) -> None:
    # Resolve the path without building a validated Config so this never requires
    # Drive/Deepgram/OpenAI secrets. When a pointer is active, report both ends.
    bootstrap, effective = resolve_effective_config_path()
    if bootstrap.resolve() != effective.resolve():
        print(f"bootstrap: {bootstrap}")
        print(f"effective: {effective}")
    else:
        print(str(effective))


def cmd_config_link(args: argparse.Namespace) -> None:
    try:
        path = link_config(
            args.dir,
            copy_prompts=args.copy_prompts,
            force=args.force,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    print(f"Linked configuration to {path}")


def cmd_config_get(args: argparse.Namespace) -> None:
    try:
        output = config_get(args.key, show_secrets=args.show_secrets)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    print(output)


def cmd_config_set(args: argparse.Namespace) -> None:
    try:
        path = config_set(args.key, args.value)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    print(f"Set {args.key} in {path}")


def cmd_config_unset(args: argparse.Namespace) -> None:
    try:
        path = config_unset(args.key)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    print(f"Unset {args.key} in {path}")


def cmd_speakers_set(args: argparse.Namespace) -> None:
    config = load_config(validate_providers=False)
    service = auth.build_drive_service(config=config)
    names = json.dumps(args.names, ensure_ascii=False)
    drive.set_file_app_properties(
        service,
        args.target,
        {drive.SPEAKER_NAMES_PROPERTY: names},
    )
    logger.info("Speaker names saved on %s", args.target)


def cmd_transcribe(args: argparse.Namespace) -> None:
    config = load_config()
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        logger.error("Audio file not found: %s", audio_path)
        raise SystemExit(1)
    cost_usd: dict[str, float | None] = {}
    text = transcribe_file(audio_path, config, cost_usd=cost_usd)
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(text, encoding="utf-8")
        logger.info("Transcript written to %s", out_path)
    else:
        print(text)
    print(f"Deepgram cost: {_format_deepgram_cost(cost_usd)}")


def cmd_relabel(args: argparse.Namespace) -> None:
    map_cfg = json.loads(Path(args.mapfile).read_text(encoding="utf-8"))
    src_text = Path(args.src).read_text(encoding="utf-8")
    result = relabel_transcript.relabel(
        src_text, map_cfg, include_header=not args.no_header
    )
    Path(args.out).write_text(result, encoding="utf-8")
    logger.info("Relabeled transcript written to %s", args.out)


def cmd_list(args: argparse.Namespace) -> None:
    config = load_config(validate_providers=False)
    folder_ids = [args.folder] if args.folder else config.folder_ids
    if not folder_ids:
        logger.error("No folders to inspect; set FOLDER_IDS or pass --folder")
        raise SystemExit(1)
    service = auth.build_drive_service(config=config)
    for folder_id in folder_ids:
        items = drive.list_folder_state(service, folder_id)
        print(f"Folder {folder_id}: {len(items)} mp4 file(s)")
        for item in items:
            name = item["file"]["name"]
            mp3 = "mp3" if item.get("has_mp3") else "---"
            txt = "txt" if item.get("has_txt") else "---"
            print(f"  [{mp3}] [{txt}] {name}")


def _add_processing_safety_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without downloading, transcribing, or uploading",
    )
    parser.add_argument(
        "--max-size",
        type=_parse_size,
        default=None,
        metavar="SIZE",
        help="Skip Drive videos larger than SIZE unless --confirm-large is passed",
    )
    parser.add_argument(
        "--confirm-large",
        action="store_true",
        help="Allow processing files that exceed --max-size",
    )


def _set_parser_safety_description(
    parser: argparse.ArgumentParser,
    *,
    summary: str,
    safety_note: str,
) -> None:
    parser.description = summary
    parser.epilog = f"Safety: {safety_note}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gdstt",
        description=(
            "Operator CLI for the Google Drive video STT service. Prefer "
            "doctor -> list -> process <file-id> --dry-run -> process <file-id> "
            "before folder-wide run-once or run."
        ),
        epilog=(
            "Safety: run and folder-wide processing can spend STT credits across pending "
            "files. Start with --dry-run when the command supports it."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            "Path to config.yml (overrides the GDSTT_CONFIG env var and the default "
            "<data_dir>/config.yml). Must appear before the subcommand."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_auth = sub.add_parser("auth", help="Run the browser or manual OAuth flow")
    p_auth.add_argument(
        "--manual",
        action="store_true",
        help="Print the authorization URL instead of opening a browser",
    )
    p_auth.add_argument(
        "response_url",
        nargs="?",
        default=None,
        help="OAuth redirect URL for the manual flow (optional)",
    )
    p_auth.set_defaults(func=cmd_auth)

    p_auth_import = sub.add_parser(
        "auth-import-credentials",
        help="Store an OAuth client JSON inline in the config (google.credentials)",
    )
    p_auth_import.add_argument(
        "path", help="Path to the downloaded OAuth client (Desktop app) JSON file"
    )
    p_auth_import.set_defaults(func=cmd_auth_import_credentials)

    p_auth_use_files = sub.add_parser(
        "auth-use-files",
        help="Switch Google auth to file mode and clear inline credentials/token",
    )
    p_auth_use_files.add_argument(
        "--credentials-file",
        required=True,
        metavar="PATH",
        help="Path the OAuth client JSON lives at (google.credentials_file)",
    )
    p_auth_use_files.add_argument(
        "--token-file",
        default=None,
        metavar="PATH",
        help="Path the saved token lives at (default: <credentials parent>/token.json)",
    )
    p_auth_use_files.set_defaults(func=cmd_auth_use_files)

    p_run = sub.add_parser(
        "run",
        help="Run the polling loop for all pending configured folders",
    )
    _set_parser_safety_description(
        p_run,
        summary="Run the polling loop continuously.",
        safety_note=(
            "this command can process every pending configured folder and spend STT "
            "credits repeatedly. Prefer run-once --dry-run or process <file-id> --dry-run "
            "before using it."
        ),
    )
    p_run.set_defaults(func=cmd_run)

    p_run_once = sub.add_parser(
        "run-once",
        help="Run one polling cycle across pending configured folders",
    )
    _set_parser_safety_description(
        p_run_once,
        summary="Run a single polling cycle across the configured folders.",
        safety_note=(
            "this command can spend STT credits across multiple pending files. Use --dry-run "
            "first and add --max-size only as an optional manual limit for larger folder runs."
        ),
    )
    _add_processing_safety_args(p_run_once)
    p_run_once.set_defaults(func=cmd_run_once)

    p_process = sub.add_parser(
        "process",
        help="Process one Drive file or folder on demand",
    )
    _set_parser_safety_description(
        p_process,
        summary="Process one Drive file or folder on demand.",
        safety_note=(
            "use --dry-run first. When the target is a folder, this command can process many "
            "files and spend STT credits. --reprocess-txt intentionally reruns STT and overwrites "
            "the linked .txt."
        ),
    )
    p_process.add_argument("target", help="Drive file ID or folder ID")
    p_process.add_argument(
        "--folder",
        action="store_true",
        help="Treat the target as a folder ID (default: auto-detect)",
    )
    p_process.add_argument(
        "--reprocess-txt",
        action="store_true",
        help="Run STT again and overwrite the existing TXT instead of skipping it",
    )
    _add_processing_safety_args(p_process)
    p_process.set_defaults(func=cmd_process)

    p_latest = sub.add_parser(
        "latest",
        help="Process the newest mp4 in a folder",
    )
    _set_parser_safety_description(
        p_latest,
        summary="Process the most recently created mp4 in a folder.",
        safety_note=(
            "this command spends STT credits on the newest mp4. Use --dry-run first to "
            "confirm which file would be processed."
        ),
    )
    p_latest.add_argument(
        "--folder",
        default=None,
        help="Folder ID to inspect (default: first of FOLDER_IDS)",
    )
    p_latest.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which file would be processed without downloading or transcribing",
    )
    p_latest.add_argument(
        "--max-size",
        type=_parse_size,
        default=None,
        metavar="SIZE",
        help="Skip the newest mp4 if it is larger than SIZE unless --confirm-large is passed",
    )
    p_latest.add_argument(
        "--confirm-large",
        action="store_true",
        help="Allow processing the newest mp4 even if it exceeds --max-size",
    )
    p_latest.set_defaults(func=cmd_latest)

    p_doctor = sub.add_parser(
        "doctor",
        help="Check local Drive/OAuth configuration without changing anything",
    )
    p_doctor.add_argument(
        "--drive",
        action="store_true",
        help="Also authenticate and list configured Drive folders",
    )
    p_doctor.set_defaults(func=cmd_doctor)

    p_config = sub.add_parser(
        "config",
        help="Manage gdstt configuration (data/config.yml)",
    )
    config_sub = p_config.add_subparsers(dest="config_command", required=True)
    p_config_migrate = config_sub.add_parser(
        "migrate",
        help="Generate data/config.yml from the current .env/environment",
    )
    p_config_migrate.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config.yml",
    )
    p_config_migrate.set_defaults(func=cmd_config_migrate)

    p_config_init = config_sub.add_parser(
        "init",
        help="Create a fresh config.yml with default presets and prompt assets",
    )
    p_config_init.add_argument(
        "--local",
        action="store_true",
        help="Write ./data/config.yml in the current directory instead of the user path",
    )
    p_config_init.add_argument(
        "--data-dir",
        default=None,
        metavar="PATH",
        help="Set data_dir in the generated config",
    )
    p_config_init.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help="Write artifacts to this local folder (sets output.target=folder)",
    )
    p_config_init.add_argument(
        "--prompt-dir",
        default=None,
        metavar="PATH",
        help="Copy prompt assets here and point prompt_file entries at this directory",
    )
    p_config_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config.yml",
    )
    p_config_init.set_defaults(func=cmd_config_init)

    p_config_path = config_sub.add_parser(
        "path",
        help="Print the resolved config.yml path without requiring provider secrets",
    )
    p_config_path.set_defaults(func=cmd_config_path)

    p_config_link = config_sub.add_parser(
        "link",
        help="Move the effective config to DIR/config.yml and leave a pointer behind",
    )
    p_config_link.add_argument("dir", help="Directory to hold the full config.yml")
    p_config_link.add_argument(
        "--copy-prompts",
        action="store_true",
        help="Copy prompt assets into DIR/prompts/ if missing",
    )
    p_config_link.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing DIR/config.yml",
    )
    p_config_link.set_defaults(func=cmd_config_link)

    p_config_get = config_sub.add_parser(
        "get",
        help="Print the effective config (secrets masked) or one dotted KEY value",
    )
    p_config_get.add_argument(
        "key",
        nargs="?",
        default=None,
        metavar="KEY",
        help="Dotted key (e.g. openai.model); omit to print the whole masked config",
    )
    p_config_get.add_argument(
        "--show-secrets",
        action="store_true",
        help="Reveal secret values (api keys, tokens) instead of masking them",
    )
    p_config_get.set_defaults(func=cmd_config_get)

    p_config_set = config_sub.add_parser(
        "set",
        help="Set a dotted KEY to VALUE in the effective config and validate it",
    )
    p_config_set.add_argument("key", metavar="KEY", help="Dotted key (e.g. openai.api_key)")
    p_config_set.add_argument("value", metavar="VALUE", help="New value for the key")
    p_config_set.set_defaults(func=cmd_config_set)

    p_config_unset = config_sub.add_parser(
        "unset",
        help="Remove an optional dotted KEY from the effective config",
    )
    p_config_unset.add_argument("key", metavar="KEY", help="Dotted key to remove")
    p_config_unset.set_defaults(func=cmd_config_unset)

    p_speakers = sub.add_parser(
        "speakers",
        help="Manage explicit speaker names stored on a Drive MP4",
    )
    speakers_sub = p_speakers.add_subparsers(dest="speakers_command", required=True)
    p_speakers_set = speakers_sub.add_parser(
        "set",
        help="Set speaker names for future post-processing of a Drive MP4",
    )
    p_speakers_set.add_argument("target", help="Drive MP4 file ID")
    p_speakers_set.add_argument("names", nargs="+", help="Speaker names in order")
    p_speakers_set.set_defaults(func=cmd_speakers_set)

    p_transcribe = sub.add_parser(
        "transcribe", help="Transcribe a local audio file with the configured provider"
    )
    p_transcribe.add_argument("audio", help="Path to a local audio file (e.g. an MP3)")
    p_transcribe.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write the transcript to this path instead of stdout",
    )
    p_transcribe.set_defaults(func=cmd_transcribe)

    p_relabel = sub.add_parser(
        "relabel",
        help="Rename transcript speakers deterministically using a MAP.json",
    )
    p_relabel.add_argument(
        "--in", dest="src", required=True, help="Path to the source transcript"
    )
    p_relabel.add_argument(
        "--out", dest="out", required=True, help="Path to write the relabeled transcript"
    )
    p_relabel.add_argument(
        "--map", dest="mapfile", required=True, help="Path to the MAP.json mapping file"
    )
    p_relabel.add_argument(
        "--no-header",
        action="store_true",
        help="Skip the MAP.json header even when one is present",
    )
    p_relabel.set_defaults(func=cmd_relabel)

    p_list = sub.add_parser(
        "list",
        aliases=["status"],
        help="Show folder state (sibling MP3/TXT presence) without doing work",
    )
    p_list.add_argument(
        "--folder",
        default=None,
        help="Folder ID to inspect (default: configured FOLDER_IDS)",
    )
    p_list.set_defaults(func=cmd_list)

    return parser


# ``auth`` keeps an optional positional ``response_url`` for the manual flow, which
# argparse cannot combine with nested subcommands. Rewrite ``auth import-credentials``
# / ``auth use-files`` into flat top-level commands so the operator still types the
# spec'd ``gdstt auth <verb>`` form while the parser stays unambiguous.
_AUTH_SUBCOMMANDS = {
    "import-credentials": "auth-import-credentials",
    "use-files": "auth-use-files",
}


def _rewrite_auth_subcommand(argv: list[str]) -> list[str]:
    # Find the first non-option token (the command), skipping the global --config PATH.
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--config":
            i += 2
            continue
        if token.startswith("--config="):
            i += 1
            continue
        break
    if i + 1 < len(argv) and argv[i] == "auth" and argv[i + 1] in _AUTH_SUBCOMMANDS:
        flat = _AUTH_SUBCOMMANDS[argv[i + 1]]
        return [*argv[:i], flat, *argv[i + 2:]]
    return argv


def main(argv: list[str] | None = None) -> None:
    _configure_console_encoding()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    argv = _rewrite_auth_subcommand(list(argv))
    args = parser.parse_args(argv)
    # --config is a bootstrap pointer to config.yml, not an application setting;
    # surface it through the same GDSTT_CONFIG env var load_config already resolves
    # so every command honors it without threading config_path through each call.
    if getattr(args, "config", None):
        os.environ[CONFIG_PATH_ENV_VAR] = args.config
    args.func(args)


if __name__ == "__main__":
    main()
