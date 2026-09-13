from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import TextIO

from src import auth, booking_gate, call_booking, change_cursor, drive, meta_doc
from src import main as main_module
from src import preset_pipeline, relabel_transcript
from src.config import (
    config_get,
    config_set,
    config_unset,
    import_google_credentials,
    init_config,
    load_config,
    parse_since,
    resolve_config_file_path,
    set_run_enabled,
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
        if "deepgram" in cost_usd:
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
    config = load_config(validate_providers=False, config_path=args.config)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    auth.run_interactive_flow(
        config=config,
        manual=args.manual,
        response_url=args.response_url,
    )
    logger.info("Token saved")


def cmd_auth_import_credentials(args: argparse.Namespace) -> None:
    try:
        path = import_google_credentials(args.path, config_path=args.config)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    print(f"Imported OAuth client credentials into {path}")


def cmd_auth_use_files(args: argparse.Namespace) -> None:
    try:
        path = use_google_files(
            args.credentials_file,
            token_file=args.token_file,
            config_path=args.config,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    print(f"Switched Google auth to file mode in {path}")


def cmd_run(args: argparse.Namespace) -> None:
    # Validate the config before enabling so a broken/missing config.yml fails fast
    # with the `gdstt config init` setup error instead of starting a doomed loop.
    load_config(config_path=args.config)
    # Explicitly resume: clear any sticky `gdstt stop` flag so an operator's
    # `gdstt run` always starts the polling loop.
    set_run_enabled(True, config_path=args.config)
    main_module.main(config_path=args.config)


def cmd_start(args: argparse.Namespace) -> None:
    try:
        path = set_run_enabled(True, config_path=args.config)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    print(
        f"Set run.enabled=true in {path}. A paused `gdstt run` loop resumes "
        f"processing on its next cycle."
    )


def _since_argument(value: str) -> str:
    """Validate `--since` at parse time so a typo fails before Drive is touched."""
    try:
        parsed = parse_since(value, source="--since")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return parsed.isoformat() if parsed is not None else ""


def cmd_run_once(args: argparse.Namespace) -> None:
    config = load_config(config_path=args.config)
    service = auth.build_drive_service(config=config)
    main_module.run_once(
        service,
        config,
        dry_run=args.dry_run,
        max_size_bytes=args.max_size,
        confirm_large=args.confirm_large,
        # No --mode means "do what the service would do", so a deployment pinned to
        # `run.discovery: walk` is not silently exercised on the other path.
        mode=args.mode or config.run_discovery,
        since=args.since or "",
    )


def cmd_process(args: argparse.Namespace) -> None:
    config = load_config(config_path=args.config)
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
    config = load_config(config_path=args.config)
    folder_id = args.folder or (config.folders[0].folder_id if config.folders else None)
    if not folder_id:
        logger.error("No folder to inspect; configure folders or pass --folder")
        raise SystemExit(1)
    if not args.folder and len(config.folders) > 1:
        logger.info(
            "%d folders configured; using the first (%s). Pass --folder to pick another.",
            len(config.folders), folder_id,
        )
    service = auth.build_drive_service(config=config)
    newest = drive.find_newest_mp4_in_tree(service, folder_id)
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


def _stage_names(config) -> list[str]:
    """Enabled presets in chain (topological) order, used as reprocess stages 1..N."""
    return preset_pipeline.topological_order(config.presets)


def _print_preset_dag(config) -> None:
    """Report the resolved preset DAG with reprocess stage numbers.

    Stage 0 is the transcript (Deepgram base); 1..N are the enabled presets in chain
    order. These numbers are what ``gdstt reprocess <target> <stages>`` accepts.
    """
    presets = {preset.name: preset for preset in config.presets}
    if not presets:
        print("Presets: none enabled (stage 0 = transcript only)")
        return
    # config.presets only ever holds enabled presets (merge_presets drops disabled
    # ones), so there is no per-preset enabled/disabled state to annotate here.
    print(f"Presets: {len(presets)} enabled (reprocess stages)")
    print("  0. transcript (Deepgram base)")
    for index, name in enumerate(_stage_names(config), start=1):
        deps = presets[name].depends_on
        src = ", ".join(deps) if deps else "transcript"
        print(f"  {index}. {name} <- {src}")


def _parse_stage_spec(spec: str | None, stage_names: list[str]) -> tuple[bool, list[str]]:
    """Resolve a reprocess stage spec into (reprocess_transcript, preset_names).

    ``spec`` accepts numbers, ranges, and lists over stages 0..N where 0 is the
    transcript and 1..N are the enabled presets in chain order: ``"3"``, ``"2-3"``,
    ``"1,3"``, ``"0"`` (transcript + everything downstream), or empty/``"all"`` for
    every preset (transcript kept). Raises ``ValueError`` with the valid range on a
    bad token. Returns ``(True, [])`` when the transcript is included (a full
    re-transcribe regenerates all presets), else ``(False, selected names)``.
    """
    text = (spec or "").strip().lower()
    if text in ("", "all"):
        return False, list(stage_names)
    numbers: set[int] = set()
    for token in text.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            lo_s, _, hi_s = token.partition("-")
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError as exc:
                raise ValueError(f"invalid stage range: {token!r}") from exc
            if lo > hi:
                lo, hi = hi, lo
            numbers.update(range(lo, hi + 1))
        else:
            try:
                numbers.add(int(token))
            except ValueError as exc:
                raise ValueError(f"invalid stage number: {token!r}") from exc
    max_stage = len(stage_names)
    for n in numbers:
        if n < 0 or n > max_stage:
            raise ValueError(
                f"stage {n} out of range; valid stages are 0..{max_stage} "
                f"(0=transcript, 1..{max_stage}=presets)"
            )
    if 0 in numbers:
        # Re-transcribing regenerates every downstream preset, so the explicit
        # preset selection collapses to a full reprocess.
        return True, []
    return False, [stage_names[n - 1] for n in sorted(numbers)]


def cmd_reprocess(args: argparse.Namespace) -> None:
    config = load_config(config_path=args.config)
    stage_names = _stage_names(config)
    try:
        reprocess_txt, preset_names = _parse_stage_spec(args.stages, stage_names)
    except ValueError as exc:
        logger.error("%s", exc)
        print("Stages: 0=transcript" + "".join(
            f", {i}={name}" for i, name in enumerate(stage_names, start=1)
        ))
        raise SystemExit(1) from exc
    if not reprocess_txt and not preset_names:
        logger.error("No presets enabled to reprocess; stage 0 (transcript) only.")
        raise SystemExit(1)
    plan = "transcript + all presets" if reprocess_txt else ", ".join(preset_names)
    print(f"Reprocess plan for {args.target}: {plan}")
    service = auth.build_drive_service(config=config)
    is_folder = True if args.folder else None
    telemetry = main_module.process_target(
        service,
        args.target,
        config,
        is_folder=is_folder,
        reprocess_txt=reprocess_txt,
        reprocess_presets=preset_names or None,
        dry_run=args.dry_run,
        max_size_bytes=args.max_size,
        confirm_large=args.confirm_large,
    )
    _print_spend_summary(telemetry, dry_run=args.dry_run)


def cmd_stop(args: argparse.Namespace) -> None:
    try:
        path = set_run_enabled(False, config_path=args.config)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    print(
        f"Set run.enabled=false in {path}. A running `gdstt run` loop pauses "
        f"(goes idle) after its current cycle and stays paused across restarts; "
        f"resume with `gdstt start` (or `gdstt run`)."
    )


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


def _describe_employee(folder) -> str:
    """Describe a folder's employee for doctor; name/email are both optional."""
    if folder.name and folder.email:
        return f"{folder.name} <{folder.email}>"
    return folder.name or folder.email or "(no employee configured)"


def _print_folder_diagnosis(service, folder_id: str, configured_ids: set[str]) -> None:
    """One line per configured folder, saying enough to spot a folder gone quiet.

    Reachability alone is what made this service look healthy while it was finding
    nothing, so this reports what the folder *is* and when it last received anything,
    not only that it answered.
    """
    ancestors: dict[str, str | None] = {}
    try:
        meta = drive.describe_folder(service, folder_id)
    except Exception as exc:  # noqa: BLE001 -- a diagnostic must report, not raise
        print(f"Folder {folder_id}: UNREACHABLE ({exc})")
        return

    name = meta.get("name") or "(no name)"
    parents = ", ".join(meta.get("parents") or []) or "none"
    trashed = " TRASHED" if meta.get("trashed") else ""
    try:
        items = main_module._without_calls_the_organizer_covers(
            service,
            drive.list_folder_tree_state(service, folder_id),
            configured_ids,
            ancestors,
        )
        subfolders = drive.list_subfolders(service, folder_id)
    except Exception as exc:  # noqa: BLE001
        print(f"Folder {folder_id}: {name!r}{trashed}, parent {parents}, listing failed ({exc})")
        return

    newest = max(
        (it["file"].get("createdTime", "") for it in items), default=""
    )
    print(
        f"Folder {folder_id}: {name!r}{trashed}, parent {parents}, "
        f"{len(subfolders)} subfolder(s), {len(items)} mp4 file(s), "
        f"newest {newest or 'never'}"
    )

    # What becomes of each call this person only attended, said out loud. Without it
    # a folder reports every recording it holds as handled while the meetings they
    # only attended -- a shortcut each -- could go missing without a trace.
    try:
        shortcuts = drive.list_recording_shortcuts(service, folder_id)
        followed = organizers = unreadable = 0
        for shortcut in shortcuts:
            target = (
                drive.get_shortcut_target(service, shortcut["target_id"])
                if shortcut["target_id"] else None
            )
            if target is None:
                unreadable += 1
            elif main_module._is_in_a_configured_folder(
                service, target.get("parents"), configured_ids, ancestors
            ):
                organizers += 1
            else:
                followed += 1
    except Exception as exc:  # noqa: BLE001
        print(f"  shortcuts to recordings: could not check ({exc})")
        return
    if not shortcuts:
        return
    line = (
        f"  {len(shortcuts)} shortcut(s) to recordings: {followed} processed from this "
        f"folder, {organizers} left to the organizer's configured folder, {unreadable} "
        "not readable by this account"
    )
    if unreadable:
        # The only outcome an operator can act on: those calls reach no folder at all.
        line += (
            " -- share those recordings with this account, or configure the "
            "organizer's folder"
        )
    print(line)


def cmd_doctor(args: argparse.Namespace) -> None:
    config_path = resolve_config_file_path(args.config)
    try:
        config = load_config(validate_providers=False, config_path=args.config)
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
    print(f"data dir: {config.data_dir}")
    print(f"credentials.json: {'OK' if credentials_path.exists() else 'missing'}")
    print(f"token.json: {'OK' if token_path.exists() else 'missing'}")
    # Report the Google auth source without ever printing secrets (client_secret /
    # token / refresh_token stay masked; only the source kind/location is shown).
    print(f"Google credentials: {_describe_google_credentials(config)}")
    print(f"Google token: {_describe_google_token(config)}")
    print(f"folders: {len(config.folders)} configured")
    for folder in config.folders:
        print(f"  {folder.folder_id}: {_describe_employee(folder)}")
    print(f"stt.provider: {config.stt_provider or 'not configured'}")
    _print_preset_dag(config)
    print(
        f"call_booking: enabled={config.call_booking_enabled}, "
        f"listen={config.call_booking_listen_host}:{config.call_booking_listen_port}, "
        f"token={'set' if config.call_booking_token else 'unset'}, "
        f"threshold_minutes={config.call_booking_threshold_minutes}, "
        f"disable_recognition={config.call_booking_disable_recognition}"
    )
    print(
        f"planfix: url={'set' if config.planfix_create_comment_url else 'unset'}, "
        f"token={'set' if config.planfix_token else 'unset'}, "
        f"presets={', '.join(config.planfix_presets) or '(none)'}"
    )
    print(f"call bookings journal: {config.call_bookings_file}")

    if not args.drive:
        print("Drive auth: not checked (use --drive)")
        return

    service = auth.build_drive_service(config=config)
    print("Drive auth: OK")
    cursor_path = change_cursor.path_for(config.data_dir)
    saved_cursor = change_cursor.read(cursor_path)
    if not saved_cursor:
        state = "absent, next cycle sweeps"
    elif change_cursor.read_folders(
        change_cursor.folders_path_for(config.data_dir)
    ) == change_cursor.fingerprint(
        folder.folder_id for folder in config.folders
    ):
        state = "set, covers the configured folders"
    else:
        state = "set, but the configured folders changed -- next cycle sweeps once"
    print(f"changes cursor: {cursor_path} ({state})")
    print(f"discovery: run.discovery={config.run_discovery}")
    print(f"since: run.since={config.run_since or 'unset, every recording in scope'}")
    configured_ids = {folder.folder_id for folder in config.folders}
    for folder in config.folders:
        _print_folder_diagnosis(service, folder.folder_id, configured_ids)


def cmd_config_init(args: argparse.Namespace) -> None:
    try:
        path = init_config(
            config_path=args.config,
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
    # Drive/Deepgram/OpenAI secrets.
    print(str(resolve_config_file_path(args.config)))


def cmd_config_get(args: argparse.Namespace) -> None:
    try:
        output = config_get(
            args.key, config_path=args.config, show_secrets=args.show_secrets
        )
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    print(output)


def cmd_config_set(args: argparse.Namespace) -> None:
    try:
        path = config_set(args.key, args.value, config_path=args.config)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    print(f"Set {args.key} in {path}")


def cmd_config_unset(args: argparse.Namespace) -> None:
    try:
        path = config_unset(args.key, config_path=args.config)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    print(f"Unset {args.key} in {path}")


def cmd_speakers_set(args: argparse.Namespace) -> None:
    config = load_config(validate_providers=False, config_path=args.config)
    service = auth.build_drive_service(config=config)
    names = json.dumps(args.names, ensure_ascii=False)
    drive.set_file_app_properties(
        service,
        args.target,
        {drive.SPEAKER_NAMES_PROPERTY: names},
    )
    logger.info("Speaker names saved on %s", args.target)


def cmd_bookings_list(args: argparse.Namespace) -> None:
    """Print the booking journal after dedupe and pruning.

    This is the first thing to look at when a recording did not match: it shows what
    the matcher actually had to work with.
    """
    config = load_config(config_path=args.config, validate_providers=False)
    bookings = call_booking.load(config.call_bookings_file)
    if not bookings:
        print(f"No bookings in {config.call_bookings_file}")
        return
    for booking in sorted(bookings, key=lambda b: b.start_time):
        print(
            f"{booking.task_id}\t{booking.manager_email}\t"
            f"{booking.start_time.isoformat()}"
        )


def cmd_bookings_rematch(args: argparse.Namespace) -> None:
    """Clear the unmatched mark so the polling loop reconsiders a recording."""
    config = load_config(config_path=args.config, validate_providers=False)
    service = auth.build_drive_service(config=config)
    booking_gate.clear_mark(service, args.target)
    print(
        f"Cleared the unmatched mark on {args.target}; "
        f"the next polling cycle will reconsider it"
    )


def cmd_planfix_sent(args: argparse.Namespace) -> None:
    """List every recording whose Planfix comment was actually delivered.

    The ``planfix_comment_task_id`` appProperty is written only after a successful
    POST, so it -- not the booking journal -- is the record of what was sent: a booking
    can exist for a comment that never landed, and the journal is pruned besides. The
    manager comes from the folder the recording sits in.

    Newest first and capped by ``--limit``, because the question this answers is almost
    always "what happened lately".
    """
    config = load_config(config_path=args.config, validate_providers=False)
    service = auth.build_drive_service(config=config)
    rows: list[tuple[str, str, str, str, str]] = []
    for folder in config.folders:
        for item in drive.list_mp4_timestamps_in_tree(service, folder.folder_id):
            task_id = (item.get("appProperties") or {}).get(
                drive.PLANFIX_COMMENT_TASK_ID_PROPERTY, ""
            )
            if task_id:
                rows.append(
                    (
                        item.get("createdTime", ""),
                        task_id,
                        folder.name,
                        item.get("name", ""),
                        # A call followed through a shortcut keeps its marker on the
                        # shortcut; the link should still open the recording.
                        (item.get("shortcutDetails") or {}).get("targetId")
                        or item.get("id", ""),
                    )
                )

    if not rows:
        print("No recording carries a sent-comment marker.")
        return

    # Newest first: this is a "what happened lately" listing, and the calls an operator
    # is asking about are the ones that just happened.
    rows.sort(reverse=True)
    shown = rows if args.limit <= 0 else rows[: args.limit]

    # One record per block rather than one per line: the two links are the point of
    # this command, and a terminal only makes them clickable when they stand alone.
    for index, (created, task_id, manager, name, file_id) in enumerate(shown):
        if index:
            print()
        print(f"{created}\t{manager}")
        print(meta_doc.task_url(config.planfix_task_url, task_id) or f"task {task_id}")
        print(meta_doc.video_url(file_id))
        print(name)

    # Say what was left out. A truncated list that looks complete is worse than a long
    # one, because nobody goes looking for the calls they were never told about.
    if len(shown) < len(rows):
        print(f"\n{len(shown)} of {len(rows)} shown; --limit 0 for all")


def cmd_bookings_restore_dates(args: argparse.Namespace) -> None:
    """Restore modifiedTime on recordings whose date the unmatched mark moved.

    Writing ``booking_match=none`` counted as an edit in Drive and moved every marked
    recording's date, which broke sorting in the shared folders. This walks the
    configured folders and puts each marked file's modifiedTime back to its
    createdTime -- the closest recoverable value, since the original was overwritten.
    """
    config = load_config(config_path=args.config, validate_providers=False)
    service = auth.build_drive_service(config=config)
    total = 0
    for folder in config.folders:
        files = drive.list_mp4_timestamps_in_tree(service, folder.folder_id)
        for file_id, name, created in booking_gate.select_stale_marks(files):
            total += 1
            if args.dry_run:
                print(f"would restore\t{file_id}\t{created}\t{name}")
                continue
            drive.set_file_modified_time(service, file_id, created)
            print(f"restored\t{file_id}\t{created}\t{name}")
    if args.dry_run:
        print(f"{total} file(s) would be restored; re-run without --dry-run to apply")
    else:
        print(f"Restored modifiedTime on {total} file(s)")


def cmd_transcribe(args: argparse.Namespace) -> None:
    config = load_config(config_path=args.config)
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


def cmd_changes(args: argparse.Namespace) -> None:
    """Show what the changes feed reports, without acting on it or moving the cursor.

    The question this answers is "does Drive think anything happened", asked before
    the next cycle rather than after it. Read-only on purpose: an operator looking
    into the feed must not consume it, or the cycle that follows would find nothing
    and the recording would be skipped.
    """
    config = load_config(validate_providers=False, config_path=args.config)
    cursor_path = change_cursor.path_for(config.data_dir)
    cursor = change_cursor.read(cursor_path)
    if cursor is None:
        print(f"No cursor at {cursor_path}; the next cycle sweeps every folder.")
        return

    # Same question the cycle asks itself. Without it this command would report
    # "nothing of ours" for a folder just added to the config and be right about the
    # feed while being useless to the operator.
    if change_cursor.read_folders(
        change_cursor.folders_path_for(config.data_dir)
    ) != change_cursor.fingerprint(
        folder.folder_id for folder in config.folders
    ):
        print(
            "The configured folders changed since this cursor was taken; the feed "
            "cannot show what was already in a folder added since. The next cycle "
            "sweeps once."
        )

    service = auth.build_drive_service(config=config)
    entries, next_cursor = drive.list_changes(service, cursor)
    print(f"{len(entries)} change(s) since the saved cursor")

    if args.raw:
        for entry in entries:
            file_info = entry.get("file") or {}
            state = "removed" if entry.get("removed") else file_info.get("mimeType", "?")
            print(f"  {entry.get('fileId')}  {state}  {file_info.get('name', '')}")
    else:
        configured = {folder.folder_id for folder in config.folders}
        ancestors: dict[str, str | None] = {}
        shown = 0
        for entry in entries:
            file_info = entry.get("file") or {}
            if entry.get("removed") or file_info.get("trashed"):
                continue
            if not drive.names_a_recording(file_info):
                continue
            parents = file_info.get("parents") or []
            if not parents:
                continue
            owner = drive.find_configured_ancestor(
                service, parents[0], configured, cache=ancestors
            )
            if owner is None:
                continue
            shown += 1
            via = ", shortcut" if file_info.get("mimeType") == drive.SHORTCUT_MIME else ""
            print(f"  {file_info.get('name')}  in {parents[0]}  (folder {owner}{via})")
        if not shown:
            print("  nothing of ours; pass --raw to see every entry")

    print(f"cursor would move to {next_cursor}; not saved")


def cmd_cursor_show(args: argparse.Namespace) -> None:
    config = load_config(validate_providers=False, config_path=args.config)
    path = change_cursor.path_for(config.data_dir)
    cursor = change_cursor.read(path)
    print(f"path: {path}")
    if cursor is None:
        print("cursor: absent -- the next cycle sweeps every folder")
        return
    print(f"cursor: {cursor}")
    watched = change_cursor.fingerprint(
        folder.folder_id for folder in config.folders
    )
    vouched = change_cursor.read_folders(
        change_cursor.folders_path_for(config.data_dir)
    )
    if vouched is None:
        print(
            "folders: not recorded -- the next cycle sweeps once and records them"
        )
    elif vouched == watched:
        print(f"folders: {len(watched.splitlines())} watched, all covered")
    else:
        added = sorted(set(watched.splitlines()) - set(vouched.splitlines()))
        dropped = sorted(set(vouched.splitlines()) - set(watched.splitlines()))
        print(
            "folders: changed since the cursor was taken -- the next cycle sweeps "
            "once so nothing already sitting in a new folder is missed"
        )
        for folder_id in added:
            print(f"  added:   {folder_id}")
        for folder_id in dropped:
            print(f"  dropped: {folder_id}")


def cmd_cursor_reset(args: argparse.Namespace) -> None:
    """Forget the cursor so the next cycle re-reads every folder.

    The one safe big hammer in this service: a sweep re-derives what is done from
    what is next to each video, so the worst it costs is a slower cycle.
    """
    config = load_config(validate_providers=False, config_path=args.config)
    path = change_cursor.path_for(config.data_dir)
    # The folder set goes with it: left behind, it would vouch for a cursor that no
    # longer exists.
    change_cursor.clear_folders(change_cursor.folders_path_for(config.data_dir))
    if change_cursor.clear(path):
        print(f"Removed {path}; the next cycle sweeps every folder.")
    else:
        print(f"No cursor at {path}; the next cycle already sweeps.")


def cmd_list(args: argparse.Namespace) -> None:
    config = load_config(validate_providers=False, config_path=args.config)
    folder_ids = [args.folder] if args.folder else [f.folder_id for f in config.folders]
    if not folder_ids:
        logger.error("No folders to inspect; configure folders or pass --folder")
        raise SystemExit(1)
    service = auth.build_drive_service(config=config)
    configured_ids = {folder.folder_id for folder in config.folders}
    ancestors: dict[str, str | None] = {}
    for folder_id in folder_ids:
        items = main_module._without_calls_the_organizer_covers(
            service,
            drive.list_folder_tree_state(service, folder_id),
            configured_ids,
            ancestors,
        )
        # Without this the report and the service disagree: `list` would show eight
        # recordings with no transcript while every cycle skipped all eight, and the
        # operator would be left wondering which one was lying.
        cutoff = parse_since(config.since_for(folder_id), source="since")
        print(f"Folder {folder_id}: {len(items)} mp4 file(s)")
        for item in items:
            name = item["file"]["name"]
            when = main_module._recording_datetime(item)
            out_of_scope = cutoff is not None and when is not None and when < cutoff
            mp3 = "mp3" if item.get("has_mp3") else "---"
            txt = "txt" if item.get("has_txt") else "---"
            # The container is worth showing even when it equals the folder asked
            # about: it is where the artifacts went, and with meeting subfolders the
            # operator can no longer assume which folder that was.
            where = item.get("container_id") or folder_id
            scope = "  before since, not processed" if out_of_scope else ""
            via = "  via shortcut" if item.get("media_id") else ""
            print(f"  [{mp3}] [{txt}] {name}  ({where}){via}{scope}")


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
            "One-shot path to a config.yml, overriding GDSTT_HOME/config.yml (or the "
            "default ./data/config.yml). Not persisted. Must appear before the subcommand."
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
    p_run_once.add_argument(
        "--mode",
        choices=("auto", "walk", "changes"),
        default=None,
        help=(
            "How to find work: 'auto' reads the changes feed when a cursor exists and "
            "sweeps otherwise; 'walk' sweeps every folder without touching the cursor; "
            "'changes' only reads the feed and fails when there is no cursor. "
            "Defaults to run.discovery from the config, which the service itself uses"
        ),
    )
    p_run_once.add_argument(
        "--since",
        type=_since_argument,
        default=None,
        metavar="DATE",
        help=(
            "Ignore recordings of calls before this date (2026-09-12 or an ISO "
            "timestamp), overriding run.since and any folder's own since for this "
            "run. The date is read from the recording's name, falling back to when "
            "Drive received it"
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

    p_reprocess = sub.add_parser(
        "reprocess",
        help="Re-run specific chain stages (0=transcript, 1..N=presets) for a target",
    )
    _set_parser_safety_description(
        p_reprocess,
        summary="Force-rerun chain stages by number for a Drive file or folder.",
        safety_note=(
            "stages are 0=transcript (Deepgram base, re-spends STT), 1..N=presets in "
            "chain order (re-spends OpenAI). See `gdstt doctor` for the numbering. Use "
            "--dry-run first; omit STAGES or pass 'all' to rerun every preset."
        ),
    )
    p_reprocess.add_argument("target", help="Drive file ID or folder ID")
    p_reprocess.add_argument(
        "stages",
        nargs="?",
        default=None,
        metavar="STAGES",
        help="Stage spec: '3', '2-3', '1,3', '0' (transcript+all), or 'all'/omit",
    )
    p_reprocess.add_argument(
        "--folder",
        action="store_true",
        help="Treat the target as a folder ID (default: auto-detect)",
    )
    _add_processing_safety_args(p_reprocess)
    p_reprocess.set_defaults(func=cmd_reprocess)

    p_stop = sub.add_parser(
        "stop",
        help="Pause the `gdstt run` loop (run.enabled=false); stays paused across restarts",
    )
    p_stop.set_defaults(func=cmd_stop)

    p_start = sub.add_parser(
        "start",
        help="Resume a paused `gdstt run` loop by setting run.enabled=true",
    )
    p_start.set_defaults(func=cmd_start)

    p_latest = sub.add_parser(
        "latest",
        help="Process the newest mp4 in a folder",
    )
    _set_parser_safety_description(
        p_latest,
        summary=(
            "Process the most recently created mp4 in a folder or any of its meeting "
            "subfolders."
        ),
        safety_note=(
            "this command spends STT credits on the newest mp4. Use --dry-run first to "
            "confirm which file would be processed."
        ),
    )
    p_latest.add_argument(
        "--folder",
        default=None,
        help="Folder ID to inspect (default: first configured folders entry)",
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
        help=(
            "Also authenticate and report each configured folder: its name, its "
            "parent, how many subfolders and recordings it holds, when it last "
            "received one, and the state of the changes cursor"
        ),
    )
    p_doctor.set_defaults(func=cmd_doctor)

    p_config = sub.add_parser(
        "config",
        help="Manage gdstt configuration (active config.yml)",
    )
    config_sub = p_config.add_subparsers(dest="config_command", required=True)

    p_config_init = config_sub.add_parser(
        "init",
        help="Create a fresh config.yml with default presets and prompt assets",
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

    p_planfix = sub.add_parser(
        "planfix",
        help="Inspect what was sent to Planfix",
    )
    planfix_sub = p_planfix.add_subparsers(dest="planfix_command", required=True)
    p_planfix_sent = planfix_sub.add_parser(
        "sent", help="List recordings whose Planfix comment was delivered"
    )
    p_planfix_sent.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many of the newest to print; 0 for all (default: 20)",
    )
    p_planfix_sent.set_defaults(func=cmd_planfix_sent)

    p_bookings = sub.add_parser(
        "bookings",
        help="Inspect received call bookings and revive skipped recordings",
    )
    bookings_sub = p_bookings.add_subparsers(dest="bookings_command", required=True)
    p_bookings_list = bookings_sub.add_parser(
        "list", help="Print the call bookings currently in the journal"
    )
    p_bookings_list.set_defaults(func=cmd_bookings_list)
    p_bookings_rematch = bookings_sub.add_parser(
        "rematch",
        help="Clear the unmatched mark on a Drive MP4 so it is reconsidered",
    )
    p_bookings_rematch.add_argument("target", help="Drive MP4 file ID")
    p_bookings_rematch.set_defaults(func=cmd_bookings_rematch)
    p_bookings_restore = bookings_sub.add_parser(
        "restore-dates",
        help="Restore modifiedTime on recordings whose date the unmatched mark moved",
    )
    p_bookings_restore.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that would be restored without writing anything",
    )
    p_bookings_restore.set_defaults(func=cmd_bookings_restore_dates)

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

    p_changes = sub.add_parser(
        "changes",
        help="Show what the changes feed reports, without consuming it",
        description=(
            "Show what Drive's changes feed reports since the saved cursor. Read-only: "
            "the cursor is not moved, so the next cycle still sees these changes. "
            "Without a cursor there is nothing to read and the next cycle sweeps."
        ),
    )
    p_changes.add_argument(
        "--raw",
        action="store_true",
        help="Show every entry, not just the videos in configured folders",
    )
    p_changes.set_defaults(func=cmd_changes)

    p_cursor = sub.add_parser(
        "cursor",
        help="Inspect or forget the changes-feed cursor",
        description=(
            "The cursor is where the changes feed resumes from, and the only "
            "discovery state this service keeps. It is safe to forget: without one "
            "a cycle reads every configured folder and takes a fresh cursor, so the "
            "worst a reset costs is one slower cycle."
        ),
    )
    cursor_sub = p_cursor.add_subparsers(dest="cursor_command", required=True)
    p_cursor_show = cursor_sub.add_parser("show", help="Print the cursor and its path")
    p_cursor_show.set_defaults(func=cmd_cursor_show)
    p_cursor_reset = cursor_sub.add_parser(
        "reset", help="Forget the cursor so the next cycle sweeps every folder"
    )
    p_cursor_reset.set_defaults(func=cmd_cursor_reset)

    p_list = sub.add_parser(
        "list",
        aliases=["status"],
        help="Show folder state (sibling MP3/TXT presence) without doing work",
        description=(
            "Show each folder's recordings and whether their MP3/TXT siblings exist, "
            "without doing any work. Reads the folder together with its meeting "
            "subfolders, prints the folder each recording actually lives in, and "
            "marks the ones a since cutoff puts out of scope."
        ),
    )
    p_list.add_argument(
        "--folder",
        default=None,
        help="Folder ID to inspect (default: configured folders)",
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
    # --config is a one-shot file override threaded explicitly into each command's
    # config calls (config_path=args.config). It is deliberately not routed through
    # an env var so the resolver stays the single source of truth and the CLI never
    # mutates global process state.
    args.func(args)


if __name__ == "__main__":
    main()
