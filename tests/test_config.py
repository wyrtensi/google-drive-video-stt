import json
from pathlib import Path

import pytest
import yaml
from dotenv import load_dotenv as real_load_dotenv

from src.config import (
    CONFIG_FILE_NAME,
    _config_to_yaml_dict,
    _user_config_path,
    config_get,
    config_set,
    config_unset,
    copy_prompt_assets,
    import_google_credentials,
    init_config,
    link_config,
    load_config,
    migrate_config,
    resolve_config_file_path,
    resolve_effective_config_path,
    use_google_files,
)
from src.presets import PACKAGED_PROMPT_ASSETS

ENV_VARS = [
    "FOLDER_IDS",
    "POLL_INTERVAL",
    "BITRATE",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "DATA_DIR",
    "PROXY_URL",
    "STT_PROVIDER",
    "STT_LANGUAGE",
    "STT_POSTPROCESS",
    "DRIVE_MP3_ARTIFACT",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_KEYPOINTS",
    "OPENAI_BATCH",
    "OPENAI_MAX_PARALLEL",
    "OUTPUT_TARGET",
    "OUTPUT_DIR",
    "DEEPGRAM_API_KEY",
    "DEEPGRAM_API_KEY_FILE",
    "DEEPGRAM_MODEL",
    "DEEPGRAM_DIARIZE_MODEL",
    "DEEPGRAM_AUDIO_SOURCE",
    "DEEPGRAM_TXT_FORMATTER",
    "DEEPGRAM_KEYTERMS_ENABLED",
    "DEEPGRAM_KEYTERMS_FILE",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("GDSTT_CONFIG", raising=False)
    monkeypatch.setattr("src.config.load_dotenv", lambda *a, **kw: False)
    # Point the config-file resolver at a throwaway per-test path so auto-migration
    # never reads or writes the repo's real data/config.yml. Tests that exercise the
    # env-loading path still hit the migration branch (the file never exists), and
    # the default keyterms file keeps resolving against the repo cwd.
    monkeypatch.setenv("GDSTT_CONFIG", str(tmp_path / "config.yml"))
    yield


def _load_drive_only_config():
    return load_config(validate_providers=False)


def test_defaults_when_no_env(monkeypatch):
    cfg = _load_drive_only_config()
    assert cfg.folder_ids == []
    assert cfg.poll_interval == 600
    assert cfg.bitrate == "96k"
    assert cfg.data_dir == Path("data")
    assert cfg.stt_provider == "deepgram"
    assert cfg.stt_language == "ru"
    assert cfg.stt_postprocess is True
    assert cfg.output_target == "drive"
    assert cfg.output_dir is None
    assert cfg.openai_keypoints is False


def test_default_deepgram_processing_requires_key(monkeypatch):
    with pytest.raises(ValueError, match="DEEPGRAM_API_KEY is required"):
        load_config()


def test_stt_provider_disabled_explicitly_turns_transcription_off(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "disabled")

    cfg = load_config()

    assert cfg.stt_provider == ""
    assert cfg.stt_language == ""


def test_stt_postprocess_can_be_disabled(monkeypatch):
    monkeypatch.setenv("STT_POSTPROCESS", "false")
    cfg = _load_drive_only_config()
    assert cfg.stt_postprocess is False


def test_drive_mp3_artifact_defaults_to_true_when_transcription_disabled(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "disabled")

    cfg = load_config()

    assert cfg.drive_mp3_artifact is True


def test_drive_mp3_artifact_defaults_to_false_for_deepgram_m4a(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")

    cfg = load_config()

    assert cfg.deepgram_audio_source == "m4a_copy"
    assert cfg.drive_mp3_artifact is False


def test_drive_mp3_artifact_can_be_enabled_for_deepgram_m4a(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("DRIVE_MP3_ARTIFACT", "true")

    cfg = load_config()

    assert cfg.drive_mp3_artifact is True


def test_openai_pipeline_defaults(monkeypatch):
    cfg = _load_drive_only_config()
    assert cfg.openai_model == "gpt-5.4-mini"
    assert cfg.openai_keypoints is False
    assert cfg.openai_batch is False


def test_openai_keypoints_requires_api_key(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    monkeypatch.setenv("OPENAI_KEYPOINTS", "true")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        load_config()


def test_validate_providers_false_skips_openai_keypoints_secret(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    monkeypatch.setenv("OPENAI_KEYPOINTS", "true")
    cfg = load_config(validate_providers=False)
    assert cfg.openai_keypoints is True
    assert cfg.openai_api_key == ""


def test_validate_providers_true_is_default(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    monkeypatch.setenv("OPENAI_KEYPOINTS", "true")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        load_config()


def test_openai_keypoints_with_api_key(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    monkeypatch.setenv("OPENAI_KEYPOINTS", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4")
    monkeypatch.setenv("OPENAI_BATCH", "true")
    cfg = load_config()
    assert cfg.openai_keypoints is True
    assert cfg.openai_api_key == "sk-test"
    assert cfg.openai_model == "gpt-5.4"
    assert cfg.openai_batch is True


def test_openai_keypoints_enabled_without_stt_provider(monkeypatch):
    # OPENAI_KEYPOINTS is independent of STT_PROVIDER selection.
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    monkeypatch.setenv("OPENAI_KEYPOINTS", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = load_config()
    assert cfg.stt_provider == ""
    assert cfg.openai_keypoints is True


def test_output_target_defaults_to_drive(monkeypatch):
    cfg = _load_drive_only_config()
    assert cfg.output_target == "drive"
    assert cfg.output_dir is None


def test_output_target_folder_requires_output_dir(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    monkeypatch.setenv("OUTPUT_TARGET", "folder")
    with pytest.raises(ValueError, match="OUTPUT_DIR"):
        load_config()


def test_output_target_folder_with_output_dir(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    monkeypatch.setenv("OUTPUT_TARGET", "folder")
    monkeypatch.setenv("OUTPUT_DIR", "/var/lib/transcripts")
    cfg = load_config()
    assert cfg.output_target == "folder"
    assert cfg.output_dir == Path("/var/lib/transcripts")


def test_invalid_output_target_raises(monkeypatch):
    monkeypatch.setenv("OUTPUT_TARGET", "s3")
    with pytest.raises(ValueError, match="OUTPUT_TARGET"):
        load_config()


def test_parses_single_folder_id(monkeypatch):
    monkeypatch.setenv("FOLDER_IDS", "abc123")
    cfg = _load_drive_only_config()
    assert cfg.folder_ids == ["abc123"]


def test_load_config_accepts_dotenv_with_utf8_bom(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.config.load_dotenv", real_load_dotenv)
    (tmp_path / ".env").write_text("FOLDER_IDS=abc123\n", encoding="utf-8-sig")

    cfg = _load_drive_only_config()

    assert cfg.folder_ids == ["abc123"]


def test_load_config_falls_back_to_checkout_env_outside_repo(monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    elsewhere = tmp_path / "elsewhere"
    checkout.mkdir()
    elsewhere.mkdir()
    (checkout / ".env").write_text(
        "FOLDER_IDS=folder-1\nDATA_DIR=data\nSTT_PROVIDER=disabled\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr("src.config.CHECKOUT_ROOT", checkout, raising=False)
    monkeypatch.setattr("src.config.load_dotenv", real_load_dotenv)

    cfg = load_config()

    assert cfg.folder_ids == ["folder-1"]
    assert cfg.data_dir == checkout / "data"


def test_parses_multiple_folder_ids(monkeypatch):
    monkeypatch.setenv("FOLDER_IDS", "id1,id2,id3")
    cfg = _load_drive_only_config()
    assert cfg.folder_ids == ["id1", "id2", "id3"]


def test_strips_whitespace_in_folder_ids(monkeypatch):
    monkeypatch.setenv("FOLDER_IDS", " id1 , id2 ,  ,id3 ")
    cfg = _load_drive_only_config()
    assert cfg.folder_ids == ["id1", "id2", "id3"]


def test_empty_folder_ids_returns_empty_list(monkeypatch):
    monkeypatch.setenv("FOLDER_IDS", "")
    cfg = _load_drive_only_config()
    assert cfg.folder_ids == []


def test_folder_ids_only_commas(monkeypatch):
    monkeypatch.setenv("FOLDER_IDS", " , , ")
    with pytest.raises(ValueError, match="FOLDER_IDS"):
        load_config()


def test_custom_poll_interval(monkeypatch):
    monkeypatch.setenv("POLL_INTERVAL", "120")
    cfg = _load_drive_only_config()
    assert cfg.poll_interval == 120


def test_invalid_poll_interval_raises(monkeypatch):
    monkeypatch.setenv("POLL_INTERVAL", "not-a-number")
    with pytest.raises(ValueError, match="POLL_INTERVAL"):
        load_config()


def test_non_positive_poll_interval_raises(monkeypatch):
    monkeypatch.setenv("POLL_INTERVAL", "0")
    with pytest.raises(ValueError, match="positive"):
        load_config()


def test_custom_bitrate(monkeypatch):
    monkeypatch.setenv("BITRATE", "128k")
    cfg = _load_drive_only_config()
    assert cfg.bitrate == "128k"


def test_blank_bitrate_uses_default(monkeypatch):
    monkeypatch.setenv("BITRATE", "")
    cfg = _load_drive_only_config()
    assert cfg.bitrate == "96k"


def test_custom_data_dir(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/var/lib/stt")
    cfg = _load_drive_only_config()
    assert cfg.data_dir == Path("/var/lib/stt")


def test_blank_data_dir_uses_default(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "")
    cfg = _load_drive_only_config()
    assert cfg.data_dir == Path("data")


def test_stt_deepgram_requires_api_key(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("STT_LANGUAGE", "ru")
    with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
        load_config()


def test_stt_deepgram_defaults_language_and_options(monkeypatch, tmp_path):
    keyterms = tmp_path / "keyterms.txt"
    keyterms.write_text(
        "# tech terms\nKubernetes\n\nRuby on Rails\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("DEEPGRAM_KEYTERMS_FILE", str(keyterms))

    cfg = load_config()

    assert cfg.stt_language == "ru"
    assert cfg.deepgram_model == "nova-3"
    assert cfg.deepgram_diarize_model == "latest"
    assert cfg.deepgram_audio_source == "m4a_copy"
    assert cfg.deepgram_txt_formatter == "word_speaker"
    assert cfg.deepgram_keyterms_enabled is True
    assert cfg.deepgram_keyterms_file == keyterms
    assert cfg.deepgram_keyterms == ("Kubernetes", "Ruby on Rails")


def test_stt_deepgram_with_api_key(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("STT_LANGUAGE", "ru")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "  dg-test  ")
    cfg = load_config()
    assert cfg.stt_provider == "deepgram"
    assert cfg.stt_language == "ru"
    assert cfg.deepgram_api_key == "dg-test"


def test_stt_deepgram_accepts_custom_options(monkeypatch, tmp_path):
    keyterms = tmp_path / "keyterms.txt"
    keyterms.write_text("React\nTypeScript\n", encoding="utf-8")
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("STT_LANGUAGE", "multi")
    monkeypatch.setenv("DEEPGRAM_MODEL", "nova-2")
    monkeypatch.setenv("DEEPGRAM_DIARIZE_MODEL", "v1")
    monkeypatch.setenv("DEEPGRAM_AUDIO_SOURCE", "mp3_96k")
    monkeypatch.setenv("DEEPGRAM_TXT_FORMATTER", "utterance")
    monkeypatch.setenv("DEEPGRAM_KEYTERMS_FILE", str(keyterms))

    cfg = load_config()

    assert cfg.stt_language == "multi"
    assert cfg.deepgram_model == "nova-2"
    assert cfg.deepgram_diarize_model == "v1"
    assert cfg.deepgram_audio_source == "mp3_96k"
    assert cfg.deepgram_txt_formatter == "utterance"
    assert cfg.deepgram_keyterms == ("React", "TypeScript")


def test_stt_deepgram_accepts_mp3_192k_audio_source(monkeypatch, tmp_path):
    keyterms = tmp_path / "keyterms.txt"
    keyterms.write_text("React\n", encoding="utf-8")
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("DEEPGRAM_AUDIO_SOURCE", "mp3_192k")
    monkeypatch.setenv("DEEPGRAM_KEYTERMS_FILE", str(keyterms))

    cfg = load_config()

    assert cfg.deepgram_audio_source == "mp3_192k"


def test_stt_deepgram_rejects_invalid_options(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")

    monkeypatch.setenv("DEEPGRAM_DIARIZE_MODEL", "bad")
    with pytest.raises(ValueError, match="DEEPGRAM_DIARIZE_MODEL"):
        load_config()
    monkeypatch.setenv("DEEPGRAM_DIARIZE_MODEL", "latest")

    monkeypatch.setenv("DEEPGRAM_AUDIO_SOURCE", "wav")
    with pytest.raises(ValueError, match="DEEPGRAM_AUDIO_SOURCE"):
        load_config()
    monkeypatch.setenv("DEEPGRAM_AUDIO_SOURCE", "m4a_copy")

    monkeypatch.setenv("DEEPGRAM_TXT_FORMATTER", "plain")
    with pytest.raises(ValueError, match="DEEPGRAM_TXT_FORMATTER"):
        load_config()


def test_stt_deepgram_can_disable_keyterms(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("DEEPGRAM_KEYTERMS_ENABLED", "false")
    monkeypatch.setenv("DEEPGRAM_KEYTERMS_FILE", "missing.txt")

    cfg = load_config()

    assert cfg.deepgram_keyterms_enabled is False
    assert cfg.deepgram_keyterms == ()


def test_stt_deepgram_rejects_missing_keyterms_file_when_enabled(monkeypatch, tmp_path):
    missing_keyterms = tmp_path / "missing-keyterms.txt"
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("DEEPGRAM_KEYTERMS_ENABLED", "true")
    monkeypatch.setenv("DEEPGRAM_KEYTERMS_FILE", str(missing_keyterms))

    with pytest.raises(ValueError, match="could not be read"):
        load_config()


def test_stt_deepgram_rejects_too_many_keyterms(monkeypatch, tmp_path):
    keyterms = tmp_path / "keyterms.txt"
    keyterms.write_text("\n".join(f"term-{i}" for i in range(101)), encoding="utf-8")
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("DEEPGRAM_KEYTERMS_FILE", str(keyterms))

    with pytest.raises(ValueError, match="100"):
        load_config()


def test_stt_deepgram_prefers_env_key_over_file(monkeypatch, tmp_path):
    key_file = tmp_path / "deepgram.json"
    key_file.write_text("file-key", encoding="utf-8")
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("STT_LANGUAGE", "ru")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "env-key")
    monkeypatch.setenv("DEEPGRAM_API_KEY_FILE", str(key_file))

    cfg = load_config()

    assert cfg.deepgram_api_key == "env-key"


def test_stt_deepgram_reads_raw_key_file(monkeypatch, tmp_path):
    key_file = tmp_path / "deepgram_api_secret.json"
    key_file.write_text("  raw-file-key  \n", encoding="utf-8")
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("STT_LANGUAGE", "ru")
    monkeypatch.setenv("DEEPGRAM_API_KEY_FILE", str(key_file))

    cfg = load_config()

    assert cfg.deepgram_api_key == "raw-file-key"


def test_stt_deepgram_reads_raw_key_file_with_utf8_bom(monkeypatch, tmp_path):
    key_file = tmp_path / "deepgram_api_secret.json"
    key_file.write_text("raw-file-key\n", encoding="utf-8-sig")
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("STT_LANGUAGE", "ru")
    monkeypatch.setenv("DEEPGRAM_API_KEY_FILE", str(key_file))

    cfg = load_config()

    assert cfg.deepgram_api_key == "raw-file-key"


def test_stt_deepgram_reads_json_key_file(monkeypatch, tmp_path):
    key_file = tmp_path / "deepgram_api_secret.json"
    key_file.write_text('{"deepgram_api_key": "json-file-key"}', encoding="utf-8")
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("STT_LANGUAGE", "ru")
    monkeypatch.setenv("DEEPGRAM_API_KEY_FILE", str(key_file))

    cfg = load_config()

    assert cfg.deepgram_api_key == "json-file-key"


def test_stt_deepgram_reads_json_key_file_with_utf8_bom(monkeypatch, tmp_path):
    key_file = tmp_path / "deepgram_api_secret.json"
    key_file.write_text('{"deepgram_api_key": "json-file-key"}', encoding="utf-8-sig")
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("STT_LANGUAGE", "ru")
    monkeypatch.setenv("DEEPGRAM_API_KEY_FILE", str(key_file))

    cfg = load_config()

    assert cfg.deepgram_api_key == "json-file-key"


def test_stt_deepgram_reads_keyterms_file_with_utf8_bom(monkeypatch, tmp_path):
    keyterms = tmp_path / "keyterms.txt"
    keyterms.write_text("# header\nKubernetes\n", encoding="utf-8-sig")
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setenv("DEEPGRAM_KEYTERMS_FILE", str(keyterms))

    cfg = load_config()

    assert cfg.deepgram_keyterms == ("Kubernetes",)


def test_stt_deepgram_key_file_is_ignored_when_transcription_disabled(monkeypatch, tmp_path):
    missing_key_file = tmp_path / "missing.json"
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    monkeypatch.setenv("DEEPGRAM_API_KEY_FILE", str(missing_key_file))

    cfg = load_config()

    assert cfg.stt_provider == ""
    assert cfg.deepgram_api_key == ""


def test_stt_unsupported_provider_raises(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "azure")
    with pytest.raises(ValueError, match="STT_PROVIDER"):
        load_config()


def test_full_env_combination(monkeypatch):
    monkeypatch.setenv("FOLDER_IDS", "f1,f2")
    monkeypatch.setenv("POLL_INTERVAL", "300")
    monkeypatch.setenv("BITRATE", "192k")
    monkeypatch.setenv("DATA_DIR", "mydata")
    cfg = _load_drive_only_config()
    assert cfg.folder_ids == ["f1", "f2"]
    assert cfg.poll_interval == 300
    assert cfg.bitrate == "192k"
    assert cfg.data_dir == Path("mydata")


# --- config.yml loading -----------------------------------------------------


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def test_loads_grouped_yaml(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {
            "folder_ids": ["abc", "def"],
            "poll_interval": 300,
            "bitrate": "128k",
            "data_dir": "mydata",
            "proxy_url": "socks5://proxy:9050",
            "output": {"target": "drive", "dir": None},
            "stt": {
                "provider": "deepgram",
                "language": "ru",
                "postprocess": False,
                "deepgram": {
                    "api_key": "dg-yaml",
                    "model": "nova-2",
                    "keyterms_enabled": False,
                },
            },
            "openai": {"api_key": "sk-yaml", "model": "gpt-5.4", "batch": True},
        },
    )

    cfg = load_config(config_path=config_file)

    assert cfg.folder_ids == ["abc", "def"]
    assert cfg.poll_interval == 300
    assert cfg.bitrate == "128k"
    assert cfg.data_dir == tmp_path / "mydata"
    assert cfg.proxy_url == "socks5://proxy:9050"
    assert cfg.stt_provider == "deepgram"
    assert cfg.deepgram_api_key == "dg-yaml"
    assert cfg.deepgram_model == "nova-2"
    assert cfg.stt_postprocess is False
    assert cfg.openai_api_key == "sk-yaml"
    assert cfg.openai_model == "gpt-5.4"
    assert cfg.openai_batch is True


def test_openai_batch_wait_defaults_true(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "openai": {"api_key": "sk", "model": "gpt-5.4"},
            "presets": {"keypoints": {"enabled": False}},
        },
    )
    cfg = load_config(config_path=config_file)
    assert cfg.openai_batch_wait is True


def test_openai_batch_wait_parsed_from_yaml(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "openai": {"api_key": "sk", "batch_wait": False},
            "presets": {"keypoints": {"enabled": False}},
        },
    )
    cfg = load_config(config_path=config_file)
    assert cfg.openai_batch_wait is False


def test_generated_yaml_omits_batch_wait_by_default(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "openai": {"api_key": "sk"},
            "presets": {"keypoints": {"enabled": False}},
        },
    )
    cfg = load_config(config_path=config_file)
    serialized = _config_to_yaml_dict(cfg, config_file)
    # batch_wait defaults to true; the serialized YAML omits it (no hidden setting).
    assert "batch_wait" not in serialized["openai"]


def test_generated_yaml_emits_batch_wait_when_disabled(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "openai": {"api_key": "sk", "batch_wait": False},
            "presets": {"keypoints": {"enabled": False}},
        },
    )
    cfg = load_config(config_path=config_file)
    serialized = _config_to_yaml_dict(cfg, config_file)
    assert serialized["openai"]["batch_wait"] is False


def test_yaml_disabled_provider_is_mp3_only(tmp_path):
    config_file = tmp_path / "config.yml"
    # Disabling the built-in keypoints preset keeps this a pure mp3-only setup; with
    # any enabled preset, an openai.api_key would be required.
    _write_yaml(
        config_file,
        {"stt": {"provider": "disabled"}, "presets": {"keypoints": {"enabled": False}}},
    )

    cfg = load_config(config_path=config_file)

    assert cfg.stt_provider == ""
    assert cfg.stt_language == ""


def test_yaml_deepgram_requires_api_key(tmp_path):
    config_file = tmp_path / "config.yml"
    # Supply an openai.api_key so the preset gate passes and the deepgram-specific
    # validation is what surfaces.
    _write_yaml(
        config_file,
        {"stt": {"provider": "deepgram"}, "openai": {"api_key": "sk-test"}},
    )

    with pytest.raises(ValueError, match="deepgram.api_key is required"):
        load_config(config_path=config_file)


def test_yaml_enabled_preset_requires_openai_api_key(tmp_path):
    config_file = tmp_path / "config.yml"
    # A preset enabled via the presets map (not the legacy openai.keypoints flag) must
    # still require an openai.api_key at load time.
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "presets": {
                "keypoints": {"enabled": False},
                "summary": {"instructions": "summarize"},
            },
        },
    )

    with pytest.raises(ValueError, match="openai.api_key is required"):
        load_config(config_path=config_file)


def test_yaml_openai_keypoints_requires_api_key(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {"stt": {"provider": "disabled"}, "openai": {"keypoints": True}},
    )

    with pytest.raises(ValueError, match="openai.api_key is required"):
        load_config(config_path=config_file)


def test_yaml_invalid_output_target_raises(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(config_file, {"stt": {"provider": "disabled"}, "output": {"target": "s3"}})

    with pytest.raises(ValueError, match="output.target"):
        load_config(config_path=config_file)


def test_yaml_folder_target_requires_dir(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {"stt": {"provider": "disabled"}, "output": {"target": "folder"}},
    )

    with pytest.raises(ValueError, match="output.dir is required"):
        load_config(config_path=config_file)


def test_yaml_rejects_non_mapping_document(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_config(config_path=config_file)


def test_yaml_validate_providers_false_skips_secrets(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {"stt": {"provider": "deepgram"}, "openai": {"keypoints": True}},
    )

    cfg = load_config(config_path=config_file, validate_providers=False)

    assert cfg.stt_provider == "deepgram"
    assert cfg.openai_keypoints is True
    assert cfg.deepgram_api_key == ""


# --- auto-migration ---------------------------------------------------------


def test_auto_migration_writes_config_yml_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("FOLDER_IDS", "mig1,mig2")
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    config_file = tmp_path / "config.yml"
    assert not config_file.exists()

    cfg = load_config(config_path=config_file, validate_providers=False)

    assert cfg.folder_ids == ["mig1", "mig2"]
    assert config_file.exists()
    written = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert written["folder_ids"] == ["mig1", "mig2"]
    assert written["stt"]["provider"] == ""
    assert "presets" in written


def test_auto_migration_triggers_on_empty_file(monkeypatch, tmp_path):
    monkeypatch.setenv("FOLDER_IDS", "only-env")
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    config_file = tmp_path / "config.yml"
    config_file.write_text("   \n", encoding="utf-8")

    cfg = load_config(config_path=config_file, validate_providers=False)

    assert cfg.folder_ids == ["only-env"]
    assert yaml.safe_load(config_file.read_text(encoding="utf-8"))["folder_ids"] == ["only-env"]


def test_existing_yaml_takes_precedence_over_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FOLDER_IDS", "from-env")
    config_file = tmp_path / "config.yml"
    _write_yaml(config_file, {"folder_ids": ["from-yaml"], "stt": {"provider": "disabled"}})

    cfg = load_config(config_path=config_file, validate_providers=False)

    assert cfg.folder_ids == ["from-yaml"]


# --- user config path -------------------------------------------------------


def test_user_config_path_windows(monkeypatch):
    monkeypatch.setattr("src.config.sys.platform", "win32")
    monkeypatch.setenv("APPDATA", "/win/appdata")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    assert _user_config_path() == Path("/win/appdata/gdstt/config.yml")


def test_user_config_path_windows_without_appdata(monkeypatch):
    monkeypatch.setattr("src.config.sys.platform", "win32")
    monkeypatch.delenv("APPDATA", raising=False)

    expected = (Path("~/AppData/Roaming") / "gdstt" / "config.yml").expanduser()
    assert _user_config_path() == expected


def test_user_config_path_macos(monkeypatch):
    monkeypatch.setattr("src.config.sys.platform", "darwin")

    expected = (
        Path("~/Library/Application Support") / "gdstt" / "config.yml"
    ).expanduser()
    assert _user_config_path() == expected


def test_user_config_path_linux_uses_xdg(monkeypatch):
    monkeypatch.setattr("src.config.sys.platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/custom/xdg")

    assert _user_config_path() == Path("/custom/xdg/gdstt/config.yml")


def test_user_config_path_linux_defaults_to_dot_config(monkeypatch):
    monkeypatch.setattr("src.config.sys.platform", "linux")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    expected = (Path("~/.config") / "gdstt" / "config.yml").expanduser()
    assert _user_config_path() == expected


# --- resolver priority ------------------------------------------------------


def test_resolve_priority_full_ordering(monkeypatch, tmp_path):
    arg = tmp_path / "arg.yml"
    env = tmp_path / "env.yml"
    data = tmp_path / "datadir"
    user = tmp_path / "user.yml"
    monkeypatch.setattr("src.config._user_config_path", lambda: user)
    monkeypatch.setenv("GDSTT_CONFIG", str(env))
    monkeypatch.setenv("DATA_DIR", str(data))

    # --config arg beats everything.
    assert resolve_config_file_path(arg) == arg

    # Without an arg, GDSTT_CONFIG wins over DATA_DIR and the user path.
    assert resolve_config_file_path() == env

    # Without GDSTT_CONFIG, DATA_DIR/config.yml is used.
    monkeypatch.delenv("GDSTT_CONFIG", raising=False)
    assert resolve_config_file_path() == data / CONFIG_FILE_NAME

    # With neither set, fall back to the per-user path (no cwd ./data default).
    monkeypatch.delenv("DATA_DIR", raising=False)
    assert resolve_config_file_path() == user


def test_resolve_data_dir_only_honored_when_set(monkeypatch, tmp_path):
    user = tmp_path / "user.yml"
    monkeypatch.setattr("src.config._user_config_path", lambda: user)
    monkeypatch.delenv("GDSTT_CONFIG", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)

    # No DATA_DIR -> user path, NOT ./data/config.yml.
    assert resolve_config_file_path() == user

    # Empty DATA_DIR is treated as unset.
    monkeypatch.setenv("DATA_DIR", "  ")
    assert resolve_config_file_path() == user

    # Explicit DATA_DIR -> its config.yml.
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "dd"))
    assert resolve_config_file_path() == tmp_path / "dd" / CONFIG_FILE_NAME


def test_resolve_container_data_dir(monkeypatch, tmp_path):
    """Container layout: DATA_DIR=/app/data resolves config into the volume.

    Guards the Docker fix - the image bakes DATA_DIR=/app/data (and compose
    sets it too) so the bootstrap config lands in the mounted ./data volume.
    Without DATA_DIR the resolver must fall back to the per-user path, never
    /app/data, so an ambient container env can't hijack a host install.
    """
    user = tmp_path / "user.yml"
    monkeypatch.setattr("src.config._user_config_path", lambda: user)
    monkeypatch.delenv("GDSTT_CONFIG", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)

    # No DATA_DIR -> per-user path, not the container volume.
    assert resolve_config_file_path() == user

    # DATA_DIR=/app/data (the baked container default) -> /app/data/config.yml.
    monkeypatch.setenv("DATA_DIR", "/app/data")
    assert resolve_config_file_path() == Path("/app/data") / CONFIG_FILE_NAME


def test_resolve_identical_from_different_cwd(monkeypatch, tmp_path):
    user = tmp_path / "user.yml"
    monkeypatch.setattr("src.config._user_config_path", lambda: user)
    monkeypatch.delenv("GDSTT_CONFIG", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)

    here = tmp_path / "here"
    there = tmp_path / "there"
    here.mkdir()
    there.mkdir()

    monkeypatch.chdir(here)
    first = resolve_config_file_path()
    monkeypatch.chdir(there)
    second = resolve_config_file_path()

    assert first == second == user


# --- pointer configs --------------------------------------------------------


def test_pointer_resolves_relative_target(monkeypatch, tmp_path):
    pointer = tmp_path / "pointer.yml"
    target = tmp_path / "real" / "config.yml"
    target.parent.mkdir()
    _write_yaml(target, {"folder_ids": ["pointed"], "stt": {"provider": "disabled"}})
    pointer.write_text("config_file: real/config.yml\n", encoding="utf-8")

    bootstrap, effective = resolve_effective_config_path(pointer)
    assert bootstrap == pointer
    assert effective == target

    cfg = load_config(config_path=pointer, validate_providers=False)
    assert cfg.folder_ids == ["pointed"]


def test_pointer_resolves_absolute_target(monkeypatch, tmp_path):
    pointer = tmp_path / "pointer.yml"
    target = tmp_path / "abs.yml"
    _write_yaml(target, {"folder_ids": ["abs"], "stt": {"provider": "disabled"}})
    pointer.write_text(f"config_file: {target}\n", encoding="utf-8")

    _, effective = resolve_effective_config_path(pointer)
    assert effective == target


def test_pointer_expands_user_and_env_vars(monkeypatch, tmp_path):
    home = tmp_path / "home"
    target = home / "gdstt.yml"
    target.parent.mkdir()
    _write_yaml(target, {"folder_ids": ["expanded"], "stt": {"provider": "disabled"}})
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("MY_CFG_DIR", str(home))

    pointer = tmp_path / "pointer.yml"
    pointer.write_text("config_file: $MY_CFG_DIR/gdstt.yml\n", encoding="utf-8")
    _, effective = resolve_effective_config_path(pointer)
    assert effective == target

    pointer.write_text("config_file: ~/gdstt.yml\n", encoding="utf-8")
    _, effective = resolve_effective_config_path(pointer)
    assert effective == target


def test_pointer_with_extra_keys_rejected(tmp_path):
    pointer = tmp_path / "pointer.yml"
    pointer.write_text(
        "config_file: real.yml\nfolder_ids: [oops]\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="extra keys"):
        resolve_effective_config_path(pointer)


def test_pointer_self_reference_rejected(tmp_path):
    pointer = tmp_path / "pointer.yml"
    pointer.write_text("config_file: pointer.yml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="loop"):
        resolve_effective_config_path(pointer)


def test_pointer_loop_rejected(tmp_path):
    a = tmp_path / "a.yml"
    b = tmp_path / "b.yml"
    a.write_text("config_file: b.yml\n", encoding="utf-8")
    b.write_text("config_file: a.yml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="loop"):
        resolve_effective_config_path(a)


def test_resolve_config_file_path_prefers_explicit_arg(monkeypatch, tmp_path):
    monkeypatch.setenv("GDSTT_CONFIG", str(tmp_path / "from-env.yml"))
    explicit = tmp_path / "explicit.yml"

    assert resolve_config_file_path(explicit) == explicit


def test_resolve_config_file_path_honors_env_var(monkeypatch, tmp_path):
    env_path = tmp_path / "from-env.yml"
    monkeypatch.setenv("GDSTT_CONFIG", str(env_path))

    assert resolve_config_file_path() == env_path


def test_resolve_config_file_path_defaults_to_data_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("GDSTT_CONFIG", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "datadir"))

    assert resolve_config_file_path() == tmp_path / "datadir" / CONFIG_FILE_NAME


def test_gdstt_config_env_var_resolves_path(monkeypatch, tmp_path):
    config_file = tmp_path / "custom-config.yml"
    _write_yaml(config_file, {"folder_ids": ["env-path"], "stt": {"provider": "disabled"}})
    monkeypatch.setenv("GDSTT_CONFIG", str(config_file))

    cfg = load_config(validate_providers=False)

    assert cfg.folder_ids == ["env-path"]


def test_config_to_yaml_dict_round_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    monkeypatch.setenv("FOLDER_IDS", "rt1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-rt")
    monkeypatch.setenv("OPENAI_KEYPOINTS", "true")
    cfg = load_config(validate_providers=False)

    data = _config_to_yaml_dict(cfg)
    config_file = tmp_path / "config.yml"
    _write_yaml(config_file, data)
    reloaded = load_config(config_path=config_file, validate_providers=False)

    assert reloaded.folder_ids == ["rt1"]
    assert reloaded.openai_api_key == "sk-rt"
    assert reloaded.openai_keypoints is True
    assert reloaded.stt_provider == ""


# --- migrate_config ---------------------------------------------------------


def test_migrate_config_writes_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FOLDER_IDS", "m1")
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    config_file = tmp_path / "config.yml"

    written_path = migrate_config(config_path=config_file)

    assert written_path == config_file
    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["folder_ids"] == ["m1"]
    assert data["presets"]["keypoints"]["enabled"] is False


def test_migrate_config_refuses_existing_without_force(monkeypatch, tmp_path):
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    config_file = tmp_path / "config.yml"
    config_file.write_text("folder_ids: [keep]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        migrate_config(config_path=config_file)

    assert config_file.read_text(encoding="utf-8") == "folder_ids: [keep]\n"


def test_migrate_config_force_overwrites(monkeypatch, tmp_path):
    monkeypatch.setenv("FOLDER_IDS", "fresh")
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    config_file = tmp_path / "config.yml"
    config_file.write_text("folder_ids: [stale]\n", encoding="utf-8")

    migrate_config(config_path=config_file, force=True)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["folder_ids"] == ["fresh"]


# --- presets DAG wiring -----------------------------------------------------


def test_yaml_presets_merge_over_builtins(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "presets": {
                "transcript-cleanup": {"instructions": "clean it"},
                "keypoints": {"depends_on": ["transcript-cleanup"]},
            },
        },
    )

    cfg = load_config(config_path=config_file, validate_providers=False)

    by_name = {p.name: p for p in cfg.presets}
    assert set(by_name) == {"keypoints", "transcript-cleanup"}
    assert by_name["keypoints"].depends_on == ("transcript-cleanup",)
    # The built-in instructions are preserved when only depends_on is overridden.
    assert by_name["keypoints"].artifact_suffix == ".keypoints.md"


def test_yaml_presets_default_to_builtins_when_absent(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(config_file, {"stt": {"provider": "disabled"}})

    cfg = load_config(config_path=config_file, validate_providers=False)

    assert {p.name for p in cfg.presets} == {"keypoints"}


def test_yaml_presets_can_disable_builtin(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {"stt": {"provider": "disabled"}, "presets": {"keypoints": {"enabled": False}}},
    )

    cfg = load_config(config_path=config_file, validate_providers=False)

    assert cfg.presets == ()


def test_yaml_presets_invalid_dag_raises(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "presets": {"summary": {"instructions": "s", "depends_on": ["ghost"]}},
        },
    )

    with pytest.raises(ValueError, match="unknown or disabled"):
        load_config(config_path=config_file, validate_providers=False)


def test_yaml_preset_prompt_file_resolves_relative_to_config(tmp_path):
    config_file = tmp_path / "config.yml"
    (tmp_path / "managers.md").write_text("manager prompt body", encoding="utf-8")
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "presets": {"managers": {"prompt_file": "managers.md"}},
        },
    )

    cfg = load_config(config_path=config_file, validate_providers=False)

    by_name = {p.name: p for p in cfg.presets}
    assert by_name["managers"].instructions == "manager prompt body"
    assert by_name["managers"].prompt_file == "managers.md"


def test_yaml_preset_prompt_file_absolute_path(tmp_path):
    config_file = tmp_path / "config.yml"
    prompt = tmp_path / "elsewhere" / "p.md"
    prompt.parent.mkdir()
    prompt.write_text("absolute prompt", encoding="utf-8")
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "presets": {"abs": {"prompt_file": str(prompt)}},
        },
    )

    cfg = load_config(config_path=config_file, validate_providers=False)

    by_name = {p.name: p for p in cfg.presets}
    assert by_name["abs"].instructions == "absolute prompt"


def test_yaml_preset_prompt_file_falls_back_to_packaged_asset(tmp_path):
    config_file = tmp_path / "config.yml"
    # The file exists only as a packaged asset, not next to the config.
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "presets": {"extra": {"prompt_file": "keypoints.md"}},
        },
    )

    cfg = load_config(config_path=config_file, validate_providers=False)

    by_name = {p.name: p for p in cfg.presets}
    assert "## Задачи" in by_name["extra"].instructions


def test_yaml_preset_instructions_win_over_prompt_file(tmp_path):
    config_file = tmp_path / "config.yml"
    (tmp_path / "managers.md").write_text("from file", encoding="utf-8")
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "presets": {
                "managers": {"instructions": "inline wins", "prompt_file": "managers.md"}
            },
        },
    )

    cfg = load_config(config_path=config_file, validate_providers=False)

    by_name = {p.name: p for p in cfg.presets}
    assert by_name["managers"].instructions == "inline wins"


def test_yaml_preset_empty_prompt_file_raises(tmp_path):
    config_file = tmp_path / "config.yml"
    (tmp_path / "blank.md").write_text("   \n", encoding="utf-8")
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "presets": {"blank": {"prompt_file": "blank.md"}},
        },
    )

    with pytest.raises(ValueError, match="empty"):
        load_config(config_path=config_file, validate_providers=False)


def test_yaml_preset_missing_prompt_file_raises(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "presets": {"gone": {"prompt_file": "no-such-prompt.md"}},
        },
    )

    with pytest.raises(ValueError, match="could not be resolved"):
        load_config(config_path=config_file, validate_providers=False)


# --- strict YAML validation -------------------------------------------------


def test_yaml_rejects_duplicate_top_level_key(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        "stt:\n  provider: disabled\nbitrate: 96k\nbitrate: 128k\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key"):
        load_config(config_path=config_file, validate_providers=False)


def test_yaml_rejects_duplicate_preset_key(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        "stt:\n"
        "  provider: disabled\n"
        "presets:\n"
        "  managers:\n"
        "    instructions: first\n"
        "  managers:\n"
        "    instructions: second\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key"):
        load_config(config_path=config_file, validate_providers=False)


def test_yaml_rejects_duplicate_artifact_suffix_among_enabled(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "presets": {
                "first": {"instructions": "a", "artifact_suffix": ".shared.md"},
                "second": {"instructions": "b", "artifact_suffix": ".shared.md"},
            },
        },
    )

    with pytest.raises(ValueError, match="artifact_suffix"):
        load_config(config_path=config_file, validate_providers=False)


def test_yaml_duplicate_artifact_suffix_ignored_when_disabled(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "presets": {
                "first": {"instructions": "a", "artifact_suffix": ".shared.md"},
                "second": {
                    "instructions": "b",
                    "artifact_suffix": ".shared.md",
                    "enabled": False,
                },
            },
        },
    )

    cfg = load_config(config_path=config_file, validate_providers=False)

    assert {p.name for p in cfg.presets} == {"keypoints", "first"}


def test_yaml_shared_prompt_file_distinct_names_and_suffixes_allowed(tmp_path):
    config_file = tmp_path / "config.yml"
    (tmp_path / "shared.md").write_text("shared prompt body", encoding="utf-8")
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "presets": {
                "first": {"prompt_file": "shared.md", "artifact_suffix": ".one.md"},
                "second": {"prompt_file": "shared.md", "artifact_suffix": ".two.md"},
            },
        },
    )

    cfg = load_config(config_path=config_file, validate_providers=False)

    by_name = {p.name: p for p in cfg.presets}
    assert by_name["first"].instructions == "shared prompt body"
    assert by_name["second"].instructions == "shared prompt body"
    assert by_name["first"].artifact_suffix == ".one.md"
    assert by_name["second"].artifact_suffix == ".two.md"


def test_env_migration_seeds_keypoints_preset(monkeypatch, tmp_path):
    monkeypatch.setenv("FOLDER_IDS", "f1")
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    monkeypatch.setenv("OPENAI_KEYPOINTS", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    config_file = tmp_path / "config.yml"

    cfg = load_config(config_path=config_file, validate_providers=False)

    assert {p.name for p in cfg.presets} == {"keypoints"}


def test_env_migration_drops_keypoints_when_gate_off(monkeypatch, tmp_path):
    monkeypatch.setenv("FOLDER_IDS", "f1")
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    monkeypatch.setenv("OPENAI_KEYPOINTS", "false")
    config_file = tmp_path / "config.yml"

    cfg = load_config(config_path=config_file, validate_providers=False)

    assert cfg.presets == ()


# --- openai.max_parallel ----------------------------------------------------


def test_max_parallel_defaults_to_four(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(config_file, {"stt": {"provider": "disabled"}})

    cfg = load_config(config_path=config_file, validate_providers=False)

    assert cfg.openai_max_parallel == 4


def test_yaml_max_parallel_is_read(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {"stt": {"provider": "disabled"}, "openai": {"max_parallel": 8}},
    )

    cfg = load_config(config_path=config_file, validate_providers=False)

    assert cfg.openai_max_parallel == 8


def test_yaml_max_parallel_must_be_positive(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {"stt": {"provider": "disabled"}, "openai": {"max_parallel": 0}},
    )

    with pytest.raises(ValueError, match="max_parallel must be positive"):
        load_config(config_path=config_file, validate_providers=False)


def test_yaml_max_parallel_rejects_bool(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {"stt": {"provider": "disabled"}, "openai": {"max_parallel": True}},
    )

    with pytest.raises(ValueError, match="max_parallel must be an integer"):
        load_config(config_path=config_file, validate_providers=False)


def test_yaml_max_parallel_rejects_non_integer_float(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {"stt": {"provider": "disabled"}, "openai": {"max_parallel": 2.9}},
    )

    with pytest.raises(ValueError, match="max_parallel must be an integer"):
        load_config(config_path=config_file, validate_providers=False)


def test_env_migration_reads_max_parallel(monkeypatch, tmp_path):
    monkeypatch.setenv("FOLDER_IDS", "f1")
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    monkeypatch.setenv("OPENAI_MAX_PARALLEL", "6")
    config_file = tmp_path / "config.yml"

    cfg = load_config(config_path=config_file, validate_providers=False)

    assert cfg.openai_max_parallel == 6


def test_max_parallel_round_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    monkeypatch.setenv("OPENAI_MAX_PARALLEL", "9")
    cfg = load_config(validate_providers=False)

    data = _config_to_yaml_dict(cfg)
    config_file = tmp_path / "config.yml"
    _write_yaml(config_file, data)
    reloaded = load_config(config_path=config_file, validate_providers=False)

    assert reloaded.openai_max_parallel == 9


# --- prompt assets & config generation --------------------------------------


def test_copy_prompt_assets_copies_all_packaged_prompts(tmp_path):
    target = tmp_path / "prompts"
    written = copy_prompt_assets(target)

    assert {p.name for p in written} == set(PACKAGED_PROMPT_ASSETS)
    for name in PACKAGED_PROMPT_ASSETS:
        assert (target / name).read_text(encoding="utf-8").strip()


def test_copy_prompt_assets_skips_existing_without_overwrite(tmp_path):
    target = tmp_path / "prompts"
    target.mkdir()
    (target / "keypoints.md").write_text("custom", encoding="utf-8")

    written = copy_prompt_assets(target)

    assert target / "keypoints.md" not in written
    assert (target / "keypoints.md").read_text(encoding="utf-8") == "custom"


def test_init_creates_config_with_prompts_and_relative_paths(tmp_path):
    config_file = tmp_path / "config.yml"

    path = init_config(config_path=config_file)

    assert path == config_file
    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["presets"]["keypoints"]["enabled"] is True
    assert data["presets"]["keypoints"]["prompt_file"] == "prompts/keypoints.md"
    # Only keypoints is enabled by default (the default transcript -> keypoints chain).
    assert [name for name, p in data["presets"].items() if p["enabled"]] == ["keypoints"]
    # The packaged prompt assets are copied beside the config so extra presets
    # (transcript-cleanup, action-items) are one edit away.
    assert (tmp_path / "prompts" / "keypoints.md").is_file()
    assert (tmp_path / "prompts" / "transcript-cleanup.md").is_file()
    # The generated config loads back without provider secrets and yields the chain.
    cfg = load_config(config_path=config_file, validate_providers=False)
    assert {p.name for p in cfg.presets} == {"keypoints"}


def test_init_local_writes_under_cwd_data(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    path = init_config(local=True)

    assert path == Path("data") / CONFIG_FILE_NAME
    assert (tmp_path / "data" / CONFIG_FILE_NAME).is_file()
    assert (tmp_path / "data" / "prompts" / "keypoints.md").is_file()


def test_init_uses_user_path_without_config(monkeypatch, tmp_path):
    user = tmp_path / "user" / "config.yml"
    monkeypatch.setattr("src.config._user_config_path", lambda: user)
    monkeypatch.delenv("GDSTT_CONFIG", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)

    path = init_config()

    assert path == user
    assert user.is_file()


def test_init_default_target_honors_data_dir(monkeypatch, tmp_path):
    """With no flags, init writes where the runtime resolver reads.

    Guards the Docker fix: the image bakes DATA_DIR=/app/data, so a bare
    ``gdstt config init`` must land inside the mounted volume (matching the
    runtime/``doctor`` read path), not the per-user config path.
    """
    user = tmp_path / "user" / "config.yml"
    monkeypatch.setattr("src.config._user_config_path", lambda: user)
    monkeypatch.delenv("GDSTT_CONFIG", raising=False)
    data_dir = tmp_path / "app" / "data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    path = init_config()

    # Writes config (and prompts) under DATA_DIR, not the per-user path.
    assert path == data_dir / CONFIG_FILE_NAME
    assert path.is_file()
    assert (data_dir / "prompts" / "keypoints.md").is_file()
    assert not user.exists()


def test_init_default_target_honors_relative_data_dir(monkeypatch, tmp_path):
    """A relative DATA_DIR in the bare default anchors to the dotenv parent.

    Guards the resolver path the Docker fix routes ``init`` through: a relative
    ``DATA_DIR`` must resolve via ``_resolve_relative_to_dotenv`` exactly as the
    runtime reader does, not against an unrelated cwd.
    """
    user = tmp_path / "user" / "config.yml"
    monkeypatch.setattr("src.config._user_config_path", lambda: user)
    monkeypatch.delenv("GDSTT_CONFIG", raising=False)
    # Pin the dotenv anchor inside tmp_path so the relative DATA_DIR is
    # deterministic regardless of any .env in the checkout.
    dotenv = tmp_path / "anchor" / ".env"
    monkeypatch.setattr("src.config._dotenv_path", lambda: dotenv)
    monkeypatch.setenv("DATA_DIR", "app/data")

    path = init_config()

    assert path == dotenv.parent / "app" / "data" / CONFIG_FILE_NAME
    assert path.is_file()
    assert not user.exists()


def test_init_default_blank_gdstt_config_falls_back_to_data_dir(monkeypatch, tmp_path):
    """A whitespace-only GDSTT_CONFIG is treated as unset, so DATA_DIR wins.

    The resolver strips ``GDSTT_CONFIG`` before honoring it; this guards the
    init default branch against regressing to a raw (non-stripped) read that
    would target an empty path instead of the volume.
    """
    user = tmp_path / "user" / "config.yml"
    monkeypatch.setattr("src.config._user_config_path", lambda: user)
    monkeypatch.setenv("GDSTT_CONFIG", "   ")
    data_dir = tmp_path / "app" / "data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    path = init_config()

    assert path == data_dir / CONFIG_FILE_NAME
    assert path.is_file()
    assert not user.exists()


def test_init_data_dir_does_not_override_explicit_targets(monkeypatch, tmp_path):
    """DATA_DIR awareness applies only to the bare default, not --local/--config."""
    user = tmp_path / "user" / "config.yml"
    monkeypatch.setattr("src.config._user_config_path", lambda: user)
    monkeypatch.delenv("GDSTT_CONFIG", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "datadir"))

    # Explicit --config still wins.
    explicit = tmp_path / "explicit" / "config.yml"
    assert init_config(config_path=explicit) == explicit

    # --local still targets ./data/config.yml under the cwd.
    monkeypatch.chdir(tmp_path)
    assert init_config(local=True) == Path("data") / CONFIG_FILE_NAME


def test_init_default_prefers_gdstt_config_over_data_dir(monkeypatch, tmp_path):
    """In the bare-default branch, GDSTT_CONFIG outranks DATA_DIR and the user path."""
    user = tmp_path / "user" / "config.yml"
    monkeypatch.setattr("src.config._user_config_path", lambda: user)
    env_target = tmp_path / "env" / "config.yml"
    monkeypatch.setenv("GDSTT_CONFIG", str(env_target))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "datadir"))

    path = init_config()

    assert path == env_target
    assert env_target.is_file()
    assert not user.exists()
    assert not (tmp_path / "datadir").exists()


def test_init_default_does_not_dereference_forwarding_pointer(monkeypatch, tmp_path):
    """init targets the bootstrap location, never silently following a pointer.

    The bootstrap target (here the per-user path) is a forwarding pointer to a
    real config. init must treat the pointer file itself as the existing config
    and refuse to overwrite it without --force - it must NOT dereference the
    pointer and create the real target behind it.
    """
    user = tmp_path / "user" / "config.yml"
    monkeypatch.setattr("src.config._user_config_path", lambda: user)
    monkeypatch.delenv("GDSTT_CONFIG", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    real_target = tmp_path / "real" / "config.yml"
    user.parent.mkdir(parents=True, exist_ok=True)
    user.write_text(f"config_file: {real_target}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        init_config()

    # The pointer was not dereferenced: the real target was never created.
    assert not real_target.exists()


def test_init_output_dir_sets_folder_target(tmp_path):
    config_file = tmp_path / "config.yml"
    out = tmp_path / "artifacts"

    init_config(config_path=config_file, output_dir=out)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["output"]["target"] == "folder"
    assert data["output"]["dir"] == "artifacts"
    cfg = load_config(config_path=config_file, validate_providers=False)
    assert cfg.output_target == "folder"
    assert cfg.output_dir == out


def test_init_data_dir_writes_relative_path(tmp_path):
    config_file = tmp_path / "config.yml"

    init_config(config_path=config_file, data_dir=tmp_path / "store")

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["data_dir"] == "store"


def test_init_prompt_dir_points_prompt_files(tmp_path):
    config_file = tmp_path / "config.yml"
    prompt_dir = tmp_path / "shared-prompts"

    init_config(config_path=config_file, prompt_dir=prompt_dir)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["presets"]["keypoints"]["prompt_file"] == "shared-prompts/keypoints.md"
    assert (prompt_dir / "keypoints.md").is_file()


def test_init_refuses_existing_without_force(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("folder_ids: [keep]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        init_config(config_path=config_file)

    assert config_file.read_text(encoding="utf-8") == "folder_ids: [keep]\n"


def test_init_force_overwrites(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("folder_ids: [stale]\n", encoding="utf-8")

    init_config(config_path=config_file, force=True)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["folder_ids"] == []


def test_link_moves_full_config_and_leaves_pointer(monkeypatch, tmp_path):
    user = tmp_path / "user" / "config.yml"
    user.parent.mkdir()
    _write_yaml(user, {"folder_ids": ["moved"], "stt": {"provider": "disabled"}})
    monkeypatch.setattr("src.config._user_config_path", lambda: user)
    monkeypatch.delenv("GDSTT_CONFIG", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)

    dest_dir = tmp_path / "linked"
    dest = link_config(dest_dir)

    assert dest == dest_dir / CONFIG_FILE_NAME
    moved = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert moved["folder_ids"] == ["moved"]
    # The user path now forwards to the moved config.
    pointer = yaml.safe_load(user.read_text(encoding="utf-8"))
    assert "config_file" in pointer
    _, effective = resolve_effective_config_path()
    assert effective.resolve() == dest.resolve()


def test_link_creates_default_when_no_config(monkeypatch, tmp_path):
    user = tmp_path / "user" / "config.yml"
    monkeypatch.setattr("src.config._user_config_path", lambda: user)
    monkeypatch.delenv("GDSTT_CONFIG", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)

    dest_dir = tmp_path / "linked"
    dest = link_config(dest_dir)

    data = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert data["presets"]["keypoints"]["enabled"] is True
    assert (dest_dir / "prompts" / "keypoints.md").is_file()
    pointer = yaml.safe_load(user.read_text(encoding="utf-8"))
    assert "config_file" in pointer


def test_link_refuses_existing_dest_without_force(monkeypatch, tmp_path):
    user = tmp_path / "user" / "config.yml"
    user.parent.mkdir()
    _write_yaml(user, {"folder_ids": ["x"], "stt": {"provider": "disabled"}})
    monkeypatch.setattr("src.config._user_config_path", lambda: user)
    monkeypatch.delenv("GDSTT_CONFIG", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)

    dest_dir = tmp_path / "linked"
    dest_dir.mkdir()
    (dest_dir / CONFIG_FILE_NAME).write_text("folder_ids: [keep]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        link_config(dest_dir)

    assert (dest_dir / CONFIG_FILE_NAME).read_text(encoding="utf-8") == (
        "folder_ids: [keep]\n"
    )


def test_link_copy_prompts_into_dest(monkeypatch, tmp_path):
    user = tmp_path / "user" / "config.yml"
    user.parent.mkdir()
    _write_yaml(user, {"folder_ids": ["x"], "stt": {"provider": "disabled"}})
    monkeypatch.setattr("src.config._user_config_path", lambda: user)
    monkeypatch.delenv("GDSTT_CONFIG", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)

    dest_dir = tmp_path / "linked"
    link_config(dest_dir, copy_prompts=True)

    assert (dest_dir / "prompts" / "keypoints.md").is_file()


def test_migrate_writes_preset_prompt_file(monkeypatch, tmp_path):
    monkeypatch.setenv("FOLDER_IDS", "m1")
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    monkeypatch.setenv("OPENAI_KEYPOINTS", "false")
    config_file = tmp_path / "config.yml"

    migrate_config(config_path=config_file)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    # OPENAI_KEYPOINTS=false -> keypoints disabled but it STILL carries a prompt_file.
    assert data["presets"]["keypoints"]["enabled"] is False
    assert data["presets"]["keypoints"]["prompt_file"] == "prompts/keypoints.md"
    assert (tmp_path / "prompts" / "keypoints.md").is_file()


def test_migrate_keypoints_enabled_writes_prompt_file(monkeypatch, tmp_path):
    monkeypatch.setenv("FOLDER_IDS", "m1")
    monkeypatch.setenv("STT_PROVIDER", "disabled")
    monkeypatch.setenv("OPENAI_KEYPOINTS", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    config_file = tmp_path / "config.yml"

    migrate_config(config_path=config_file)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["presets"]["keypoints"]["enabled"] is True


# --- config get / set / unset -----------------------------------------------


def _base_config_file(tmp_path: Path) -> Path:
    """Create a valid full config (provider disabled) for get/set/unset tests."""
    config_file = tmp_path / "config.yml"
    init_config(config_path=config_file)
    # The default init writes provider=deepgram; disable it so loads don't require
    # a Deepgram key and so openai.api_key isn't required (keypoints stays enabled).
    config_set("stt.provider", "disabled", config_path=config_file)
    config_set("openai.api_key", "sk-base", config_path=config_file)
    return config_file


def test_config_set_openai_api_key_and_model(tmp_path):
    config_file = _base_config_file(tmp_path)

    config_set("openai.api_key", "sk-new", config_path=config_file)
    config_set("openai.model", "gpt-5.4", config_path=config_file)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["openai"]["api_key"] == "sk-new"
    assert data["openai"]["model"] == "gpt-5.4"
    assert config_get("openai.model", config_path=config_file) == "gpt-5.4"


def test_config_set_output_dir_sets_folder_target(tmp_path):
    config_file = _base_config_file(tmp_path)

    config_set("output.dir", "out", config_path=config_file)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["output"]["dir"] == "out"
    assert data["output"]["target"] == "folder"


def test_config_set_output_drive_true_sets_drive_target(tmp_path):
    config_file = _base_config_file(tmp_path)
    config_set("output.dir", "out", config_path=config_file)

    config_set("output.drive", "true", config_path=config_file)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["output"]["target"] == "drive"


def test_config_set_output_drive_false_requires_dir(tmp_path):
    config_file = _base_config_file(tmp_path)
    before = config_file.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="config set output.dir"):
        config_set("output.drive", "false", config_path=config_file)

    assert config_file.read_text(encoding="utf-8") == before


def test_config_set_preset_prompt_file(tmp_path):
    config_file = _base_config_file(tmp_path)

    config_set("presets.keypoints.prompt_file", "prompts/keypoints.md", config_path=config_file)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["presets"]["keypoints"]["prompt_file"] == "prompts/keypoints.md"


def test_config_set_preset_depends_on_parses_list(tmp_path):
    config_file = _base_config_file(tmp_path)
    # Make transcript-cleanup a real enabled dependency target.
    config_set(
        "presets.transcript-cleanup.prompt_file",
        "prompts/transcript-cleanup.md",
        config_path=config_file,
    )
    config_set("presets.transcript-cleanup.enabled", "true", config_path=config_file)

    config_set(
        "presets.keypoints.depends_on", "transcript-cleanup", config_path=config_file
    )

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["presets"]["keypoints"]["depends_on"] == ["transcript-cleanup"]


def test_config_set_preset_depends_on_accepts_json_list(tmp_path):
    config_file = _base_config_file(tmp_path)
    config_set(
        "presets.transcript-cleanup.prompt_file",
        "prompts/transcript-cleanup.md",
        config_path=config_file,
    )
    config_set("presets.transcript-cleanup.enabled", "true", config_path=config_file)

    config_set(
        "presets.keypoints.depends_on",
        '["transcript-cleanup"]',
        config_path=config_file,
    )

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["presets"]["keypoints"]["depends_on"] == ["transcript-cleanup"]


def test_config_unset_removes_key(tmp_path):
    config_file = _base_config_file(tmp_path)
    config_set("proxy_url", "http://proxy", config_path=config_file)

    config_unset("proxy_url", config_path=config_file)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert "proxy_url" not in data


def test_config_unset_missing_key_raises(tmp_path):
    config_file = _base_config_file(tmp_path)

    with pytest.raises(ValueError, match="not set"):
        config_unset("nope.missing", config_path=config_file)


def test_config_get_whole_masks_secrets(tmp_path):
    config_file = _base_config_file(tmp_path)
    config_set("openai.api_key", "sk-secret", config_path=config_file)

    output = config_get(config_path=config_file)

    assert "sk-secret" not in output
    assert "***" in output


def test_config_get_single_secret_is_masked_by_default(tmp_path):
    config_file = _base_config_file(tmp_path)
    config_set("openai.api_key", "sk-secret", config_path=config_file)

    # A single-key get of a secret leaf must not leak the value to stdout/logs.
    assert config_get("openai.api_key", config_path=config_file) == "***"


def test_config_get_single_secret_revealed_with_show_secrets(tmp_path):
    config_file = _base_config_file(tmp_path)
    config_set("openai.api_key", "sk-secret", config_path=config_file)

    assert (
        config_get("openai.api_key", config_path=config_file, show_secrets=True)
        == "sk-secret"
    )


def test_config_get_single_nonsecret_is_plain(tmp_path):
    config_file = _base_config_file(tmp_path)
    config_set("openai.model", "gpt-x", config_path=config_file)

    assert config_get("openai.model", config_path=config_file) == "gpt-x"


def test_config_get_missing_key_raises(tmp_path):
    config_file = _base_config_file(tmp_path)

    with pytest.raises(ValueError, match="not set"):
        config_get("openai.nope", config_path=config_file)


def test_config_set_follows_pointer_to_effective_file(tmp_path):
    real = tmp_path / "real" / "config.yml"
    real.parent.mkdir()
    init_config(config_path=real)
    config_set("stt.provider", "disabled", config_path=real)
    config_set("openai.api_key", "sk-base", config_path=real)
    pointer = tmp_path / "pointer.yml"
    pointer.write_text(f"config_file: {real}\n", encoding="utf-8")

    config_set("openai.model", "gpt-pointer", config_path=pointer)

    # The effective (real) file changed; the pointer file is untouched.
    assert pointer.read_text(encoding="utf-8") == f"config_file: {real}\n"
    data = yaml.safe_load(real.read_text(encoding="utf-8"))
    assert data["openai"]["model"] == "gpt-pointer"


def test_config_set_invalid_leaves_file_unchanged(tmp_path):
    config_file = _base_config_file(tmp_path)
    before = config_file.read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        config_set("output.target", "s3", config_path=config_file)

    assert config_file.read_text(encoding="utf-8") == before


# --- google auth config ------------------------------------------------------


def test_yaml_google_inline_credentials_and_token(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "presets": {"keypoints": {"enabled": False}},
            "google": {
                "credentials": {"installed": {"client_id": "cid", "client_secret": "x"}},
                "token": {"token": "t", "refresh_token": "r"},
            },
        },
    )

    cfg = load_config(config_path=config_file)

    assert cfg.google_credentials == {
        "installed": {"client_id": "cid", "client_secret": "x"}
    }
    assert cfg.google_token == {"token": "t", "refresh_token": "r"}
    assert cfg.google_credentials_file is None
    assert cfg.google_token_file is None
    assert cfg.config_file == config_file


def test_yaml_google_file_paths_resolve_relative(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "presets": {"keypoints": {"enabled": False}},
            "google": {
                "credentials_file": "secrets/creds.json",
                "token_file": "/abs/token.json",
            },
        },
    )

    cfg = load_config(config_path=config_file)

    assert cfg.google_credentials is None
    assert cfg.google_credentials_file == tmp_path / "secrets" / "creds.json"
    assert cfg.google_token_file == Path("/abs/token.json")


def test_yaml_google_back_compat_data_dir_fallback(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {"stt": {"provider": "disabled"}, "presets": {"keypoints": {"enabled": False}}},
    )

    cfg = load_config(config_path=config_file)

    assert cfg.google_credentials is None
    assert cfg.google_token is None
    assert cfg.google_credentials_file is None
    assert cfg.google_token_file is None


def test_yaml_google_credentials_both_inline_and_file_fails(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "presets": {"keypoints": {"enabled": False}},
            "google": {
                "credentials": {"installed": {"client_id": "cid"}},
                "credentials_file": "creds.json",
            },
        },
    )

    with pytest.raises(ValueError, match="both set"):
        load_config(config_path=config_file)


def test_yaml_google_token_both_inline_and_file_fails(tmp_path):
    config_file = tmp_path / "config.yml"
    _write_yaml(
        config_file,
        {
            "stt": {"provider": "disabled"},
            "presets": {"keypoints": {"enabled": False}},
            "google": {
                "token": {"token": "t"},
                "token_file": "token.json",
            },
        },
    )

    with pytest.raises(ValueError, match="both set"):
        load_config(config_path=config_file)


def test_import_google_credentials_writes_inline(tmp_path):
    config_file = _base_config_file(tmp_path)
    creds_src = tmp_path / "download.json"
    creds_src.write_text(
        json.dumps({"installed": {"client_id": "cid", "client_secret": "sek"}})
    )

    import_google_credentials(creds_src, config_path=config_file)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["google"]["credentials"] == {
        "installed": {"client_id": "cid", "client_secret": "sek"}
    }
    assert "credentials_file" not in data["google"]


def test_import_google_credentials_clears_file_pointer(tmp_path):
    config_file = _base_config_file(tmp_path)
    config_set("google.credentials_file", "old/creds.json", config_path=config_file)
    creds_src = tmp_path / "download.json"
    creds_src.write_text(json.dumps({"installed": {"client_id": "cid"}}))

    import_google_credentials(creds_src, config_path=config_file)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert "credentials_file" not in data["google"]
    assert data["google"]["credentials"] == {"installed": {"client_id": "cid"}}


def test_use_google_files_switches_and_clears_inline(tmp_path):
    config_file = _base_config_file(tmp_path)
    creds_src = tmp_path / "download.json"
    creds_src.write_text(json.dumps({"installed": {"client_id": "cid"}}))
    import_google_credentials(creds_src, config_path=config_file)
    config_set("google.token.token", "inline-tok", config_path=config_file)

    creds_file = tmp_path / "creds" / "client.json"
    use_google_files(creds_file, config_path=config_file)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert "credentials" not in data["google"]
    assert "token" not in data["google"]
    assert data["google"]["credentials_file"] == str(creds_file)
    assert data["google"]["token_file"] == str(creds_file.parent / "token.json")


def test_use_google_files_honors_explicit_token_file(tmp_path):
    config_file = _base_config_file(tmp_path)
    creds_file = tmp_path / "client.json"
    token_file = tmp_path / "elsewhere" / "tok.json"

    use_google_files(creds_file, token_file=token_file, config_path=config_file)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["google"]["token_file"] == str(token_file)


def test_config_get_masks_inline_google_secrets(tmp_path):
    config_file = _base_config_file(tmp_path)
    creds_src = tmp_path / "download.json"
    creds_src.write_text(
        json.dumps({"installed": {"client_id": "cid", "client_secret": "sup3rsecret"}})
    )
    import_google_credentials(creds_src, config_path=config_file)
    config_set("google.token.refresh_token", "rt-secret", config_path=config_file)
    config_set("google.token.token", "tok-secret", config_path=config_file)

    output = config_get(config_path=config_file)

    assert "sup3rsecret" not in output
    assert "rt-secret" not in output
    assert "tok-secret" not in output
    assert "***" in output


def test_generated_config_prefers_inline_and_omits_file_keys(tmp_path):
    config_file = tmp_path / "config.yml"
    init_config(config_path=config_file)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    # The generated config ships an (empty) google block without file pointers.
    assert "google" in data
    assert "credentials_file" not in data["google"]
    assert "token_file" not in data["google"]
