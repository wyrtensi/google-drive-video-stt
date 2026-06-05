from pathlib import Path

import pytest
import yaml
from dotenv import load_dotenv as real_load_dotenv

from src.config import (
    CONFIG_FILE_NAME,
    _config_to_yaml_dict,
    load_config,
    migrate_config,
    resolve_config_file_path,
)

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
