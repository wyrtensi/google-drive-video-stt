from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def test_compose_is_config_only_and_does_not_require_env_file():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    service = compose["services"]["google-drive-video-stt"]

    assert "env_file" not in service
    assert service["environment"]["DATA_DIR"] == "/app/data"
    assert service["volumes"] == ["./data:/app/data"]


def test_docker_smoke_is_config_only_and_initializes_clean_volume():
    smoke = (ROOT / "scripts" / "docker-smoke.sh").read_text(encoding="utf-8")

    assert "--env-file" not in smoke
    assert "gdstt config init" in smoke
    assert "from src.config import load_config" in smoke
