from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from google.auth.exceptions import RefreshError

from src import auth


def _assert_secure_token_mode(path: Path) -> None:
    # Windows does not expose POSIX owner-only permission bits through st_mode.
    if os.name == "nt":
        assert path.exists()
        return
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def _write_token(path: Path, payload: dict | None = None) -> None:
    payload = payload or {
        "token": "access",
        "refresh_token": "refresh",
        "client_id": "cid",
        "client_secret": "csec",
        "token_uri": "https://oauth2.googleapis.com/token",
        "scopes": list(auth.SCOPES),
    }
    path.write_text(json.dumps(payload))


def test_load_credentials_missing_token_raises(tmp_path):
    with pytest.raises(auth.AuthError, match="Token file not found"):
        auth.load_credentials(tmp_path)


def test_load_credentials_returns_valid_creds(tmp_path, mocker):
    token_file = tmp_path / "token.json"
    _write_token(token_file)

    fake_creds = MagicMock()
    fake_creds.valid = True
    fake_creds.expired = False
    fake_creds.refresh_token = "refresh"
    fake_creds.scopes = list(auth.SCOPES)

    from_file = mocker.patch(
        "src.auth.Credentials.from_authorized_user_file", return_value=fake_creds
    )

    result = auth.load_credentials(tmp_path)

    assert result is fake_creds
    from_file.assert_called_once_with(str(token_file), auth.SCOPES)


def test_load_credentials_refreshes_expired(tmp_path, mocker):
    token_file = tmp_path / "token.json"
    _write_token(token_file)

    fake_creds = MagicMock()
    fake_creds.valid = False
    fake_creds.expired = True
    fake_creds.refresh_token = "refresh"
    fake_creds.scopes = list(auth.SCOPES)
    fake_creds.to_json.return_value = '{"refreshed": true}'

    mocker.patch("src.auth.Credentials.from_authorized_user_file", return_value=fake_creds)
    request_cls = mocker.patch("src.auth.Request")

    result = auth.load_credentials(tmp_path)

    fake_creds.refresh.assert_called_once_with(request_cls.return_value)
    assert result is fake_creds
    assert token_file.read_text() == '{"refreshed": true}'


def test_load_credentials_refresh_error_raises(tmp_path, mocker):
    token_file = tmp_path / "token.json"
    _write_token(token_file)

    fake_creds = MagicMock()
    fake_creds.valid = False
    fake_creds.expired = True
    fake_creds.refresh_token = "refresh"
    fake_creds.scopes = list(auth.SCOPES)
    fake_creds.refresh.side_effect = RefreshError("boom")

    mocker.patch("src.auth.Credentials.from_authorized_user_file", return_value=fake_creds)
    mocker.patch("src.auth.Request")

    with pytest.raises(auth.AuthError, match="could not be refreshed"):
        auth.load_credentials(tmp_path)


def test_load_credentials_malformed_token_raises_auth_error(tmp_path, mocker):
    token_file = tmp_path / "token.json"
    _write_token(token_file)

    mocker.patch(
        "src.auth.Credentials.from_authorized_user_file",
        side_effect=ValueError("bad json"),
    )

    with pytest.raises(auth.AuthError, match="malformed"):
        auth.load_credentials(tmp_path)


def test_load_credentials_invalid_no_refresh_raises(tmp_path, mocker):
    token_file = tmp_path / "token.json"
    _write_token(token_file)

    fake_creds = MagicMock()
    fake_creds.valid = False
    fake_creds.expired = False
    fake_creds.refresh_token = None
    fake_creds.scopes = list(auth.SCOPES)

    mocker.patch("src.auth.Credentials.from_authorized_user_file", return_value=fake_creds)

    with pytest.raises(auth.AuthError, match="invalid"):
        auth.load_credentials(tmp_path)


def test_load_credentials_missing_scope_raises(tmp_path, mocker):
    token_file = tmp_path / "token.json"
    # Real google-auth reads scopes from the file when the caller passes
    # scopes=None, but our loader passes scopes=SCOPES, which forces
    # creds.scopes to mirror SCOPES regardless of saved scopes. The check
    # therefore must inspect the file's "scopes" field directly.
    _write_token(
        token_file,
        {
            "token": "access",
            "refresh_token": "refresh",
            "client_id": "cid",
            "client_secret": "csec",
            "token_uri": "https://oauth2.googleapis.com/token",
            "scopes": ["https://www.googleapis.com/auth/userinfo.email"],
        },
    )

    # No need to mock from_authorized_user_file — the scope check fires before
    # we attempt to construct a Credentials object.
    with pytest.raises(auth.AuthError, match="missing required scopes"):
        auth.load_credentials(tmp_path)


def test_load_credentials_missing_scope_field_raises(tmp_path):
    token_file = tmp_path / "token.json"
    token_file.write_text(
        json.dumps(
            {
                "token": "access",
                "refresh_token": "refresh",
                "client_id": "cid",
                "client_secret": "csec",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )
    )

    with pytest.raises(auth.AuthError, match="missing required scopes"):
        auth.load_credentials(tmp_path)


def test_load_credentials_scopes_as_space_string(tmp_path, mocker):
    token_file = tmp_path / "token.json"
    token_file.write_text(
        json.dumps(
            {
                "token": "access",
                "refresh_token": "refresh",
                "client_id": "cid",
                "client_secret": "csec",
                "token_uri": "https://oauth2.googleapis.com/token",
                "scopes": " ".join(auth.SCOPES),
            }
        )
    )

    fake_creds = MagicMock()
    fake_creds.valid = True
    fake_creds.expired = False
    fake_creds.refresh_token = "refresh"
    fake_creds.scopes = list(auth.SCOPES)

    mocker.patch("src.auth.Credentials.from_authorized_user_file", return_value=fake_creds)

    result = auth.load_credentials(tmp_path)
    assert result is fake_creds


def test_load_credentials_token_not_a_dict_raises(tmp_path):
    token_file = tmp_path / "token.json"
    token_file.write_text(json.dumps([]))

    with pytest.raises(auth.AuthError, match="expected a JSON object"):
        auth.load_credentials(tmp_path)


def test_load_credentials_scopes_wrong_type_raises(tmp_path):
    token_file = tmp_path / "token.json"
    token_file.write_text(
        json.dumps(
            {
                "token": "access",
                "refresh_token": "refresh",
                "client_id": "cid",
                "client_secret": "csec",
                "token_uri": "https://oauth2.googleapis.com/token",
                "scopes": 42,
            }
        )
    )

    with pytest.raises(auth.AuthError, match="'scopes' must be a list or string"):
        auth.load_credentials(tmp_path)


def test_load_credentials_scopes_with_non_string_members_raises(tmp_path):
    token_file = tmp_path / "token.json"
    token_file.write_text(
        json.dumps(
            {
                "token": "access",
                "refresh_token": "refresh",
                "client_id": "cid",
                "client_secret": "csec",
                "token_uri": "https://oauth2.googleapis.com/token",
                "scopes": [{}],
            }
        )
    )

    with pytest.raises(auth.AuthError, match="must contain only strings"):
        auth.load_credentials(tmp_path)


def test_scopes_are_drive_only():
    assert "https://www.googleapis.com/auth/drive" in auth.SCOPES
    assert "https://www.googleapis.com/auth/cloud-platform" not in auth.SCOPES


def test_build_drive_service_uses_credentials(tmp_path, mocker):
    fake_creds = MagicMock()
    mocker.patch("src.auth.load_credentials", return_value=fake_creds)
    build_mock = mocker.patch("src.auth.build", return_value="service")

    result = auth.build_drive_service(tmp_path)

    assert result == "service"
    build_mock.assert_called_once_with(
        "drive", "v3", credentials=fake_creds, cache_discovery=False
    )


def test_build_drive_service_uses_config_when_no_dir(mocker, tmp_path):
    fake_cfg = MagicMock()
    fake_cfg.data_dir = tmp_path
    load_config_mock = mocker.patch("src.auth.load_config", return_value=fake_cfg)
    fake_creds = MagicMock()
    load_mock = mocker.patch("src.auth.load_credentials", return_value=fake_creds)
    mocker.patch("src.auth.build", return_value="service")

    auth.build_drive_service()

    load_config_mock.assert_called_once_with(validate_providers=False)
    load_mock.assert_called_once_with(config=fake_cfg)


def test_auth_main_uses_drive_only_config(mocker, tmp_path):
    fake_cfg = MagicMock()
    fake_cfg.data_dir = tmp_path
    load_config_mock = mocker.patch("src.auth.load_config", return_value=fake_cfg)
    flow_mock = mocker.patch("src.auth.run_interactive_flow")
    mocker.patch("src.auth.argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(manual=False, response_url=None))

    auth.main()

    load_config_mock.assert_called_once_with(validate_providers=False)
    flow_mock.assert_called_once_with(tmp_path, manual=False, response_url=None)


def test_run_interactive_flow_missing_credentials(tmp_path):
    with pytest.raises(auth.AuthError, match="Missing"):
        auth.run_interactive_flow(tmp_path)


def test_run_interactive_flow_writes_token(tmp_path, mocker):
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text(json.dumps({"installed": {"client_id": "cid"}}))

    fake_creds = MagicMock()
    fake_creds.to_json.return_value = '{"new": "token"}'

    flow = MagicMock()
    flow.run_local_server.return_value = fake_creds
    flow_cls = mocker.patch(
        "src.auth.InstalledAppFlow.from_client_config", return_value=flow
    )

    auth.run_interactive_flow(tmp_path)

    flow_cls.assert_called_once_with({"installed": {"client_id": "cid"}}, auth.SCOPES)
    flow.run_local_server.assert_called_once_with(port=0, open_browser=True)
    token_file = tmp_path / "token.json"
    assert token_file.read_text() == '{"new": "token"}'
    _assert_secure_token_mode(token_file)


def test_run_interactive_flow_accepts_credentials_json_with_utf8_bom(tmp_path, mocker):
    creds_file = tmp_path / "credentials.json"
    client_config = {
        "installed": {
            "client_id": "cid.apps.googleusercontent.com",
            "client_secret": "csec",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }
    creds_file.write_text(json.dumps(client_config), encoding="utf-8-sig")

    fake_creds = MagicMock()
    fake_creds.to_json.return_value = '{"new": "token"}'
    run_local_server = mocker.patch(
        "src.auth.InstalledAppFlow.run_local_server", return_value=fake_creds
    )

    auth.run_interactive_flow(tmp_path)

    run_local_server.assert_called_once_with(port=0, open_browser=True)
    assert (tmp_path / "token.json").read_text() == '{"new": "token"}'


def test_run_interactive_flow_manual_prints_url_and_exits(tmp_path, mocker, capsys):
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text(json.dumps({"installed": {"client_id": "cid"}}))

    flow = MagicMock()
    flow.authorization_url.return_value = ("https://accounts.example/auth", "state-1")
    mocker.patch("src.auth.InstalledAppFlow.from_client_config", return_value=flow)

    with pytest.raises(SystemExit) as excinfo:
        auth.run_interactive_flow(tmp_path, manual=True)

    assert excinfo.value.code == 0
    assert flow.redirect_uri == "http://localhost"
    flow.authorization_url.assert_called_once_with(access_type="offline", prompt="consent")
    out = capsys.readouterr().out
    assert "https://accounts.example/auth" in out


def test_run_interactive_flow_manual_accepts_localhost_callback_response_url(
    tmp_path,
    mocker,
    monkeypatch,
):
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text(json.dumps({"installed": {"client_id": "cid"}}))

    fake_creds = MagicMock()
    fake_creds.to_json.return_value = '{"new": "token"}'

    flow = MagicMock()
    flow.credentials = fake_creds
    monkeypatch.delenv("OAUTHLIB_INSECURE_TRANSPORT", raising=False)

    def assert_loopback_transport_enabled(**kwargs):
        assert kwargs == {
            "authorization_response": "http://localhost/?state=state-1&code=abc"
        }
        assert os.environ["OAUTHLIB_INSECURE_TRANSPORT"] == "1"

    flow.fetch_token.side_effect = assert_loopback_transport_enabled
    mocker.patch("src.auth.InstalledAppFlow.from_client_config", return_value=flow)

    auth.run_interactive_flow(
        tmp_path,
        response_url="http://localhost/?state=state-1&code=abc",
    )

    assert flow.redirect_uri == "http://localhost"
    flow.fetch_token.assert_called_once_with(
        authorization_response="http://localhost/?state=state-1&code=abc"
    )
    assert "OAUTHLIB_INSECURE_TRANSPORT" not in os.environ
    assert (tmp_path / "token.json").read_text() == '{"new": "token"}'


def test_load_credentials_refresh_writes_token_with_secure_mode(tmp_path, mocker):
    token_file = tmp_path / "token.json"
    _write_token(token_file)

    fake_creds = MagicMock()
    fake_creds.valid = False
    fake_creds.expired = True
    fake_creds.refresh_token = "refresh"
    fake_creds.scopes = list(auth.SCOPES)
    fake_creds.to_json.return_value = '{"refreshed": true}'

    mocker.patch("src.auth.Credentials.from_authorized_user_file", return_value=fake_creds)
    mocker.patch("src.auth.Request")

    auth.load_credentials(tmp_path)

    _assert_secure_token_mode(token_file)


# --- config-owned (inline-first) Google auth ---------------------------------


def _make_config(tmp_path, **overrides):
    from dataclasses import replace

    from tests.test_main import make_config

    base = make_config(data_dir=tmp_path)
    return replace(base, **overrides)


def _inline_token_payload():
    return {
        "token": "access",
        "refresh_token": "refresh",
        "client_id": "cid",
        "client_secret": "csec",
        "token_uri": "https://oauth2.googleapis.com/token",
        "scopes": list(auth.SCOPES),
    }


def test_load_credentials_uses_inline_token(tmp_path, mocker):
    config = _make_config(tmp_path, google_token=_inline_token_payload())

    fake_creds = MagicMock()
    fake_creds.valid = True
    fake_creds.expired = False
    from_info = mocker.patch(
        "src.auth.Credentials.from_authorized_user_info", return_value=fake_creds
    )

    result = auth.load_credentials(config=config)

    assert result is fake_creds
    from_info.assert_called_once_with(_inline_token_payload(), auth.SCOPES)


def test_load_credentials_inline_token_refresh_persists_into_config(tmp_path, mocker):
    config_file = tmp_path / "config.yml"
    config_file.write_text("google:\n  token:\n    token: old\n", encoding="utf-8")
    config = _make_config(
        tmp_path,
        google_token=_inline_token_payload(),
        config_file=config_file,
    )

    fake_creds = MagicMock()
    fake_creds.valid = False
    fake_creds.expired = True
    fake_creds.refresh_token = "refresh"
    fake_creds.to_json.return_value = '{"token": "fresh", "scopes": ["x"]}'
    mocker.patch("src.auth.Credentials.from_authorized_user_info", return_value=fake_creds)
    mocker.patch("src.auth.Request")

    auth.load_credentials(config=config)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["google"]["token"] == {"token": "fresh", "scopes": ["x"]}
    # The inline token was never written into a file beside data_dir.
    assert not (tmp_path / "token.json").exists()


def test_load_credentials_uses_token_file_path(tmp_path, mocker):
    token_file = tmp_path / "nested" / "token.json"
    token_file.parent.mkdir()
    _write_token(token_file)
    config = _make_config(tmp_path, google_token_file=token_file)

    fake_creds = MagicMock()
    fake_creds.valid = True
    fake_creds.expired = False
    from_file = mocker.patch(
        "src.auth.Credentials.from_authorized_user_file", return_value=fake_creds
    )

    result = auth.load_credentials(config=config)

    assert result is fake_creds
    from_file.assert_called_once_with(str(token_file), auth.SCOPES)


def test_load_credentials_config_falls_back_to_data_dir(tmp_path, mocker):
    token_file = tmp_path / "token.json"
    _write_token(token_file)
    config = _make_config(tmp_path)  # no google.* set

    fake_creds = MagicMock()
    fake_creds.valid = True
    fake_creds.expired = False
    from_file = mocker.patch(
        "src.auth.Credentials.from_authorized_user_file", return_value=fake_creds
    )

    auth.load_credentials(config=config)

    from_file.assert_called_once_with(str(token_file), auth.SCOPES)


def test_run_interactive_flow_uses_inline_credentials(tmp_path, mocker):
    config = _make_config(
        tmp_path,
        google_credentials={"installed": {"client_id": "cid"}},
        google_token={"token": "old", "scopes": list(auth.SCOPES)},
        config_file=tmp_path / "config.yml",
    )
    (tmp_path / "config.yml").write_text("google: {}\n", encoding="utf-8")

    fake_creds = MagicMock()
    fake_creds.to_json.return_value = '{"token": "new", "scopes": ["x"]}'
    flow = MagicMock()
    flow.run_local_server.return_value = fake_creds
    flow_cls = mocker.patch(
        "src.auth.InstalledAppFlow.from_client_config", return_value=flow
    )

    auth.run_interactive_flow(config=config)

    flow_cls.assert_called_once_with({"installed": {"client_id": "cid"}}, auth.SCOPES)
    # No token file written; the fresh token went back into google.token.
    assert not (tmp_path / "token.json").exists()
    data = yaml.safe_load((tmp_path / "config.yml").read_text(encoding="utf-8"))
    assert data["google"]["token"] == {"token": "new", "scopes": ["x"]}


def test_run_interactive_flow_file_mode_writes_token_file(tmp_path, mocker):
    creds_file = tmp_path / "creds.json"
    creds_file.write_text(json.dumps({"installed": {"client_id": "cid"}}))
    token_file = tmp_path / "tok.json"
    config = _make_config(
        tmp_path,
        google_credentials_file=creds_file,
        google_token_file=token_file,
    )

    fake_creds = MagicMock()
    fake_creds.to_json.return_value = '{"new": "token"}'
    flow = MagicMock()
    flow.run_local_server.return_value = fake_creds
    mocker.patch("src.auth.InstalledAppFlow.from_client_config", return_value=flow)

    auth.run_interactive_flow(config=config)

    assert token_file.read_text() == '{"new": "token"}'
    _assert_secure_token_mode(token_file)
