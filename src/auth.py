from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import yaml
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from src.config import Config, load_config
from src.config import _write_config_text as _write_secret_config

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/drive",
]


class AuthError(Exception):
    pass


def _credentials_path(data_dir: Path) -> Path:
    return data_dir / "credentials.json"


def _token_path(data_dir: Path) -> Path:
    return data_dir / "token.json"


def _write_token(path: Path, payload: str) -> None:
    path.write_text(payload)
    path.chmod(0o600)


def _fetch_manual_token(flow, response_url: str) -> None:
    parsed = urlparse(response_url)
    allow_loopback_http = (
        parsed.scheme == "http"
        and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    )
    previous = os.environ.get("OAUTHLIB_INSECURE_TRANSPORT")
    if allow_loopback_http:
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    try:
        flow.fetch_token(authorization_response=response_url)
    finally:
        if allow_loopback_http:
            if previous is None:
                os.environ.pop("OAUTHLIB_INSECURE_TRANSPORT", None)
            else:
                os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = previous


def _load_client_config(path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise AuthError(
            f"OAuth client credentials at {path} are malformed: {exc}. "
            "Download a fresh Desktop app credentials JSON from Google Cloud Console."
        ) from exc
    if not isinstance(config, dict):
        raise AuthError(
            f"OAuth client credentials at {path} are malformed: expected a JSON object."
        )
    return config


def _resolve_token_source(
    config: Config | None, data_dir: Path | None
) -> tuple[dict | None, Path | None]:
    """Resolve the saved-token source: inline mapping wins, then file, then data_dir.

    Returns ``(inline_token, token_file)`` with exactly one set. ``inline_token`` is
    a copy of the YAML mapping. When no inline token and no ``google.token_file`` are
    configured, the legacy ``<data_dir>/token.json`` path is used.
    """
    if config is not None:
        if config.google_token is not None:
            return dict(config.google_token), None
        if config.google_token_file is not None:
            return None, config.google_token_file
    base = data_dir if data_dir is not None else (config.data_dir if config else None)
    if base is None:
        raise AuthError("no token source configured")
    return None, _token_path(base)


def _resolve_client_source(
    config: Config | None, data_dir: Path | None
) -> tuple[dict | None, Path | None]:
    """Resolve the OAuth client config source: inline mapping wins, then file, then data_dir."""
    if config is not None and config.google_credentials is not None:
        return dict(config.google_credentials), None
    if config is not None and config.google_credentials_file is not None:
        return None, config.google_credentials_file
    base = data_dir if data_dir is not None else (config.data_dir if config else None)
    if base is None:
        raise AuthError("no OAuth client credentials source configured")
    return None, _credentials_path(base)


def _validate_token_scopes(token_data: dict, where: str) -> None:
    saved_scopes = token_data.get("scopes")
    if isinstance(saved_scopes, str):
        saved_scopes = saved_scopes.split(" ")
    elif saved_scopes is not None and not isinstance(saved_scopes, list):
        raise AuthError(
            f"Token at {where} is malformed: 'scopes' must be a list or "
            f"string, got {type(saved_scopes).__name__}. "
            "Re-run `python -m src.auth` to re-authorize."
        )
    if saved_scopes and not all(isinstance(s, str) for s in saved_scopes):
        raise AuthError(
            f"Token at {where} is malformed: 'scopes' must contain only "
            "strings. Re-run `python -m src.auth` to re-authorize."
        )
    granted = set(saved_scopes or [])
    missing = [s for s in SCOPES if s not in granted]
    if missing:
        raise AuthError(
            f"Token at {where} is missing required scopes: {missing}. "
            "Re-run `python -m src.auth` to re-authorize."
        )


def _persist_inline_token(config: Config, token_json: str) -> None:
    """Write a refreshed inline token back into the effective config's google.token.

    An inline token must never be spilled into a pointer file; it is rewritten into
    the YAML mapping under ``google.token`` so the next run reads the fresh token.
    A missing config file (no config_file path) is logged and skipped rather than
    raising, so a transient refresh still returns valid in-memory credentials.
    """
    config_file = config.config_file
    if config_file is None or not config_file.exists():
        logger.warning(
            "Refreshed inline Google token could not be persisted: no config file."
        )
        return
    try:
        text = config_file.read_text(encoding="utf-8-sig")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("config is not a mapping")
        google = data.setdefault("google", {})
        if not isinstance(google, dict):
            google = {}
            data["google"] = google
        google["token"] = json.loads(token_json)
        # Route through config's secure writer so the rewritten config (now holding a
        # fresh inline refresh_token) is restricted to owner-only, not left at umask.
        _write_secret_config(
            config_file, yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        )
    except (OSError, ValueError) as exc:
        logger.warning("Could not persist refreshed inline Google token: %s", exc)


def _credentials_from_inline_token(token_data: dict, where: str, config: Config) -> Credentials:
    _validate_token_scopes(token_data, where)
    try:
        creds = Credentials.from_authorized_user_info(token_data, SCOPES)
    except ValueError as exc:
        raise AuthError(
            f"Token at {where} is malformed: {exc}. "
            "Re-run `python -m src.auth` to re-authorize."
        ) from exc

    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            raise AuthError(
                f"Token at {where} could not be refreshed: {exc}. "
                "Re-run `python -m src.auth` to re-authorize."
            ) from exc
        _persist_inline_token(config, creds.to_json())
        return creds
    raise AuthError(
        f"Token at {where} is invalid and has no refresh token. "
        "Re-run `python -m src.auth` to re-authorize."
    )


def _credentials_from_token_file(token_file: Path) -> Credentials:
    if not token_file.exists():
        raise AuthError(
            f"Token file not found at {token_file}. "
            "Run `python -m src.auth` to perform interactive OAuth."
        )

    # Inspect the token file directly: Credentials.from_authorized_user_file
    # overwrites creds.scopes with whatever scopes argument we pass, so checking
    # creds.scopes after the fact only echoes our request — it does not reveal
    # what the saved token was actually authorized for.
    try:
        token_data = json.loads(token_file.read_text())
    except (OSError, ValueError) as exc:
        raise AuthError(
            f"Token at {token_file} is malformed: {exc}. "
            "Re-run `python -m src.auth` to re-authorize."
        ) from exc

    if not isinstance(token_data, dict):
        raise AuthError(
            f"Token at {token_file} is malformed: expected a JSON object, "
            f"got {type(token_data).__name__}. "
            "Re-run `python -m src.auth` to re-authorize."
        )

    _validate_token_scopes(token_data, token_file)

    try:
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    except ValueError as exc:
        raise AuthError(
            f"Token at {token_file} is malformed: {exc}. "
            "Re-run `python -m src.auth` to re-authorize."
        ) from exc

    if creds.valid:
        return creds

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            raise AuthError(
                f"Token at {token_file} could not be refreshed: {exc}. "
                "Re-run `python -m src.auth` to re-authorize."
            ) from exc
        _write_token(token_file, creds.to_json())
        return creds

    raise AuthError(
        f"Token at {token_file} is invalid and has no refresh token. "
        "Re-run `python -m src.auth` to re-authorize."
    )


def load_credentials(
    data_dir: Path | None = None,
    *,
    config: Config | None = None,
) -> Credentials:
    """Load Drive OAuth credentials, inline-first then file then data_dir fallback.

    A bare ``data_dir`` (the legacy call) reads ``data_dir/token.json`` exactly as
    before. When a ``config`` is supplied the token source follows the config-owned
    resolution (inline ``google.token`` > ``google.token_file`` > data_dir fallback);
    a refreshed inline token is persisted back into the config's ``google.token``.
    """
    inline_token, token_file = _resolve_token_source(config, data_dir)
    if inline_token is not None:
        assert config is not None  # inline tokens only come from a Config
        return _credentials_from_inline_token(inline_token, "google.token", config)
    assert token_file is not None
    return _credentials_from_token_file(token_file)


def build_drive_service(
    data_dir: Path | None = None,
    *,
    config: Config | None = None,
):
    if config is None and data_dir is None:
        config = load_config(validate_providers=False)
    if config is not None:
        creds = load_credentials(config=config)
    else:
        creds = load_credentials(data_dir)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def run_interactive_flow(
    data_dir: Path | None = None,
    *,
    config: Config | None = None,
    manual: bool = False,
    response_url: str | None = None,
) -> Credentials:
    base = data_dir if data_dir is not None else (config.data_dir if config else None)
    if base is not None:
        base.mkdir(parents=True, exist_ok=True)

    inline_client, client_file = _resolve_client_source(config, data_dir)
    if inline_client is not None:
        client_config = inline_client
    else:
        assert client_file is not None
        if not client_file.exists():
            raise AuthError(
                f"Missing {client_file}. Download OAuth client credentials from Google "
                "Cloud Console and place them at this path."
            )
        client_config = _load_client_config(client_file)

    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)

    if manual or response_url:
        flow.redirect_uri = "http://localhost"
        flow.autogenerate_code_verifier = False
        flow.code_verifier = None
        if response_url:
            _fetch_manual_token(flow, response_url)
            creds = flow.credentials
        else:
            auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent")
            print(f"Open this URL in a browser:\n\n{auth_url}\n")
            print(
                "Then re-run with `gdstt auth <paste-redirect-url>` or "
                "`gdstt auth --manual <paste-redirect-url>`."
            )
            raise SystemExit(0)
    else:
        creds = flow.run_local_server(port=0, open_browser=True)

    # Persist the new token to whichever sink the config selects: an inline token
    # config writes back into google.token; otherwise the token file (config-owned
    # token_file or the data_dir fallback).
    if config is not None and (
        config.google_token is not None or config.google_credentials is not None
    ):
        _persist_inline_token(config, creds.to_json())
    else:
        _, token_file = _resolve_token_source(config, data_dir)
        assert token_file is not None
        token_file.parent.mkdir(parents=True, exist_ok=True)
        _write_token(token_file, creds.to_json())
    return creds


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(validate_providers=False)
    data_dir = config.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(prog="python -m src.auth")
    parser.add_argument("--manual", action="store_true", help="Print the auth URL instead of opening a browser")
    parser.add_argument("response_url", nargs="?", default=None, help="Redirect URL pasted from the manual flow")
    args = parser.parse_args()
    try:
        run_interactive_flow(
            data_dir,
            manual=args.manual,
            response_url=args.response_url,
        )
    except AuthError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc
    logger.info("Token saved to %s", _token_path(data_dir))


if __name__ == "__main__":
    main()
