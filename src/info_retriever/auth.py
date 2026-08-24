"""Credential minting for the Claude API.

Two modes, selected by ``ANTHROPIC_AUTH_MODE``:

``appleconnect`` (default)
    Mint a short-lived OAuth token with the appleconnect CLI and reach Claude
    through Apple's Floodgate gateway.

``default``
    Let the SDK resolve credentials itself — ``ANTHROPIC_API_KEY``,
    ``ANTHROPIC_AUTH_TOKEN``, or an ``ant auth login`` profile. Used by the test
    suite, and for running against the public API.

The appleconnect token is short-lived, so it is cached with an expiry rather than
fetched once per process: this app runs as a server for hours, well past the life
of a single token. Where the token is a JWT its own ``exp`` claim drives the
refresh; otherwise a conservative fixed TTL applies.

The token is a bearer credential for the caller's Apple identity. It is never
logged, printed, or included in an exception message.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import threading
import time

from .config import settings

#: Refresh this many seconds before the token actually expires, so a request that
#: starts just under the wire does not finish just over it.
_EXPIRY_MARGIN_SECONDS = 300.0

_lock = threading.Lock()
_cached_token: str | None = None
#: Wall-clock (``time.time()``) basis, deliberately not ``time.monotonic()``:
#: CLOCK_MONOTONIC does not advance while macOS is asleep, so a laptop that slept
#: for hours would consider a long-dead token still fresh.
_expires_at: float = 0.0


class AuthError(RuntimeError):
    """Could not mint a credential."""


def _decode_jwt_expiry(token: str) -> float | None:
    """Return the ``exp`` claim as a POSIX timestamp, or None if unreadable.

    The signature is not verified and must not be — this is a local hint used only
    to schedule a refresh. The gateway is what actually validates the token.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)  # restore base64url padding
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except Exception:  # noqa: BLE001 - any malformed segment just means "no hint"
        return None
    expiry = claims.get("exp")
    return float(expiry) if isinstance(expiry, (int, float)) else None


def _mint() -> tuple[str, float]:
    """Run appleconnect and return ``(token, expires_at_wall_clock)``."""
    cfg = settings()

    if shutil.which(cfg.appleconnect_path) is None:
        raise AuthError(
            f"appleconnect not found at {cfg.appleconnect_path}. Install AppleConnect, "
            "or set ANTHROPIC_AUTH_MODE=default to use an API key instead."
        )

    command = [
        cfg.appleconnect_path,
        "getToken",
        "-C",
        cfg.appleconnect_app_id,
        "--token-type=oauth",
        # Non-interactive: fail fast rather than block a web request on a GUI prompt.
        "--interactivity-type=none",
        "-E",
        cfg.appleconnect_env,
        "-G",
        "pkce",
        "-o",
        cfg.appleconnect_scopes,
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=cfg.appleconnect_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AuthError(
            f"appleconnect timed out after {cfg.appleconnect_timeout_seconds}s."
        ) from exc
    except OSError as exc:
        raise AuthError(f"Could not run appleconnect: {exc}") from exc

    if completed.returncode != 0:
        # stderr only: stdout carries the token on success and may echo it on
        # partial failure, and this message may reach a log or an HTTP response.
        detail = (completed.stderr or "").strip().splitlines()
        hint = detail[-1] if detail else f"exit status {completed.returncode}"
        raise AuthError(
            f"appleconnect could not mint a token ({hint}). "
            "Your AppleConnect session may have expired — run "
            f"`{cfg.appleconnect_path} login` and retry."
        )

    fields = (completed.stdout or "").split()
    if not fields:
        raise AuthError("appleconnect returned no token.")
    token = fields[-1]

    expiry = _decode_jwt_expiry(token)
    if expiry is not None:
        # Trust the token's own expiry, minus a safety margin. If it is already
        # inside the margin, use it once and re-mint on the next call.
        return token, max(expiry - _EXPIRY_MARGIN_SECONDS, time.time())

    return token, time.time() + float(cfg.appleconnect_token_ttl_seconds)


def auth_token(*, force_refresh: bool = False) -> str:
    """Return a valid appleconnect OAuth token, minting or refreshing as needed."""
    global _cached_token, _expires_at

    with _lock:
        if not force_refresh and _cached_token and time.time() < _expires_at:
            return _cached_token
        _cached_token, _expires_at = _mint()
        return _cached_token


def invalidate() -> None:
    """Drop the cached token so the next call mints a fresh one.

    Call this after an authentication failure — the gateway rejected a token we
    still believed was valid.
    """
    global _cached_token, _expires_at

    with _lock:
        _cached_token = None
        _expires_at = 0.0


def describe() -> dict[str, object]:
    """Non-secret auth state, safe to expose over HTTP."""
    cfg = settings()
    with _lock:
        cached = _cached_token is not None
        remaining = max(0.0, _expires_at - time.time()) if cached else 0.0
    return {
        "auth_mode": cfg.auth_mode,
        "base_url": cfg.base_url,
        "token_cached": cached,
        "token_refresh_in_seconds": round(remaining),
    }
