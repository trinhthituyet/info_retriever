"""Auth tests. No appleconnect binary required — the subprocess call is stubbed."""

from __future__ import annotations

import base64
import json
import subprocess
import time

import pytest

SECRET = "eyJhbGciOiJub25lIn0.super-secret-token-value.sig"


def _jwt(exp: float | None) -> str:
    """A token whose payload is real base64url JSON, so the exp parser sees it."""
    claims = {"sub": "abc"} if exp is None else {"sub": "abc", "exp": exp}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"header.{payload}.signature"


@pytest.fixture
def fresh_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "appleconnect")

    from info_retriever import auth, extract
    from info_retriever.config import settings

    settings.cache_clear()
    auth.invalidate()
    extract.reset_client()
    # Pretend the binary exists; individual tests decide what running it does.
    monkeypatch.setattr(auth.shutil, "which", lambda path: path)
    yield auth
    auth.invalidate()
    extract.reset_client()
    settings.cache_clear()


def _stub_run(monkeypatch, auth, *, stdout="", stderr="", returncode=0, calls=None):
    def fake_run(command, **kwargs):
        if calls is not None:
            calls.append(command)
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr(auth.subprocess, "run", fake_run)


# --------------------------------------------------------------------------- #
# expiry parsing
# --------------------------------------------------------------------------- #


def test_jwt_expiry_is_read_from_the_token(fresh_auth):
    deadline = time.time() + 3600
    assert fresh_auth._decode_jwt_expiry(_jwt(deadline)) == pytest.approx(deadline)


@pytest.mark.parametrize(
    "token",
    ["not-a-jwt", "only.two", "a.b.c", "a.!!!not-base64!!!.c", _jwt(None)],
    ids=["opaque", "two-segments", "non-json-payload", "bad-base64", "no-exp-claim"],
)
def test_unreadable_expiry_returns_none(fresh_auth, token):
    assert fresh_auth._decode_jwt_expiry(token) is None


# --------------------------------------------------------------------------- #
# minting and caching
# --------------------------------------------------------------------------- #


def test_token_is_the_last_whitespace_separated_field(fresh_auth, monkeypatch):
    # appleconnect prints preamble lines before the token.
    _stub_run(monkeypatch, fresh_auth, stdout=f"Fetching token...\n{_jwt(time.time() + 3600)}\n")
    assert fresh_auth.auth_token().startswith("header.")


def test_token_is_cached_across_calls(fresh_auth, monkeypatch):
    calls: list[list[str]] = []
    _stub_run(monkeypatch, fresh_auth, stdout=_jwt(time.time() + 3600), calls=calls)

    first = fresh_auth.auth_token()
    second = fresh_auth.auth_token()
    assert first == second
    assert len(calls) == 1, "second call should have hit the cache"


def test_command_carries_the_configured_appleconnect_arguments(fresh_auth, monkeypatch):
    calls: list[list[str]] = []
    _stub_run(monkeypatch, fresh_auth, stdout=_jwt(time.time() + 3600), calls=calls)
    fresh_auth.auth_token()

    command = calls[0]
    assert command[1] == "getToken"
    assert "--token-type=oauth" in command
    assert "--interactivity-type=none" in command
    assert "pkce" in command
    assert "openid,dsid,accountname,profile,groups" in command


def test_a_token_expiring_inside_the_margin_is_reminted_next_call(fresh_auth, monkeypatch):
    """A token valid for less than the safety margin must not be cached."""
    calls: list[list[str]] = []
    soon = time.time() + 60  # margin is 300s
    _stub_run(monkeypatch, fresh_auth, stdout=_jwt(soon), calls=calls)

    fresh_auth.auth_token()
    fresh_auth.auth_token()
    assert len(calls) == 2, "near-expired token should not have been reused"


def test_opaque_token_falls_back_to_the_configured_ttl(fresh_auth, monkeypatch):
    monkeypatch.setenv("APPLECONNECT_TOKEN_TTL_SECONDS", "900")
    from info_retriever.config import settings

    settings.cache_clear()

    calls: list[list[str]] = []
    _stub_run(monkeypatch, fresh_auth, stdout="opaque-token-no-jwt", calls=calls)

    assert fresh_auth.auth_token() == "opaque-token-no-jwt"
    assert fresh_auth.auth_token() == "opaque-token-no-jwt"
    assert len(calls) == 1
    assert fresh_auth.describe()["token_refresh_in_seconds"] == pytest.approx(900, abs=5)


def test_force_refresh_and_invalidate_both_remint(fresh_auth, monkeypatch):
    calls: list[list[str]] = []
    _stub_run(monkeypatch, fresh_auth, stdout=_jwt(time.time() + 3600), calls=calls)

    fresh_auth.auth_token()
    fresh_auth.auth_token(force_refresh=True)
    assert len(calls) == 2

    fresh_auth.invalidate()
    assert fresh_auth.describe()["token_cached"] is False
    fresh_auth.auth_token()
    assert len(calls) == 3


# --------------------------------------------------------------------------- #
# failures
# --------------------------------------------------------------------------- #


def test_missing_binary_names_the_path_and_the_way_out(fresh_auth, monkeypatch):
    monkeypatch.setattr(fresh_auth.shutil, "which", lambda path: None)
    with pytest.raises(fresh_auth.AuthError) as excinfo:
        fresh_auth.auth_token()
    message = str(excinfo.value)
    assert "/usr/local/bin/appleconnect" in message
    assert "ANTHROPIC_AUTH_MODE=default" in message


def test_nonzero_exit_suggests_re_authenticating(fresh_auth, monkeypatch):
    _stub_run(monkeypatch, fresh_auth, returncode=1, stderr="no active session\n")
    with pytest.raises(fresh_auth.AuthError) as excinfo:
        fresh_auth.auth_token()
    assert "no active session" in str(excinfo.value)
    assert "login" in str(excinfo.value)


def test_empty_stdout_is_an_auth_error_not_an_index_error(fresh_auth, monkeypatch):
    """The `.split()[-1]` shape this replaces raised IndexError on empty output."""
    _stub_run(monkeypatch, fresh_auth, stdout="   \n")
    with pytest.raises(fresh_auth.AuthError, match="no token"):
        fresh_auth.auth_token()


def test_timeout_is_reported_as_an_auth_error(fresh_auth, monkeypatch):
    def raise_timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 30)

    monkeypatch.setattr(fresh_auth.subprocess, "run", raise_timeout)
    with pytest.raises(fresh_auth.AuthError, match="timed out"):
        fresh_auth.auth_token()


def test_a_failure_message_never_contains_the_token(fresh_auth, monkeypatch):
    """appleconnect may echo the token on stdout even when it exits non-zero; the
    message we build (and log, and return over HTTP) must not carry it."""
    _stub_run(monkeypatch, fresh_auth, returncode=2, stdout=SECRET, stderr="partial failure")
    with pytest.raises(fresh_auth.AuthError) as excinfo:
        fresh_auth.auth_token()
    assert SECRET not in str(excinfo.value)
    assert "super-secret-token-value" not in str(excinfo.value)


def test_describe_never_exposes_the_token(fresh_auth, monkeypatch):
    _stub_run(monkeypatch, fresh_auth, stdout=SECRET)
    fresh_auth.auth_token()
    described = json.dumps(fresh_auth.describe())
    assert SECRET not in described
    assert "super-secret-token-value" not in described
    assert fresh_auth.describe()["token_cached"] is True


# --------------------------------------------------------------------------- #
# client construction
# --------------------------------------------------------------------------- #


def test_client_uses_floodgate_and_the_minted_token(fresh_auth, monkeypatch):
    from info_retriever import extract
    from info_retriever.config import FLOODGATE_BASE_URL

    _stub_run(monkeypatch, fresh_auth, stdout=_jwt(time.time() + 3600))
    client = extract.client()

    assert str(client.base_url).rstrip("/") == FLOODGATE_BASE_URL
    assert client.auth_token.startswith("header.")
    assert client.api_key is None


def test_ambient_anthropic_base_url_cannot_redirect_the_apple_token(fresh_auth, monkeypatch):
    """ANTHROPIC_BASE_URL is commonly set to a local proxy or mock. In appleconnect
    mode it must be ignored, or an ambient variable would redirect an Apple identity
    token to an unintended host. Override requires FLOODGATE_BASE_URL."""
    from info_retriever import extract
    from info_retriever.config import FLOODGATE_BASE_URL, settings

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:7079")
    settings.cache_clear()
    extract.reset_client()
    _stub_run(monkeypatch, fresh_auth, stdout=_jwt(time.time() + 3600))

    assert str(extract.client().base_url).rstrip("/") == FLOODGATE_BASE_URL

    # The dedicated variable is honoured.
    monkeypatch.setenv("FLOODGATE_BASE_URL", "https://gateway.example.internal/anthropic")
    settings.cache_clear()
    extract.reset_client()
    assert "gateway.example.internal" in str(extract.client().base_url)


def test_client_is_rebuilt_when_the_token_rotates(fresh_auth, monkeypatch):
    from info_retriever import extract

    _stub_run(monkeypatch, fresh_auth, stdout=_jwt(time.time() + 3600))
    first = extract.client()
    assert extract.client() is first, "same token should reuse the client"

    # New token on the next mint: the client must be rebuilt, not reused.
    extract.reset_client()
    _stub_run(monkeypatch, fresh_auth, stdout="second-token")
    second = extract.client()
    assert second is not first
    assert second.auth_token == "second-token"


def test_default_mode_bypasses_appleconnect_entirely(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "default")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    from info_retriever import auth, extract
    from info_retriever.config import settings

    settings.cache_clear()
    extract.reset_client()

    def explode(*args, **kwargs):
        raise AssertionError("default mode must not invoke appleconnect")

    monkeypatch.setattr(auth.subprocess, "run", explode)

    client = extract.client()
    assert client.api_key == "sk-test"
    assert "floodgate" not in str(client.base_url)

    extract.reset_client()
    settings.cache_clear()


def test_invalid_auth_mode_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "sso")

    from info_retriever.config import settings

    settings.cache_clear()
    with pytest.raises(ValueError, match="ANTHROPIC_AUTH_MODE"):
        settings()
    settings.cache_clear()
