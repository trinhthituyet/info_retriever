"""Anthropic client construction: credential, gateway and CA bundle."""

from __future__ import annotations

import json

import pytest

SECRET = "sk-ant-super-secret-key-value"


@pytest.fixture
def fresh_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    from info_retriever import extract
    from info_retriever.config import settings

    settings.cache_clear()
    extract.reset_client()
    yield extract
    extract.reset_client()
    settings.cache_clear()


def test_client_uses_the_api_key_and_the_public_api(fresh_client):
    client = fresh_client.client()
    assert client.api_key == SECRET
    assert "api.anthropic.com" in str(client.base_url)


def test_anthropic_base_url_is_honoured(fresh_client, monkeypatch):
    from info_retriever.config import settings

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.internal/anthropic")
    settings.cache_clear()
    fresh_client.reset_client()
    assert "gateway.example.internal" in str(fresh_client.client().base_url)


def test_client_is_reused_until_reset(fresh_client):
    first = fresh_client.client()
    assert fresh_client.client() is first
    fresh_client.reset_client()
    assert fresh_client.client() is not first


def test_ca_bundle_defaults_to_certifi(fresh_client):
    import certifi

    from info_retriever.config import settings

    assert settings().ca_bundle == certifi.where()


def test_ssl_cert_file_overrides_the_ca_bundle(fresh_client, monkeypatch):
    from info_retriever.config import settings

    monkeypatch.setenv("SSL_CERT_FILE", "/custom/cacert.pem")
    settings.cache_clear()
    assert settings().ca_bundle == "/custom/cacert.pem"


def test_describe_never_exposes_the_api_key(fresh_client):
    from info_retriever import llm

    fresh_client.client()
    described = json.dumps(llm.provider().describe())
    assert SECRET not in described
    assert "super-secret" not in described
