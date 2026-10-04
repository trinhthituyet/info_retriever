"""Runtime configuration, loaded from the environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    blob_dir: Path

    db_backend: str
    postgres_dsn: str | None

    llm_provider: str

    auth_mode: str
    base_url: str | None
    ca_bundle: str | None
    appleconnect_path: str
    appleconnect_app_id: str
    appleconnect_env: str
    appleconnect_scopes: str
    appleconnect_timeout_seconds: float
    appleconnect_token_ttl_seconds: int

    extract_model: str
    agent_model: str
    agent_effort: str

    vllm_base_url: str
    vllm_api_key: str
    vllm_model: str
    vllm_max_tokens: int
    vllm_temperature: float
    vllm_pdf_dpi: int
    vllm_timeout: float

    history_max_turns: int
    history_max_chars: int
    query_rewrite: bool
    agent_max_rounds: int
    cite_max_documents: int

    embed_model: str
    embed_dim: int
    embed_query_prefix: str
    embed_passage_prefix: str

    @property
    def is_anthropic(self) -> bool:
        return self.llm_provider == "anthropic"

    @property
    def active_model(self) -> str:
        """The model that will actually serve requests, whichever provider is on."""
        return self.agent_model if self.is_anthropic else self.vllm_model


FLOODGATE_BASE_URL = "https://floodgate.g.apple.com/api/anthropic"
APPLECONNECT_APP_ID = "hvys3fcwcteqrvw3qzkvtk86viuoqv"
APPLECONNECT_SCOPES = "openid,dsid,accountname,profile,groups"

LLM_PROVIDERS = ("anthropic", "vllm")
DB_BACKENDS = ("sqlite", "postgres")
AUTH_MODES = ("appleconnect", "default")


def _one_of(name: str, value: str, allowed: tuple[str, ...]) -> str:
    normalised = value.strip().lower()
    if normalised not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}, got {value.strip()!r}")
    return normalised


def _resolve_ca_bundle() -> str | None:
    """Path to a CA bundle that trusts Apple's internal roots, if available.

    Floodgate presents a certificate signed by an Apple internal CA, which the
    stock certifi bundle does not carry — hence ``apple-certifi``. An explicit
    ``SSL_CERT_FILE`` wins, so an unusual setup can still override.
    """
    override = os.getenv("SSL_CERT_FILE", "").strip()
    if override:
        return override
    try:
        import apple_certifi
    except ModuleNotFoundError:
        return None
    return apple_certifi.where()


@lru_cache(maxsize=1)
def settings() -> Settings:
    load_dotenv()

    data_dir = Path(os.getenv("DATA_DIR", "./data")).expanduser().resolve()
    blob_dir = data_dir / "blobs"
    data_dir.mkdir(parents=True, exist_ok=True)
    blob_dir.mkdir(parents=True, exist_ok=True)

    llm_provider = _one_of("LLM_PROVIDER", os.getenv("LLM_PROVIDER", "anthropic"), LLM_PROVIDERS)
    db_backend = _one_of("DB_BACKEND", os.getenv("DB_BACKEND", "sqlite"), DB_BACKENDS)

    # DATABASE_URL is the near-universal name, so accept it as a fallback.
    postgres_dsn = (os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL") or "").strip()
    if db_backend == "postgres" and not postgres_dsn:
        raise ValueError(
            "DB_BACKEND=postgres requires POSTGRES_DSN (or DATABASE_URL), e.g. "
            "postgresql://user:pass@localhost:5432/info_retriever"
        )
    auth_mode = _one_of(
        "ANTHROPIC_AUTH_MODE", os.getenv("ANTHROPIC_AUTH_MODE", "appleconnect"), AUTH_MODES
    )

    if auth_mode == "appleconnect":
        # Deliberately NOT ANTHROPIC_BASE_URL. That variable is commonly set in
        # shells and CI to point at a local proxy or mock, and honouring it here
        # would silently send an Apple identity token to that host. The gateway an
        # appleconnect token is minted for must be named explicitly.
        base_url = os.getenv("FLOODGATE_BASE_URL", FLOODGATE_BASE_URL).strip()
    else:
        base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()

    return Settings(
        data_dir=data_dir,
        db_path=data_dir / "documents.db",
        blob_dir=blob_dir,
        db_backend=db_backend,
        postgres_dsn=postgres_dsn or None,
        llm_provider=llm_provider,
        auth_mode=auth_mode,
        base_url=base_url or None,
        ca_bundle=_resolve_ca_bundle(),
        appleconnect_path=os.getenv("APPLECONNECT_PATH", "/usr/local/bin/appleconnect"),
        appleconnect_app_id=os.getenv("APPLECONNECT_APP_ID", APPLECONNECT_APP_ID),
        appleconnect_env=os.getenv("APPLECONNECT_ENV", "prod"),
        appleconnect_scopes=os.getenv("APPLECONNECT_SCOPES", APPLECONNECT_SCOPES),
        appleconnect_timeout_seconds=float(os.getenv("APPLECONNECT_TIMEOUT_SECONDS", "30")),
        appleconnect_token_ttl_seconds=int(os.getenv("APPLECONNECT_TOKEN_TTL_SECONDS", "1800")),
        extract_model=os.getenv("EXTRACT_MODEL", "claude-opus-5"),
        agent_model=os.getenv("AGENT_MODEL", "claude-opus-5"),
        agent_effort=os.getenv("AGENT_EFFORT", "medium"),
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/"),
        vllm_api_key=os.getenv("VLLM_API_KEY", "not-needed"),
        vllm_model=os.getenv("VLLM_MODEL", ""),
        vllm_max_tokens=int(os.getenv("VLLM_MAX_TOKENS", "4096")),
        vllm_temperature=float(os.getenv("VLLM_TEMPERATURE", "0.2")),
        vllm_pdf_dpi=int(os.getenv("VLLM_PDF_DPI", "150")),
        vllm_timeout=float(os.getenv("VLLM_TIMEOUT", "180")),
        history_max_turns=int(os.getenv("HISTORY_MAX_TURNS", "12")),
        history_max_chars=int(os.getenv("HISTORY_MAX_CHARS", "12000")),
        query_rewrite=os.getenv("QUERY_REWRITE", "1").strip().lower()
        not in ("0", "false", "no", "off"),
        agent_max_rounds=max(1, int(os.getenv("AGENT_MAX_ROUNDS", "3"))),
        cite_max_documents=max(1, int(os.getenv("CITE_MAX_DOCUMENTS", "8"))),
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
        embed_dim=int(os.getenv("EMBED_DIM", "1024")),
        embed_query_prefix=os.getenv("EMBED_QUERY_PREFIX", ""),
        embed_passage_prefix=os.getenv("EMBED_PASSAGE_PREFIX", ""),
    )
