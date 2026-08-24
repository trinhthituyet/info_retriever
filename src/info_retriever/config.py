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

    auth_mode: str
    base_url: str | None
    appleconnect_path: str
    appleconnect_app_id: str
    appleconnect_env: str
    appleconnect_scopes: str
    appleconnect_timeout_seconds: float
    appleconnect_token_ttl_seconds: int

    extract_model: str
    agent_model: str
    agent_effort: str

    embed_model: str
    embed_dim: int
    embed_query_prefix: str
    embed_passage_prefix: str


FLOODGATE_BASE_URL = "https://floodgate.g.apple.com/api/anthropic"
APPLECONNECT_APP_ID = "hvys3fcwcteqrvw3qzkvtk86viuoqv"
APPLECONNECT_SCOPES = "openid,dsid,accountname,profile,groups"


@lru_cache(maxsize=1)
def settings() -> Settings:
    load_dotenv()

    data_dir = Path(os.getenv("DATA_DIR", "./data")).expanduser().resolve()
    blob_dir = data_dir / "blobs"
    data_dir.mkdir(parents=True, exist_ok=True)
    blob_dir.mkdir(parents=True, exist_ok=True)

    auth_mode = os.getenv("ANTHROPIC_AUTH_MODE", "appleconnect").strip().lower()
    if auth_mode not in ("appleconnect", "default"):
        raise ValueError(
            f"ANTHROPIC_AUTH_MODE must be 'appleconnect' or 'default', got {auth_mode!r}"
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
        auth_mode=auth_mode,
        base_url=base_url or None,
        appleconnect_path=os.getenv("APPLECONNECT_PATH", "/usr/local/bin/appleconnect"),
        appleconnect_app_id=os.getenv("APPLECONNECT_APP_ID", APPLECONNECT_APP_ID),
        appleconnect_env=os.getenv("APPLECONNECT_ENV", "prod"),
        appleconnect_scopes=os.getenv("APPLECONNECT_SCOPES", APPLECONNECT_SCOPES),
        appleconnect_timeout_seconds=float(os.getenv("APPLECONNECT_TIMEOUT_SECONDS", "30")),
        appleconnect_token_ttl_seconds=int(os.getenv("APPLECONNECT_TOKEN_TTL_SECONDS", "1800")),
        extract_model=os.getenv("EXTRACT_MODEL", "claude-opus-5"),
        agent_model=os.getenv("AGENT_MODEL", "claude-opus-5"),
        agent_effort=os.getenv("AGENT_EFFORT", "medium"),
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
        embed_dim=int(os.getenv("EMBED_DIM", "1024")),
        embed_query_prefix=os.getenv("EMBED_QUERY_PREFIX", ""),
        embed_passage_prefix=os.getenv("EMBED_PASSAGE_PREFIX", ""),
    )
