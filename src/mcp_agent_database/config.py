"""Central configuration.

Everything is read from environment variables prefixed with ``FORGE_`` (or a ``.env`` file),
so the same code runs locally, in Docker and in CI without edits.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Provider = Literal["ollama", "openai", "anthropic"]
ApprovalMode = Literal["ask", "auto_approve", "auto_deny"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FORGE_", env_file=".env", extra="ignore")

    # --- LLM -------------------------------------------------------------------------------
    llm_provider: Provider = "ollama"
    model: str = "qwen3:8b"
    temperature: float = 0.0
    llm_timeout_s: float = 180.0
    ollama_base_url: str = "http://localhost:11434"
    ollama_think: bool | None = Field(
        default=False, description="Enable 'thinking' for reasoning models (qwen3, deepseek-r1)."
    )
    # Any OpenAI-compatible endpoint works: OpenAI, vLLM, LM Studio, Groq, OpenRouter, Ollama /v1.
    openai_base_url: str | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    max_tokens: int = 2048

    # --- Agent -----------------------------------------------------------------------------
    servers_config: Path = Path("mcp_servers.json")
    max_steps: int = 10
    approval_mode: ApprovalMode = "ask"
    approval_timeout_s: float = 300.0
    # Fully-qualified tool names (server__tool) that never need approval, e.g. "workspace__write_file".
    trusted_tools: Annotated[list[str], NoDecode] = Field(default_factory=list)
    max_tool_output_chars: int = 12_000
    audit_log: Path = Path("data/audit.jsonl")

    # --- MCP servers -----------------------------------------------------------------------
    db_path: Path = Path("data/shop.duckdb")
    sql_max_rows: int = 200
    sql_timeout_s: float = 10.0
    docs_dir: Path = Path("data/docs")
    embed_model: str | None = Field(
        default=None, description="Ollama embedding model for hybrid search, e.g. 'nomic-embed-text'."
    )
    sandbox_dir: Path = Path("data/sandbox")
    max_file_bytes: int = 200_000
    github_token: SecretStr | None = None
    github_api_url: str = "https://api.github.com"

    # --- API -------------------------------------------------------------------------------
    api_token: SecretStr | None = Field(default=None, description="If set, required as a Bearer token.")

    @field_validator("trusted_tools", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
