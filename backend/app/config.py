from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py -> app -> backend -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Provider selection ───────────────────────────────────────
    default_provider: str = "ollama"

    # ── Ollama ───────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen2.5:7b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout: float = 180.0

    # ── Optional cloud providers ─────────────────────────────────
    # A provider is registered only when its key is present. No keys set
    # means fully local, no network.

    gemini_api_key: str | None = None
    # Google retires model ids often; override if this one 404s.
    gemini_model: str = "gemini-3.7-flash"
    # Google searches server-side. The only route to the live web here.
    gemini_enable_search: bool = False

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5"
    # Retry a safety-declined request on another model. Beta header; set
    # false if your key rejects it.
    anthropic_enable_fallbacks: bool = True

    # OPENAI_BASE_URL reuses this provider for Groq/Together/OpenRouter/vLLM.
    openai_api_key: str | None = None
    openai_model: str = "gpt-5"
    openai_base_url: str | None = None
    # Newer models require max_completion_tokens; older ones reject it.
    openai_max_tokens_param: str = "max_completion_tokens"

    #: Shared request timeout for cloud providers (seconds).
    cloud_timeout: float = 120.0

    # ── RAG tuning ───────────────────────────────────────────────
    # Bounds are guardrails: under ~100 chars a chunk carries too little
    # context; over ~4000 several of them exceed a 7B context window.
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    rag_top_k: int = Field(default=5, ge=1, le=20)

    # Chroma enforces this too, but only at collection-creation time and
    # with an opaque traceback. Validating here fails readably at startup.
    chroma_collection: str = Field(
        default="documents",
        min_length=3,
        max_length=512,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$",
    )

    # ── Analyst tuning ───────────────────────────────────────────
    # None = in-memory DuckDB, dying with the process. A file path keeps
    # loaded tables between runs.
    analyst_db_path: str | None = None

    # Rows shown TO THE MODEL. A context budget, not a display limit -
    # every row costs tokens on every later turn of the loop.
    analyst_max_rows: int = Field(default=50, ge=1, le=500)

    # Model calls before the loop gives up. ~10s per call on a local 7B.
    analyst_max_turns: int = Field(default=5, ge=1, le=20)

    # Seconds one query may run, so a runaway join cannot hang the client.
    analyst_query_timeout: float = Field(default=10.0, gt=0)

    # Charts and exports. None = <data_dir>/processed. Must be ABSOLUTE:
    # a relative path resolves against the working directory, so output
    # would follow wherever the process was launched from.
    analyst_output_dir: Path | None = None

    # Write the full result to .xlsx when the model only saw part of it.
    analyst_autosave_results: bool = True

    # jpg blurs text and thin lines; png is the right default for charts.
    analyst_chart_format: str = Field(default="png", pattern=r"^(png|jpg|svg|pdf)$")

    # 150 is screen-sharp; 200+ for documents and slides.
    analyst_chart_dpi: int = Field(default=150, ge=50, le=600)

    # ── Paths ────────────────────────────────────────────────────
    data_dir: Path = PROJECT_ROOT / "data"

    # ── API server ───────────────────────────────────────────────
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"

    # ── Derived paths ────────────────────────────────────────────
    @property
    def raw_dir(self) -> Path:
        """Where uploaded source documents are kept."""
        return self.data_dir / "raw"

    @property
    def vectorstore_dir(self) -> Path:
        """Chroma's on-disk persistence directory."""
        return self.data_dir / "vectorstore"

    @property
    def processed_dir(self) -> Path:
        """Charts and exports. Absolute, so output does not follow the cwd."""
        return self.analyst_output_dir or (self.data_dir / "processed")

    @property
    def sqlite_url(self) -> str:
        """SQLAlchemy URL for the metadata / chat-history database."""
        return f"sqlite:///{(self.data_dir / 'app.db').as_posix()}"

    def ensure_dirs(self) -> None:
        """Create the directories we write into. Safe to call repeatedly."""
        for path in (self.data_dir, self.raw_dir, self.vectorstore_dir, self.processed_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Settings singleton.

    Cached because .env should be read once per process, and because
    FastAPI's Depends(get_settings) would otherwise re-parse the file on
    every single request.
    """
    return Settings()
