"""Application configuration loaded from environment and sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Runtime settings for CampusAsk."""

    project_root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    data_benchmark: Path = PROJECT_ROOT / "data" / "benchmark"
    indexes_dir: Path = PROJECT_ROOT / "indexes"
    results_dir: Path = PROJECT_ROOT / "results"
    sources_registry: Path = PROJECT_ROOT / "data" / "raw" / "sources.json"

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    top_k: int = int(os.getenv("TOP_K", "8"))
    # Tuned so catalog policy Qs (thesis/GPA) pass while off-topic stays out.
    evidence_min_score: float = float(os.getenv("EVIDENCE_MIN_SCORE", "0.38"))

    # "anthropic" → Claude via the Anthropic SDK; "openai" → any OpenAI-compatible
    # endpoint (Groq by default) via LLM_BASE_URL + LLM_API_KEY.
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    # Claude effort: low | medium | high | xhigh | max (Opus 5.5 default is medium).
    llm_effort: str = os.getenv("LLM_EFFORT", "medium").strip().lower()
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))

    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv(
        "LLM_BASE_URL", "https://api.groq.com/openai/v1"
    )
    llm_model: str = os.getenv(
        "LLM_MODEL",
        "claude-opus-5-5"
        if os.getenv("LLM_PROVIDER", "").strip().lower() == "anthropic"
        else "openai/gpt-oss-20b",
    )

    @property
    def llm_configured(self) -> bool:
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        return bool(self.llm_api_key)

    faiss_index_path: Path = PROJECT_ROOT / "indexes" / "campusask.faiss"
    metadata_path: Path = PROJECT_ROOT / "indexes" / "chunks.jsonl"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_raw,
            self.data_processed,
            self.data_benchmark,
            self.indexes_dir,
            self.results_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
