"""Runtime settings, read from environment variables (and .env if present)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    llm: str = _env("RECON_LLM", "mistral")
    model: str = _env("RECON_MODEL", "mistral-small-latest")
    api_key: str = _env("MISTRAL_API_KEY", "")
    max_steps: int = int(_env("RECON_MAX_STEPS", "8"))
    request_timeout_s: float = float(_env("RECON_REQUEST_TIMEOUT_S", "60"))
    llm_timeout_s: float = float(_env("RECON_LLM_TIMEOUT_S", "30"))
    price_in_per_m: float = float(_env("RECON_PRICE_IN_PER_M", "0.10"))
    price_out_per_m: float = float(_env("RECON_PRICE_OUT_PER_M", "0.30"))
    mock_latency_ms: int = int(_env("RECON_MOCK_LATENCY_MS", "0"))

    data_dir: Path = ROOT / "data"
    trace_dir: Path = ROOT / "traces"
    eval_dir: Path = ROOT / "eval"

    @property
    def invoices_path(self) -> Path:
        return self.data_dir / "invoices.jsonl"

    @property
    def ledger_path(self) -> Path:
        return self.data_dir / "ledger.db"

    @property
    def questions_path(self) -> Path:
        return self.eval_dir / "questions.jsonl"

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.price_in_per_m + completion_tokens * self.price_out_per_m) / 1e6


settings = Settings()
