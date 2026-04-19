"""
client_factory.py — 依 LLMConfig.backend 建立對應的 LLM client。

支援的 backend：
  - "openai" → LLMClient（AsyncOpenAI wrapper，JSON schema 強制）
  - "ollama" → OllamaClient（httpx + regex JSON 提取）

兩者皆 duck-typing 相容以下介面：
  - call_agent(sys_prompt, user_prompt, context) → {thought, action_id}
  - call_planner_observe(...) → {thought, society_comment}
  - call_planner_tax(...) → {thought, tax_brackets}
  - call_consolidation(prompt) → str
  - get_token_usage() → dict
  - aclose()（僅 OllamaClient 有；OpenAI client 不需手動關）
"""
from __future__ import annotations

from .config import LLMConfig
from .llm_client import LLMClient
from .ollama_client import OllamaClient


def make_llm_client(cfg: LLMConfig):
    """根據 cfg.backend 建立對應 client。

    Args:
        cfg: LLMConfig，至少需含 backend 與 model。

    Returns:
        LLMClient 或 OllamaClient 實例。

    Raises:
        ValueError: 不支援的 backend 名稱。
    """
    backend = (cfg.backend or "openai").lower()

    if backend == "openai":
        return LLMClient(cfg)

    if backend == "ollama":
        return OllamaClient(
            model=cfg.model,
            base_url=cfg.base_url or "http://localhost:11434",
            max_retries=cfg.max_retries,
            temperature=cfg.temperature,
            timeout=cfg.timeout if cfg.timeout and cfg.timeout > 30 else 120,
        )

    raise ValueError(
        f"Unknown LLM backend: {backend!r} (expected 'openai' or 'ollama')"
    )
