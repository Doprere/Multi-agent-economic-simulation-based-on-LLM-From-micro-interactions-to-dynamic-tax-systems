"""
ollama_client.py — 非同步 Ollama HTTP 客戶端。

與 LLMClient 相同的公開方法簽名（duck typing），
讓 planner.py / agent.py 無需修改即可切換後端。

支援模型：llama3:8b（及任何 Ollama 上的模型）
JSON 萃取：prompt engineering + regex（不依賴 response_format）
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  JSON 萃取工具
# ──────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict[str, Any]:
    """
    從 LLM 輸出文字中萃取第一個完整 JSON 物件。
    策略：
      1. 嘗試直接 json.loads
      2. 用 regex 找 {...} 最外層 block（包含巢狀大括號）
      3. 都失敗 → raise ValueError
    """
    text = text.strip()

    # 策略1：直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 策略2：找最外層 {...}（貪婪，支援巢狀）
    # 從第一個 { 開始，追蹤大括號深度
    start = text.find("{")
    if start == -1:
        raise ValueError(f"找不到 JSON 物件，原始輸出：{text[:200]}")

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"JSON 解析失敗：{e}\n原始片段：{candidate[:300]}"
                    ) from e

    raise ValueError(f"未找到完整 JSON 物件，原始輸出：{text[:200]}")


# ──────────────────────────────────────────────────────────────
#  Prompt Templates（引導本地模型輸出 JSON）
# ──────────────────────────────────────────────────────────────

_AGENT_JSON_HINT = (
    "\n\nYou MUST respond with ONLY a valid JSON object, no markdown, no explanation:\n"
    '{"thought": "<your reasoning>", "action_id": <integer 0-49>}'
)

_OBSERVE_JSON_HINT = (
    "\n\nYou MUST respond with ONLY a valid JSON object, no markdown, no explanation:\n"
    '{"thought": "<your reasoning and analysis>", "society_comment": "<economic assessment>"}'
)

_TAX_JSON_HINT = (
    "\n\nYou MUST respond with ONLY a valid JSON object, no markdown, no explanation:\n"
    '{"thought": "<your reasoning>", "tax_brackets": [<list of integers 0-21>]}'
)

_CONSOLIDATION_JSON_HINT = (
    "\n\nYou MUST respond with ONLY a valid JSON object, no markdown, no explanation:\n"
    '{"summary": "<1-2 sentence summary>"}'
)


# ──────────────────────────────────────────────────────────────
#  OllamaClient
# ──────────────────────────────────────────────────────────────

class OllamaClient:
    """
    非同步 Ollama 客戶端。
    公開方法與 LLMClient 相同（duck typing 相容）。

    Args:
        model: Ollama 模型名稱（e.g. "llama3:8b"）
        base_url: Ollama API 服務位址（e.g. "http://localhost:11434"）
        max_retries: 最大重試次數
        temperature: 溫度（0.0~1.0）
        timeout: HTTP 超時秒數
    """

    def __init__(
        self,
        model: str = "llama3:8b",
        base_url: str = "http://localhost:11434",
        max_retries: int = 3,
        temperature: float = 0.7,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.temperature = temperature
        self._client = httpx.AsyncClient(timeout=timeout)

    # ── 核心 HTTP 呼叫 ────────────────────────────────────────

    async def _generate(self, prompt: str) -> str:
        """
        呼叫 Ollama /api/generate 端點，回傳完整回應文字。
        使用 stream=False 取得單次完整回應。
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 512,
            },
        }
        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        json_hint: str,
        context: str = "",
    ) -> dict[str, Any]:
        """
        呼叫 Ollama 並以 regex 萃取 JSON，最多重試 max_retries 次。

        Args:
            system_prompt: 角色設定
            user_prompt: 當步觀察
            json_hint: 引導輸出格式的 prompt 後綴
            context: 可選背景（短期記憶）

        Returns:
            解析後的 JSON dict

        Raises:
            RuntimeError: 所有重試均失敗
        """
        # 組裝完整 prompt（Ollama /api/generate 是單字串輸入）
        parts = [f"[SYSTEM]\n{system_prompt}"]
        if context:
            parts.append(f"[BACKGROUND MEMORY]\n{context}")
        parts.append(f"[USER]\n{user_prompt}{json_hint}")
        full_prompt = "\n\n".join(parts)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                raw = await self._generate(full_prompt)
                result = _extract_json(raw)
                return result

            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"[Ollama] HTTP 錯誤（第 {attempt+1} 次）：{e.response.status_code} – {e}"
                )
                last_error = e
                await asyncio.sleep(2 ** attempt)

            except httpx.RequestError as e:
                logger.warning(
                    f"[Ollama] 連線錯誤（第 {attempt+1} 次）：{e}. "
                    "請確認 Ollama 服務已啟動（podman run -p 11434:11434 ollama/ollama）"
                )
                last_error = e
                await asyncio.sleep(2 ** attempt)

            except (ValueError, KeyError) as e:
                logger.warning(f"[Ollama] JSON 萃取失敗（第 {attempt+1} 次）：{e}")
                last_error = e
                # 不 sleep，直接重試（可能模型輸出格式略有不同）

        raise RuntimeError(
            f"Ollama 呼叫失敗（已重試 {self.max_retries} 次），最後錯誤：{last_error}"
        )

    # ── 與 LLMClient 相同的公開介面（duck typing）─────────────

    async def call_agent(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str = "",
    ) -> dict[str, Any]:
        """MobileAgent 決策：回傳 {thought, action_id}"""
        return await self.call(
            system_prompt, user_prompt, _AGENT_JSON_HINT, context
        )

    async def call_planner_observe(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Planner 非稅收日觀察：回傳 {thought, society_comment}"""
        return await self.call(
            system_prompt, user_prompt, _OBSERVE_JSON_HINT, context
        )

    async def call_planner_tax(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Planner 稅收日決策：回傳 {thought, tax_brackets}"""
        return await self.call(
            system_prompt, user_prompt, _TAX_JSON_HINT, context
        )

    async def call_consolidation(self, prompt: str) -> str:
        """長期記憶彙整：回傳 summary 字串"""
        result = await self.call(
            system_prompt=(
                "You are a memory management assistant. "
                "Summarize the provided decision history concisely."
            ),
            user_prompt=prompt,
            json_hint=_CONSOLIDATION_JSON_HINT,
        )
        return result.get("summary", "")

    async def aclose(self) -> None:
        """關閉 HTTP session"""
        await self._client.aclose()
