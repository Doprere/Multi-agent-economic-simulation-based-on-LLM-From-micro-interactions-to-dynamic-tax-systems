"""
llm_client.py — 非同步 LLM API 客戶端，含 JSON schema 強制輸出與 retry 機制。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from typing import Any

import openai

from .config import LLMConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  JSON Schema 定義
# ──────────────────────────────────────────────────────────────

AGENT_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_action",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "基於當前視野與資源狀況的推理過程"
                },
                "action_id": {
                    "type": "integer",
                    "description": "必須為環境定義之 Action ID (0-49)"
                }
            },
            "required": ["thought", "action_id"],
            "additionalProperties": False
        }
    }
}

PLANNER_TAX_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "planner_tax_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "稅率設定的推理與社會考量"
                },
                "tax_brackets": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "各稅率區間的索引值（0-21），長度須符合環境稅階數"
                }
            },
            "required": ["thought", "tax_brackets"],
            "additionalProperties": False
        }
    }
}

PLANNER_OBSERVE_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "planner_observation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "社會規劃師的觀察與思考"
                },
                "society_comment": {
                    "type": "string",
                    "description": "對當前社會經濟狀況的評估與洞見"
                }
            },
            "required": ["thought", "society_comment"],
            "additionalProperties": False
        }
    }
}

CONSOLIDATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "memory_consolidation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "對過去決策思維的精簡總結（1-2句）"
                }
            },
            "required": ["summary"],
            "additionalProperties": False
        }
    }
}


# ──────────────────────────────────────────────────────────────
#  LLM 客戶端
# ──────────────────────────────────────────────────────────────

class LLMClient:
    """
    非同步 LLM API 客戶端。
    - 強制 JSON schema 輸出
    - 自動 retry（最多 max_retries 次）
    - Exponential backoff on rate limit / timeout
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY", "")
        base_url = cfg.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=cfg.timeout,
        )

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict,
        context: str = "",
    ) -> dict[str, Any]:
        """
        呼叫 LLM，回傳解析後的 JSON dict。
        自動重試最多 max_retries 次。

        Args:
            system_prompt: 系統角色設定
            user_prompt: 用戶訊息（當步觀察）
            response_format: JSON schema 格式定義
            context: 可選的額外上下文（短期記憶等）

        Returns:
            解析後的 JSON dict

        Raises:
            RuntimeError: 所有重試均失敗
        """
        messages = [{"role": "system", "content": system_prompt}]
        if context:
            messages.append({"role": "user", "content": context})
            messages.append({
                "role": "assistant",
                "content": "我已了解以上背景資訊，請告訴我當前狀態。"
            })
        messages.append({"role": "user", "content": user_prompt})

        last_error: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                response = await self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,  # type: ignore[arg-type]
                    response_format=response_format,  # type: ignore[arg-type]
                    temperature=self.cfg.temperature,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("LLM 回傳空回應")
                result = json.loads(content)
                return result

            except openai.RateLimitError as e:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"[LLM] Rate limit（第 {attempt+1} 次），等待 {wait:.1f}s... {e}")
                last_error = e
                await asyncio.sleep(wait)

            except openai.APITimeoutError as e:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"[LLM] Timeout（第 {attempt+1} 次），等待 {wait:.1f}s... {e}")
                last_error = e
                await asyncio.sleep(wait)

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"[LLM] JSON 解析失敗（第 {attempt+1} 次）：{e}")
                last_error = e
                # 不 sleep，立刻重試

            except openai.APIError as e:
                logger.error(f"[LLM] API 錯誤（第 {attempt+1} 次）：{e}")
                last_error = e
                await asyncio.sleep(1)

        raise RuntimeError(
            f"LLM 呼叫失敗（已重試 {self.cfg.max_retries} 次），最後錯誤：{last_error}"
        )

    async def call_agent(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str = "",
    ) -> dict[str, Any]:
        """呼叫 MobileAgent 決策（強制 {thought, action_id}）。"""
        return await self.call(system_prompt, user_prompt, AGENT_JSON_SCHEMA, context)

    async def call_planner_observe(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str = "",
    ) -> dict[str, Any]:
        """呼叫 Planner 觀察（強制 {thought, society_comment}，非稅收日）。"""
        return await self.call(system_prompt, user_prompt, PLANNER_OBSERVE_JSON_SCHEMA, context)

    async def call_planner_tax(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str = "",
    ) -> dict[str, Any]:
        """呼叫 Planner 稅率決策（強制 {thought, tax_brackets}，稅收日）。"""
        return await self.call(system_prompt, user_prompt, PLANNER_TAX_JSON_SCHEMA, context)

    async def call_consolidation(
        self,
        prompt: str,
    ) -> str:
        """呼叫長期記憶彙整（回傳 summary 字串）。"""
        result = await self.call(
            system_prompt="你是一個記憶管理助手，請精確總結所提供的過去決策思維。",
            user_prompt=prompt,
            response_format=CONSOLIDATION_SCHEMA,
        )
        return result.get("summary", "")


# ──────────────────────────────────────────────────────────────
#  CLI 測試
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from llm_agent.config import load_config

    async def _test():
        cfg = load_config()
        client = LLMClient(cfg.llm)

        print("=== 測試 Agent 決策 ===")
        result = await client.call_agent(
            system_prompt="你是一個 AI Economist 的 Agent，請根據狀態選擇動作。",
            user_prompt=(
                "當前狀態：Coin=10, Wood=2, Stone=0，位於 (5,5)。\n"
                "合法動作：[0] NOOP、[1] Build（木石不足，被 mask）、[46] 移動 Left\n"
                "請選擇 action_id。"
            ),
        )
        print(f"結果：{result}")
        assert "thought" in result
        assert "action_id" in result
        print("[PASS] Agent 決策測試通過")

        print("\n=== 測試 Retry（模擬）===")
        print("（若 API key 無效，retry 機制應在 3 次後 raise RuntimeError）")

    asyncio.run(_test())
