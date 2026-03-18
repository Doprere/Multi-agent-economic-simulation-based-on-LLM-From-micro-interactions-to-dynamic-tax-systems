"""
agent.py — MobileAgent LLM 決策層。
具備短期記憶（滑動視窗）+ 長期記憶（異步彙整）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from .action_map import (
    get_masked_description,
    get_random_valid_action,
    is_valid_action,
)
from .config import AppConfig, PersonaConfig
from .llm_client import LLMClient
from .memory import AgentMemory
from .translator import ObsTranslator

logger = logging.getLogger(__name__)

# avoid circular import: logger imported at runtime inside methods
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .logger import SimulationLogger

# ──────────────────────────────────────────────────────────────
#  Few-shot 範例
# ──────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
Here are a few response examples (action_id MUST be from the valid action list):

Example 1 (gather):
{{"thought": "Wood is 2 tiles to the East, my Wood=0. I need it to build. Moving right to collect.", "action_id": 47}}

Example 2 (build):
{{"thought": "I have Wood=2, Stone=1. Current tile has no landmark. Building now earns 15 Coin.", "action_id": 1}}

Example 3 (sell on market):
{{"thought": "I have Stone=3 which exceeds build needs. highest Stone bid is 7 Coin. Selling at ask 6 for profit.", "action_id": 40}}

Example 4 (NOOP):
{{"thought": "No nearby resources, cannot build, open orders at limit. Waiting.", "action_id": 0}}
"""


# ──────────────────────────────────────────────────────────────
#  MobileAgent LLM
# ──────────────────────────────────────────────────────────────

class MobileAgentLLM:
    """
    單一 MobileAgent 的 LLM 決策邏輯。
    每步決策：翻譯觀察 → 組裝 prompt → 呼叫 LLM → 驗證 → 更新記憶。
    長期記憶彙整：非阻塞異步背景任務。
    """

    def __init__(
        self,
        persona: PersonaConfig,
        llm_client: LLMClient,
        mem_cfg: Any,
        sim_logger: "SimulationLogger | None" = None,
    ) -> None:
        self.persona = persona
        self.client = llm_client
        self.sim_logger = sim_logger
        self.memory = AgentMemory(
            agent_id=persona.id,
            short_term_window=mem_cfg.agent_short_term_window,
            long_term_trigger=mem_cfg.agent_long_term_trigger,
        )
        self.translator = ObsTranslator()
        self._consolidation_task: asyncio.Task | None = None

    # ── System Prompt ─────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        p = self.persona
        base = (
            f"You are an Agent in the AI Economist simulation, representing '{p.display_name}'.\n"
            f"Your role: {p.role.strip()}\n\n"
            "Your goal is to maximize your own utility (diminishing marginal utility of Coin minus labor cost).\n\n"
            "Decision rules:\n"
            "1. Choose exactly ONE action_id per step. It MUST appear in the valid action list.\n"
            "2. Actions not in the valid list are masked by the environment and will have no effect.\n"
            "3. Output MUST be strict JSON: {\"thought\": \"...\", \"action_id\": <integer>}\n\n"
            f"{FEW_SHOT_EXAMPLES}"
        )
        if self.memory.has_long_term:
            base += f"\n[Long-term Memory]\n{self.memory.long_term}\n"
        return base

    # ── 決策 ──────────────────────────────────────────────────

    async def decide(
        self,
        obs: dict,
        mask: np.ndarray,
        env,
        step: int,
    ) -> int:
        """
        執行一步決策。

        Returns:
            合法的 action_id（int）
        """
        # 1. 翻譯觀察
        state_desc = self.translator.translate_agent_obs(
            obs=obs,
            agent_id=self.persona.id,
            agent_name=self.persona.display_name,
            agent_role=self.persona.role,
            env=env,
            step=step,
        )

        # 2. 組裝context（短期記憶）
        short_ctx = self.memory.get_short_term_block()

        # 3. 呼叫 LLM（含 retry）
        thought, action_id = await self._call_with_validation(
            state_desc=state_desc,
            short_ctx=short_ctx,
            mask=mask,
        )

        # 4. 更新記憶
        self.memory.add_thought(thought)

        # 5. 記錄到 sim_logger（Excel）
        if self.sim_logger is not None:
            self.sim_logger.log_thought(
                step=step,
                agent_id=str(self.persona.id),
                agent_name=self.persona.display_name,
                role=self.persona.role,
                thought=thought,
                action_id=action_id,
                short_term_memory=self.memory.get_short_term_context(),
                long_term_memory=self.memory.long_term,
            )

        # 6. 觸發長期記憶彙整（非阻塞）
        if self.memory.should_consolidate():
            self._trigger_consolidation()

        logger.info(
            f"[Agent {self.persona.id}] step={step} action={action_id} "
            f"thought={thought[:60]}..."
        )
        return action_id

    async def _call_with_validation(
        self,
        state_desc: str,
        short_ctx: str,
        mask: np.ndarray,
    ) -> tuple[str, int]:
        """
        呼叫 LLM 並驗證 action_id 是否合法。
        若輸出非法，在 retry_prompt 中加入錯誤提示重試。
        """
        from .config import get_config
        max_retries = get_config().llm.max_retries

        retry_hint = ""
        for attempt in range(max_retries):
            prompt = state_desc
            if retry_hint:
                prompt += f"\n\n⚠️ 上次回應無效：{retry_hint}\n請重新選擇合法的 action_id。"

            try:
                result = await self.client.call_agent(
                    system_prompt=self._build_system_prompt(),
                    user_prompt=prompt,
                    context=short_ctx if attempt == 0 else "",
                )
                thought = str(result.get("thought", ""))
                action_id = int(result.get("action_id", -1))

                if is_valid_action(action_id, mask):
                    return thought, action_id
                else:
                    valid_ids = [i for i, m in enumerate(mask) if m == 1]
                    retry_hint = (
                        f"action_id={action_id} is NOT in the valid action list. "
                        f"Valid action_ids: {valid_ids[:10]}..."
                    )
                    logger.warning(
                        f"[Agent {self.persona.id}] illegal action_id={action_id}, "
                        f"retry {attempt+1}/{max_retries}"
                    )

            except Exception as e:
                retry_hint = f"JSON parse error: {e}"
                logger.warning(
                    f"[Agent {self.persona.id}] call error attempt {attempt+1}: {e}"
                )

        # Fallback
        fallback = get_random_valid_action(mask)
        logger.error(
            f"[Agent {self.persona.id}] all retries failed, fallback to random action {fallback}"
        )
        return "（fallback：LLM 重試失敗）", fallback

    # ── 長期記憶彙整（非阻塞）────────────────────────────────

    def _trigger_consolidation(self) -> None:
        """Launch non-blocking background long-term memory consolidation task."""
        if self._consolidation_task and not self._consolidation_task.done():
            return  # previous consolidation still running

        consolidation_prompt = self.memory.build_consolidation_prompt(
            self.persona.display_name
        )

        async def _consolidate():
            try:
                summary = await self.client.call_consolidation(consolidation_prompt)
                self.memory.set_long_term(summary)
                logger.info(
                    f"[Agent {self.persona.id}] long-term memory updated: {summary[:60]}..."
                )
                # 記錄長期記憶快照
                if self.sim_logger is not None:
                    self.sim_logger.log_memory_snapshot(
                        step=-1,  # 非同步，step 不確定，用 -1 標示
                        agent_id=str(self.persona.id),
                        agent_name=self.persona.display_name,
                        long_term_memory=summary,
                        trigger="consolidation",
                    )
            except Exception as e:
                logger.error(f"[Agent {self.persona.id}] long-term memory consolidation failed: {e}")

        self._consolidation_task = asyncio.create_task(_consolidate())


# ──────────────────────────────────────────────────────────────
#  批次並行決策
# ──────────────────────────────────────────────────────────────

async def decide_batch(
    agents: list[MobileAgentLLM],
    obs: dict,
    env,
    step: int,
) -> dict[str, int]:
    """
    並行呼叫所有 MobileAgent 的 decide，回傳 {agent_id_str: action_id}。
    """
    async def _decide_one(agent: MobileAgentLLM) -> tuple[str, int]:
        agent_key = str(agent.persona.id)
        agent_obs = obs.get(agent_key, {})
        raw_mask = agent_obs.get("action_mask", np.zeros(50))
        mask = np.array(raw_mask, dtype=np.float32)
        action_id = await agent.decide(
            obs=agent_obs,
            mask=mask,
            env=env,
            step=step,
        )
        return agent_key, action_id

    results = await asyncio.gather(*[_decide_one(a) for a in agents])
    return dict(results)
