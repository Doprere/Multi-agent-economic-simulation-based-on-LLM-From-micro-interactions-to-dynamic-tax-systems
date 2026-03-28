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

Example 1 (gather — move toward resource):
{{"thought": "Wood is 2 tiles to the Right, my Wood=0. I need it to build. Moving right to collect.", "action_id": 47}}

Example 2 (gather — move toward distant resource):
{{"thought": "Stone is 3 steps away (2 Down, 1 Right). No resources nearby. Moving Down first to get closer.", "action_id": 49}}

Example 3 (build):
{{"thought": "I have Wood=1, Stone=1. Building now earns 15 Coin — worth the labor.", "action_id": 1}}

Example 4 (buy on market):
{{"thought": "I need Stone to build. Lowest ask for Stone is 4 Coin. I'll bid 4 to match and buy immediately.", "action_id": 6}}

Example 5 (sell on market):
{{"thought": "I have Stone=3 which exceeds my build needs. Highest bid for Stone is 5 Coin. I'll ask 5 to sell immediately.", "action_id": 18}}

Example 6 (NOOP — only when truly idle):
{{"thought": "No resources visible, cannot build, all orders pending. Nothing productive to do this step.", "action_id": 0}}
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

        # Map labor_cost_modifier to natural-language stamina description
        lcm = getattr(p, "labor_cost_modifier", 1.0)
        if lcm <= 0.7:
            stamina = "You have good stamina — physical tasks cost you relatively little effort."
        elif lcm <= 1.0:
            stamina = "Physical tasks require a moderate amount of effort."
        elif lcm <= 1.3:
            stamina = "Physical tasks require more effort than they used to."
        else:
            stamina = "Physical tasks cost you more effort than most — pace yourself accordingly."

        base = (
            f"You are an Agent in the AI Economist simulation, representing '{p.display_name}'.\n"
            f"Your role: {p.role.strip()}\n\n"
            "Your happiness = Coin earned minus effort spent.\n"
            "- Gather Wood and Stone, then Build houses or Sell on market to earn Coin.\n"
            "- Every action (moving, gathering, building, trading) costs some labor.\n"
            "- Spending labor to gather and build is an investment — it pays off in Coin.\n"
            f"{stamina}\n\n"
            "Decision rules:\n"
            "1. Choose exactly ONE action_id per step. It MUST appear in the [Valid Actions] list.\n"
            "2. IMPORTANT: If an action_id is NOT in the [Valid Actions] list, it is BLOCKED by the environment and your turn will be wasted. Always check the list before choosing.\n"
            "3. Movement directions may be blocked by walls, water, or other agents. If a direction is not in [Valid Actions], do NOT attempt it — choose a different direction or action.\n"
            "4. Output MUST be strict JSON: {\"thought\": \"...\", \"action_id\": <integer>}\n\n"
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

        # 2.5 記錄完整 prompt（供診斷）
        if self.sim_logger is not None:
            self.sim_logger.log_prompt(
                step=step,
                agent_id=str(self.persona.id),
                agent_name=self.persona.display_name,
                system_prompt=self._build_system_prompt(),
                user_prompt=state_desc,
                context=short_ctx,
            )

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
                prompt += f"\n\nWARNING: Your last response was invalid: {retry_hint}\nPlease choose a valid action_id."

            try:
                result = await self.client.call_agent(
                    system_prompt=self._build_system_prompt(),
                    user_prompt=prompt,
                    context=short_ctx,
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
        return "(fallback: LLM retries exhausted)", fallback

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
