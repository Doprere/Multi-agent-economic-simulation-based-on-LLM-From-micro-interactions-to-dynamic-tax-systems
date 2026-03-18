"""
planner.py — Planner LLM 決策層。
- 每步執行觀察（輸出 thought + society_comment）
- 滑動視窗記憶（PlannerMemory）
- 每 100 步（稅收日）輸出稅率決策
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import AppConfig, PlannerConfig
from .llm_client import LLMClient
from .memory import PlannerMemory
from .translator import ObsTranslator, _gini

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .logger import SimulationLogger


# ──────────────────────────────────────────────────────────────
#  PlannerLLM
# ──────────────────────────────────────────────────────────────

class PlannerLLM:
    """
    Planner 每步接收全局觀察，透過滑動視窗累積短期記憶。
    在稅收日（step % tax_period == 0）輸出稅率決策並更新環境。
    非稅收日僅輸出觀察結論，向環境提交 NOOP。
    """

    def __init__(
        self,
        planner_cfg: PlannerConfig,
        llm_client: LLMClient,
        mem_cfg: Any,
        tax_period: int = 100,
        sim_logger: "SimulationLogger | None" = None,
    ) -> None:
        self.cfg = planner_cfg
        self.client = llm_client
        self.sim_logger = sim_logger
        self.memory = PlannerMemory(window=mem_cfg.planner_short_term_window)
        self.tax_period = tax_period
        self.translator = ObsTranslator()
        self._n_tax_brackets: int | None = None  # 稅階數，第一次執行時偵測

    # ── System Prompts ────────────────────────────────────────

    def _observe_system_prompt(self) -> str:
        return (
            f"You are the '{self.cfg.display_name}'. {self.cfg.role.strip()}\n\n"
            "Each step you observe the socioeconomic state and accumulate understanding through memory.\n"
            "This is NOT a tax-setting step — focus on observation and analysis only.\n"
            'Output format: {"thought": "...", "society_comment": "..."}'
        )

    def _tax_system_prompt(self) -> str:
        return (
            f"You are the '{self.cfg.display_name}'. {self.cfg.role.strip()}\n\n"
            "TAX ADJUSTMENT TIME! Set the optimal tax brackets based on your observations.\n\n"
            "Tax bracket rules:\n"
            "- tax_brackets is an integer list; each element is the tax rate index (0-21) for that bracket.\n"
            "- Index 0 = 0% tax rate, index 21 = 100% tax rate (~5% per step).\n"
            "- Progressive taxation is recommended: lower brackets get lower rates.\n"
            "- List length must match the number of tax brackets in the environment (usually 7, US Federal).\n\n"
            'Output format: {"thought": "...", "tax_brackets": [<integer list>]}'
        )

    # ── 每步執行 ──────────────────────────────────────────────

    async def step(
        self,
        obs: dict,
        env,
        step: int,
    ) -> list[int] | None:
        """
        每步執行：
        - 生成觀察描述
        - 非稅收日：呼叫 observe LLM，更新記憶，回傳 None（NOOP）
        - 稅收日：呼叫 tax LLM，更新記憶，回傳稅率 list

        Returns:
            None：非稅收日（Planner 提交 NOOP）
            list[int]：稅收日（稅率索引 list）
        """
        # 稅收日：step=0（初始設稅）以及每隔 tax_period 步（預設100）
        # 非決策日：其餘步數提交 NOOP，並呼叫 observe LLM 寫入短期記憶
        is_tax_day = (step % self.tax_period == 0)

        # 翻譯觀察
        planner_obs = obs.get(env.world.planner.idx, {})
        state_desc = self.translator.translate_planner_obs(
            obs=planner_obs,
            env=env,
            step=step,
            is_tax_day=is_tax_day,
        )

        # 記憶背景
        prev_entry = self.memory.get_last_entry()
        memory_ctx = self.memory.get_context_block() if not self.memory.is_empty else ""

        if not is_tax_day:
            return await self._observe_step(state_desc, memory_ctx, prev_entry, env, step)  # type: ignore[return-value]
        else:
            return await self._tax_step(state_desc, memory_ctx, prev_entry, env, step)

    async def _observe_step(
        self,
        state_desc: str,
        memory_ctx: str,
        prev_entry: str,
        env,
        step: int,
    ) -> None:
        """非稅收日：觀察並更新記憶，提交 NOOP。"""
        try:
            result = await self.client.call_planner_observe(
                system_prompt=self._observe_system_prompt(),
                user_prompt=state_desc,
                context=memory_ctx,
            )
            thought = result.get("thought", "")
            comment = result.get("society_comment", "")
            obs_summary = f"{thought} | {comment}"

        except Exception as e:
            logger.error(f"[Planner] observe step failed step={step}: {e}")
            obs_summary = f"(observation failed: {e})"
            thought = "(none)"
            comment = str(e)

        self.memory.add_entry(obs_summary, prev_entry)

        if self.sim_logger is not None:
            self.sim_logger.log_thought(
                step=step,
                agent_id="planner",
                agent_name=self.cfg.display_name,
                role=self.cfg.role,
                thought=thought,
                action_id=-1,
                society_comment=comment,
                short_term_memory=memory_ctx,
                is_tax_day=False,
            )

        logger.info(f"[Planner] step={step} memory updated")
        return None

    async def _tax_step(
        self,
        state_desc: str,
        memory_ctx: str,
        prev_entry: str,
        env,
        step: int,
    ) -> list[int]:
        """稅收日：根據記憶輸出稅率，更新記憶與環境。"""
        # 自動偵測稅階數
        n_brackets = self._get_n_brackets(env)

        user_prompt = (
            f"{state_desc}\n\n"
            f"Please set {n_brackets} tax bracket indices (each 0-21). "
            f"Output a list of exactly {n_brackets} integers."
        )

        try:
            result = await self.client.call_planner_tax(
                system_prompt=self._tax_system_prompt(),
                user_prompt=user_prompt,
                context=memory_ctx,
            )
            thought = result.get("thought", "")
            tax_brackets = result.get("tax_brackets", [])

            # 驗證並修正
            tax_brackets = self._validate_brackets(tax_brackets, n_brackets)

            obs_summary = (
                f"[TAX ADJUSTMENT] {thought} | brackets={tax_brackets}"
            )
            self.memory.add_entry(obs_summary, prev_entry)

            if self.sim_logger is not None:
                self.sim_logger.log_thought(
                    step=step,
                    agent_id="planner",
                    agent_name=self.cfg.display_name,
                    role=self.cfg.role,
                    thought=thought,
                    action_id=-1,
                    tax_brackets=tax_brackets,
                    short_term_memory=memory_ctx,
                    is_tax_day=True,
                )

            logger.info(
                f"[Planner] step={step} tax decision: {tax_brackets} | {thought[:60]}..."
            )
            return tax_brackets

        except Exception as e:
            logger.error(f"[Planner] tax step failed step={step}: {e}")
            fallback = [0] * n_brackets
            obs_summary = f"[TAX ADJUSTMENT FAILED, fallback all-zero] error: {e}"
            self.memory.add_entry(obs_summary, prev_entry)
            if self.sim_logger is not None:
                self.sim_logger.log_thought(
                    step=step,
                    agent_id="planner",
                    agent_name=self.cfg.display_name,
                    role=self.cfg.role,
                    thought=f"(tax step failed: {e})",
                    action_id=-1,
                    tax_brackets=fallback,
                    short_term_memory=memory_ctx,
                    is_tax_day=True,
                )
            return fallback

    def _get_n_brackets(self, env) -> int:
        """自動從環境取得稅階數量。"""
        if self._n_tax_brackets is not None:
            return self._n_tax_brackets
        try:
            planner = env.world.planner
            dims = planner.action_spaces
            if hasattr(dims, '__len__'):
                self._n_tax_brackets = len(dims)
            else:
                self._n_tax_brackets = int(dims)
        except Exception:
            self._n_tax_brackets = 7  # US Federal 預設 7 個區間
        return self._n_tax_brackets

    def _validate_brackets(self, brackets: list, n: int) -> list[int]:
        """
        驗證並修正稅率 brackets：
        - 確保長度為 n
        - 確保每個值在 0-21 之間
        """
        # 修正長度
        if len(brackets) < n:
            brackets = brackets + [0] * (n - len(brackets))
        elif len(brackets) > n:
            brackets = brackets[:n]

        # 修正範圍
        brackets = [max(0, min(21, int(b))) for b in brackets]
        return brackets
