"""
memory.py — AgentMemory（短期+長期）與 PlannerMemory（滑動視窗複合摘要）。
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Awaitable


# ──────────────────────────────────────────────────────────────
#  AgentMemory：MobileAgent 雙層記憶
# ──────────────────────────────────────────────────────────────

class AgentMemory:
    """
    短期記憶：固定長度滑動視窗，記錄最近 N 步 LLM 輸出的 thought。
    長期記憶：當短期記憶累積到 trigger 步後，呼叫外部 consolidator 函數
              彙整成一段長期摘要，並重置計數器。
    """

    def __init__(
        self,
        agent_id: int,
        short_term_window: int = 8,
        long_term_trigger: int = 10,
    ) -> None:
        self.agent_id = agent_id
        self._window = short_term_window
        self._trigger = long_term_trigger

        self._short_term: deque[str] = deque(maxlen=short_term_window)
        self._long_term: str = ""          # 長期記憶字串
        self._counter: int = 0             # 自上次彙整後的累積步數

    # ── 短期記憶 ──────────────────────────────────────────────

    def add_thought(self, thought: str) -> None:
        """加入一步 thought 到短期記憶，並增加長期計數。"""
        self._short_term.append(thought)
        self._counter += 1

    def get_short_term_context(self) -> str:
        """格式化最近 N 步 thought，供 prompt 使用。"""
        if not self._short_term:
            return "(no short-term memory yet)"
        lines = [
            f"  [{i+1}] {t}"
            for i, t in enumerate(self._short_term)
        ]
        return "\n".join(lines)

    # ── 長期記憶 ──────────────────────────────────────────────

    @property
    def long_term(self) -> str:
        return self._long_term

    @property
    def has_long_term(self) -> bool:
        return bool(self._long_term)

    def should_consolidate(self) -> bool:
        """是否達到長期記憶彙整觸發條件。"""
        return self._counter >= self._trigger

    def set_long_term(self, summary: str) -> None:
        """由外部（LLM 彙整結果）更新長期記憶並重置計數器。"""
        self._long_term = summary
        self._counter = 0

    def build_consolidation_prompt(self, persona_name: str) -> str:
        """產生長期記憶彙整的 user prompt。"""
        thoughts_text = "\n".join(
            f"[Step {i+1}] {t}"
            for i, t in enumerate(self._short_term)
        )
        return (
            f"You are {persona_name}. Below are your recent {len(self._short_term)} decision thoughts:\n"
            f"{thoughts_text}\n\n"
            "Summarize your current strategic tendencies and resource situation in 1-2 sentences "
            "for long-term memory storage. Output only the summary text, no prefixes or suffixes."
        )

    # ── 組裝 Prompt 片段 ──────────────────────────────────────

    def get_long_term_block(self) -> str:
        """供 system prompt 使用的長期記憶區塊。"""
        if not self._long_term:
            return ""
        return f"\n[Long-term Memory]\n{self._long_term}\n"

    def get_short_term_block(self) -> str:
        """供 user prompt 使用的短期記憶區塊。"""
        ctx = self.get_short_term_context()
        return f"\n[Recent Decision Memory (last {self._window} steps)]\n{ctx}\n"


# ──────────────────────────────────────────────────────────────
#  PlannerMemory：滑動視窗複合摘要
# ──────────────────────────────────────────────────────────────

class PlannerMemory:
    """
    Planner 每步（含非稅收日）產生一份「當步社會觀察 + 前一步摘要」的複合摘要，
    以滑動視窗形式保留最近 N 步，作為稅收日決策時的背景記憶。
    """

    def __init__(self, window: int = 3) -> None:
        self._window = window
        self._entries: deque[str] = deque(maxlen=window)

    def add_entry(self, obs_summary: str, prev_entry: str = "") -> None:
        """
        生成複合摘要並加入視窗。

        Args:
            obs_summary: 本步的社會觀察摘要（文字）
            prev_entry: 前一步的記憶條目（可選，用於延伸記憶）
        """
        if prev_entry:
            entry = f"[Observation] {obs_summary}\n[Prior Context] {prev_entry}"
        else:
            entry = f"[Observation] {obs_summary}"
        self._entries.append(entry)

    def get_last_entry(self) -> str:
        """取得最新一條記憶條目，用於下一步的 prev_entry。"""
        if not self._entries:
            return ""
        return self._entries[-1]

    def get_context(self) -> str:
        """回傳最近 N 步記憶的格式化字串，供稅收日 prompt 使用。"""
        if not self._entries:
            return "(no observation history yet)"
        lines = []
        total = len(self._entries)
        for i, entry in enumerate(self._entries):
            lines.append(f"--- Memory {i+1}/{total} ---\n{entry}")
        return "\n\n".join(lines)

    def get_context_block(self) -> str:
        """完整格式化的記憶區塊，直接嵌入 prompt。"""
        ctx = self.get_context()
        return f"\n[Social Observation Memory (last {self._window} steps)]\n{ctx}\n"

    @property
    def is_empty(self) -> bool:
        return len(self._entries) == 0
