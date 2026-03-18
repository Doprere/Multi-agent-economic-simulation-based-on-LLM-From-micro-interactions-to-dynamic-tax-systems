"""
logger.py — 模擬指標記錄與輸出。
記錄每步的 utility、social welfare、Gini、productivity、action log。
並將 agent/planner 的 thought 與記憶輸出至 Excel（.xlsx）供研究者閱讀。
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False

logger = logging.getLogger(__name__)


class SimulationLogger:
    """
    記錄模擬過程中的各項指標，並在模擬結束後輸出至 CSV 和 JSON。
    """

    def __init__(self, output_dir: str = "simulation_results", run_name: str | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"run_{ts}"

        self._step_logs: list[dict] = []
        self._action_logs: list[dict] = []
        self._tax_logs: list[dict] = []
        self._planner_logs: list[dict] = []
        # thought / 記憶 log（給 Excel）
        self._thought_logs: list[dict] = []  # agent 每步 thought
        self._memory_snapshots: list[dict] = []  # agent 記憶快照（長期記憶更新時）

    # ── 每步記錄 ──────────────────────────────────────────────

    def log_step(
        self,
        step: int,
        rewards: dict,
        env,
        agent_actions: dict[str, int],
        planner_action: list[int] | None = None,
    ) -> None:
        """記錄單步的指標與動作。"""
        agents = env.world.agents

        # 資產
        coin_endowments = np.array([a.total_endowment("Coin") for a in agents])
        total_coin = float(np.sum(coin_endowments))
        mean_coin = float(np.mean(coin_endowments))
        gini = _gini(coin_endowments)

        # 各 agent 效用（reward 累積）
        agent_rewards = {
            str(a.idx): float(rewards.get(str(a.idx), 0.0))
            for a in agents
        }

        # Social Welfare（Planner reward）
        planner_reward = float(rewards.get(env.world.planner.idx, 0.0))

        step_record: dict[str, Any] = {
            "step": step,
            "total_coin": total_coin,
            "mean_coin": round(mean_coin, 3),
            "gini": round(gini, 5),
            "planner_reward": round(planner_reward, 5),
            **{f"reward_agent_{i}": round(v, 5) for i, v in agent_rewards.items()},
            **{
                f"coin_agent_{str(a.idx)}": round(float(a.inventory.get("Coin", 0)), 3)
                for a in agents
            },
            **{
                f"wood_agent_{str(a.idx)}": int(a.inventory.get("Wood", 0))
                for a in agents
            },
            **{
                f"stone_agent_{str(a.idx)}": int(a.inventory.get("Stone", 0))
                for a in agents
            },
            **{
                f"labor_agent_{str(a.idx)}": round(float(a.endogenous.get("Labor", 0)), 3)
                for a in agents
            },
        }
        self._step_logs.append(step_record)

        # Action log
        action_record = {
            "step": step,
            **{f"action_agent_{k}": v for k, v in agent_actions.items()},
            "planner_action": json.dumps(planner_action) if planner_action else "NOOP",
        }
        self._action_logs.append(action_record)

        # 每 100 步打印摘要
        if step % 100 == 0:
            self._print_summary(step, mean_coin, gini, planner_reward, agent_rewards)

    def log_tax(self, step: int, tax_brackets: list[int]) -> None:
        """記錄稅率設定。"""
        self._tax_logs.append({
            "step": step,
            "tax_brackets": json.dumps(tax_brackets),
        })

    def log_planner_thought(self, step: int, thought: str, comment: str) -> None:
        """記錄 Planner 的思考過程。"""
        self._planner_logs.append({
            "step": step,
            "thought": thought,
            "society_comment": comment,
        })

    def log_thought(
        self,
        step: int,
        agent_id: str,
        agent_name: str,
        role: str,
        thought: str,
        action_id: int,
        short_term_memory: str = "",
        long_term_memory: str = "",
        is_tax_day: bool = False,
        tax_brackets: list[int] | None = None,
        society_comment: str = "",
    ) -> None:
        """
        記錄單步的 agent thought 與記憶狀態（主要供 Excel 輸出）。

        agent_id: '0'~'3' 代表 MobileAgent；'planner' 代表 Planner
        """
        self._thought_logs.append({
            "step": step,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "role": role,
            "is_tax_day": is_tax_day,
            "thought": thought,
            "action_id": action_id if action_id is not None else "",
            "tax_brackets": json.dumps(tax_brackets) if tax_brackets else "",
            "society_comment": society_comment,
            "short_term_memory": short_term_memory,
            "long_term_memory": long_term_memory,
        })

    def log_memory_snapshot(
        self,
        step: int,
        agent_id: str,
        agent_name: str,
        long_term_memory: str,
        trigger: str = "consolidation",
    ) -> None:
        """記錄長期記憶更新快照。"""
        self._memory_snapshots.append({
            "step": step,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "trigger": trigger,
            "long_term_memory": long_term_memory,
        })

    # ── 輸出 ──────────────────────────────────────────────────

    def save(self) -> None:
        """將所有記錄輸出至 CSV、JSON 和 Excel。"""
        run_dir = self.output_dir / self.run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        _write_csv(run_dir / "step_metrics.csv", self._step_logs)
        _write_csv(run_dir / "action_log.csv", self._action_logs)
        _write_csv(run_dir / "tax_log.csv", self._tax_logs)
        _write_csv(run_dir / "planner_thoughts.csv", self._planner_logs)

        # 整合輸出
        summary = {
            "run_name": self.run_name,
            "total_steps": len(self._step_logs),
            "final_metrics": self._step_logs[-1] if self._step_logs else {},
            "tax_count": len(self._tax_logs),
        }
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # Excel 輸出（thought + 記憶）
        if _HAS_OPENPYXL and self._thought_logs:
            xlsx_path = run_dir / "agent_thoughts.xlsx"
            _write_thoughts_excel(xlsx_path, self._thought_logs, self._memory_snapshots, self._step_logs)
            print(f"  - agent_thoughts.xlsx ({len(self._thought_logs)} 筆 thought)")
        elif not _HAS_OPENPYXL:
            print("  [警告] 未安裝 openpyxl，跳過 Excel 輸出。請執行：pip install openpyxl")

        print(f"\n[Logger] 結果已儲存至：{run_dir}")
        print(f"  - step_metrics.csv ({len(self._step_logs)} 筆)")
        print(f"  - action_log.csv ({len(self._action_logs)} 筆)")
        print(f"  - tax_log.csv ({len(self._tax_logs)} 筆)")
        print(f"  - summary.json")

    def _print_summary(
        self,
        step: int,
        mean_coin: float,
        gini: float,
        planner_reward: float,
        agent_rewards: dict,
    ) -> None:
        rewards_str = ", ".join(
            f"A{k}={v:.3f}" for k, v in sorted(agent_rewards.items())
        )
        print(
            f"[Step {step:>4}] "
            f"均 Coin={mean_coin:.1f} | "
            f"Gini={gini:.4f} | "
            f"Planner Reward={planner_reward:.4f} | "
            f"Agent Rewards: {rewards_str}"
        )


# ──────────────────────────────────────────────────────────────
#  工具函數
# ──────────────────────────────────────────────────────────────

def _gini(values: np.ndarray) -> float:
    values = np.sort(np.abs(values))
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(
        (2 * np.dot(idx, values) - (n + 1) * np.sum(values))
        / (n * np.sum(values))
    )


def _write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def _write_thoughts_excel(
    path: Path,
    thought_logs: list[dict],
    memory_snapshots: list[dict],
    step_logs: list[dict],
) -> None:
    """
    輸出研究者友善的 Excel：
    Sheet 1 - Agent Thoughts：MobileAgent 每步 thought / action / 記憶
    Sheet 2 - Planner Thoughts：Planner 每步 thought / 稅率決策
    Sheet 3 - Memory Snapshots：長期記憶彙整歷史
    Sheet 4 - Step Metrics Summary：關鍵指標摘要（Coin/Gini/Reward）
    """
    if not _HAS_OPENPYXL:
        return

    wb = openpyxl.Workbook()

    # ── 通用樣式 ─────────────────────────────────────────────
    HEADER_FILL = PatternFill("solid", fgColor="2E4057")
    HEADER_FONT = Font(bold=True, color="FFFFFF", size=11)
    AGENT_FILLS = {
        "planner": PatternFill("solid", fgColor="E8F4F8"),
        "0": PatternFill("solid", fgColor="FFF8E7"),
        "1": PatternFill("solid", fgColor="F0FFF0"),
        "2": PatternFill("solid", fgColor="FFF0F0"),
        "3": PatternFill("solid", fgColor="F5F0FF"),
    }
    TAX_DAY_FONT = Font(bold=True, color="C0392B")
    WRAP = Alignment(wrap_text=True, vertical="top")
    CENTER = Alignment(horizontal="center", vertical="top")
    thin = Side(style="thin", color="CCCCCC")
    BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)

    def _header_row(ws, headers: list[str]) -> None:
        ws.append(headers)
        for cell in ws[1]:
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.alignment = CENTER
            cell.border = BORDER
        ws.row_dimensions[1].height = 20

    def _apply_border_wrap(ws, row_idx: int, cols: int) -> None:
        for c in range(1, cols + 1):
            cell = ws.cell(row=row_idx, column=c)
            cell.border = BORDER
            if cell.alignment.wrap_text is None:
                cell.alignment = WRAP

    # ── Sheet 1: Agent Thoughts ──────────────────────────────
    agent_logs = [r for r in thought_logs if r["agent_id"] != "planner"]
    if agent_logs:
        ws1 = wb.active
        ws1.title = "Agent Thoughts"
        headers1 = [
            "Step", "Agent ID", "Agent Name", "Role",
            "Thought（決策思維）", "Action ID",
            "短期記憶（最近N步）", "長期記憶（彙整）",
        ]
        _header_row(ws1, headers1)

        col_widths1 = [7, 9, 14, 16, 60, 10, 50, 50]
        for i, w in enumerate(col_widths1, 1):
            ws1.column_dimensions[get_column_letter(i)].width = w

        for rec in agent_logs:
            row = [
                rec["step"],
                rec["agent_id"],
                rec["agent_name"],
                rec["role"],
                rec["thought"],
                rec["action_id"],
                rec["short_term_memory"],
                rec["long_term_memory"],
            ]
            ws1.append(row)
            ri = ws1.max_row
            fill = AGENT_FILLS.get(str(rec["agent_id"]), PatternFill())
            for c in range(1, len(headers1) + 1):
                cell = ws1.cell(row=ri, column=c)
                cell.fill = fill
                cell.border = BORDER
                cell.alignment = WRAP
            ws1.row_dimensions[ri].height = 60
    else:
        ws1 = wb.active
        ws1.title = "Agent Thoughts"

    # ── Sheet 2: Planner Thoughts ────────────────────────────
    planner_logs = [r for r in thought_logs if r["agent_id"] == "planner"]
    ws2 = wb.create_sheet("Planner Thoughts")
    headers2 = [
        "Step", "Is Tax Day", "Thought（決策思維）",
        "Society Comment（社會觀察）", "Tax Brackets（稅率）",
        "短期記憶",
    ]
    _header_row(ws2, headers2)
    col_widths2 = [7, 11, 60, 50, 25, 60]
    for i, w in enumerate(col_widths2, 1):
        ws2.column_dimensions[get_column_letter(i)].width = w

    for rec in planner_logs:
        is_tax = rec.get("is_tax_day", False)
        row = [
            rec["step"],
            "✅ 稅收日" if is_tax else "觀察",
            rec["thought"],
            rec.get("society_comment", ""),
            rec.get("tax_brackets", ""),
            rec.get("short_term_memory", ""),
        ]
        ws2.append(row)
        ri = ws2.max_row
        fill = PatternFill("solid", fgColor="FDEBD0") if is_tax else AGENT_FILLS["planner"]
        for c in range(1, len(headers2) + 1):
            cell = ws2.cell(row=ri, column=c)
            cell.fill = fill
            cell.border = BORDER
            cell.alignment = WRAP
            if is_tax:
                cell.font = TAX_DAY_FONT
        ws2.row_dimensions[ri].height = 60

    # ── Sheet 3: Memory Snapshots ────────────────────────────
    ws3 = wb.create_sheet("Memory Snapshots")
    headers3 = ["Step", "Agent ID", "Agent Name", "Trigger", "長期記憶內容"]
    _header_row(ws3, headers3)
    col_widths3 = [7, 9, 14, 14, 80]
    for i, w in enumerate(col_widths3, 1):
        ws3.column_dimensions[get_column_letter(i)].width = w

    for rec in memory_snapshots:
        row = [
            rec["step"], rec["agent_id"], rec["agent_name"],
            rec["trigger"], rec["long_term_memory"],
        ]
        ws3.append(row)
        ri = ws3.max_row
        fill = AGENT_FILLS.get(str(rec["agent_id"]), PatternFill())
        for c in range(1, len(headers3) + 1):
            cell = ws3.cell(row=ri, column=c)
            cell.fill = fill
            cell.border = BORDER
            cell.alignment = WRAP
        ws3.row_dimensions[ri].height = 45

    # ── Sheet 4: Step Metrics Summary ────────────────────────
    ws4 = wb.create_sheet("Step Metrics")
    if step_logs:
        # 只取關鍵欄位
        key_cols = ["step", "mean_coin", "gini", "planner_reward", "total_coin"]
        agent_reward_cols = [k for k in step_logs[0] if k.startswith("reward_agent_")]
        agent_coin_cols = [k for k in step_logs[0] if k.startswith("coin_agent_")]
        display_cols = key_cols + agent_reward_cols + agent_coin_cols
        headers4 = [c.replace("_", " ").title() for c in display_cols]
        _header_row(ws4, headers4)
        for i in range(1, len(headers4) + 1):
            ws4.column_dimensions[get_column_letter(i)].width = 14
        for rec in step_logs:
            ws4.append([rec.get(c, "") for c in display_cols])
            ri = ws4.max_row
            for c in range(1, len(headers4) + 1):
                ws4.cell(row=ri, column=c).border = BORDER
                ws4.cell(row=ri, column=c).alignment = CENTER

    wb.save(path)
    logger.info(f"[Logger] Excel 已儲存：{path}")


def setup_logging(level: int = logging.INFO) -> None:

    """Configure global logging with UTF-8 encoding."""
    import io as _io
    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        # Force UTF-8 on stdout-based handler to avoid Windows garbling
        stream = _io.TextIOWrapper(
            _io.open(sys.__stderr__.fileno(), mode="wb", closefd=False),
            encoding="utf-8",
            errors="replace",
        ) if hasattr(sys, "__stderr__") and sys.__stderr__ else sys.stderr

        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)
