"""
logger.py — 模擬指標記錄與輸出。
記錄每步的 utility、social welfare、Gini、productivity、action log。
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

    # ── 輸出 ──────────────────────────────────────────────────

    def save(self) -> None:
        """將所有記錄輸出至 CSV 和 JSON。"""
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
