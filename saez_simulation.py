"""
saez_simulation.py — Saez 最適稅率公式實驗腳本。

與 ollama_simulation.py 的差異：
  - 不使用 LLM Planner（稅率由 Foundation 內建 Saez 公式自動計算）
  - Agent 仍使用 Ollama LLM 做決策（llama3.1:8b）
  - 用於比較 AI Planner vs 傳統經濟學最適稅率

用法：
    python saez_simulation.py --steps 1000 --model llama3.1:8b
    python saez_simulation.py --steps 1000 --model llama3.1:8b --run-name saez_run1
    python saez_simulation.py --dry-run --steps 5
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr（Windows 中文環境）
# 只在尚未包裝時執行，避免與 ollama_simulation import 衝突
if hasattr(sys.stdout, "buffer") and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer") and not isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from copy import deepcopy

import numpy as np

_PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))
_AI_ECON_PKG = _PROJECT_ROOT / "ai_economist"
if _AI_ECON_PKG.is_dir():
    sys.path.insert(0, str(_AI_ECON_PKG))

from ai_economist import foundation

from llm_agent.action_map import get_random_valid_action
from llm_agent.agent import MobileAgentLLM, decide_batch
from llm_agent.config import load_config, make_env_config
from llm_agent.logger import SimulationLogger, setup_logging
from llm_agent.ollama_client import OllamaClient

# 從 ollama_simulation 重用不修改原檔的工具函式
from ollama_simulation import (
    _apply_age_group_skills,
    _apply_labor_modifier,
    _build_planner_action,
    _check_resource_adjacency,
    _validate_action_order,
)

logger = logging.getLogger(__name__)


def _sample_random_actions(env, obs) -> dict[str, int]:
    """Dry-run 用：隨機 agent 動作（不含 planner）。"""
    agent_actions: dict[str, int] = {}
    for agent in env.world.agents:
        key = str(agent.idx)
        if key not in obs:
            continue
        mask = np.array(obs[key].get("action_mask", np.zeros(50)))
        agent_actions[key] = get_random_valid_action(mask)
    return agent_actions


# ──────────────────────────────────────────────────────────────
#  主模擬迴圈（Saez 版）
# ──────────────────────────────────────────────────────────────

async def run_episode(
    config_path: str,
    ollama_url: str,
    ollama_model: str,
    max_steps: int | None = None,
    dry_run: bool = False,
    run_name: str | None = None,
    output_dir: str = "simulation_results",
) -> None:
    """
    執行 Saez 公式 Planner 實驗。

    與 ollama_simulation.run_episode 的差異：
    - tax_model="saez"，Foundation 自動計算最適稅率
    - 不實例化 PlannerLLM，planner 永遠 NOOP
    """
    # 強制重新載入 config（避免 singleton cache 問題）
    import llm_agent.config as _cfg_mod
    _cfg_mod._CONFIG = None
    cfg = load_config(config_path)

    print("\n" + "=" * 60)
    print(" AI Economist — Saez Formula Planner Experiment")
    print("=" * 60)
    print(f"Agent 模型：{ollama_model} @ {ollama_url}")
    print(f"Planner：Saez 最適稅率公式（Foundation 內建）")
    print(f"Episode 長度：{cfg.environment.episode_length}")
    print(f"Dry-run：{dry_run}")

    # ── 環境初始化 ──────────────────────────────────────────
    env_config = make_env_config(cfg)
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()
    _apply_age_group_skills(env, cfg)
    _validate_action_order(env)

    # 取得稅率元件，確認 Saez 模式
    tax_component = env.get_component("PeriodicBracketTax")
    assert tax_component.tax_model == "saez", (
        f"Expected tax_model='saez', got '{tax_component.tax_model}'. "
        f"Please use config_saez.yaml."
    )

    print(f"\n[Init] 環境初始化完成，tax_model=saez")
    print(f"[Init] Planner action_spaces={env.world.planner.action_spaces}（Saez 模式下應為空）")

    # ── 初始化 LLM Agent ──────────────────────────────────
    sim_logger = SimulationLogger(output_dir=output_dir, run_name=run_name)

    ollama_client: OllamaClient | None = None

    if not dry_run:
        ollama_client = OllamaClient(
            model=ollama_model,
            base_url=ollama_url,
            max_retries=cfg.llm.max_retries,
            temperature=cfg.llm.temperature,
        )
        agents = [
            MobileAgentLLM(
                persona,
                ollama_client,
                cfg.memory,
                sim_logger=sim_logger,
            )  # type: ignore[arg-type]
            for persona in cfg.personas
        ]
        print(f"[Init] {len(agents)} 個 Ollama Agent 已建立（無 LLM Planner）")
    else:
        agents = []
        print("[Init] Dry-run 模式：使用隨機動作，不呼叫 LLM")

    # ── 主迴圈 ──────────────────────────────────────────────
    episode_length = min(
        max_steps or cfg.environment.episode_length,
        cfg.environment.episode_length,
    )
    print(f"[Start] 開始模擬，共 {episode_length} 步\n")

    step = 0
    try:
        for step in range(episode_length):

            if dry_run:
                agent_actions = _sample_random_actions(env, obs)
            else:
                # Agent 並行決策（同 ollama_simulation）
                agent_actions = await decide_batch(agents, obs, env, step)

            # 資源鄰近偵測
            should_snapshot = (step + 1) % 20 == 0 or step == 0

            if not dry_run:
                recent_thoughts: dict[str, str] = {}
                for rec in reversed(sim_logger._thought_logs):
                    if rec["step"] == step and rec["agent_id"] != "planner":
                        recent_thoughts[rec["agent_id"]] = rec["thought"]
                    if len(recent_thoughts) >= len(cfg.personas):
                        break

                has_adjacent = _check_resource_adjacency(
                    env=env, obs=obs, cfg=cfg, sim_logger=sim_logger,
                    agent_actions=agent_actions, agent_thoughts=recent_thoughts,
                    step=step,
                )
                if has_adjacent:
                    should_snapshot = True

            if should_snapshot:
                sim_logger.save_map_snapshot(step=step, env=env)

            # 建構 actions dict — planner 永遠 NOOP
            planner_env_action = _build_planner_action(env, None)
            actions: dict = {
                **agent_actions,
                env.world.planner.idx: planner_env_action,
            }

            # 環境推進（Saez 稅率在 component_step 內自動計算）
            obs, rewards, done, info = env.step(actions)
            _apply_labor_modifier(env)

            # 記錄 step metrics
            sim_logger.log_step(
                step=step,
                rewards=rewards,
                env=env,
                agent_actions=agent_actions,
                planner_action=None,  # Saez 模式無 planner action
            )

            # 記錄 Saez 計算的稅率（每個 tax period 開始時 + 初始）
            if step % tax_component.period == 0:
                saez_rates = [
                    round(float(r), 4)
                    for r in tax_component.curr_marginal_rates
                ]
                sim_logger.log_tax(step, saez_rates)

            # 定期存檔（每 100 步）
            if (step + 1) % 100 == 0:
                sim_logger.save()
                print(f"[Checkpoint] Step {step+1} intermediate results saved")

            if done.get("__all__", False):
                print(f"\n[Done] Episode 在步驟 {step} 結束（all done）")
                break

    finally:
        # ── 結束：確保資源釋放 ──────────────────────────────
        print(f"\n[End] 模擬完成！共執行 {step + 1} 步")
        sim_logger.save()

        if not dry_run:
            # 等待背景記憶彙整
            pending = [
                a._consolidation_task
                for a in agents
                if a._consolidation_task and not a._consolidation_task.done()
            ]
            if pending:
                print(f"[Cleanup] 等待 {len(pending)} 個記憶彙整任務完成...")
                await asyncio.gather(*pending, return_exceptions=True)

            if ollama_client:
                await ollama_client.aclose()


# ──────────────────────────────────────────────────────────────
#  CLI 入口
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Economist — Saez Formula Planner Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="最大模擬步數（覆蓋 config）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="不呼叫 LLM，使用隨機動作測試",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="輸出資料夾名稱（預設自動加時間戳）",
    )
    parser.add_argument(
        "--output-dir", type=str, default="simulation_results",
        help="輸出根目錄（預設 simulation_results）",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(_PROJECT_ROOT / "llm_agent" / "config_saez.yaml"),
        help="config YAML 路徑（預設 llm_agent/config_saez.yaml）",
    )
    parser.add_argument(
        "--ollama-url", type=str, default="http://localhost:11434",
        help="Ollama API 服務位址",
    )
    parser.add_argument(
        "--model", type=str, default="llama3.1:8b",
        help="Ollama 模型名稱（用於 Agent）",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="啟用 DEBUG 日誌",
    )
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.debug else logging.INFO)

    asyncio.run(
        run_episode(
            config_path=args.config,
            ollama_url=args.ollama_url,
            ollama_model=args.model,
            max_steps=args.steps,
            dry_run=args.dry_run,
            run_name=args.run_name,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
