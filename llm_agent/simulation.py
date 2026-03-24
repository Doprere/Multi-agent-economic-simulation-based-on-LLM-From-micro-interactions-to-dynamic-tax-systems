"""
simulation.py — Main simulation loop.
Runs the LLM-driven AI Economist socioeconomic simulation.

Usage:
    python -m llm_agent.simulation                      # full 1000-step episode
    python -m llm_agent.simulation --steps 10           # quick test
    python -m llm_agent.simulation --dry-run --steps 5  # random actions, no LLM
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr so Chinese log messages don't garble on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from copy import deepcopy

import numpy as np

# 確保可從 project 根目錄執行
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_economist import foundation

from llm_agent.action_map import get_random_valid_action
from llm_agent.agent import MobileAgentLLM, decide_batch
from llm_agent.config import AppConfig, load_config, make_env_config
from llm_agent.logger import SimulationLogger, setup_logging
from llm_agent.llm_client import LLMClient
from llm_agent.planner import PlannerLLM

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  年齡族群 Pareto Alpha 注入
# ──────────────────────────────────────────────────────────────

def _apply_age_group_skills(env, cfg: AppConfig) -> None:
    """
    根據 Persona 的 skill_pareto_alpha 與 labor_cost_modifier，
    重新設定各 Agent 的建造技能與勞動消耗係數，並注入初始 Coin 稟賦。

    此函數在 env.reset() 之後呼叫，覆蓋預設的技能抽樣結果。
    """
    build_comp = env.get_component("Build")
    gather_comp = env.get_component("Gather")

    for persona in cfg.personas:
        agent = env.get_agent(str(persona.id))

        # — 建造技能：根據 Pareto alpha 重新抽樣 —
        alpha = persona.skill_pareto_alpha
        pmsm = build_comp.payment_max_skill_multiplier

        sampled_skill = np.random.pareto(alpha)
        pay_rate = np.minimum(pmsm, (pmsm - 1) * sampled_skill + 1)
        agent.state["build_payment"] = float(pay_rate * build_comp.payment)
        agent.state["build_skill"] = float(sampled_skill)
        build_comp.sampled_skills[agent.idx] = sampled_skill

        # — 勞動消耗：以 modifier 縮放 Gather 元件的勞動係數在狀態上 —
        # Foundation 不直接支援 per-agent 的 move_labor，
        # 我們透過修改 agent 的 bonus_gather_prob 間接表示效率差異，
        # 並在 prompt 中告知 LLM 各族群的體力狀況。
        # 對數值影響：將 labor_cost_modifier 存入 agent state，
        # 在 simulation loop 中每步結束後補償勞動消耗。
        agent.state["labor_cost_modifier"] = persona.labor_cost_modifier

        # --- Initial coin endowment (lifecycle savings) ---
        lo = persona.endowment_coin_min
        hi = persona.endowment_coin_max
        assert lo >= 0 and hi >= 0 and lo <= hi, (
            f"Invalid endowment range for {persona.name}: [{lo}, {hi}]"
        )
        if hi > 0:
            endowment = float(np.random.uniform(lo, hi))
            agent.state["inventory"]["Coin"] += endowment
        else:
            endowment = 0.0

        logger.info(
            f"[Init] Agent {persona.id}（{persona.display_name}）"
            f"build_payment={agent.state['build_payment']:.2f}, "
            f"labor_modifier={persona.labor_cost_modifier}, "
            f"initial_coin={agent.state['inventory']['Coin']:.1f}"
        )

    # --- Recalculate reward baseline after endowment injection ---
    # Without this, step 0 reward would include a massive one-time spike
    # from isoelastic_utility(coin_with_endowment) - isoelastic_utility(0)
    curr = env.get_current_optimization_metrics()
    env.curr_optimization_metric = deepcopy(curr)
    env.init_optimization_metric = deepcopy(curr)
    env.prev_optimization_metric = deepcopy(curr)
    logger.info("[Init] Optimization metric baseline recalculated after endowment injection")


def _apply_labor_modifier(env) -> None:
    """
    每步結束後，根據 labor_cost_modifier 調整各 Agent 的累積勞動。
    modifier > 1 → 消耗更多體力（老年）；modifier < 1 → 消耗更少體力（青年）。

    注意：Foundation 的 Labor 是累積值，原始公式本步觸發的勞動已「過量」記錄。
    我們在此通過補差值的方式實現 per-agent 係數。
    """
    # 此功能暫以 prompt 語義層面說明為主，
    # 數值注入可能影響環境 reward 計算的一致性，
    # 若需嚴格數值精度，建議直接 fork Foundation 的 Gather/Build 元件。
    pass


# ──────────────────────────────────────────────────────────────
#  Planner 動作轉換
# ──────────────────────────────────────────────────────────────

def _build_planner_action(env, tax_brackets: list[int] | None) -> list[int]:
    """生成 Planner 提交給環境的動作。"""
    planner = env.world.planner
    dims = planner.action_spaces

    if tax_brackets is None:
        # NOOP：全部為 0
        return [0] * len(dims)

    # 確保長度一致
    n = len(dims)
    if len(tax_brackets) < n:
        tax_brackets = tax_brackets + [0] * (n - len(tax_brackets))
    elif len(tax_brackets) > n:
        tax_brackets = tax_brackets[:n]

    # 確保值域
    return [max(0, min(int(d) - 1, int(t))) for d, t in zip(dims, tax_brackets)]


# ──────────────────────────────────────────────────────────────
#  隨機動作（dry-run 模式）
# ──────────────────────────────────────────────────────────────

def _sample_random_actions(env, obs) -> tuple[dict[str, int], list[int]]:
    """為所有 Agent 生成隨機合法動作（dry-run 用）。"""
    agent_actions: dict[str, int] = {}
    for agent in env.world.agents:
        key = str(agent.idx)
        if key not in obs:
            continue
        mask = np.array(obs[key].get("action_mask", np.zeros(50)))
        agent_actions[key] = get_random_valid_action(mask)

    # Planner 也隨機
    planner = env.world.planner
    dims = planner.action_spaces
    planner_action = [int(np.random.randint(0, int(d))) for d in dims]

    return agent_actions, planner_action


# ──────────────────────────────────────────────────────────────
#  主模擬迴圈
# ──────────────────────────────────────────────────────────────

async def run_episode(
    cfg: AppConfig,
    max_steps: int | None = None,
    dry_run: bool = False,
    run_name: str | None = None,
) -> None:
    """
    執行一個完整 Episode 的模擬。

    Args:
        cfg: 全局設定
        max_steps: 最大步數（覆蓋 config 中的 episode_length）
        dry_run: True → 不呼叫 LLM，使用隨機動作
        run_name: 輸出資料夾名稱
    """
    # ── 初始化環境 ──────────────────────────────────────────
    print("\n" + "="*60)
    print(" AI Economist LLM Simulation")
    print("="*60)
    print(f"模型：{cfg.llm.model}")
    print(f"Episode 長度：{cfg.environment.episode_length}")
    print(f"Dry-run：{dry_run}")

    env_config = make_env_config(cfg)
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()

    # 注入年齡族群技能
    _apply_age_group_skills(env, cfg)

    print(f"\n[Init] 環境初始化完成，Planner action_spaces={env.world.planner.action_spaces}")

    # ── 初始化 LLM 元件 ────────────────────────────────────
    sim_logger = SimulationLogger(run_name=run_name)

    if not dry_run:
        llm_client = LLMClient(cfg.llm)
        agents = [
            MobileAgentLLM(persona, llm_client, cfg.memory)
            for persona in cfg.personas
        ]
        planner = PlannerLLM(
            planner_cfg=cfg.planner,
            llm_client=llm_client,
            mem_cfg=cfg.memory,
            tax_period=cfg.planner.tax_period,
        )
        print(f"[Init] {len(agents)} 個 LLM Agent + 1 個 LLM Planner 已建立")
    else:
        agents = []
        planner = None
        print("[Init] Dry-run 模式：使用隨機動作，不呼叫 LLM")

    # ── 主迴圈 ──────────────────────────────────────────────
    episode_length = min(max_steps or cfg.environment.episode_length, cfg.environment.episode_length)
    print(f"[Start] 開始模擬，共 {episode_length} 步\n")

    planner_action_cache: list[int] | None = None  # 上一次稅收日的稅率（供非稅收日使用）

    for step in range(episode_length):

        if dry_run:
            # 隨機動作
            agent_actions, planner_brackets = _sample_random_actions(env, obs)
        else:
            # ─ Planner 決策（每步） ─
            planner_brackets = await planner.step(obs=obs, env=env, step=step)

            # ─ Agent 並行決策 ─
            agent_actions = await decide_batch(agents, obs, env, step)

        # ─ 建構完整 actions dict ─
        planner_env_action = _build_planner_action(env, planner_brackets)
        actions: dict = {
            **agent_actions,
            env.world.planner.idx: planner_env_action,
        }

        # ─ 環境推進 ─
        obs, rewards, done, info = env.step(actions)

        # ─ 記錄 ─
        sim_logger.log_step(
            step=step,
            rewards=rewards,
            env=env,
            agent_actions=agent_actions,
            planner_action=planner_brackets,
        )
        if planner_brackets is not None:
            sim_logger.log_tax(step, planner_brackets)

        if done.get("__all__", False):
            print(f"\n[Done] Episode 在步驟 {step} 結束（all done）")
            break

    # ── 結束 ────────────────────────────────────────────────
    print(f"\n[End] 模擬完成！共執行 {step+1} 步")
    sim_logger.save()

    # 等待所有背景記憶彙整任務完成
    if not dry_run:
        pending = [
            a._consolidation_task for a in agents
            if a._consolidation_task and not a._consolidation_task.done()
        ]
        if pending:
            print(f"[Cleanup] 等待 {len(pending)} 個記憶彙整任務完成...")
            await asyncio.gather(*pending, return_exceptions=True)


# ──────────────────────────────────────────────────────────────
#  CLI 入口
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AI Economist LLM Simulation")
    parser.add_argument("--steps", type=int, default=None, help="最大模擬步數（覆蓋 config）")
    parser.add_argument("--dry-run", action="store_true", help="不呼叫 LLM，使用隨機動作測試")
    parser.add_argument("--run-name", type=str, default=None, help="輸出資料夾名稱")
    parser.add_argument("--config", type=str, default=None, help="config.yaml 路徑")
    parser.add_argument("--debug", action="store_true", help="啟用 DEBUG 日誌")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.debug else logging.INFO)

    cfg = load_config(args.config)

    asyncio.run(
        run_episode(
            cfg=cfg,
            max_steps=args.steps,
            dry_run=args.dry_run,
            run_name=args.run_name,
        )
    )


if __name__ == "__main__":
    main()
