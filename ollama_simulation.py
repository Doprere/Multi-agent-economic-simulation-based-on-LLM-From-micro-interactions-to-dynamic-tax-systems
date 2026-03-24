"""
ollama_simulation.py — 使用 Ollama 本地 LLM（llama3:8b）的模擬腳本。

對應關係：
  - simulation.py   → GPT / dry-run（隨機）模式
  - ollama_simulation.py → 本地 Ollama 模式

用法：
    # 在 project 根目錄執行
    python ollama_simulation.py --steps 10 --run-name ollama_test
    python ollama_simulation.py --steps 200 --model llama3:8b --ollama-url http://localhost:11434
    python ollama_simulation.py --dry-run --steps 5   # 隨機動作，測試環境
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr（Windows 中文環境）
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from copy import deepcopy

import numpy as np

_PROJECT_ROOT = Path(__file__).parent.resolve()
# 確保 project 根目錄在 path（讓 llm_agent 可 import）
sys.path.insert(0, str(_PROJECT_ROOT))
# 額外插入 ai_economist 套件目錄（本地未 pip install 時需要）
# project/ai_economist/ 裡有 ai_economist/ 子目錄（package 本體）
_AI_ECON_PKG = _PROJECT_ROOT / "ai_economist"
if _AI_ECON_PKG.is_dir():
    sys.path.insert(0, str(_AI_ECON_PKG))

from ai_economist import foundation

from llm_agent.action_map import get_random_valid_action
from llm_agent.agent import MobileAgentLLM, decide_batch
from llm_agent.config import AppConfig, load_config, make_env_config
from llm_agent.logger import SimulationLogger, setup_logging
from llm_agent.ollama_client import OllamaClient
from llm_agent.planner import PlannerLLM
from llm_agent.translator import _extract_resource_positions, _direction_desc

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  年齡族群技能注入（與 simulation.py 相同）
# ──────────────────────────────────────────────────────────────

def _apply_age_group_skills(env, cfg: AppConfig) -> None:
    build_comp = env.get_component("Build")

    for persona in cfg.personas:
        agent = env.get_agent(str(persona.id))

        alpha = persona.skill_pareto_alpha
        pmsm = build_comp.payment_max_skill_multiplier

        sampled_skill = np.random.pareto(alpha)
        pay_rate = np.minimum(pmsm, (pmsm - 1) * sampled_skill + 1)
        agent.state["build_payment"] = float(pay_rate * build_comp.payment)
        agent.state["build_skill"] = float(sampled_skill)
        build_comp.sampled_skills[agent.idx] = sampled_skill
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


# ──────────────────────────────────────────────────────────────
#  Planner 動作轉換（與 simulation.py 相同）
# ──────────────────────────────────────────────────────────────

def _build_planner_action(env, tax_brackets: list[int] | None) -> list[int]:
    planner = env.world.planner
    dims = planner.action_spaces

    if tax_brackets is None:
        return [0] * len(dims)

    n = len(dims)
    if len(tax_brackets) < n:
        tax_brackets = tax_brackets + [0] * (n - len(tax_brackets))
    elif len(tax_brackets) > n:
        tax_brackets = tax_brackets[:n]

    return [max(0, min(int(d) - 1, int(t))) for d, t in zip(dims, tax_brackets)]


# ──────────────────────────────────────────────────────────────
#  隨機動作（dry-run）
# ──────────────────────────────────────────────────────────────

def _sample_random_actions(env, obs) -> tuple[dict[str, int], list[int]]:
    agent_actions: dict[str, int] = {}
    for agent in env.world.agents:
        key = str(agent.idx)
        if key not in obs:
            continue
        mask = np.array(obs[key].get("action_mask", np.zeros(50)))
        agent_actions[key] = get_random_valid_action(mask)

    planner = env.world.planner
    dims = planner.action_spaces
    planner_action = [int(np.random.randint(0, int(d))) for d in dims]

    return agent_actions, planner_action


# ──────────────────────────────────────────────────────────────
#  資源鄰近偵測（驗證用）
# ──────────────────────────────────────────────────────────────

def _check_resource_adjacency(
    env,
    obs: dict,
    cfg: AppConfig,
    sim_logger: SimulationLogger,
    agent_actions: dict[str, int],
    agent_thoughts: dict[str, str],
    step: int,
) -> bool:
    """
    檢查是否有任何 agent 在資源旁（Manhattan dist=1）。
    若有，記錄 adjacency event 並回傳 True（觸發地圖截圖）。
    """
    found_any = False

    # 取得 channel 名稱順序
    channel_names: list[str] | None = None
    try:
        channel_names = list(env.world.maps._maps.keys())
    except AttributeError:
        pass

    for persona in cfg.personas:
        agent_key = str(persona.id)
        agent_obs = obs.get(agent_key, {})
        _vm = agent_obs.get("world-map")
        visible_map = _vm if _vm is not None else agent_obs.get("map")
        if visible_map is None:
            continue

        resource_pos = _extract_resource_positions(
            np.array(visible_map), channel_names=channel_names
        )

        for res_type, positions in resource_pos.items():
            for r, c in positions:
                dist = abs(r) + abs(c)
                if dist == 1:
                    direction = _direction_desc(r, c)
                    sim_logger.log_adjacency_event(
                        step=step,
                        agent_id=agent_key,
                        agent_name=persona.display_name,
                        resource_type=res_type,
                        direction=direction,
                        agent_action=agent_actions.get(agent_key),
                        agent_thought=agent_thoughts.get(agent_key, ""),
                    )
                    found_any = True
                    logger.info(
                        f"[Adjacency] Step {step}: Agent {agent_key}（{persona.display_name}）"
                        f"旁有 {res_type}（{direction}），實際動作={agent_actions.get(agent_key)}"
                    )

    return found_any


# ──────────────────────────────────────────────────────────────
#  主模擬迴圈（Ollama 版）
# ──────────────────────────────────────────────────────────────

async def run_episode(
    cfg: AppConfig,
    ollama_url: str,
    ollama_model: str,
    max_steps: int | None = None,
    dry_run: bool = False,
    run_name: str | None = None,
) -> None:
    """
    執行一個完整 Episode（Ollama LLM 版）。

    Args:
        cfg: 全局設定（從 config.yaml 載入）
        ollama_url: Ollama API 服務位址
        ollama_model: 使用的 Ollama 模型
        max_steps: 最大步數
        dry_run: True → 不呼叫 LLM，使用隨機動作
        run_name: 輸出資料夾名稱
    """
    print("\n" + "=" * 60)
    print(" AI Economist — Ollama LLM Simulation")
    print("=" * 60)
    print(f"模型：{ollama_model} @ {ollama_url}")
    print(f"Episode 長度：{cfg.environment.episode_length}")
    print(f"Dry-run：{dry_run}")

    # ── 環境初始化 ──────────────────────────────────────────
    env_config = make_env_config(cfg)
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()
    _apply_age_group_skills(env, cfg)

    print(f"\n[Init] 環境初始化完成，Planner action_spaces={env.world.planner.action_spaces}")

    # ── 初始化 LLM 元件 ────────────────────────────────────
    sim_logger = SimulationLogger(run_name=run_name)

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
        planner = PlannerLLM(
            planner_cfg=cfg.planner,
            llm_client=ollama_client,  # type: ignore[arg-type]
            mem_cfg=cfg.memory,
            tax_period=cfg.planner.tax_period,
            sim_logger=sim_logger,
        )
        print(f"[Init] {len(agents)} 個 Ollama Agent + 1 個 Ollama Planner 已建立")
    else:
        agents = []
        planner = None
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
                agent_actions, planner_brackets = _sample_random_actions(env, obs)
            else:
                # Planner 每步執行（稅收日決策 / 非稅收日 NOOP+觀察）
                planner_brackets = await planner.step(obs=obs, env=env, step=step)

                # Agent 並行決策
                agent_actions = await decide_batch(agents, obs, env, step)

            # 資源鄰近偵測（使用 agent 決策時看到的 obs）
            if not dry_run:
                # 從最近的 thought_logs 中取出本步各 agent 的 thought
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
                    sim_logger.save_map_snapshot(step=step, env=env)

            # 建構 actions dict
            planner_env_action = _build_planner_action(env, planner_brackets)
            actions: dict = {
                **agent_actions,
                env.world.planner.idx: planner_env_action,
            }

            # 環境推進
            obs, rewards, done, info = env.step(actions)

            # 記錄
            sim_logger.log_step(
                step=step,
                rewards=rewards,
                env=env,
                agent_actions=agent_actions,
                planner_action=planner_brackets,
            )
            if planner_brackets is not None:
                sim_logger.log_tax(step, planner_brackets)

            # 每 20 步輸出地圖快照
            if (step + 1) % 20 == 0 or step == 0:
                sim_logger.save_map_snapshot(step=step, env=env)

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

            # 關閉 HTTP session
            if ollama_client:
                await ollama_client.aclose()


# ──────────────────────────────────────────────────────────────
#  CLI 入口
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Economist — Ollama LLM Simulation",
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
        "--config", type=str, default=None,
        help="config.yaml 路徑（預設 llm_agent/config.yaml）",
    )
    parser.add_argument(
        "--ollama-url", type=str, default="http://localhost:11434",
        help="Ollama API 服務位址",
    )
    parser.add_argument(
        "--model", type=str, default="llama3:8b",
        help="Ollama 模型名稱",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="啟用 DEBUG 日誌",
    )
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.debug else logging.INFO)

    cfg = load_config(args.config)

    asyncio.run(
        run_episode(
            cfg=cfg,
            ollama_url=args.ollama_url,
            ollama_model=args.model,
            max_steps=args.steps,
            dry_run=args.dry_run,
            run_name=args.run_name,
        )
    )


if __name__ == "__main__":
    main()
