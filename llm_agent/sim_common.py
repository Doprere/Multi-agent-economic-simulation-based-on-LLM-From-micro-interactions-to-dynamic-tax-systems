"""
sim_common.py — 抽出 simulation 共用函式，供新統一入口 run_simulation.py 使用。

內容來源：llm_agent/simulation.py 與 project/ollama_simulation.py
（兩者原本 90% 重疊，抽到此處避免漂移）。

提供：
  - _apply_age_group_skills(env, cfg)
  - _validate_action_order(env)
  - _apply_labor_modifier(env)
  - _build_planner_action(env, tax_brackets)
  - _sample_random_actions(env, obs)
  - _check_resource_adjacency(env, obs, cfg, sim_logger, agent_actions, agent_thoughts, step)
"""
from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np

from .action_map import get_random_valid_action
from .config import AppConfig
from .logger import SimulationLogger
from .translator import _direction_desc, _extract_resource_positions

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  年齡族群技能 / 勞動修正器 / 初始 Coin 稟賦注入
# ──────────────────────────────────────────────────────────────

def _apply_age_group_skills(env, cfg: AppConfig) -> None:
    """根據 Persona 的 skill_pareto_alpha、labor_cost_modifier、endowment_coin_* 注入 agent state。

    在 env.reset() 之後呼叫，覆蓋預設的技能抽樣結果，並補上生命週期儲蓄。
    最後重算 optimization metric baseline，避免 step 0 因初始 coin 產生虛假 reward spike。
    """
    build_comp = env.get_component("Build")

    for persona in cfg.personas:
        agent = env.get_agent(str(persona.id))

        # — 建造技能：以 Pareto alpha 重新抽樣 —
        alpha = persona.skill_pareto_alpha
        pmsm = build_comp.payment_max_skill_multiplier

        sampled_skill = np.random.pareto(alpha)
        pay_rate = np.minimum(pmsm, (pmsm - 1) * sampled_skill + 1)
        agent.state["build_payment"] = float(pay_rate * build_comp.payment)
        agent.state["build_skill"] = float(sampled_skill)
        build_comp.sampled_skills[agent.idx] = sampled_skill

        # — 勞動修正器（delta-based scaling 會在每步結束用到）—
        agent.state["labor_cost_modifier"] = persona.labor_cost_modifier
        agent.state["_prev_labor"] = 0.0

        # — 初始 Coin 稟賦（lifecycle savings）—
        lo = persona.endowment_coin_min
        hi = persona.endowment_coin_max
        assert lo >= 0 and hi >= 0 and lo <= hi, (
            f"Invalid endowment range for {persona.name}: [{lo}, {hi}]"
        )
        if hi > 0:
            endowment = float(np.random.uniform(lo, hi))
            agent.state["inventory"]["Coin"] += endowment

        logger.info(
            f"[Init] Agent {persona.id}（{persona.display_name}）"
            f"build_payment={agent.state['build_payment']:.2f}, "
            f"labor_modifier={persona.labor_cost_modifier}, "
            f"initial_coin={agent.state['inventory']['Coin']:.1f}"
        )

    # — Recalculate reward baseline after endowment injection —
    curr = env.get_current_optimization_metrics()
    env.curr_optimization_metric = deepcopy(curr)
    env.init_optimization_metric = deepcopy(curr)
    env.prev_optimization_metric = deepcopy(curr)
    logger.info("[Init] Optimization metric baseline recalculated after endowment injection")


def _validate_action_order(env) -> None:
    """驗證 Foundation 的 action_names 順序與 action_map.py 預期一致（Stone before Wood）。"""
    agent = env.get_agent("0")
    names = agent._action_names
    expected = [
        "Build",
        "ContinuousDoubleAuction.Buy_Stone",
        "ContinuousDoubleAuction.Sell_Stone",
        "ContinuousDoubleAuction.Buy_Wood",
        "ContinuousDoubleAuction.Sell_Wood",
        "Gather",
    ]
    assert names == expected, (
        f"Action name order mismatch! Expected {expected}, got {names}. "
        f"action_map.py may need updating."
    )
    logger.info("[Init] Action order validation passed")


def _apply_labor_modifier(env) -> None:
    """每步結束後依 labor_cost_modifier 調整本步新增 Labor（delta-based，不縮放歷史累積）。"""
    for agent in env.world.agents:
        modifier = agent.state.get("labor_cost_modifier", 1.0)
        if modifier == 1.0:
            continue

        current_labor = agent.state["endogenous"]["Labor"]
        prev_labor = agent.state.get("_prev_labor", 0.0)
        delta = current_labor - prev_labor

        if delta > 0:
            adjusted_delta = delta * modifier
            agent.state["endogenous"]["Labor"] = prev_labor + adjusted_delta

        agent.state["_prev_labor"] = agent.state["endogenous"]["Labor"]


# ──────────────────────────────────────────────────────────────
#  Planner 動作轉換
# ──────────────────────────────────────────────────────────────

def _build_planner_action(env, tax_brackets: list[int] | None) -> list[int]:
    """把 PlannerLLM 回傳的 tax_brackets 轉為環境接受的動作向量（對齊 action_spaces 維度）。"""
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
#  隨機動作（dry-run 模式）
# ──────────────────────────────────────────────────────────────

def _sample_random_actions(env, obs) -> tuple[dict[str, int], list[int]]:
    """為所有 Agent + Planner 生成隨機合法動作（dry-run 用，不呼叫 LLM）。"""
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
#  資源鄰近偵測（診斷 + 觸發地圖截圖）
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
    """檢查是否有任一 agent 在資源旁（Manhattan dist=1）；是→記錄事件並回傳 True 觸發截圖。"""
    found_any = False

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
