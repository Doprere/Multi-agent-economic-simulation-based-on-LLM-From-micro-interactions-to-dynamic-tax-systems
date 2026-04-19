"""
run_simulation.py — AI Economist LLM 模擬統一入口

════════════════════════════════════════════════════════════════════════════
  使用快速入門
════════════════════════════════════════════════════════════════════════════

1. 最簡單：依 config.yaml 的 `llm:` 區塊決定 backend（預設 OpenAI）

     $env:OPENAI_API_KEY = "sk-proj-你的key"     # Windows PowerShell
     python run_simulation.py --steps 200 --run-name baseline

2. 只用本地 Ollama（Agents + Planner 共用）

     # 先啟動 Ollama 並拉模型：ollama pull llama3:8b
     python run_simulation.py --steps 200 \
       --agent-backend ollama --agent-model llama3:8b \
       --ollama-url http://localhost:11434 \
       --run-name all_ollama

3. 混搭（推薦的不對稱實驗）：Planner 用雲端強模型，4 個 Agent 用本地弱模型

     python run_simulation.py --steps 200 \
       --agent-backend   ollama --agent-model   gemma2:2b \
       --planner-backend openai --planner-model gpt-4o-mini \
       --ollama-url http://localhost:11434 \
       --run-name mix_gemma_gpt4o

4. 不呼叫 LLM 的環境煙測（隨機合法動作）

     python run_simulation.py --dry-run --steps 5 --run-name dry_smoke

════════════════════════════════════════════════════════════════════════════
  換模型 / 換 backend 的優先順序
════════════════════════════════════════════════════════════════════════════

  CLI flag  >  config.yaml 的 `llm_planner:` / `llm:`  >  dataclass 預設值

  - 若沒給 `--planner-backend` / `--planner-model`，且 config 無 `llm_planner:`
    區塊 → Planner 與 Agents **共用同一 client 實例**（向後相容，與舊版行為一致）。
  - 若要讓 Planner 走獨立設定但不想動 CLI，可直接在 config.yaml 取消註解
    `llm_planner:` 區塊填寫即可。

════════════════════════════════════════════════════════════════════════════
  風險與建議
════════════════════════════════════════════════════════════════════════════

  - Planner 用 Ollama 時 `tax_brackets` 的 regex 解析較易失敗（沒有 mask 可
    fallback），建議 Planner 優先選 OpenAI；Agents 用 Ollama 相對安全（有
    action_mask 隨機 fallback）。
  - token_usage 的 summary.json 格式為 nested：
        {"agents": {...}, "planner": {...}}
    共用 client 時 `planner` 會標註 `"shared_with_agents": true`。

════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
from pathlib import Path

# 強制 UTF-8 stdout/stderr（Windows 中文環境）
# 檢查既有 encoding 避免重複 wrap —— 被 deprecation stub 委派時，stub 已 wrap 過，
# 再 wrap 一次會讓舊 wrapper GC 時關閉底層 buffer，導致後續 print 炸 I/O closed。
if hasattr(sys.stdout, "buffer") and (sys.stdout.encoding or "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer") and (sys.stderr.encoding or "").lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))
_AI_ECON_PKG = _PROJECT_ROOT / "ai_economist"
if _AI_ECON_PKG.is_dir():
    sys.path.insert(0, str(_AI_ECON_PKG))

from ai_economist import foundation

from llm_agent.agent import MobileAgentLLM, decide_batch
from llm_agent.client_factory import make_llm_client
from llm_agent.config import AppConfig, LLMConfig, load_config, make_env_config
from llm_agent.logger import SimulationLogger, setup_logging
from llm_agent.planner import PlannerLLM
from llm_agent.sim_common import (
    _apply_age_group_skills,
    _apply_labor_modifier,
    _build_planner_action,
    _check_resource_adjacency,
    _sample_random_actions,
    _validate_action_order,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  CLI override 套用
# ──────────────────────────────────────────────────────────────

def _apply_cli_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    """將 CLI 參數覆蓋到 cfg 上（CLI > config.yaml > 預設值）。"""

    # Agents 的 LLM 設定
    if args.agent_backend:
        cfg.llm.backend = args.agent_backend
    if args.agent_model:
        cfg.llm.model = args.agent_model
    if args.ollama_url and cfg.llm.backend == "ollama":
        cfg.llm.base_url = args.ollama_url

    # Planner 的 LLM 設定（若指定任一 Planner 參數 → 建立 / 使用 llm_planner）
    planner_cli_given = any([args.planner_backend, args.planner_model])
    if planner_cli_given:
        if cfg.llm_planner is None:
            # 以 agents 設定作為起點 clone；base_url 不繼承（避免跨 backend 沾染），
            # 由後續 backend 判定分派。
            cfg.llm_planner = LLMConfig(
                backend=cfg.llm.backend,
                model=cfg.llm.model,
                base_url="",  # 後面依 backend 填合適預設
                api_key=cfg.llm.api_key,
                max_retries=cfg.llm.max_retries,
                temperature=cfg.llm.temperature,
                timeout=cfg.llm.timeout,
            )
        if args.planner_backend:
            cfg.llm_planner.backend = args.planner_backend
        if args.planner_model:
            cfg.llm_planner.model = args.planner_model

        # base_url 分派：
        #   - ollama → --ollama-url（若給）或預設 http://localhost:11434
        #   - openai → 留空讓 LLMClient fallback 到 OPENAI_BASE_URL / api.openai.com
        if cfg.llm_planner.backend == "ollama":
            cfg.llm_planner.base_url = args.ollama_url or "http://localhost:11434"
        elif cfg.llm_planner.backend == "openai":
            # 若 base_url 是 ollama 風格（localhost:11434）則清掉
            if "11434" in (cfg.llm_planner.base_url or ""):
                cfg.llm_planner.base_url = ""

    return cfg


# ──────────────────────────────────────────────────────────────
#  主模擬迴圈
# ──────────────────────────────────────────────────────────────

async def run_episode(
    cfg: AppConfig,
    max_steps: int | None = None,
    dry_run: bool = False,
    run_name: str | None = None,
    output_dir: str = "simulation_results",
) -> None:
    """執行一個完整 Episode。

    若 cfg.llm_planner 為 None → Planner 與 Agents 共用同一 client。
    若 cfg.llm_planner 存在    → Planner 使用獨立 client（可跨 backend）。
    """
    # ── 解析實際步數（CLI --steps 覆蓋 config）──────────────
    episode_length = min(
        max_steps or cfg.environment.episode_length,
        cfg.environment.episode_length,
    )

    # ── 頭部資訊 ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" AI Economist LLM Simulation (unified entry)")
    print("=" * 60)
    print(f"Agents  backend/model: {cfg.llm.backend} / {cfg.llm.model}")
    if cfg.llm_planner is None:
        print("Planner backend/model: (shared with agents)")
    else:
        print(f"Planner backend/model: {cfg.llm_planner.backend} / {cfg.llm_planner.model}")
    if max_steps and max_steps < cfg.environment.episode_length:
        print(f"Episode 長度：{episode_length}（--steps 覆蓋 config {cfg.environment.episode_length}）")
    else:
        print(f"Episode 長度：{episode_length}")
    print(f"Dry-run：{dry_run}")

    # ── 環境初始化 ──────────────────────────────────────────
    env_config = make_env_config(cfg)
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()
    _apply_age_group_skills(env, cfg)
    _validate_action_order(env)
    print(f"\n[Init] 環境初始化完成，Planner action_spaces={env.world.planner.action_spaces}")

    # ── 初始化 LLM client（雙 client 或共用）────────────────
    sim_logger = SimulationLogger(output_dir=output_dir, run_name=run_name)

    agent_client = None
    planner_client = None
    shared_client = False

    if not dry_run:
        agent_client = make_llm_client(cfg.llm)
        if cfg.llm_planner is None:
            planner_client = agent_client
            shared_client = True
        else:
            planner_client = make_llm_client(cfg.llm_planner)

        agents = [
            MobileAgentLLM(
                persona,
                agent_client,  # type: ignore[arg-type]
                cfg.memory,
                sim_logger=sim_logger,
            )
            for persona in cfg.personas
        ]
        planner = PlannerLLM(
            planner_cfg=cfg.planner,
            llm_client=planner_client,  # type: ignore[arg-type]
            mem_cfg=cfg.memory,
            tax_period=cfg.planner.tax_period,
            sim_logger=sim_logger,
        )
        mode_desc = "共用同一 client" if shared_client else "獨立 client（混搭）"
        print(f"[Init] {len(agents)} 個 Agent + 1 個 Planner 已建立（{mode_desc}）")
    else:
        agents = []
        planner = None
        print("[Init] Dry-run 模式：使用隨機動作，不呼叫 LLM")

    # ── 主迴圈 ──────────────────────────────────────────────
    print(f"[Start] 開始模擬，共 {episode_length} 步\n")

    step = 0
    try:
        for step in range(episode_length):

            if dry_run:
                agent_actions, planner_brackets = _sample_random_actions(env, obs)
            else:
                planner_brackets = await planner.step(obs=obs, env=env, step=step)
                agent_actions = await decide_batch(agents, obs, env, step)

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

            planner_env_action = _build_planner_action(env, planner_brackets)
            actions: dict = {
                **agent_actions,
                env.world.planner.idx: planner_env_action,
            }

            obs, rewards, done, info = env.step(actions)
            _apply_labor_modifier(env)

            sim_logger.log_step(
                step=step,
                rewards=rewards,
                env=env,
                agent_actions=agent_actions,
                planner_action=planner_brackets,
            )
            if planner_brackets is not None:
                sim_logger.log_tax(step, planner_brackets)

            if (step + 1) % 100 == 0 and not dry_run:
                sim_logger.save()
                print(f"[Checkpoint] Step {step+1} intermediate results saved")

            if done.get("__all__", False):
                print(f"\n[Done] Episode 在步驟 {step} 結束（all done）")
                break

    finally:
        print(f"\n[End] 模擬完成！共執行 {step + 1} 步")
        sim_logger.save()

        if not dry_run:
            _write_token_usage_summary(
                sim_logger=sim_logger,
                agent_client=agent_client,
                planner_client=planner_client,
                agent_cfg=cfg.llm,
                planner_cfg=cfg.llm_planner,
                shared=shared_client,
            )

            # 等待背景記憶彙整任務
            pending = [
                a._consolidation_task
                for a in agents
                if a._consolidation_task and not a._consolidation_task.done()
            ]
            if pending:
                print(f"[Cleanup] 等待 {len(pending)} 個記憶彙整任務完成...")
                await asyncio.gather(*pending, return_exceptions=True)

            # 關閉 client（去重：共用時只關一次）
            await _close_clients_once(agent_client, planner_client)


# ──────────────────────────────────────────────────────────────
#  Token usage summary 寫入
# ──────────────────────────────────────────────────────────────

def _write_token_usage_summary(
    sim_logger: SimulationLogger,
    agent_client,
    planner_client,
    agent_cfg: LLMConfig,
    planner_cfg: LLMConfig | None,
    shared: bool,
) -> None:
    """把 token usage 以 nested 格式寫進 summary.json，並在終端列印。"""
    import json as _json

    agent_usage = agent_client.get_token_usage() if agent_client else {}
    agent_usage = {**agent_usage, "backend": agent_cfg.backend, "model": agent_cfg.model}

    if shared:
        planner_usage = {"shared_with_agents": True}
    else:
        planner_usage = planner_client.get_token_usage() if planner_client else {}
        if planner_cfg:
            planner_usage = {
                **planner_usage,
                "backend": planner_cfg.backend,
                "model": planner_cfg.model,
            }

    print("\n" + "─" * 50)
    print(" Token 用量摘要")
    print("─" * 50)
    print(f"  [Agents]  {agent_cfg.backend}/{agent_cfg.model}")
    print(f"    api_calls       : {agent_usage.get('api_calls', 0):>10,}")
    print(f"    prompt_tokens   : {agent_usage.get('prompt_tokens', 0):>10,}")
    print(f"    completion_tokens: {agent_usage.get('completion_tokens', 0):>10,}")
    print(f"    total_tokens    : {agent_usage.get('total_tokens', 0):>10,}")

    if shared:
        print("  [Planner] shared with agents (counts already included above)")
    elif planner_cfg:
        print(f"  [Planner] {planner_cfg.backend}/{planner_cfg.model}")
        print(f"    api_calls       : {planner_usage.get('api_calls', 0):>10,}")
        print(f"    prompt_tokens   : {planner_usage.get('prompt_tokens', 0):>10,}")
        print(f"    completion_tokens: {planner_usage.get('completion_tokens', 0):>10,}")
        print(f"    total_tokens    : {planner_usage.get('total_tokens', 0):>10,}")
    print("─" * 50)

    summary_path = sim_logger.output_dir / sim_logger.run_name / "summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = _json.load(f)
        summary["token_usage"] = {
            "agents": agent_usage,
            "planner": planner_usage,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            _json.dump(summary, f, ensure_ascii=False, indent=2)


async def _close_clients_once(*clients) -> None:
    """對有 aclose() 的 client 呼叫關閉；用 id() 去重，避免共用實例被關兩次。"""
    seen: set[int] = set()
    for c in clients:
        if c is None:
            continue
        if id(c) in seen:
            continue
        seen.add(id(c))
        aclose = getattr(c, "aclose", None)
        if aclose is None:
            continue
        try:
            await aclose()
        except Exception as exc:
            logger.warning(f"[Cleanup] Client aclose failed: {exc}")


# ──────────────────────────────────────────────────────────────
#  CLI 入口
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Economist LLM Simulation — unified entry (OpenAI / Ollama / mix)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 通用選項
    parser.add_argument("--steps", type=int, default=None, help="最大模擬步數（覆蓋 config）")
    parser.add_argument("--dry-run", action="store_true", help="不呼叫 LLM，使用隨機動作")
    parser.add_argument("--run-name", type=str, default=None, help="輸出資料夾名稱（自動加時戳）")
    parser.add_argument("--output-dir", type=str, default="simulation_results", help="輸出根目錄")
    parser.add_argument("--config", type=str, default=None, help="config.yaml 路徑")
    parser.add_argument("--debug", action="store_true", help="啟用 DEBUG 日誌")

    # LLM backend / model overrides
    parser.add_argument(
        "--agent-backend", type=str, default=None, choices=["openai", "ollama"],
        help="Agents 的 LLM backend（覆蓋 config.llm.backend）",
    )
    parser.add_argument(
        "--agent-model", type=str, default=None,
        help="Agents 的模型名稱（覆蓋 config.llm.model）",
    )
    parser.add_argument(
        "--planner-backend", type=str, default=None, choices=["openai", "ollama"],
        help="Planner 的 LLM backend；指定後 Planner 與 Agents 不共用 client",
    )
    parser.add_argument(
        "--planner-model", type=str, default=None,
        help="Planner 的模型名稱",
    )
    parser.add_argument(
        "--ollama-url", type=str, default=None,
        help="Ollama API 服務位址（backend=ollama 時生效）",
    )

    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.debug else logging.INFO)

    cfg = load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)

    asyncio.run(
        run_episode(
            cfg=cfg,
            max_steps=args.steps,
            dry_run=args.dry_run,
            run_name=args.run_name,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
