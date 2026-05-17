"""Calibration-seeded dynamic Saez baseline runner.

This runner loads random-tax calibration samples as the initial Foundation Saez
global buffer. The formal run then uses Foundation's Saez formula dynamically
from step 0 and updates the buffer after each tax period. It does not use the
cold-start random-tax branch.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "buffer") and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer") and not isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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
from llm_agent.saez_policy import (
    DEFAULT_ELASTICITY,
    build_saez_schedule_from_calibration,
    load_saez_buffer_from_calibration,
)
from ollama_simulation import (
    _apply_age_group_skills,
    _apply_labor_modifier,
    _build_planner_action,
    _check_resource_adjacency,
    _validate_action_order,
)


logger = logging.getLogger(__name__)


def _sample_random_actions(env, obs) -> dict[str, int]:
    agent_actions: dict[str, int] = {}
    for agent in env.world.agents:
        key = str(agent.idx)
        if key not in obs:
            continue
        mask = np.array(obs[key].get("action_mask", np.zeros(50)))
        agent_actions[key] = get_random_valid_action(mask)
    return agent_actions


def _write_saez_audit(
    sim_logger: SimulationLogger,
    preview,
    tax_component,
    income_filter: str,
    initial_buffer_size: int,
) -> None:
    run_dir = sim_logger.output_dir / sim_logger.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    fixed_rates = getattr(tax_component, "fixed_bracket_rates", None)
    audit = preview.to_dict()
    audit.update({
        "runner": "saez_simulation.py",
        "design": "calibration_seeded_dynamic_saez",
        "income_filter": income_filter,
        "initial_global_saez_buffer_size": initial_buffer_size,
        "current_saez_buffer_size": len(tax_component.saez_buffer),
        "local_saez_buffer_size": len(tax_component.get_local_saez_buffer()),
        "reached_min_samples": bool(getattr(tax_component, "_reached_min_samples", False)),
        "fixed_elasticity": getattr(tax_component, "_saez_fixed_elas", None),
        "foundation_tax_model": getattr(tax_component, "tax_model", ""),
        "foundation_fixed_bracket_rates": [
            float(v) for v in fixed_rates
        ] if fixed_rates is not None else [],
        "foundation_current_marginal_rates": [
            float(v) for v in tax_component.curr_marginal_rates
        ],
    })
    with (run_dir / "saez_schedule_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)


async def run_episode(
    config_path: str,
    calibration_csv: str,
    ollama_url: str,
    ollama_model: str,
    elasticity: float = DEFAULT_ELASTICITY,
    income_filter: str = "full",
    max_steps: int | None = None,
    dry_run: bool = False,
    run_name: str | None = None,
    output_dir: str = "simulation_results",
) -> None:
    import llm_agent.config as _cfg_mod
    _cfg_mod._CONFIG = None
    cfg = load_config(config_path)

    preview = build_saez_schedule_from_calibration(
        calibration_csv=calibration_csv,
        elasticity=elasticity,
        income_filter=income_filter,
    )
    initial_saez_buffer = load_saez_buffer_from_calibration(
        calibration_csv=calibration_csv,
        income_filter=income_filter,
    )

    print("\n" + "=" * 60)
    print(" AI Economist - Calibration-Seeded Dynamic Saez Experiment")
    print("=" * 60)
    print(f"Agent model: {ollama_model} @ {ollama_url}")
    print(f"Planner: dynamic Saez seeded from calibration | e={elasticity} | filter={income_filter}")
    print(f"Calibration CSV: {calibration_csv}")
    print(f"Initial calibration buffer rows: {len(initial_saez_buffer)}")
    print(f"Preview initial Saez rates: {[round(float(r), 4) for r in preview.clipped_bracket_rates]}")
    print(f"Episode length: {cfg.environment.episode_length}")
    print(f"Dry-run: {dry_run}")

    env_config = make_env_config(cfg)
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()
    _apply_age_group_skills(env, cfg)
    _validate_action_order(env)

    tax_component = env.get_component("PeriodicBracketTax")
    assert tax_component.tax_model == "saez", (
        f"Expected tax_model='saez', got {tax_component.tax_model!r}"
    )
    assert getattr(tax_component, "_saez_fixed_elas", None) == float(elasticity), (
        "Config elasticity and CLI elasticity differ. Update config_saez.yaml or "
        "run with --elasticity matching the config."
    )
    tax_component.set_global_saez_buffer(initial_saez_buffer)

    sim_logger = SimulationLogger(output_dir=output_dir, run_name=run_name)
    _write_saez_audit(
        sim_logger,
        preview,
        tax_component,
        income_filter,
        len(initial_saez_buffer),
    )

    ollama_client: OllamaClient | None = None
    if not dry_run:
        ollama_client = OllamaClient(
            model=ollama_model,
            base_url=ollama_url,
            max_retries=cfg.llm.max_retries,
            temperature=cfg.llm.temperature,
        )
        agents = [
            MobileAgentLLM(persona, ollama_client, cfg.memory, sim_logger=sim_logger)
            for persona in cfg.personas
        ]
        print(f"[Init] {len(agents)} Ollama agents ready; no LLM planner.")
    else:
        agents = []
        print("[Init] Dry-run mode: random valid agent actions, no LLM calls.")

    episode_length = min(
        max_steps or cfg.environment.episode_length,
        cfg.environment.episode_length,
    )
    print(f"[Start] Running simulation for {episode_length} steps")

    step = 0
    try:
        for step in range(episode_length):
            if dry_run:
                agent_actions = _sample_random_actions(env, obs)
            else:
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
                    env=env,
                    obs=obs,
                    cfg=cfg,
                    sim_logger=sim_logger,
                    agent_actions=agent_actions,
                    agent_thoughts=recent_thoughts,
                    step=step,
                )
                if has_adjacent:
                    should_snapshot = True

            if should_snapshot:
                sim_logger.save_map_snapshot(step=step, env=env)

            planner_env_action = _build_planner_action(env, None)
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
                planner_action=None,
            )

            if step % tax_component.period == 0:
                sim_logger.log_tax(
                    step,
                    [round(float(r), 4) for r in tax_component.curr_marginal_rates],
                )

            if (step + 1) % 100 == 0:
                sim_logger.save()
                _write_saez_audit(
                    sim_logger,
                    preview,
                    tax_component,
                    income_filter,
                    len(initial_saez_buffer),
                )
                print(f"[Checkpoint] Step {step + 1} intermediate results saved")

            if done.get("__all__", False):
                print(f"\n[Done] Episode ended at step {step} (all done)")
                break

    finally:
        print(f"\n[End] Simulation completed: {step + 1} steps")
        sim_logger.save()
        _write_saez_audit(
            sim_logger,
            preview,
            tax_component,
            income_filter,
            len(initial_saez_buffer),
        )

        if not dry_run:
            pending = [
                a._consolidation_task
                for a in agents
                if a._consolidation_task and not a._consolidation_task.done()
            ]
            if pending:
                print(f"[Cleanup] Waiting for {len(pending)} memory tasks...")
                await asyncio.gather(*pending, return_exceptions=True)

            if ollama_client:
                await ollama_client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Economist - Calibration-Seeded Dynamic Saez Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="simulation_results")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_PROJECT_ROOT / "llm_agent" / "config_saez.yaml"),
    )
    parser.add_argument("--calibration-csv", type=str, required=True)
    parser.add_argument("--elasticity", type=float, default=DEFAULT_ELASTICITY)
    parser.add_argument(
        "--income-filter",
        choices=["full", "nonnegative", "positive"],
        default="full",
    )
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--model", type=str, default="gemma4:e2b")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.debug else logging.INFO)

    asyncio.run(
        run_episode(
            config_path=args.config,
            calibration_csv=args.calibration_csv,
            ollama_url=args.ollama_url,
            ollama_model=args.model,
            elasticity=args.elasticity,
            income_filter=args.income_filter,
            max_steps=args.steps,
            dry_run=args.dry_run,
            run_name=args.run_name,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
