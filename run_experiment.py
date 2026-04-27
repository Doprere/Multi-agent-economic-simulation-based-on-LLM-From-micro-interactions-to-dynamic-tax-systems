"""Overnight batch experiment runner.

Usage:
    python run_experiment.py              # Run all configured experiments
    python run_experiment.py --dry-run    # Quick test: 5 steps, 1 run each

This batch is configured for the main thesis experiments:
    - 100 runs: gemma4:e2b agents + gpt-5.4-mini planner
    - 100 runs: gemma4:e2b agents + Saez planner
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


EXPERIMENTS: list[dict] = [
    {
        "name": "main_gpt54mini",
        "script": "run_simulation",
        "agent_backend": "ollama",
        "agent_model": "gemma4:e2b",
        "planner_backend": "openai",
        "planner_model": "gpt-5.4-mini",
        "steps": 1000,
        "runs": 100,
        "label": "Main planner: Planner=gpt-5.4-mini / Agents=gemma4:e2b",
    },
    {
        "name": "main_saez",
        "script": "saez_simulation",
        "agent_backend": "ollama",
        "agent_model": "gemma4:e2b",
        "planner_model": "saez",
        "steps": 1000,
        "runs": 100,
        "label": "Main Saez baseline: Planner=Saez / Agents=gemma4:e2b",
    },
]

OLLAMA_URL = "http://localhost:11434"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def check_ollama(url: str) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=5):
            return True
    except Exception:
        return False


def _sanitize(s: str) -> str:
    return s.replace(":", "_").replace(".", "_").replace("/", "_")


def _build_run_name(exp: dict, i: int) -> str:
    agent_tag = _sanitize(exp["agent_model"])
    planner_tag = _sanitize(exp.get("planner_model", exp["agent_model"]))
    return f"{exp['name']}_planner_{planner_tag}_agent_{agent_tag}_run{i}"


def _build_command(
    exp: dict,
    steps: int,
    dry_run: bool,
    run_name: str,
    ollama_url: str,
    script_dir: Path,
    output_dir: Path,
) -> list[str]:
    script = exp.get("script", "run_simulation")

    if script == "saez_simulation":
        cmd = [
            sys.executable,
            str(script_dir / "saez_simulation.py"),
            "--steps",
            str(steps),
            "--model",
            exp["agent_model"],
            "--run-name",
            run_name,
            "--output-dir",
            str(output_dir),
        ]
    elif script == "run_simulation":
        cmd = [
            sys.executable,
            str(script_dir / "run_simulation.py"),
            "--steps",
            str(steps),
            "--agent-backend",
            exp["agent_backend"],
            "--agent-model",
            exp["agent_model"],
            "--run-name",
            run_name,
            "--output-dir",
            str(output_dir),
        ]
        if exp.get("planner_backend"):
            cmd += ["--planner-backend", exp["planner_backend"]]
        if exp.get("planner_model"):
            cmd += ["--planner-model", exp["planner_model"]]
    else:
        raise ValueError(f"Unknown experiment script: {script}")

    needs_ollama = exp["agent_backend"] == "ollama" or exp.get("planner_backend") == "ollama"
    if needs_ollama:
        cmd += ["--ollama-url", ollama_url]
    if dry_run:
        cmd += ["--dry-run"]

    return cmd


def run_single(
    exp: dict,
    steps: int,
    dry_run: bool,
    run_name: str,
    ollama_url: str,
    script_dir: Path,
    output_dir: Path,
) -> int:
    cmd = _build_command(exp, steps, dry_run, run_name, ollama_url, script_dir, output_dir)
    print(f"  $ {' '.join(cmd)}")
    print()
    proc = subprocess.Popen(cmd, cwd=str(script_dir), stdout=sys.stdout, stderr=sys.stderr)
    proc.wait()
    return proc.returncode


def _needs_ollama(experiments: list[dict]) -> bool:
    return any(
        e["agent_backend"] == "ollama" or e.get("planner_backend") == "ollama"
        for e in experiments
    )


def _needs_openai(experiments: list[dict]) -> bool:
    return any(
        e["agent_backend"] == "openai" or e.get("planner_backend") == "openai"
        for e in experiments
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential overnight experiment runner")
    parser.add_argument("--dry-run", action="store_true", help="5 steps, 1 run per experiment")
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / "simulation_results" / batch_ts

    print("=" * 60)
    print("  AI Economist - Overnight Experiment Runner")
    print(f"  Started: {timestamp()}")
    print(f"  Output:  {output_dir}")
    if args.dry_run:
        print("  *** DRY-RUN MODE ***")
    print("=" * 60)
    print()

    if not args.dry_run and _needs_ollama(EXPERIMENTS):
        if not check_ollama(args.ollama_url):
            print(f"[ERROR] Ollama not reachable at {args.ollama_url}")
            sys.exit(1)
        print(f"[OK] Ollama is running at {args.ollama_url}")

    if not args.dry_run and _needs_openai(EXPERIMENTS):
        if not os.environ.get("OPENAI_API_KEY"):
            print("[ERROR] OPENAI_API_KEY not set in environment")
            sys.exit(1)
        print("[OK] OPENAI_API_KEY is set")

    print()

    results: list[tuple[str, int]] = []
    total_start = time.time()

    for exp in EXPERIMENTS:
        runs = 1 if args.dry_run else exp["runs"]
        steps = 5 if args.dry_run else exp["steps"]

        print("=" * 60)
        print(f"  {exp['label']}  |  {steps} steps x {runs} runs")
        print("=" * 60)

        for i in range(1, runs + 1):
            run_name = _build_run_name(exp, i)
            print(f"\n--- Run {i}/{runs} | {run_name} | {timestamp()} ---")

            t0 = time.time()
            rc = run_single(exp, steps, args.dry_run, run_name, args.ollama_url, script_dir, output_dir)
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))

            tag = "OK" if rc == 0 else f"FAIL (exit {rc})"
            print(f"\n--- {tag} | elapsed {elapsed} | {timestamp()} ---\n")
            results.append((run_name, rc))

    total_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - total_start))
    print("=" * 60)
    print(f"  All done at {timestamp()}  |  Total: {total_str}")
    print(f"  Results saved to: {output_dir}")
    print("=" * 60)
    for name, rc in results:
        print(f"  [{'OK' if rc == 0 else 'FAIL'}] {name}")

    failed = sum(1 for _, rc in results if rc != 0)
    if failed:
        print(f"\n  WARNING: {failed} run(s) failed!")
    print()


if __name__ == "__main__":
    main()
