"""Pair-wise parallel batch experiment runner.

Usage:
    python run_experiment.py              # Run all configured experiments
    python run_experiment.py --dry-run    # Quick test: 5 steps, 1 run each

This batch is configured for the main thesis experiments:
    - 100 runs: gemma4:e2b agents + gpt-4o-mini planner
    - 100 runs: gemma4:e2b agents + random-tax baseline

Runs are launched pair-wise: run i for each experiment starts in parallel,
then the runner waits for both to finish before launching run i+1.
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
        "name": "main_gpt4omini",
        "script": "run_simulation",
        "agent_backend": "ollama",
        "agent_model": "gemma4:e2b",
        "planner_backend": "openai",
        "planner_model": "gpt-4o-mini",
        "steps": 1000,
        "runs": 100,
        "label": "Main planner: Planner=gpt-4o-mini / Agents=gemma4:e2b",
    },
    {
        "name": "main_random_tax",
        "script": "random_tax_simulation",
        "agent_backend": "ollama",
        "agent_model": "gemma4:e2b",
        "planner_model": "random-tax",
        "steps": 1000,
        "runs": 100,
        "label": "Main random-tax baseline: Planner=random-tax / Agents=gemma4:e2b",
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

    if script == "random_tax_simulation":
        cmd = [
            sys.executable,
            str(script_dir / "random_tax_simulation.py"),
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


def _format_elapsed(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def _runs_for(exp: dict, dry_run: bool) -> int:
    return 1 if dry_run else int(exp["runs"])


def _steps_for(exp: dict, dry_run: bool) -> int:
    return 5 if dry_run else int(exp["steps"])


def start_single(
    exp: dict,
    run_index: int,
    steps: int,
    dry_run: bool,
    ollama_url: str,
    script_dir: Path,
    output_dir: Path,
    log_dir: Path,
) -> dict:
    run_name = _build_run_name(exp, run_index)
    cmd = _build_command(exp, steps, dry_run, run_name, ollama_url, script_dir, output_dir)
    log_path = log_dir / f"{run_name}.log"
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    log_file.write(f"$ {' '.join(cmd)}\n\n")
    proc = subprocess.Popen(
        cmd,
        cwd=str(script_dir),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "exp": exp,
        "run_name": run_name,
        "cmd": cmd,
        "proc": proc,
        "log_file": log_file,
        "log_path": log_path,
        "started_at": time.time(),
    }


def wait_single(job: dict) -> tuple[str, int, str, Path]:
    rc = job["proc"].wait()
    elapsed = _format_elapsed(time.time() - job["started_at"])
    job["log_file"].close()
    return job["run_name"], rc, elapsed, job["log_path"]


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
    parser = argparse.ArgumentParser(description="Pair-wise parallel overnight experiment runner")
    parser.add_argument("--dry-run", action="store_true", help="5 steps, 1 run per experiment")
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / "simulation_results" / batch_ts
    log_dir = output_dir / "_batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

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
    total_runs = max(_runs_for(exp, args.dry_run) for exp in EXPERIMENTS)

    print("=" * 60)
    print("  Experiment plan")
    print("=" * 60)
    for exp in EXPERIMENTS:
        print(f"  {exp['label']}  |  {_steps_for(exp, args.dry_run)} steps x {_runs_for(exp, args.dry_run)} runs")
    print(f"  Batch logs: {log_dir}")
    print("=" * 60)

    for i in range(1, total_runs + 1):
        active_exps = [exp for exp in EXPERIMENTS if i <= _runs_for(exp, args.dry_run)]
        print(f"\n=== Pair {i}/{total_runs} | launching {len(active_exps)} run(s) | {timestamp()} ===")

        jobs = []
        for exp in active_exps:
            steps = _steps_for(exp, args.dry_run)
            job = start_single(
                exp=exp,
                run_index=i,
                steps=steps,
                dry_run=args.dry_run,
                ollama_url=args.ollama_url,
                script_dir=script_dir,
                output_dir=output_dir,
                log_dir=log_dir,
            )
            jobs.append(job)
            print(f"  [START] {job['run_name']}")
            print(f"          log: {job['log_path']}")

        for job in jobs:
            run_name, rc, elapsed, log_path = wait_single(job)
            tag = "OK" if rc == 0 else f"FAIL (exit {rc})"
            print(f"  [{tag}] {run_name} | elapsed {elapsed} | log: {log_path}")
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
