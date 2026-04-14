"""
run_experiment.py — Overnight batch experiment runner.

Usage:
    python run_experiment.py              # Run all experiments
    python run_experiment.py --dry-run    # Quick test (5 steps, 1 run each)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

EXPERIMENTS: list[dict] = [
    {
        "name": "expC",
        "model": "llama3.1:8b",
        "steps": 1000,
        "runs": 1,
        "label": "Experiment C: llama3.1:8b — 1000 steps",
    },
    {
        "name": "expG",
        "model": "gemma4:e2b",
        "steps": 1000,
        "runs": 1,
        "label": "Experiment G: gemma4:e2b — 1000 steps",
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


def run_single(
    model: str, steps: int, run_name: str, ollama_url: str, script_dir: Path,
    output_dir: Path,
) -> int:
    cmd = [
        sys.executable,
        str(script_dir / "ollama_simulation.py"),
        "--steps", str(steps),
        "--model", model,
        "--ollama-url", ollama_url,
        "--run-name", run_name,
        "--output-dir", str(output_dir),
    ]
    print(f"  $ {' '.join(cmd)}")
    print()
    proc = subprocess.Popen(cmd, cwd=str(script_dir), stdout=sys.stdout, stderr=sys.stderr)
    proc.wait()
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight experiment runner")
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

    if not check_ollama(args.ollama_url):
        print(f"[ERROR] Ollama not reachable at {args.ollama_url}")
        sys.exit(1)
    print(f"[OK] Ollama is running at {args.ollama_url}\n")

    results: list[tuple[str, int]] = []
    total_start = time.time()

    for exp in EXPERIMENTS:
        runs = 1 if args.dry_run else exp["runs"]
        steps = 5 if args.dry_run else exp["steps"]

        print("=" * 60)
        print(f"  {exp['label']}  |  {steps} steps x {runs} runs")
        print("=" * 60)

        for i in range(1, runs + 1):
            run_name = f"{exp['name']}_{exp['model'].replace(':', '_').replace('.', '_')}_run{i}"
            print(f"\n--- Run {i}/{runs} | {run_name} | {timestamp()} ---")

            t0 = time.time()
            rc = run_single(exp["model"], steps, run_name, args.ollama_url, script_dir, output_dir)
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
