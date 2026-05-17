"""Run calibration-seeded dynamic Saez baseline episodes.

Each run starts from the same calibration CSV and then evolves its own Saez
buffer during that episode. Buffers are intentionally not shared across runs,
so repeated runs remain independent samples under the same initial policy
calibration.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from run_experiment import check_ollama


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_CALIBRATION_CSV = (
    "simulation_results/random_calibration_20260515_090034/calibration_samples.csv"
)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_elapsed(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibration-seeded dynamic Saez batch runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--model", type=str, default="gemma4:e2b")
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--elasticity", type=float, default=0.4)
    parser.add_argument(
        "--income-filter",
        choices=["full", "nonnegative", "positive"],
        default="full",
    )
    parser.add_argument(
        "--calibration-csv",
        type=Path,
        default=Path(DEFAULT_CALIBRATION_CSV),
        help="Calibration CSV used to seed every independent Saez run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("simulation_results"),
        help="Root directory for Saez batch output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one 105-step dry-run with random valid agent actions.",
    )
    return parser.parse_args()


def build_command(
    script_dir: Path,
    output_dir: Path,
    run_index: int,
    steps: int,
    model: str,
    ollama_url: str,
    calibration_csv: Path,
    elasticity: float,
    income_filter: str,
    dry_run: bool,
) -> tuple[str, list[str], Path]:
    run_name = f"saez_calibrated_run{run_index}"
    cmd = [
        sys.executable,
        str(script_dir / "saez_simulation.py"),
        "--steps",
        str(steps),
        "--calibration-csv",
        str(calibration_csv),
        "--ollama-url",
        ollama_url,
        "--model",
        model,
        "--elasticity",
        str(elasticity),
        "--income-filter",
        income_filter,
        "--run-name",
        run_name,
        "--output-dir",
        str(output_dir),
    ]
    if dry_run:
        cmd.append("--dry-run")
    return run_name, cmd, output_dir / "_batch_logs" / f"{run_name}.log"


def start_job(run_name: str, cmd: list[str], log_path: Path, cwd: Path) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    log_file.write(f"$ {' '.join(cmd)}\n\n")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "run_name": run_name,
        "cmd": cmd,
        "proc": proc,
        "log_file": log_file,
        "log_path": log_path,
        "started_at": time.time(),
    }


def main() -> None:
    args = parse_args()
    runs = 1 if args.dry_run else args.runs
    steps = 105 if args.dry_run else args.steps
    parallelism = max(1, int(args.parallelism))

    script_dir = Path(__file__).resolve().parent
    calibration_csv = args.calibration_csv
    if not calibration_csv.is_absolute():
        calibration_csv = script_dir / calibration_csv
    if not calibration_csv.exists():
        print(f"[ERROR] Calibration CSV not found: {calibration_csv}")
        sys.exit(1)

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = script_dir / output_root
    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"saez_calibrated_{batch_ts}"

    print("=" * 60)
    print("  AI Economist - Calibration-Seeded Dynamic Saez Runner")
    print(f"  Started: {timestamp()}")
    print(f"  Output:  {output_dir}")
    print(f"  Plan:    {runs} runs x {steps} steps | parallelism={parallelism}")
    print(f"  Agent:   {args.model}")
    print(f"  Saez:    e={args.elasticity} | income_filter={args.income_filter}")
    print(f"  Seed CSV:{calibration_csv}")
    print("  Buffer:  reset to the same calibration CSV at the start of every run")
    if args.dry_run:
        print("  *** DRY-RUN MODE ***")
    print("=" * 60)

    if not args.dry_run and not check_ollama(args.ollama_url):
        print(f"[ERROR] Ollama not reachable at {args.ollama_url}")
        sys.exit(1)

    pending_run_indices = list(range(1, runs + 1))
    active: list[dict] = []
    results: list[tuple[str, int, str, Path]] = []
    total_start = time.time()

    while pending_run_indices or active:
        while pending_run_indices and len(active) < parallelism:
            run_index = pending_run_indices.pop(0)
            run_name, cmd, log_path = build_command(
                script_dir=script_dir,
                output_dir=output_dir,
                run_index=run_index,
                steps=steps,
                model=args.model,
                ollama_url=args.ollama_url,
                calibration_csv=calibration_csv,
                elasticity=args.elasticity,
                income_filter=args.income_filter,
                dry_run=args.dry_run,
            )
            job = start_job(run_name, cmd, log_path, script_dir)
            active.append(job)
            print(f"[START] {run_name} | log: {log_path}")

        still_active = []
        for job in active:
            rc = job["proc"].poll()
            if rc is not None:
                elapsed = format_elapsed(time.time() - job["started_at"])
                job["log_file"].close()
                tag = "OK" if rc == 0 else f"FAIL exit={rc}"
                print(f"[{tag}] {job['run_name']} | elapsed {elapsed}")
                results.append((job["run_name"], rc, elapsed, job["log_path"]))
            else:
                still_active.append(job)
        active = still_active
        if active:
            time.sleep(1)

    failed = sum(1 for _, rc, _, _ in results if rc != 0)
    manifest = {
        "created_at": timestamp(),
        "runs_requested": runs,
        "runs_completed": len(results) - failed,
        "runs_failed": failed,
        "steps": steps,
        "model": args.model,
        "ollama_url": args.ollama_url,
        "elasticity": args.elasticity,
        "income_filter": args.income_filter,
        "parallelism": parallelism,
        "calibration_csv": str(calibration_csv),
        "buffer_policy": "reset_same_calibration_csv_per_run",
        "run_logs": [str(log_path) for _, _, _, log_path in results],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "saez_batch_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"  All done at {timestamp()} | Total: {format_elapsed(time.time() - total_start)}")
    print(f"  Results saved to: {output_dir}")
    print(f"  Failed runs: {failed}")
    print("=" * 60)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
