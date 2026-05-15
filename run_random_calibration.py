"""Run random-tax calibration episodes and merge period-income samples.

This script is intentionally separate from the main experiment runner. It only
collects empirical income samples for the future calibrated Saez planner.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from llm_agent.calibration_logger import CALIBRATION_FIELDS
from run_experiment import check_ollama


DEFAULT_OLLAMA_URL = "http://localhost:11434"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_elapsed(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random-tax calibration sample runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--model", type=str, default="gemma4:e2b")
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("simulation_results"),
        help="Root directory for calibration batch output.",
    )
    parser.add_argument("--dry-run", action="store_true", help="105 steps, 1 random-action run")
    return parser.parse_args()


def build_command(
    script_dir: Path,
    output_dir: Path,
    sample_dir: Path,
    run_index: int,
    steps: int,
    model: str,
    ollama_url: str,
    dry_run: bool,
) -> tuple[str, list[str], Path, Path]:
    run_name = f"random_tax_calibration_run{run_index}"
    calibration_csv = sample_dir / f"{run_name}_samples.csv"
    cmd = [
        sys.executable,
        str(script_dir / "random_tax_simulation.py"),
        "--steps",
        str(steps),
        "--model",
        model,
        "--ollama-url",
        ollama_url,
        "--run-name",
        run_name,
        "--output-dir",
        str(output_dir),
        "--calibration-csv",
        str(calibration_csv),
    ]
    if dry_run:
        cmd.append("--dry-run")
    return run_name, cmd, calibration_csv, output_dir / "_batch_logs" / f"{run_name}.log"


def start_job(run_name: str, cmd: list[str], log_path: Path) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    log_file.write(f"$ {' '.join(cmd)}\n\n")
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parent),
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


def merge_samples(sample_dir: Path, merged_csv: Path) -> int:
    rows_written = 0
    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    with merged_csv.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=CALIBRATION_FIELDS)
        writer.writeheader()
        for sample_csv in sorted(sample_dir.glob("*_samples.csv")):
            with sample_csv.open("r", newline="", encoding="utf-8") as src:
                reader = csv.DictReader(src)
                for row in reader:
                    writer.writerow({field: row.get(field, "") for field in CALIBRATION_FIELDS})
                    rows_written += 1
    return rows_written


def main() -> None:
    args = parse_args()
    runs = 1 if args.dry_run else args.runs
    steps = 105 if args.dry_run else args.steps
    parallelism = max(1, int(args.parallelism))

    script_dir = Path(__file__).resolve().parent
    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = script_dir / output_root
    output_dir = output_root / f"random_calibration_{batch_ts}"
    sample_dir = output_dir / "_calibration_samples_by_run"
    merged_csv = output_dir / "calibration_samples.csv"

    print("=" * 60)
    print("  Random-Tax Calibration Runner")
    print(f"  Started: {timestamp()}")
    print(f"  Output:  {output_dir}")
    print(f"  Plan:    {runs} runs x {steps} steps | parallelism={parallelism}")
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
            run_name, cmd, _sample_csv, log_path = build_command(
                script_dir=script_dir,
                output_dir=output_dir,
                sample_dir=sample_dir,
                run_index=run_index,
                steps=steps,
                model=args.model,
                ollama_url=args.ollama_url,
                dry_run=args.dry_run,
            )
            job = start_job(run_name, cmd, log_path)
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

    sample_rows = merge_samples(sample_dir, merged_csv)
    failed = sum(1 for _, rc, _, _ in results if rc != 0)
    manifest = {
        "created_at": timestamp(),
        "runs_requested": runs,
        "runs_completed": len(results) - failed,
        "runs_failed": failed,
        "steps": steps,
        "model": args.model,
        "ollama_url": args.ollama_url,
        "parallelism": parallelism,
        "sample_rows": sample_rows,
        "merged_csv": str(merged_csv),
        "run_logs": [str(log_path) for _, _, _, log_path in results],
    }
    with (output_dir / "calibration_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"  All done at {timestamp()} | Total: {format_elapsed(time.time() - total_start)}")
    print(f"  Merged calibration CSV: {merged_csv}")
    print(f"  Sample rows: {sample_rows}")
    print(f"  Failed runs: {failed}")
    print("=" * 60)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
