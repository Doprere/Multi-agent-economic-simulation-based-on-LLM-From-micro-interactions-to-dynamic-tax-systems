#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PARENT_DIR="$(dirname "$PROJECT_ROOT")"
PROJECT_DIR="$(basename "$PROJECT_ROOT")"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${PROJECT_ROOT}/llm_econ_linux_package_${TS}.tar.gz"

cd "$PARENT_DIR"

tar \
  --exclude="${PROJECT_DIR}/.git" \
  --exclude="${PROJECT_DIR}/venv" \
  --exclude="${PROJECT_DIR}/.venv" \
  --exclude="${PROJECT_DIR}/__pycache__" \
  --exclude="${PROJECT_DIR}/simulation_results" \
  --exclude="${PROJECT_DIR}/linux_simulation_results" \
  --exclude="${PROJECT_DIR}/thesis" \
  --exclude="${PROJECT_DIR}/*.pdf" \
  --exclude="${PROJECT_DIR}/*.docx" \
  --exclude="${PROJECT_DIR}/*.doc" \
  --exclude="${PROJECT_DIR}/.env" \
  --exclude="${PROJECT_DIR}/*.env" \
  --exclude="${PROJECT_DIR}/credentials.json" \
  -czf "$OUT" \
  "$PROJECT_DIR/run_simulation.py" \
  "$PROJECT_DIR/run_experiment.py" \
  "$PROJECT_DIR/visualize_experiments.py" \
  "$PROJECT_DIR/requirements.txt" \
  "$PROJECT_DIR/README.md" \
  "$PROJECT_DIR/AGENTS.md" \
  "$PROJECT_DIR/llm_agent" \
  "$PROJECT_DIR/ai_economist" \
  "$PROJECT_DIR/deployment/linux"

echo "Created package: $OUT"
