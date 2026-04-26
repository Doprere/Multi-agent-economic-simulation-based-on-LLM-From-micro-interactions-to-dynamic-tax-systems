#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
AGENT_MODEL="${AGENT_MODEL:-gemma4:e2b}"
PLANNER_MODEL="${PLANNER_MODEL:-gpt-5.4-mini}"

echo "[1/2] Dry-run smoke test"
python run_simulation.py \
  --dry-run \
  --steps 5 \
  --run-name linux_dry_test

echo
echo "[2/2] LLM smoke test"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY is not set. Load .env or export the key first." >&2
  exit 1
fi

python run_simulation.py \
  --steps 5 \
  --agent-backend ollama \
  --agent-model "$AGENT_MODEL" \
  --planner-backend openai \
  --planner-model "$PLANNER_MODEL" \
  --ollama-url "$OLLAMA_URL" \
  --run-name linux_llm_test

echo
echo "[OK] Smoke tests completed."
