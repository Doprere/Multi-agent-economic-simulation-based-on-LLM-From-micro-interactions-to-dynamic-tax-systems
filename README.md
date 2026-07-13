# LLM-Driven Multi-Agent Economic Simulation

> From Micro Interactions to Dynamic Tax Systems

An extension of the [AI Economist](https://github.com/salesforce/ai-economist) framework (Zheng et al., 2020) that replaces reinforcement learning policies with **Large Language Model (LLM)** driven agents for multi-agent socioeconomic simulation.

## Research Motivation

Traditional RL-based economic simulations require extensive training and produce opaque policies. This project investigates whether LLM-driven agents can:

- Make economically rational decisions through natural language reasoning
- Exhibit emergent social dynamics (inequality, trade patterns, specialization)
- Respond to tax policy changes set by an LLM-driven social planner

## Architecture

```
┌─────────────────────────────────────────────────┐
│              AI Economist Environment            │
│         (25×25 grid, Wood, Stone, Market)        │
└──────────────────────┬──────────────────────────┘
                       │ obs / actions
        ┌──────────────┴──────────────┐
        ▼                             ▼
 ┌──────────────┐            ┌──────────────┐
 │ 4 Mobile     │            │ Social       │
 │ Agents       │            │ Planner      │
 │ (Age Groups) │            │ (Tax Policy) │
 └──────┬───────┘            └──────┬───────┘
        │                           │
        ▼                           ▼
 ┌──────────────────────────────────────────┐
 │           LLM Decision Engine            │
 │  Translator → Prompt → LLM → Validate   │
 │  (OpenAI GPT-4o-mini / Ollama llama3)   │
 └──────────────────────────────────────────┘
```

### Agent Decision Pipeline

1. **Observation Translation** — Raw env observations → English natural language (resource positions with directions & action IDs, market status with pricing tips, valid actions with blocked directions, game rules)
2. **Memory Assembly** — Short-term (sliding window, 8 steps) + Long-term (async LLM consolidation every 10 steps)
3. **LLM Call** — System prompt (persona + happiness framing + few-shot) + User prompt (state + rules) + Context (memory) → JSON response
4. **Action Validation** — Verify `action_id` against environment action mask; retry with error hint (max 3), then fallback to random valid action

### Agent Personas (Age-Based Heterogeneity)

| Agent | Persona | Skill Level | Labor Cost | Coin Endowment | Strategy Tendency |
|-------|---------|-------------|------------|----------------|-------------------|
| 0 | Youth (≤20) | Low (alpha=6.0) | Low (0.7x) | 0 | Exploration, learning |
| 1 | Young Adult (21-40) | Moderate (alpha=4.0) | Normal (1.0x) | 0 | Balanced, versatile |
| 2 | Middle-aged (41-60) | High (alpha=2.5) | High (1.3x) | 20-50 | Efficiency, trading |
| 3 | Senior (>60) | Highest (alpha=1.5) | Highest (1.8x) | 50-100 | Stationary, market-focused |

- **Labor cost modifier**: Applied via delta-based scaling after each `env.step()`. Only the new labor incurred per step is multiplied by the modifier, preventing compounding errors on historical accumulation.
- **Coin endowment**: Middle-aged and Senior agents start with initial Coin (drawn uniformly from their range) representing lifecycle savings. This is injected at environment reset and the optimization baseline is recalculated accordingly.

### Utility / Happiness Design

Agent utility follows the AI Economist's original formulation: **diminishing marginal utility of Coin minus labor cost**. However, the LLM prompt frames this as:

> *"Your happiness = Coin earned minus effort spent."*

Key prompt design decisions:
- **Labor as investment**: "Spending labor to gather and build is an investment — it pays off in Coin." This prevents small models from interpreting labor avoidance as optimal (which caused NOOP-heavy behavior in early experiments).
- **Stamina description**: Each agent's `labor_cost_modifier` is translated to natural language (e.g., "You have good stamina" for 0.7x, "Physical tasks cost you more effort than most" for 1.8x) rather than exposing the raw numeric modifier.
- **No explicit recommendation**: The prompt describes the environment fully but never tells agents what action to take.

### Action Space

50 discrete actions, ordered alphabetically by Foundation convention (**Stone before Wood**):

| Range | Action |
|-------|--------|
| 0 | NOOP |
| 1 | Build (requires 1 Wood + 1 Stone) |
| 2-12 | Buy Stone (bid price 0-10) |
| 13-23 | Sell Stone (ask price 0-10) |
| 24-34 | Buy Wood (bid price 0-10) |
| 35-45 | Sell Wood (ask price 0-10) |
| 46-49 | Move Left / Right / Up / Down |

An `_validate_action_order()` check runs at startup to confirm Foundation's internal ordering matches `action_map.py`.

### Market (Continuous Double Auction)

- `max_num_orders: 5` — each agent has at most 5 outstanding bid/ask orders per resource.
- `order_duration: 20` — unmatched orders auto-expire after 20 steps. This prevents the 5-order ceiling from locking agents into NOOP when the market goes quiet; expired slots are reclaimed automatically.
- Translator surfaces both the ceiling warning and the expiry behavior, plus an "avoid price 0" note (bidding 0 Coin = asking for a free gift; asking 0 Coin = giving the resource away — neither happens in a normal economy). Agents are told to reference `market_rate` when pricing.

### Social Planner

- Observes all agents' wealth, inventory, and inequality (Gini coefficient) every step
- Sets progressive tax brackets every 100 steps
- Guided by two objectives: **fairness** (Gini, wealth gaps, resource access) and **productivity** (total Coin, building/market activity)

## Project Structure

```
project/
├── llm_agent/                  # Core package
│   ├── agent.py                # Mobile agent LLM decision layer
│   ├── planner.py              # Social planner LLM decision layer
│   ├── translator.py           # Observation → English text
│   ├── memory.py               # Dual-layer memory system
│   ├── config.py               # Configuration loading (+ optional `llm_planner`)
│   ├── config.yaml             # Simulation parameters
│   ├── llm_client.py           # OpenAI async client
│   ├── ollama_client.py        # Ollama local client (duck-typed)
│   ├── client_factory.py       # Backend factory (openai | ollama)
│   ├── sim_common.py           # Shared helpers (age skill, action order, labor modifier, planner action build, adjacency)
│   ├── logger.py               # Metrics, Excel, map snapshots, nested token_usage
│   ├── action_map.py           # Action ID ↔ name mapping
│   └── simulation.py           # DEPRECATED stub → run_simulation
├── ollama_simulation.py        # DEPRECATED stub → run_simulation
├── run_simulation.py           # Unified entry with asymmetric backend support (recommended)
├── run_experiment.py           # Batch runner (multiple runs / models)
├── random_tax_simulation.py    # Legacy cold-start random-tax baseline
├── run_random_calibration.py   # Random-tax calibration sample batch runner
├── run_saez_experiment.py      # Calibration-seeded dynamic Saez batch runner
├── saez_simulation.py          # Single-run Saez baseline experiment entry
├── preview_saez_schedule.py    # Preview Saez schedule from calibration samples
├── validate_calibration_csv.py # Validate calibration CSV before Saez runs
├── ai_economist/               # Foundation framework (Zheng et al.)
└── simulation_results/         # Output directory (git-ignored)
```

## Quick Start

### Prerequisites

- Python 3.9+
- [AI Economist Foundation](https://github.com/salesforce/ai-economist) (included as `ai_economist/`)

### Installation

```bash
pip install openai httpx openpyxl matplotlib numpy pandas pyyaml
```

### Unified Entry: `run_simulation.py` (Recommended)

```bash
# 1) Config-driven (reads `llm:` / optional `llm_planner:` in config.yaml)
export OPENAI_API_KEY="sk-proj-..."
python run_simulation.py --steps 200 --run-name baseline

# 2) All-local Ollama (Agents + Planner share one client)
python run_simulation.py --steps 200 \
  --agent-backend ollama --agent-model llama3:8b \
  --ollama-url http://localhost:11434 \
  --run-name all_ollama

# 3) Asymmetric: Planner = cloud-strong, Agents = local-weak
python run_simulation.py --steps 1000 \
  --agent-backend   ollama --agent-model   gemma4:e4b \
  --planner-backend openai --planner-model gpt-4o-mini \
  --ollama-url http://localhost:11434 \
  --run-name ge4e4b_gpt4omi_1000_run1

# 4) Dry-run (random legal actions, no LLM calls)
python run_simulation.py --dry-run --steps 5
```

**Precedence:** `CLI > config.yaml (llm_planner / llm) > dataclass defaults`. If neither CLI nor `llm_planner:` is set, Planner and Agents share one client (backward compatible).

**Deprecation stubs** — old entry points still work; they translate legacy flags and delegate to `run_simulation.main()`:

```bash
python ollama_simulation.py --steps 200 --model llama3:8b --ollama-url http://localhost:11434
python -m llm_agent.simulation --steps 200
```

### Config schema

```yaml
llm:                           # Agents default (also Planner when llm_planner is absent)
  backend: "ollama"            # "openai" | "ollama"
  model: "gemma4:e2b"
  base_url: "http://localhost:11434"
  ...

llm_planner:                   # optional — if present, Planner uses this independently
  backend: "openai"
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com/v1"
  ...
```

**Asymmetric rationale:** In a 1000-step episode, Planner runs ~10 tax decisions vs Agents' 4000 step decisions. Routing the strong model to Planner (high-value, infrequent) and the weak local model to Agents (low-value, frequent) isolates "decision-layer vs execution-layer" effects on social dynamics.

**Known risk:** When Planner uses Ollama, `tax_brackets` regex parsing has no fallback — prefer OpenAI for Planner. Agent-side Ollama is safer (`action_mask` random fallback).

## Output

Each run generates a timestamped directory under `simulation_results/`:

| File | Description |
|------|-------------|
| `agent_thoughts.xlsx` | 6-sheet Excel: agent thoughts, planner thoughts, memory snapshots, metrics, LLM prompts, resource adjacency events |
| `step_metrics.csv` | Per-step Coin, Gini coefficient, rewards, inventory |
| `action_log.csv` | Action IDs per agent per step |
| `tax_log.csv` | Tax bracket changes |
| `summary.json` | Run metadata + nested `token_usage` (`agents` / `planner`) |
| `maps/*.png` | Map snapshots (global + egocentric views) |

The `token_usage` block records `{model, backend, api_calls, tokens_in, tokens_out}` for **Agents** and **Planner** separately. When Planner shares the Agents' client, the `planner` entry is `{"shared_with_agents": true}`.

## Experiment Validation Utilities

These utilities are kept in the repository because they support experiment execution or pre-run validation:

```bash
# Validate random-tax calibration samples before running the calibrated Saez baseline
python validate_calibration_csv.py simulation_results/random_calibration_YYYYMMDD_HHMMSS/calibration_samples.csv --min-samples 2000

# Preview the initial Saez marginal tax schedule generated from a calibration CSV
python preview_saez_schedule.py simulation_results/random_calibration_YYYYMMDD_HHMMSS/calibration_samples.csv
```

Local diagnostics, prompt-audit scratch files, thesis-only analysis scripts, plots, and result archives are intentionally git-ignored. They are useful during thesis analysis but are not required to reproduce or run the experiments.

## Key Design Decisions

- **LLM as policy**: Natural language reasoning replaces RL gradient updates
- **Duck-typed backends**: OpenAI and Ollama share identical interfaces — easy to swap or extend
- **Non-blocking memory**: Long-term consolidation runs as async background task
- **Information-rich, non-prescriptive**: Translator provides full environmental context without recommending actions
- **Comprehensive logging**: Every LLM prompt and response recorded for research reproducibility
- **Small-model optimized prompts**: Prompt design tested with 3B-8B parameter models (qwen2.5:3b, llama3:8b). Few-shot examples prioritize gather actions (primacy effect), trading rules are compressed, and directions use action-aligned language (Up/Down/Left/Right with action IDs instead of cardinal N/S/E/W)
- **Delta-based labor scaling**: Per-agent labor cost modifiers apply only to each step's new labor increment, avoiding compounding errors
- **Market self-healing**: CDA `order_duration: 20` auto-expires stale orders, so the per-resource 5-order ceiling never permanently blocks an agent. The translator states this explicitly so agents don't retreat to NOOP on a temporarily full book.
- **Pricing guardrail**: The `[Game Rules]` section warns against bidding/asking price 0 and directs agents to use the dynamic `market_rate` as a pricing anchor, cutting down on wasted "free gift" orders seen with small models.
- **Asymmetric backends**: Planner and agents can draw from different providers/models via the optional `llm_planner:` config block. Implemented as a factory (`client_factory.make_llm_client`) that returns a duck-typed client from the `backend` string; both code paths share `sim_common.py` helpers so agent/planner/memory/translator modules remain backend-agnostic.

## Roadmap

**Completed (Phase 1 — English text, rules injection, diagnostic tools)**
- English-only LLM-facing prompts; observation translator with action-ID-aligned directions
- Retry + action mask validation with random valid fallback; startup action-order check
- 6-sheet Excel logger, LLM prompt sheet, resource adjacency sheet, map snapshots

**Completed (Phase 2 — prompt balancing & small-model tuning)**
- Happiness framing rewritten: labor framed as *investment* (prevents NOOP-collapse in small models)
- CDA action-ordering fix (Stone before Wood), delta-based labor modifier
- Initial Coin endowment for Middle-aged / Senior (lifecycle savings)
- CDA `order_duration: 20` — auto-expire unmatched orders so the 5-order ceiling self-recovers
- Translator "avoid price 0" warning — bid/ask 0 framed as free gifts; recommend `market_rate` anchor
- Validated models: llama3:8b (8B), qwen2.5:3b (3B, runs on GTX 1650 4GB VRAM)

**Completed (Phase 3 — asymmetric model architecture)**
- `llm_planner:` optional config block + `client_factory.make_llm_client()` + unified `run_simulation.py`
- Shared helpers extracted to `sim_common.py` (age skill, action-order validation, labor modifier, planner action build, adjacency detection)
- Backward compatible: absent `llm_planner:` → Planner reuses the Agents' client instance; `simulation.py` / `ollama_simulation.py` remain as deprecation stubs that translate flags and delegate
- `agent.py` / `planner.py` / `memory.py` / `translator.py` / `llm_client.py` / `ollama_client.py` unchanged (already duck-typed)
- `summary.json` → `token_usage` is now nested: `{"agents": {...}, "planner": {...}}` (with `{"shared_with_agents": true}` when one client is shared)
- Known risk: Planner-on-Ollama has no `tax_brackets` fallback — prefer OpenAI for Planner

**Next (Phase 4 — controlled experiments)**
- Smoke (100 steps) → full (1000 steps) per model combo; see `.claude/plans/witty-hugging-pearl.md`
- Primary comparison: `ge4e4b + gpt-4o-mini` vs baseline `ge4e2b + gpt-4o-mini` (cumulative builds, Gini, total Coin)

## References

- Zheng, S., Trott, A., Srinivasa, S., et al. (2020). *The AI Economist: Improving Equality and Productivity with AI-Driven Tax Policies.* arXiv:2004.13332
- Salesforce AI Economist: https://github.com/salesforce/ai-economist

## License

This project is for academic research purposes.
