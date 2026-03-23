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

1. **Observation Translation** — Raw env observations → English natural language descriptions (resource locations, market status, valid actions, game rules)
2. **Memory Assembly** — Short-term (sliding window, 8 steps) + Long-term (async LLM consolidation every 10 steps)
3. **LLM Call** — System prompt (persona + rules) + User prompt (state) + Context (memory) → JSON response
4. **Action Validation** — Verify `action_id` against environment action mask; retry or fallback on invalid

### Agent Personas (Age-Based Heterogeneity)

| Agent | Persona | Skill Level | Labor Cost | Strategy Tendency |
|-------|---------|-------------|------------|-------------------|
| 0 | Youth (≤20) | Low (alpha=6.0) | Low (0.7x) | Exploration, learning |
| 1 | Young Adult (21-40) | Moderate (alpha=4.0) | Normal (1.0x) | Balanced, versatile |
| 2 | Middle-aged (41-60) | High (alpha=2.5) | High (1.3x) | Efficiency, trading |
| 3 | Senior (>60) | Highest (alpha=1.5) | Highest (1.8x) | Stationary, market-focused |

### Social Planner

- Observes all agents' wealth and inequality (Gini coefficient) every step
- Sets progressive tax brackets every 100 steps
- Objective: maximize social welfare (equality × productivity)

## Project Structure

```
project/
├── llm_agent/                  # Core package
│   ├── agent.py                # Mobile agent LLM decision layer
│   ├── planner.py              # Social planner LLM decision layer
│   ├── translator.py           # Observation → English text
│   ├── memory.py               # Dual-layer memory system
│   ├── config.py               # Configuration loading
│   ├── config.yaml             # Simulation parameters
│   ├── llm_client.py           # OpenAI async client
│   ├── ollama_client.py        # Ollama local client (duck-typed)
│   ├── logger.py               # Metrics, Excel, map snapshots
│   ├── action_map.py           # Action ID ↔ name mapping
│   └── simulation.py           # OpenAI backend entry point
├── ollama_simulation.py        # Ollama backend entry point
├── ai_economist/               # Foundation framework (Zheng et al.)
└── simulation_results/         # Output directory (git-ignored)
```

## Quick Start

### Prerequisites

- Python 3.9+
- [AI Economist Foundation](https://github.com/salesforce/ai-economist) (included as `ai_economist/`)

### Installation

```bash
pip install openai httpx openpyxl matplotlib numpy pyyaml
```

### Run with Ollama (Local)

```bash
# Start Ollama server first
ollama serve

# Run simulation
python ollama_simulation.py --steps 200 --model llama3:8b --ollama-url http://localhost:11434
```

### Run with OpenAI

```bash
export OPENAI_API_KEY="your-key-here"
python -m llm_agent.simulation --steps 200
```

### Dry Run (Environment Test)

```bash
python ollama_simulation.py --dry-run --steps 5
```

## Output

Each run generates a timestamped directory under `simulation_results/`:

| File | Description |
|------|-------------|
| `agent_thoughts.xlsx` | 6-sheet Excel: agent thoughts, planner thoughts, memory snapshots, metrics, LLM prompts, resource adjacency events |
| `step_metrics.csv` | Per-step Coin, Gini coefficient, rewards, inventory |
| `action_log.csv` | Action IDs per agent per step |
| `tax_log.csv` | Tax bracket changes |
| `summary.json` | Run metadata |
| `maps/*.png` | Map snapshots (global + egocentric views) |

## Key Design Decisions

- **LLM as policy**: Natural language reasoning replaces RL gradient updates
- **Duck-typed backends**: OpenAI and Ollama share identical interfaces — easy to swap or extend
- **Non-blocking memory**: Long-term consolidation runs as async background task
- **Information-rich, non-prescriptive**: Translator provides full environmental context without recommending actions
- **Comprehensive logging**: Every LLM prompt and response recorded for research reproducibility

## References

- Zheng, S., Trott, A., Srinivasa, S., et al. (2020). *The AI Economist: Improving Equality and Productivity with AI-Driven Tax Policies.* arXiv:2004.13332
- Salesforce AI Economist: https://github.com/salesforce/ai-economist

## License

This project is for academic research purposes.
