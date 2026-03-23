"""
config.py — 讀取 config.yaml，提供全局設定物件。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ──────────────────────────────────────────────────────────────
#  Config 資料類別
# ──────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_retries: int = 3
    temperature: float = 0.7
    timeout: int = 30


@dataclass
class MemoryConfig:
    agent_short_term_window: int = 8
    agent_long_term_trigger: int = 10
    planner_short_term_window: int = 3


@dataclass
class PersonaConfig:
    id: int
    name: str
    display_name: str
    age_group: str
    role: str
    skill_pareto_alpha: float
    labor_cost_modifier: float


@dataclass
class PlannerConfig:
    name: str = "Social Planner"
    display_name: str = "Social Planner"
    role: str = ""
    tax_period: int = 100


@dataclass
class EnvironmentConfig:
    scenario_name: str = "layout_from_file/simple_wood_and_stone"
    n_agents: int = 4
    episode_length: int = 1000
    world_size: list = field(default_factory=lambda: [25, 25])
    dense_log_frequency: int = 1
    env_layout_file: str = "quadrant_25x25_20each_30clump.txt"
    flatten_observations: bool = False
    flatten_masks: bool = True
    multi_action_mode_agents: bool = False
    multi_action_mode_planner: bool = True
    components: list = field(default_factory=list)


@dataclass
class AppConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    personas: list[PersonaConfig] = field(default_factory=list)
    planner: PlannerConfig = field(default_factory=PlannerConfig)


# ──────────────────────────────────────────────────────────────
#  載入設定
# ──────────────────────────────────────────────────────────────

_CONFIG: AppConfig | None = None
_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _resolve_env(value: Any) -> Any:
    """遞迴展開 ${ENV_VAR} 語法。"""
    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}"):
            key = value[2:-1]
            return os.environ.get(key, "")
        return value
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def _build_component_list(raw: list) -> list:
    """將 YAML 元件格式轉成 Foundation 接受的 (name, kwargs) list。"""
    result = []
    for item in raw:
        for comp_name, comp_kwargs in item.items():
            result.append((comp_name, comp_kwargs or {}))
    return result


def load_config(path: Path | str | None = None) -> AppConfig:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    cfg_path = Path(path) if path else _CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw = _resolve_env(raw)

    # LLM 設定：api_key 優先使用環境變數
    llm_raw = raw.get("llm", {})
    if not llm_raw.get("api_key"):
        llm_raw["api_key"] = os.environ.get("OPENAI_API_KEY", "")
    if not llm_raw.get("base_url") or llm_raw["base_url"] == "https://api.openai.com/v1":
        env_url = os.environ.get("OPENAI_BASE_URL", "")
        if env_url:
            llm_raw["base_url"] = env_url

    # 環境設定
    env_raw = raw.get("environment", {})
    components = _build_component_list(env_raw.pop("components", []))
    env_cfg = EnvironmentConfig(**env_raw)
    env_cfg.components = components  # type: ignore[assignment]

    # 記憶設定
    mem_raw = raw.get("memory", {})
    mem_cfg = MemoryConfig(**mem_raw)

    # Persona 設定
    personas = [
        PersonaConfig(**p) for p in raw.get("agents", {}).get("personas", [])
    ]

    # Planner 設定
    planner_raw = raw.get("planner", {})
    planner_cfg = PlannerConfig(**planner_raw)

    _CONFIG = AppConfig(
        environment=env_cfg,
        llm=LLMConfig(**llm_raw),
        memory=mem_cfg,
        personas=personas,
        planner=planner_cfg,
    )
    return _CONFIG


def get_config() -> AppConfig:
    """取得全局設定，若未載入則自動從預設路徑載入。"""
    return load_config()


def make_env_config(cfg: AppConfig) -> dict:
    """將 AppConfig 轉成 foundation.make_env_instance 所需的 dict。"""
    env = cfg.environment
    return {
        "scenario_name": env.scenario_name,
        "n_agents": env.n_agents,
        "episode_length": env.episode_length,
        "world_size": env.world_size,
        "dense_log_frequency": env.dense_log_frequency,
        "env_layout_file": env.env_layout_file,
        "flatten_observations": env.flatten_observations,
        "flatten_masks": env.flatten_masks,
        "multi_action_mode_agents": env.multi_action_mode_agents,
        "multi_action_mode_planner": env.multi_action_mode_planner,
        "components": env.components,
    }
