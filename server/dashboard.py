"""
The InventoryGym dashboard: describe an inventory, watch an agent run it, compare agents.

Three steps on one page:

1. Your inventory  - build one in plain words in the quick builder, or start from a built-in
                     world, an example or a blank starter; edit
                     storage, products, tasks and events in tables, or edit / upload
                     the world file itself. Mistakes are explained in place.
2. Run an episode  - any task of that world, with or without news, played by a rule
                     agent, a reference agent or an LLM, with charts redrawn every
                     simulated day and the baseline and oracle on the same seed.
3. Compare agents  - several agents over several seeds, paired seed by seed.

Every run builds its own environment in-process, so viewers never share an episode and
the benchmark API is unaffected. Agents only see the observation; the charts also show
the hidden truth (orange) so a viewer can judge the decisions.

Worlds typed or uploaded here are parsed with the safe YAML loader and size limits
(world.load_world_text), and no file path from the browser is ever opened. An API token
is used for the calls of one run and never stored or logged.
"""

from __future__ import annotations

import html
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import yaml

import inference
from agents import (Agent, BaselineAgent, History, LLMAgent, NewsLLMAgent, NewsPlanAgent, NewsRuleAgent,  # noqa: F401
                    NothingAgent, OracleAgent, baseline_policy)
from models import EpisodeConfig, InventoryState, PharmaAction
from news import find_events
from tasks import PHARMA_WORLD, TEST_SEEDS, average_step_reward, get_task_config, weighted_fill
from world import World, WorldError, default_world, load_world_text

try:
    from my_env_environment import PharmaEnvironment
    import world_forms as forms
    import world_wizard as wizard
except ModuleNotFoundError:
    from server.my_env_environment import PharmaEnvironment
    from server import world_forms as forms
    from server import world_wizard as wizard

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Choices
# ---------------------------------------------------------------------------

TEMPLATE_PHARMA = "Pharma warehouse (built-in)"
TEMPLATE_GROCERY = "Grocery store (example)"
TEMPLATE_STARTER = "Small shop (blank starter)"
TEMPLATES = (TEMPLATE_PHARMA, TEMPLATE_GROCERY, TEMPLATE_STARTER)

AGENT_BASELINE = "Baseline rule (no news, no LLM)"
AGENT_NEWS_PLAN = "News plan (reads the news perfectly; reference)"
AGENT_NEWS_RULE = "Extra-days news rule (reference)"
AGENT_ORACLE = "Oracle (perfect foresight; the ceiling)"
AGENT_NOTHING = "Order nothing (the floor)"
AGENT_LLM = "LLM: adjusts the baseline's orders"
AGENT_LLM_NEWS = "LLM: reads the news only (hybrid)"
RULE_AGENTS = {
    AGENT_BASELINE: BaselineAgent, AGENT_NEWS_PLAN: NewsPlanAgent, AGENT_NEWS_RULE: NewsRuleAgent,
    AGENT_ORACLE: OracleAgent, AGENT_NOTHING: NothingAgent,
}
LLM_AGENTS = (AGENT_LLM, AGENT_LLM_NEWS)
AGENT_CHOICES = [*RULE_AGENTS, *LLM_AGENTS]

NEWS_OFF, NEWS_EXACT, NEWS_NATURAL = "Off", "Exact", "Natural"
NEWS_INFO = "Exact: one fixed template with the exact days. Natural: varied wording, in your inventory's own words."
NEWS_LEVELS = {NEWS_OFF: 0, NEWS_EXACT: 1, NEWS_NATURAL: 2}

# Models offered for the LLM agents: suggestions only, any model the endpoint serves can be typed.
MODEL_SUGGESTIONS = [inference.MODEL_NAME, "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
                     "microsoft/Phi-4-mini-instruct", "google/gemma-2-2b-it"]

# Stop an LLM run after this many failed calls in a row (bad token, wrong model...).
MAX_CONSECUTIVE_LLM_ERRORS = 3
# The theme of the dashboard page (server/app.py passes it when it mounts the page): blue accents to
# match the charts, instead of Gradio's default indigo.
THEME = gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.teal,
                       neutral_hue=gr.themes.colors.slate, radius_size=gr.themes.sizes.radius_lg)
# Compare tab: seeds per agent, and fewer for LLM agents (each seed is one model call per day).
MAX_COMPARE_SEEDS = 30
MAX_COMPARE_SEEDS_LLM = 3

# Chart colours. Blue is what the agent sees or does, orange is hidden ground truth,
# aqua is deliveries; grey is a reference. Pools use the categorical list.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#7a5cd6"
INK_MUTED = "#898781"
GRID = "rgba(137, 135, 129, 0.25)"
CATEGORICAL = [BLUE, AQUA, VIOLET, "#c2478f", "#8a8f2a", "#3aa6c9", "#a8672e", "#5b6ee1"]
# Single-hue sequential ramp for the unmet-demand heatmap (near-zero recedes).
UNMET_SCALE = [[0.0, "#e8f1fc"], [0.25, "#9ec5f4"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1.0, "#0d366b"]]

STYLE = """
<style>
:root, .gradio-container {
  --ig-blue: #2a78d6; --ig-blue-dark: #1f63b4; --ig-teal: #1baf7a; --ig-orange: #eb6834; --ig-violet: #7a5cd6;
  --ig-radius: 14px;
  --button-primary-background-fill: #2a78d6 !important;
  --button-primary-background-fill-hover: #1f63b4 !important;
  --button-primary-border-color: #2a78d6 !important;
  --button-primary-text-color: #ffffff !important;
  --color-accent: #2a78d6 !important;
  --slider-color: #2a78d6 !important;
  --checkbox-background-color-selected: #2a78d6 !important;
  --checkbox-border-color-selected: #2a78d6 !important;
}
.gradio-container { max-width: 1180px !important; margin-left: auto !important; margin-right: auto !important; }
.gradio-container button[role="tab"] { font-size: 15px !important; font-weight: 600 !important; padding: 10px 16px !important; }
.ig-dice { align-self: flex-end; margin-bottom: 4px; }
.ig-cta, button.ig-cta, .ig-cta button { font-size: 16px !important; font-weight: 700 !important; border-radius: 12px !important; min-height: 48px; }

/* header */
.ig-hero { border-radius: 18px; padding: 24px 28px; color: #fff; margin-bottom: 4px;
           background: linear-gradient(115deg, #1b4f9c 0%, #2a78d6 48%, #1baf7a 100%);
           box-shadow: 0 8px 28px rgba(42, 120, 214, 0.25); }
.ig-hero-top { display: flex; align-items: center; gap: 16px; }
.ig-logo { font-size: 38px; width: 64px; height: 64px; flex: none; display: flex; align-items: center;
           justify-content: center; border-radius: 16px; background: rgba(255, 255, 255, 0.18); }
.ig-hero h1 { margin: 0; font-size: 30px; line-height: 1.15; font-weight: 700; color: #fff; letter-spacing: -0.01em; }
.ig-hero p { margin: 6px 0 0 0; font-size: 15px; color: #fff; opacity: 0.93; }
.ig-steps { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 18px; }
.ig-step { display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px 6px 6px; border-radius: 999px;
           background: rgba(255, 255, 255, 0.16); font-size: 14px; font-weight: 500; color: #fff; }
.ig-step b { display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px;
             border-radius: 50%; background: #fff; color: #1f63b4; font-size: 13px; }
.ig-arrow { opacity: 0.7; color: #fff; }
.ig-legend { margin-top: 14px; font-size: 13px; color: #fff; opacity: 0.95; display: flex; flex-wrap: wrap; gap: 6px 20px; }
.ig-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 7px; }
.ig-current { display: flex; align-items: center; flex-wrap: wrap; gap: 6px 12px; padding: 8px 16px; border-radius: 999px;
              width: fit-content; max-width: 100%; border: 1px solid var(--border-color-primary);
              background: var(--block-background-fill); font-size: 14px; color: var(--body-text-color); }
.ig-current b { color: var(--ig-blue); }

/* cards, chips, tiles */
.ig-card { padding: 16px 20px; border-radius: var(--ig-radius); border: 1px solid var(--border-color-primary);
           background: var(--block-background-fill); box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05); color: var(--body-text-color); }
.ig-card h4 { margin: 0 0 6px 0; }
.ig-ok { border-left: 5px solid #1baf7a; } .ig-error { border-left: 5px solid #c8472b; }
.ig-error pre { white-space: pre-wrap; margin: 6px 0 0 0; font-size: 13px; }
.ig-sub { font-size: 12.5px; color: var(--body-text-color-subdued); margin-top: 2px; }
.ig-good { color: #1a8f63; } .ig-bad { color: #c8472b; }
.ig-sum-head { display: flex; align-items: center; gap: 14px; margin-bottom: 8px; }
.ig-check { width: 34px; height: 34px; flex: none; border-radius: 50%; background: #1baf7a; color: #fff;
            display: flex; align-items: center; justify-content: center; font-weight: 700; }
.ig-sum-title { font-size: 18px; font-weight: 700; }
.ig-row-label { margin-top: 10px; font-size: 12px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
                color: var(--body-text-color-subdued); }
.ig-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.ig-chip { padding: 3px 11px; border-radius: 999px; font-size: 13px; border: 1px solid var(--border-color-primary);
           color: var(--body-text-color); background: var(--background-fill-secondary); }
.ig-chip-cold { border-color: rgba(42, 120, 214, 0.45); background: rgba(42, 120, 214, 0.10); }
.ig-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.ig-tile { position: relative; overflow: hidden; padding: 14px 16px 12px; border-radius: var(--ig-radius);
           border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
           box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05); }
.ig-tile::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 4px; background: var(--ig-blue); }
.ig-tile.ig-a-teal::before { background: var(--ig-teal); } .ig-tile.ig-a-violet::before { background: var(--ig-violet); }
.ig-tile.ig-a-orange::before { background: var(--ig-orange); }
.ig-label { font-size: 12px; color: var(--body-text-color-subdued); }
.ig-value { font-size: 26px; font-weight: 700; color: var(--body-text-color); margin-top: 4px; line-height: 1.1; }
.ig-chips-note { font-size: 12px; }

/* news and verdict */
.ig-news { display: flex; flex-direction: column; gap: 8px; max-height: 260px; overflow-y: auto; }
.ig-news-item { padding: 9px 14px; border-radius: 10px; border: 1px solid var(--border-color-primary);
                background: var(--block-background-fill); font-size: 14px; color: var(--body-text-color); }
.ig-news-new { border-left: 4px solid var(--ig-orange); }
.ig-day { display: inline-block; min-width: 54px; font-weight: 600; color: var(--body-text-color-subdued); }
.ig-verdict { font-size: 16px; }

/* progress and empty state */
.ig-bar { height: 8px; border-radius: 999px; margin-top: 10px; background: var(--border-color-primary); overflow: hidden; }
.ig-bar > div { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #2a78d6, #1baf7a); transition: width 0.25s; }
.ig-feats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 14px; }
.ig-feat { padding: 12px 14px; border-radius: 12px; font-size: 13.5px; background: var(--background-fill-secondary);
           color: var(--body-text-color); }
.ig-feat b { display: block; margin-bottom: 2px; }
.ig-table { margin: 4px 0 0 !important; width: 100%; border-collapse: collapse; font-size: 14px; color: var(--body-text-color); }
.ig-table, .ig-table th, .ig-table td { border-left: none !important; border-right: none !important; border-top: none !important; }
.ig-table th { text-align: left; padding: 8px 12px; font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase;
               color: var(--body-text-color-subdued); border-bottom: 2px solid var(--border-color-primary); }
.ig-table td { padding: 10px 12px; border-bottom: 1px solid var(--border-color-primary); }
.ig-table tbody tr:first-child td { font-weight: 600; }
.ig-hint { font-size: 13px; color: var(--body-text-color-subdued); padding: 0 4px 6px; }
.ig-section { display: flex; align-items: center; gap: 10px; margin: 16px 0 4px; font-size: 16px; font-weight: 600;
              color: var(--body-text-color); }
.ig-section b { width: 26px; height: 26px; flex: none; border-radius: 50%; background: var(--ig-blue); color: #fff;
                display: inline-flex; align-items: center; justify-content: center; font-size: 13px; }
.ig-section span { font-weight: 400; font-size: 13px; color: var(--body-text-color-subdued); }
</style>
"""


# ---------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------

def template_text(name: str) -> str:
    """The world file text of a template."""
    if name == TEMPLATE_GROCERY:
        return (ROOT / "examples" / "worlds" / "grocery.yaml").read_text(encoding="utf-8")
    if name == TEMPLATE_STARTER:
        return dump_world(forms.starter_world())
    return (ROOT / "worlds" / "pharma.yaml").read_text(encoding="utf-8")


def dump_world(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)


def parse_world(text: Optional[str]) -> World:
    """The world of an editor's text; the built-in pharma world when there is none."""
    if not text or not text.strip():
        return PHARMA_WORLD
    return load_world_text(text)


def is_builtin(world: World) -> bool:
    return world == default_world()


def world_summary_html(world: Optional[World], error: str = "") -> str:
    """A card that says what the current world is, or what is wrong with the text."""
    if error or world is None:
        return ('<div class="ig-card ig-error"><h4>This inventory cannot be used yet</h4>'
                f"<pre>{html.escape(error)}</pre></div>")
    context = world.context()
    cold = {name for name, pool in world.storage.items() if pool.cold_chain}
    products = "".join(
        f'<span class="ig-chip{" ig-chip-cold" if p.storage in cold else ""}">'
        f'{"🧊" if p.storage in cold else "📦"} {html.escape(context.products[sku_id].name)}</span>'
        for sku_id, p in world.products.items())
    pools = "".join(
        f'<span class="ig-chip{" ig-chip-cold" if pool.cold_chain else ""}">'
        f'{"🧊" if pool.cold_chain else "🏬"} {html.escape(name)} · room for {pool.capacity_days:g} days of sales</span>'
        for name, pool in world.storage.items())
    tasks = "".join(f'<span class="ig-chip">🎯 {html.escape(t)}</span>' for t in world.task_names)
    business = html.escape(world.business[:1].upper() + world.business[1:])
    tag = " · the built-in world, with its original prompts" if is_builtin(world) else ""
    return f"""<div class="ig-card ig-ok">
<div class="ig-sum-head"><div class="ig-check">✓</div><div>
<div class="ig-sum-title">{html.escape(world.name)} is ready</div>
<div class="ig-sub">{business} · {world.days}-day episodes{tag}</div></div></div>
<div class="ig-row-label">Products</div><div class="ig-chips">{products}</div>
<div class="ig-row-label">Storage</div><div class="ig-chips">{pools}</div>
<div class="ig-row-label">Practice scenarios</div><div class="ig-chips">{tasks}</div></div>"""


def current_chip_html(world: World) -> str:
    """The slim bar above the tabs: which inventory the next run will use."""
    return (f'<div class="ig-current">📦 <span>Inventory in use: <b>{html.escape(world.name)}</b></span>'
            f'<span class="ig-sub">{len(world.products)} products · {len(world.tasks)} scenarios · {world.days} days</span></div>')


def note_html(text: str, kind: str = "error") -> str:
    """A short message as a card ('error' or 'info')."""
    css = "ig-error" if kind == "error" else ""
    return f'<div class="ig-card {css}">{html.escape(text)}</div>'


def section_html(number: str, title: str, hint: str = "") -> str:
    return f'<div class="ig-section"><b>{html.escape(number)}</b>{html.escape(title)}<span>{html.escape(hint)}</span></div>'


def ready_html() -> str:
    """What step 2 shows before the first run."""
    return """<div class="ig-card"><div style="font-size:17px;font-weight:700">👋 Ready when you are</div>
<div class="ig-sub" style="font-size:14px">Pick a task and an agent, then press <b>▶ Run episode</b>.</div>
<div class="ig-feats">
<div class="ig-feat"><b>📈 A live score</b>How the agent does against a simple rule and against perfect foresight, day by day.</div>
<div class="ig-feat"><b>📰 The news</b>Disruptions announced a few days ahead, in your inventory's own words.</div>
<div class="ig-feat"><b>🔍 The hidden truth</b>Orange shows what really happened, which the agent never sees.</div></div></div>"""


AGENT_HELP = {
    AGENT_BASELINE: "A simple rule: keep stock for the delivery time plus 3 days. It ignores the news.",
    AGENT_NEWS_PLAN: "A reference that reads the news perfectly (it sees the events behind it). Shows what the news is worth.",
    AGENT_NEWS_RULE: "An older reference: the baseline plus extra days for announced events.",
    AGENT_ORACLE: "Knows the future, so nobody can beat it. It marks the best possible score.",
    AGENT_NOTHING: "Never orders. The floor: shows how bad it gets.",
    AGENT_LLM: "A language model adds extra days of stock on top of the baseline's orders. Needs a token (More options).",
    AGENT_LLM_NEWS: "A language model reads only the news; the baseline places the orders. Needs a token, and news switched on.",
}


def agent_hint_html(label: str) -> str:
    return f'<div class="ig-hint">ℹ️ {html.escape(AGENT_HELP.get(label, ""))}</div>'


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

@dataclass
class EpisodeTrace:
    """Everything the dashboard draws, accumulated one simulated day at a time."""

    task_name: str
    seed: int
    config: EpisodeConfig
    state: InventoryState
    rows: List[Dict[str, Any]] = field(default_factory=list)
    # Every order the environment accepted: (sku_id, qty, order_day, true_arrival_day).
    orders: List[Tuple[str, float, int, int]] = field(default_factory=list)
    llm_errors: int = 0
    finished: bool = False
    stopped_reason: str = ""
    news_level: int = 0
    # Score so far, day by day, of reference agents on this very episode: {"Baseline": [...], "Oracle": [...]}.
    references: Dict[str, List[float]] = field(default_factory=dict)


def run_episode(task_name: str, seed: int, agent: Any, world: Optional[World] = None, news: int = 0,
                scenario: str = "fixed") -> Iterator[EpisodeTrace]:
    """Play one episode, yielding the (same, growing) trace after reset and after every day."""
    env = PharmaEnvironment()
    obs = env.reset(task_name=task_name, seed=seed, news=news, scenario=scenario, world=world)
    if hasattr(agent, "start_episode"):
        agent.start_episode(env)
    trace = EpisodeTrace(task_name=task_name, seed=seed, config=env._episode_config, state=obs, news_level=news)
    history: History = []
    consecutive_errors = 0
    yield trace

    while not obs.done:
        day = obs.current_date
        delivered_before = {sku_id: len(h) for sku_id, h in obs.lead_time_history.items()}
        news_today = list(obs.news)
        news_before = set(trace.rows[-1]["news"]) if trace.rows else set()

        reply, error = agent.act(obs, history)
        obs = env.step(PharmaAction(message=reply))

        placed = [o for o in env._open_orders if o[2] == day]
        trace.orders.extend(placed)
        trace.state = obs
        trace.rows.append({
            "day": day,
            "reply": getattr(agent, "last_model_reply", None) or reply,
            "error": error,
            "orders": {sku_id: qty for sku_id, qty, _, _ in placed},
            "reward": obs.reward,
            "fill": weighted_fill(obs.skus),
            "waste": obs.inventory_excess_today,
            "news": news_today,
            "new_news": [n for n in news_today if n not in news_before],
            "storage": {name: pool.ratio for name, pool in obs.storage.items()},
            "skus": {
                sku_id: {
                    "inventory": sku.inventory_on_hand,
                    "demand": obs.demand_history[sku_id][-1],
                    "fill": sku.demand_fulfilled_today,
                    "agent_lead_time": sku.avg_lead_time,
                    "observed_lead_times": obs.lead_time_history[sku_id][delivered_before[sku_id]:],
                }
                for sku_id, sku in obs.skus.items()
            },
        })
        history.append((day, reply, inference.build_feedback(obs)))

        if error:
            trace.llm_errors += 1
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_LLM_ERRORS:
                trace.stopped_reason = (
                    f"Stopped: the LLM call failed {consecutive_errors} times in a row. "
                    f"Last error: {error}"
                )
                yield trace
                return
        else:
            consecutive_errors = 0
        yield trace

    trace.finished = True
    yield trace


def score_so_far(trace: EpisodeTrace) -> float:
    """The final-score formula over the days played so far (equals the final score at the end)."""
    return average_step_reward(trace.state, len(trace.rows))


def running_score(rewards: Sequence[float]) -> List[float]:
    """Score so far after each day: the mean of the step rewards up to it, kept within 0..1."""
    out, total = [], 0.0
    for i, reward in enumerate(rewards, start=1):
        total += reward
        out.append(min(1.0, max(0.0, total / i)))
    return out


def play_score_curve(agent: Any, task_name: str, seed: int, world: Optional[World], news: int,
                     scenario: str) -> List[float]:
    """A rule agent's score so far, day by day, on one episode."""
    final = None
    for final in run_episode(task_name, seed, agent, world=world, news=news, scenario=scenario):
        pass
    return running_score([r["reward"] for r in final.rows])


def reference_curves(task_name: str, seed: int, world: Optional[World], news: int, scenario: str) -> Dict[str, List[float]]:
    """The baseline's and the oracle's score so far on this episode, to judge an agent against."""
    return {"Baseline": play_score_curve(BaselineAgent(), task_name, seed, world, news, scenario),
            "Oracle": play_score_curve(OracleAgent(), task_name, seed, world, news, scenario)}


def gap_closed(score: float, baseline: float, oracle: float) -> Optional[float]:
    """Share of the way from the baseline to the oracle; None when the two are (almost) the same."""
    return (score - baseline) / (oracle - baseline) if oracle - baseline > 1e-6 else None


def make_agent(label: str, model: str = "", token: str = "") -> Any:
    """The agent behind a label of AGENT_CHOICES. LLM agents need a model name and a token."""
    if label in RULE_AGENTS:
        return RULE_AGENTS[label]()
    if label == AGENT_LLM:
        return LLMAgent(model=model, api_key=token, base_url=inference.API_BASE_URL, action="adjust")
    if label == AGENT_LLM_NEWS:
        return NewsLLMAgent(model=model, api_key=token, base_url=inference.API_BASE_URL)
    raise ValueError(f"unknown agent {label!r}")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _base_figure(days: int, y_title: str, height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        # Gradio draws the chart title as a badge in the top-left corner, so the
        # legend goes below the x-axis instead.
        margin=dict(l=56, r=16, t=40, b=96),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=INK_MUTED),
        legend=dict(orientation="h", x=0, y=-0.22, xanchor="left", yanchor="top"),
        hovermode="x unified",
        xaxis=dict(title="Day", range=[-0.5, days - 0.5], showgrid=False, zeroline=False, showline=True, linecolor=GRID),
        yaxis=dict(title=y_title, gridcolor=GRID, zeroline=False, rangemode="tozero"),
    )
    return fig


def _figure(trace: EpisodeTrace, y_title: str, height: int = 360) -> go.Figure:
    return _base_figure(trace.config.no_of_days, y_title, height)


def _mark_today(fig: go.Figure, trace: EpisodeTrace) -> None:
    if trace.rows and not trace.finished:
        fig.add_vline(x=trace.rows[-1]["day"], line_width=1, line_color=INK_MUTED, opacity=0.6)


def _mark_events(fig: go.Figure, config: EpisodeConfig, sku_id: str, kinds: Sequence[str]) -> None:
    """
    Shade the days of a product's disruptions (hidden truth, orange) and, when the episode
    has news, mark the day each one was announced.
    """
    labels = {"supplier_delay": "delay", "demand_surge": "surge"}
    for event in find_events(config):
        if event["sku_id"] != sku_id or event["kind"] not in kinds:
            continue
        fig.add_vrect(x0=event["start_day"] - 0.5, x1=event["end_day"] + 0.5, fillcolor=ORANGE, opacity=0.08,
                      line_width=0, annotation_text=f"{labels[event['kind']]} ×{event['multiplier']:.1f}",
                      annotation_position="top left", annotation_font=dict(size=11, color=ORANGE))
    for event in config.news:
        if event.sku_id == sku_id and event.kind in kinds:
            fig.add_vline(x=event.announce_day, line_width=1.5, line_dash="dot", line_color=VIOLET,
                          annotation_text="news", annotation_position="bottom right",
                          annotation_font=dict(size=11, color=VIOLET))


def stock_figure(trace: EpisodeTrace, sku_id: str) -> go.Figure:
    cfg = trace.config.skus[sku_id]
    curve = cfg.demand_curve or [1.0] * trace.config.no_of_days
    fig = _figure(trace, "Units")
    _mark_events(fig, trace.config, sku_id, ("demand_surge", "supplier_delay"))
    fig.add_scatter(
        x=list(range(len(curve))), y=[cfg.base_demand * c for c in curve],
        name="Expected demand (hidden from agent)", mode="lines",
        line=dict(color=ORANGE, width=2), hovertemplate="%{y:,.0f}",
    )
    fig.add_scatter(
        x=[r["day"] for r in trace.rows], y=[r["skus"][sku_id]["demand"] for r in trace.rows],
        name="Actual demand (with daily noise)", mode="markers",
        marker=dict(color=ORANGE, size=5), hovertemplate="%{y:,.0f}",
    )
    fig.add_scatter(
        x=[r["day"] for r in trace.rows], y=[r["skus"][sku_id]["inventory"] for r in trace.rows],
        name="Stock on hand", mode="lines", line=dict(color=BLUE, width=2.5), fill="tozeroy",
        fillcolor="rgba(42, 120, 214, 0.08)", hovertemplate="%{y:,.0f}",
    )
    _mark_today(fig, trace)
    return fig


def lead_time_figure(trace: EpisodeTrace, sku_id: str) -> go.Figure:
    cfg = trace.config.skus[sku_id]
    curve = cfg.lead_time_curve or [1.0] * trace.config.no_of_days
    fig = _figure(trace, "Days")
    _mark_events(fig, trace.config, sku_id, ("supplier_delay",))
    fig.add_scatter(
        x=list(range(len(curve))), y=[max(1, round(cfg.base_lead_time * c)) for c in curve],
        name="True lead time if ordered that day (hidden)", mode="lines", line_shape="hv",
        line=dict(color=ORANGE, width=2), hovertemplate="%{y:.0f} days",
    )
    fig.add_scatter(
        x=[r["day"] for r in trace.rows], y=[r["skus"][sku_id]["agent_lead_time"] for r in trace.rows],
        name="Agent's lead-time estimate", mode="lines", line_shape="hv",
        line=dict(color=BLUE, width=2), hovertemplate="%{y:.1f} days",
    )
    delivered = [(r["day"], lt) for r in trace.rows for lt in r["skus"][sku_id]["observed_lead_times"]]
    fig.add_scatter(
        x=[d for d, _ in delivered], y=[lt for _, lt in delivered],
        name="Lead time seen on delivery", mode="markers",
        marker=dict(color=AQUA, size=9), hovertemplate="delivered after %{y:.0f} days",
    )
    _mark_today(fig, trace)
    return fig


def orders_figure(trace: EpisodeTrace, sku_id: str) -> go.Figure:
    today = trace.rows[-1]["day"] if trace.rows else -1
    ordered: Dict[int, float] = {}
    delivered: Dict[int, float] = {}
    for sku, qty, order_day, arrival_day in trace.orders:
        if sku != sku_id:
            continue
        ordered[order_day] = ordered.get(order_day, 0.0) + qty
        if arrival_day <= today:
            delivered[arrival_day] = delivered.get(arrival_day, 0.0) + qty
    fig = _figure(trace, "Units")
    fig.add_bar(
        x=list(ordered), y=list(ordered.values()), name="Ordered",
        marker=dict(color=BLUE, cornerradius=4), hovertemplate="%{y:,.0f}",
    )
    fig.add_bar(
        x=list(delivered), y=list(delivered.values()), name="Delivered",
        marker=dict(color=AQUA, cornerradius=4), hovertemplate="%{y:,.0f}",
    )
    fig.update_layout(barmode="group", bargap=0.15, bargroupgap=0.1)
    return fig


def reward_figure(trace: EpisodeTrace) -> go.Figure:
    rewards = [r["reward"] for r in trace.rows]
    fig = _figure(trace, "Reward")
    fig.add_scatter(
        x=[r["day"] for r in trace.rows], y=rewards, name="Step reward", mode="lines",
        line=dict(color=BLUE, width=2), hovertemplate="%{y:.3f}", showlegend=False,
    )
    fig.update_yaxes(range=[min([-0.05, *rewards]) - 0.05, 1.05], rangemode="normal")
    fig.add_hline(y=0, line_width=1, line_color=GRID)
    return fig


def score_figure(trace: EpisodeTrace) -> go.Figure:
    """
    The agent's score so far, day by day, against the baseline and the oracle on the same
    episode. The shaded zone between the baseline and the oracle is the room to improve.
    """
    fig = _figure(trace, "Score so far")
    days = list(range(len(next(iter(trace.references.values()), []))))
    baseline, oracle = trace.references.get("Baseline"), trace.references.get("Oracle")
    if baseline and oracle:
        fig.add_scatter(x=days, y=oracle, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=days, y=baseline, name="Baseline rule (the shaded zone is the room to improve)", mode="lines",
                        line=dict(color=INK_MUTED, width=2, dash="dot"), fill="tonexty",
                        fillcolor="rgba(235, 104, 52, 0.10)", hovertemplate="%{y:.3f}")
        fig.add_scatter(x=days, y=oracle, name="Oracle (knows the future)", mode="lines",
                        line=dict(color=ORANGE, width=2, dash="dash"), hovertemplate="%{y:.3f}")
    mine = running_score([r["reward"] for r in trace.rows])
    fig.add_scatter(x=[r["day"] for r in trace.rows], y=mine, name="This agent", mode="lines",
                    line=dict(color=BLUE, width=3.5), hovertemplate="%{y:.3f}")
    # Zoom in on where the lines are: the first days (no stock yet) would otherwise flatten them.
    settled = [v for curve in (baseline or [], oracle or [], mine) for v in curve[4:]]
    if settled:
        fig.update_yaxes(range=[max(0.0, min(settled) - 0.04), min(1.02, max(settled) + 0.02)], rangemode="normal")
    else:
        fig.update_yaxes(range=[0, 1.05], rangemode="normal")
    _mark_today(fig, trace)
    return fig


def storage_figure(trace: EpisodeTrace) -> go.Figure:
    """How full each storage pool is, day by day; deliveries beyond 100% are wasted."""
    fig = _figure(trace, "Pool in use (%)")
    for i, name in enumerate(trace.config.storage_pools):
        pool = trace.config.storage_pools[name]
        fig.add_scatter(
            x=[r["day"] for r in trace.rows], y=[100.0 * r["storage"].get(name, 0.0) for r in trace.rows],
            name=f"{name}{' (refrigerated)' if pool.cold_chain else ''}", mode="lines",
            line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=2.5), hovertemplate="%{y:.0f}%",
        )
    wasted = [(r["day"], r["waste"]) for r in trace.rows if r["waste"] > 0]
    if wasted:
        fig.add_scatter(x=[d for d, _ in wasted], y=[100.0] * len(wasted), name="Delivery did not fit (wasted)",
                        mode="markers", marker=dict(color="#c8472b", size=10, symbol="x"),
                        text=[f"{w:,.0f} units wasted" for _, w in wasted], hovertemplate="%{text}")
    fig.add_hline(y=100, line_width=1.5, line_dash="dash", line_color="#c8472b")
    fig.update_yaxes(range=[0, 108], rangemode="normal")
    _mark_today(fig, trace)
    return fig


def unmet_heatmap(trace: EpisodeTrace) -> go.Figure:
    sku_ids = list(trace.config.skus)
    days = trace.config.no_of_days
    z: List[List[Optional[float]]] = [[None] * days for _ in sku_ids]
    for r in trace.rows:
        for i, sku_id in enumerate(sku_ids):
            z[i][r["day"]] = round((1.0 - r["skus"][sku_id]["fill"]) * 100, 1)
    fig = _figure(trace, "", height=100 + 36 * len(sku_ids))
    fig.add_heatmap(
        z=z, x=list(range(days)), y=sku_ids, zmin=0, zmax=100, colorscale=UNMET_SCALE,
        xgap=2, ygap=2, hoverongaps=False,
        colorbar=dict(title="Unmet %", thickness=12, outlinewidth=0),
        hovertemplate="%{y}, day %{x}: %{z:.0f}% of demand unmet<extra></extra>",
    )
    fig.update_layout(hovermode="closest", margin=dict(b=48))
    fig.update_yaxes(title=None, gridcolor="rgba(0,0,0,0)", autorange="reversed")
    return fig


def preview_figure(config: EpisodeConfig) -> go.Figure:
    """What a task does to each product: demand and lead time as multiples of normal (the hidden plan)."""
    fig = _base_figure(config.no_of_days, "× normal", height=380)
    touched = False
    for i, (sku_id, sku) in enumerate(config.skus.items()):
        colour = CATEGORICAL[i % len(CATEGORICAL)]
        demand = sku.demand_curve or []
        lead = sku.lead_time_curve or []
        if demand and (max(demand) > 1.001 or min(demand) < 0.999):
            touched = True
            fig.add_scatter(x=list(range(len(demand))), y=demand, name=f"{sku_id}: demand", mode="lines",
                            line=dict(color=colour, width=2.5), hovertemplate="%{y:.2f}×")
        if lead and max(lead) > 1.001:
            touched = True
            fig.add_scatter(x=list(range(len(lead))), y=lead, name=f"{sku_id}: lead time", mode="lines",
                            line_shape="hv", line=dict(color=colour, width=2.5, dash="dash"), hovertemplate="%{y:.2f}×")
    if not touched:
        fig.add_annotation(text="Nothing unusual happens in this task: steady demand, normal lead times.",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=14))
    fig.add_hline(y=1, line_width=1, line_color=GRID)
    return fig


def comparison_figure(table: pd.DataFrame) -> go.Figure:
    """Mean score per agent with its 95% interval."""
    fig = go.Figure()
    if len(table):
        colours = [ORANGE if "Oracle" in a else INK_MUTED if a in (AGENT_BASELINE, AGENT_NOTHING) else BLUE
                   for a in table["Agent"]]
        fig.add_bar(x=table["Mean score"], y=table["Agent"], orientation="h",
                    marker=dict(color=colours, cornerradius=4),
                    error_x=dict(type="data", array=table["± 95%"], color=INK_MUTED, thickness=1.5),
                    text=[f"{v:.3f}" for v in table["Mean score"]], textposition="inside", insidetextanchor="start",
                    hovertemplate="%{y}: %{x:.3f}<extra></extra>")
    fig.update_layout(
        height=120 + 46 * max(len(table), 1), margin=dict(l=16, r=16, t=40, b=48),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=INK_MUTED),
        xaxis=dict(title="Mean score (0 to 1)", range=[0, 1.02], gridcolor=GRID, zeroline=False),
        yaxis=dict(autorange="reversed", automargin=True),
    )
    return fig


# ---------------------------------------------------------------------------
# Tiles, tables and text
# ---------------------------------------------------------------------------

def kpi_html(trace: EpisodeTrace) -> str:
    rows = trace.rows
    sku_days = [s for r in rows for s in r["skus"].values()]
    fill = sum(r.get("fill", 0.0) for r in rows) / len(rows) if rows else 0.0
    stockouts = sum(1 for s in sku_days if s["fill"] < 0.999)
    waste = sum(r["waste"] for r in rows)
    score = score_so_far(trace)
    tiles = [
        ("Day", f"{len(rows)} / {trace.config.no_of_days}", ""),
        ("Score so far", f"{score:.3f}", ""),
    ]
    baseline = trace.references.get("Baseline")
    if baseline and rows:
        delta = score - baseline[len(rows) - 1]
        css = "ig-good" if delta > 0.0005 else "ig-bad" if delta < -0.0005 else ""
        tiles.append(("Against the baseline", f'<span class="{css}">{delta:+.3f}</span>', "same episode, same day"))
    tiles += [
        ("Demand served", f"{fill:.0%}", "weighted by shortage cost"),
        ("Shortage product-days", f"{stockouts}", ""),
        ("Units wasted", f"{waste:,.0f}", "deliveries that did not fit"),
    ]
    if trace.llm_errors:
        tiles.append(("LLM errors", f"{trace.llm_errors}", ""))
    accents = ["blue", "teal", "violet", "teal", "orange", "orange", "orange"]
    cells = "".join(
        f'<div class="ig-tile ig-a-{accents[i % len(accents)]}"><div class="ig-label">{label}</div>'
        f'<div class="ig-value">{value}</div><div class="ig-sub">{sub}</div></div>'
        for i, (label, value, sub) in enumerate(tiles)
    )
    return f'<div class="ig-kpis">{cells}</div>'


def news_html(trace: EpisodeTrace) -> str:
    """The news published so far, newest first; items new today are marked."""
    if trace.news_level == 0:
        return '<div class="ig-hint">News is off for this run. Switch it on to announce disruptions a few days ahead.</div>'
    seen: Dict[str, bool] = {}
    for r in trace.rows:
        for item in r["news"]:
            seen[item] = item in r["new_news"] and r is trace.rows[-1]
    if not trace.rows:
        seen = {item: True for item in trace.state.news}
    if not seen:
        return '<div class="ig-hint">No news yet. Disruptions are announced 5 to 10 days before they start.</div>'
    items = []
    for item, is_new in reversed(list(seen.items())):
        day, _, text = item.partition(": ")
        items.append(f'<div class="ig-news-item{" ig-news-new" if is_new else ""}">'
                     f'<span class="ig-day">{html.escape(day)}</span>{html.escape(text)}</div>')
    return f'<div class="ig-news">{"".join(items)}</div>'


def verdict_html(trace: EpisodeTrace, agent_label: str) -> str:
    """After the episode: the score next to the baseline and the oracle on the same seed."""
    if not trace.finished or not trace.references:
        return ""
    score = score_so_far(trace)
    baseline, oracle = trace.references["Baseline"][-1], trace.references["Oracle"][-1]
    share = gap_closed(score, baseline, oracle)
    if agent_label in (AGENT_BASELINE, AGENT_ORACLE) or share is None:
        note = "The baseline and the oracle mark the range a good agent plays in."
    elif share >= 1:
        note = "It matched the oracle, which knows the future."
    elif share > 0.02:
        note = f"It closed <b>{share:.0%}</b> of the gap between the baseline and the oracle."
    elif share < -0.02:
        note = "It did <b>worse than the baseline rule</b> on this episode."
    else:
        note = "It did about as well as the baseline rule."
    return (f'<div class="ig-card ig-ok ig-verdict">🏁 Final score <b>{score:.3f}</b> · baseline {baseline:.3f} · '
            f"oracle {oracle:.3f}<div class='ig-sub' style='font-size:14px'>{note}</div></div>")


def decisions_table(trace: EpisodeTrace) -> pd.DataFrame:
    records = []
    for r in reversed(trace.rows):
        reply = " ".join(r["reply"].split())
        records.append({
            "Day": r["day"],
            "Agent reply": reply if len(reply) <= 200 else reply[:197] + "...",
            "Orders accepted": ", ".join(f"{k}: {v:,.0f}" for k, v in r["orders"].items()) or "none",
            "Worst fill": f"{min(s['fill'] for s in r['skus'].values()):.0%}",
            "Reward": round(r["reward"], 3),
            "Error": r["error"] or "",
        })
    return pd.DataFrame(records, columns=["Day", "Agent reply", "Orders accepted", "Worst fill", "Reward", "Error"])


def sku_table(trace: EpisodeTrace, sku_id: str) -> pd.DataFrame:
    """Table view of the detail charts, so no value is only readable by colour or hover."""
    ordered: Dict[int, float] = {}
    for sku, qty, order_day, _ in trace.orders:
        if sku == sku_id:
            ordered[order_day] = ordered.get(order_day, 0.0) + qty
    records = []
    for r in trace.rows:
        s = r["skus"][sku_id]
        records.append({
            "Day": r["day"],
            "Demand": round(s["demand"], 1),
            "Stock on hand": round(s["inventory"], 1),
            "Fill %": round(s["fill"] * 100, 1),
            "Ordered": round(ordered.get(r["day"], 0.0), 1),
            "Agent lead-time estimate": round(s["agent_lead_time"], 2),
            "Lead times seen on delivery": ", ".join(f"{x:.0f}" for x in s["observed_lead_times"]),
        })
    return pd.DataFrame(records)


def status_html(trace: EpisodeTrace, agent_label: str) -> str:
    """What step 2 says about the run: which episode, how far it is, and how it ended."""
    done = len(trace.rows)
    head = f"<b>{html.escape(trace.task_name)}</b> · seed {trace.seed} · {html.escape(agent_label)}"
    if trace.stopped_reason:
        return f'<div class="ig-card ig-error">{head}<pre>⚠️ {html.escape(trace.stopped_reason)}</pre></div>'
    pct = 100.0 * done / trace.config.no_of_days
    tail = (f"✅ Episode finished · final score <b>{score_so_far(trace):.3f}</b>" if trace.finished
            else f"⏳ Day {done} of {trace.config.no_of_days}")
    return (f'<div class="ig-card"><div>{head}</div><div class="ig-sub" style="font-size:13.5px">{tail}</div>'
            f'<div class="ig-bar"><div style="width:{pct:.0f}%"></div></div></div>')


def render_detail(trace: EpisodeTrace, sku_id: str) -> Tuple[go.Figure, go.Figure, go.Figure, pd.DataFrame]:
    if sku_id not in trace.config.skus:
        sku_id = next(iter(trace.config.skus))
    return stock_figure(trace, sku_id), lead_time_figure(trace, sku_id), orders_figure(trace, sku_id), sku_table(trace, sku_id)


def focus_product(config: EpisodeConfig) -> str:
    """The product an episode is mostly about: the first one a disruption hits."""
    events = find_events(config)
    return events[0]["sku_id"] if events else next(iter(config.skus))


# ---------------------------------------------------------------------------
# Comparing agents
# ---------------------------------------------------------------------------

def _mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    mean = statistics.fmean(values)
    ci = 1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, ci


def final_score(agent: Any, task_name: str, seed: int, world: Optional[World], news: int, scenario: str) -> Tuple[float, int]:
    """(final score, failed model calls) of one episode. An LLM run that stops early scores its unplayed days as 0."""
    trace = None
    for trace in run_episode(task_name, seed, agent, world=world, news=news, scenario=scenario):
        pass
    return average_step_reward(trace.state, trace.config.no_of_days), trace.llm_errors


COMPARE_COLUMNS = ["Agent", "Mean score", "± 95%", "Episodes", "vs baseline", "Gap to oracle closed"]


def comparison_html(table: pd.DataFrame) -> str:
    """The comparison as a plain HTML table (best agent first), or a hint while there is nothing to show."""
    if not len(table):
        return '<div class="ig-hint">The results appear here, best agent first.</div>'
    head = "".join(f"<th>{html.escape(c)}</th>" for c in COMPARE_COLUMNS)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row[c]))}</td>" for c in COMPARE_COLUMNS) + "</tr>"
        for _, row in table.iterrows())
    return f'<table class="ig-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def comparison_table(scores: Dict[str, List[float]]) -> pd.DataFrame:
    """
    One row per agent: mean score with its 95% interval, the mean difference to the
    baseline seed by seed (when the baseline was run) and the share of the baseline-to-oracle
    gap it closes (when both were run). Best first.
    """
    baseline, oracle = scores.get(AGENT_BASELINE), scores.get(AGENT_ORACLE)
    rows = []
    for agent, values in scores.items():
        mean, ci = _mean_ci(values)
        row = {"Agent": agent, "Mean score": round(mean, 4), "± 95%": round(ci, 4), "Episodes": len(values),
               "vs baseline": "", "Gap to oracle closed": ""}
        if baseline and agent != AGENT_BASELINE and len(baseline) >= len(values):
            diff, diff_ci = _mean_ci([v - b for v, b in zip(values, baseline)])
            row["vs baseline"] = f"{diff:+.4f} ± {diff_ci:.4f}"
            if oracle and agent != AGENT_ORACLE and len(oracle) >= len(values):
                possible = sum(o - b for o, b in zip(oracle, baseline))
                gained = sum(v - b for v, b in zip(values, baseline))
                if possible > 1e-9:
                    row["Gap to oracle closed"] = f"{gained / possible:.0%}"
        rows.append(row)
    table = pd.DataFrame(rows, columns=COMPARE_COLUMNS)
    return table.sort_values("Mean score", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

HERO = """
<div class="ig-hero">
  <div class="ig-hero-top"><div class="ig-logo">📦</div>
    <div><h1>InventoryGym</h1><p>Describe an inventory, watch an agent run it, and see what reading the news is worth.</p></div></div>
  <div class="ig-steps">
    <span class="ig-step"><b>1</b>Describe your inventory</span><span class="ig-arrow">→</span>
    <span class="ig-step"><b>2</b>Run an agent</span><span class="ig-arrow">→</span>
    <span class="ig-step"><b>3</b>Compare agents</span></div>
  <div class="ig-legend">
    <span><span class="ig-dot" style="background:#7fb6f5"></span>Blue: what the agent sees or does</span>
    <span><span class="ig-dot" style="background:#ffb18a"></span>Orange: the hidden truth it never sees</span></div>
</div>
"""

STEP1 = ("**Pick a starting point, or build your own.** The examples load at once. In the **Quick builder** you describe your "
         "products in plain words. **Tables** and **File** are for fine-tuning. Nothing changes until you press the button of "
         "the tab you used.")
BUILDER_HELP = ("Plain words in, a working inventory out. **Create my inventory** replaces the inventory above; you can then "
                "fine-tune it under **Tables** or **File**.")
STEP2 = ("Choose a task and an agent and press **Run episode**. With news on, disruptions are announced 5 to 10 days ahead: "
         "agents that read it can prepare, the baseline cannot.")
STEP3 = ("Run several agents on the same seeds of one task. Every agent faces exactly the same episodes, so differences are "
         "taken seed by seed.")
LLM_NOTE = (f"LLM agents call `{inference.API_BASE_URL}` with your token (on the Space, the `HF_TOKEN` secret is used when "
            "the box is empty). The token is used for this run only and is never stored. One model call per simulated "
            "day; the hybrid agent only calls the model on days with news.")


def _tables_update(data: Dict[str, Any]) -> List[Any]:
    t = forms.world_to_tables(data)
    return [t["name"], t["business"], t["days"],
            pd.DataFrame(t["storage"], columns=forms.STORAGE_COLUMNS),
            pd.DataFrame(t["products"], columns=forms.PRODUCT_COLUMNS),
            pd.DataFrame(t["tasks"], columns=forms.TASK_COLUMNS),
            pd.DataFrame(t["events"], columns=forms.EVENT_COLUMNS)]


def build_dashboard(web_manager=None, action_fields=None, metadata=None, is_chat_env=False, title="", quick_start_md="") -> gr.Blocks:
    """The dashboard. The arguments are openenv's gradio_builder signature and unused here."""
    start_text = template_text(TEMPLATE_PHARMA)
    start_world = PHARMA_WORLD
    start_task = "flu_season" if "flu_season" in start_world.tasks else start_world.task_names[0]
    # The tables start with the world's values. They are not filled by a page-load event:
    # that made the page freeze in an endless update loop (Svelte effect_update_depth_exceeded).
    initial = forms.world_to_tables(yaml.safe_load(start_text))

    with gr.Blocks(title="InventoryGym") as demo:
        gr.HTML(STYLE + HERO, apply_default_css=False)
        current = gr.HTML(current_chip_html(start_world))
        world_text = gr.State(start_text)        # the world in use, as file text
        trace_state = gr.State(None)

        with gr.Tabs():
            # ------------------------------------------------------------------ 1
            with gr.Tab("1 · Your inventory"):
                gr.Markdown(STEP1)
                with gr.Row():
                    template = gr.Dropdown(list(TEMPLATES), value=TEMPLATE_PHARMA, label="Start from an example", scale=3)
                    upload = gr.UploadButton("⬆ Upload a world file", file_types=[".yaml", ".yml", ".json"],
                                             type="binary", scale=1)
                summary = gr.HTML(world_summary_html(start_world))
                with gr.Tabs():
                    with gr.Tab("✨ Quick builder"):
                        gr.Markdown(BUILDER_HELP)
                        gr.HTML(section_html("1", "Your business"))
                        with gr.Row():
                            g_name = gr.Textbox(label="Name your inventory", value="my_shop", scale=1)
                            g_business = gr.Textbox(label="What kind of business is it?", value="a small shop", scale=2,
                                                    info="It goes into the LLM's instructions, e.g. a neighbourhood bakery")
                            g_days = gr.Slider(wizard.MIN_DAYS, wizard.MAX_DAYS, value=45, step=5, label="Days in an episode", scale=1)
                        gr.HTML(section_html("2", "Your products", f"one per row; leave unused rows empty (at least {wizard.MIN_PRODUCTS} products)"))
                        g_products = gr.Dataframe(
                            value=pd.DataFrame(wizard.starter_rows(), columns=wizard.GUIDED_COLUMNS, dtype=object),
                            headers=wizard.GUIDED_COLUMNS, datatype=wizard.GUIDED_DATATYPES, interactive=True,
                            row_count=wizard.GUIDED_ROWS, column_count=(len(wizard.GUIDED_COLUMNS), "fixed"), wrap=True,
                            show_label=False)
                        gr.Markdown(wizard.CRITICALITY_HELP)
                        with gr.Accordion("Storage room (optional)", open=False):
                            with gr.Row():
                                g_shelf = gr.Number(value=21, precision=0, label="Shelf: days of sales that fit")
                                g_fridge = gr.Number(value=14, precision=0, label="Fridge: days of sales that fit")
                        gr.HTML(section_html("3", "What can go wrong", "each one becomes a practice scenario; the product hit and the timing change with every seed"))
                        g_scenarios = gr.CheckboxGroup(wizard.SCENARIO_CHOICES, value=wizard.SCENARIO_CHOICES, show_label=False)
                        g_severity = gr.Radio(list(wizard.SEVERITIES), value="Medium", label="How bad should it get?")
                        g_btn = gr.Button("Create my inventory", variant="primary", elem_classes="ig-cta")
                        g_status = gr.Markdown()
                    with gr.Tab("🧮 Tables"):
                        with gr.Row():
                            f_name = gr.Textbox(label="World name", value=initial["name"], scale=1)
                            f_business = gr.Textbox(label="Whose inventory is it? (used in the LLM's prompt)", value=initial["business"], scale=2)
                            f_days = gr.Number(label="Days in an episode", value=initial["days"], precision=0, scale=1)
                        f_storage = gr.Dataframe(value=pd.DataFrame(initial["storage"], columns=forms.STORAGE_COLUMNS), headers=forms.STORAGE_COLUMNS, label="Storage pools", interactive=True,
                                                 row_count=2, column_count=(len(forms.STORAGE_COLUMNS), "fixed"), wrap=True)
                        f_products = gr.Dataframe(value=pd.DataFrame(initial["products"], columns=forms.PRODUCT_COLUMNS), headers=forms.PRODUCT_COLUMNS, label="Products (amounts: a number, or a range like 80-120 drawn per seed)",
                                                  interactive=True, row_count=3,
                                                  column_count=(len(forms.PRODUCT_COLUMNS), "fixed"), wrap=True)
                        f_tasks = gr.Dataframe(value=pd.DataFrame(initial["tasks"], columns=forms.TASK_COLUMNS), headers=forms.TASK_COLUMNS, label="Tasks (the same inventory under different events)",
                                               interactive=True, row_count=2,
                                               column_count=(len(forms.TASK_COLUMNS), "fixed"), wrap=True)
                        f_events = gr.Dataframe(value=pd.DataFrame(initial["events"], columns=forms.EVENT_COLUMNS), headers=forms.EVENT_COLUMNS, label="Events", interactive=True,
                                                row_count=2, column_count=(len(forms.EVENT_COLUMNS), "fixed"), wrap=True)
                        gr.Markdown(forms.EVENT_HELP)
                        apply_tables = gr.Button("Use these tables", variant="primary", elem_classes="ig-cta")
                    with gr.Tab("📄 File (everything)"):
                        editor = gr.Code(value=start_text, language="yaml", label="World file", lines=24, max_lines=40, interactive=True)
                        apply_file = gr.Button("Use this file", variant="primary", elem_classes="ig-cta")
                table_parts = [f_name, f_business, f_days, f_storage, f_products, f_tasks, f_events]

                gr.HTML(section_html("🔍", "Preview what can go wrong", "what a task does to each product (the plan the agent never sees), and the news an agent would read"))
                with gr.Row():
                    p_task = gr.Dropdown(start_world.task_names, value=start_task, label="Task")
                    p_seed = gr.Number(value=42, precision=0, label="Seed")
                    p_btn = gr.Button("Preview")
                p_plot = gr.Plot(label="Demand and lead time, as multiples of normal")
                p_news = gr.HTML()

            # ------------------------------------------------------------------ 2
            with gr.Tab("2 · Run an episode"):
                gr.Markdown(STEP2)
                with gr.Row():
                    task = gr.Dropdown(start_world.task_names, value=start_task, label="Task", scale=2)
                    agent = gr.Dropdown(AGENT_CHOICES, value=AGENT_BASELINE, label="Agent", scale=3)
                agent_hint = gr.HTML(agent_hint_html(AGENT_BASELINE))
                with gr.Row():
                    seed = gr.Number(value=42, precision=0, label="Seed", info="same seed, same episode", scale=2)
                    dice = gr.Button("🎲 New seed", scale=1, min_width=120, elem_classes="ig-dice")
                    news_level = gr.Radio(list(NEWS_LEVELS), value=NEWS_NATURAL, label="News", info=NEWS_INFO, scale=4)
                with gr.Accordion("⚙ More options", open=False):
                    with gr.Row():
                        delay = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Pause per day (seconds)", scale=2)
                        random_scenario = gr.Checkbox(value=False, label="Random scenario (pharma world only)", scale=1)
                    gr.Markdown("**LLM settings** (only for the LLM agents)")
                    with gr.Row():
                        model = gr.Dropdown(MODEL_SUGGESTIONS, value=inference.MODEL_NAME, allow_custom_value=True,
                                            label="Model (pick one or type any model the endpoint serves)")
                        token = gr.Textbox(type="password", label="API token (optional)",
                                           placeholder="Leave blank to use the server's HF_TOKEN secret")
                    gr.Markdown(LLM_NOTE)
                with gr.Row():
                    run_btn = gr.Button("▶ Run episode", variant="primary", elem_classes="ig-cta", scale=3)
                    stop_btn = gr.Button("■ Stop", scale=1)

                status = gr.HTML(ready_html())
                with gr.Column(visible=False) as results:
                    kpis = gr.HTML()
                    verdict = gr.HTML()
                    with gr.Row():
                        score_plot = gr.Plot(label="Score so far: this agent, the baseline and the oracle")
                        storage_plot = gr.Plot(label="Storage in use")
                    gr.HTML(section_html("📰", "News feed"))
                    news_feed = gr.HTML()
                    gr.HTML(section_html("🔍", "One product in detail"))
                    focus = gr.Dropdown(list(start_world.products), value=None, label="Product shown in the charts below")
                    with gr.Row():
                        stock_plot = gr.Plot(label="Stock on hand vs. demand")
                        lead_plot = gr.Plot(label="Lead time: truth vs. the agent's estimate")
                    with gr.Row():
                        orders_plot = gr.Plot(label="Orders placed and delivered")
                        heat_plot = gr.Plot(label="Unmet demand by product and day")
                    with gr.Accordion("Every decision (newest first)", open=False):
                        log = gr.Dataframe(wrap=True, column_widths=["6%", "38%", "28%", "9%", "8%", "11%"])
                    with gr.Accordion("Data table for the selected product", open=False):
                        table = gr.Dataframe()

            # ------------------------------------------------------------------ 3
            with gr.Tab("3 · Compare agents"):
                gr.Markdown(STEP3)
                with gr.Row():
                    c_task = gr.Dropdown(start_world.task_names, value=start_task, label="Task", scale=3)
                    c_news = gr.Radio(list(NEWS_LEVELS), value=NEWS_NATURAL, label="News", info=NEWS_INFO, scale=4)
                    c_seeds = gr.Slider(3, MAX_COMPARE_SEEDS, value=10, step=1, label="Episodes per agent (test seeds)", scale=3)
                c_agents = gr.CheckboxGroup(AGENT_CHOICES, value=[AGENT_BASELINE, AGENT_NEWS_PLAN, AGENT_ORACLE],
                                            label=f"Agents (LLM agents run at most {MAX_COMPARE_SEEDS_LLM} episodes, "
                                                  "with the model and token from step 2)")
                with gr.Row():
                    c_btn = gr.Button("▶ Compare", variant="primary", elem_classes="ig-cta", scale=3)
                    c_stop = gr.Button("■ Stop", scale=1)
                c_status = gr.Markdown("Pick the agents and press **Compare**. A good first comparison: the baseline, the news plan and the oracle.")
                c_plot = gr.Plot(label="Mean score per agent, with 95% interval")
                gr.HTML(section_html("📊", "Results", "differences are taken seed by seed"))
                c_table = gr.HTML(comparison_html(pd.DataFrame(columns=COMPARE_COLUMNS)), apply_default_css=False)

        # ====================================================================== callbacks
        selectors = [task, c_task, p_task, focus, random_scenario]

        def _selectors(world: World) -> List[Any]:
            names = world.task_names
            first = "flu_season" if "flu_season" in names else names[0]
            return [gr.update(choices=names, value=first), gr.update(choices=names, value=first),
                    gr.update(choices=names, value=first), gr.update(choices=list(world.products), value=None),
                    gr.update(visible=is_builtin(world), value=False)]

        def use_text(text: str):
            """Check a world file's text and, if it is valid, make it the world in use."""
            try:
                world = load_world_text(text or "")
                data = yaml.safe_load(text)
            except WorldError as exc:
                return [gr.skip(), world_summary_html(None, str(exc)), gr.skip(), *[gr.skip()] * (len(table_parts) + len(selectors))]
            return [text, world_summary_html(world), text, *_tables_update(data), *_selectors(world)]

        def on_template(name: str):
            return use_text(template_text(name))

        def on_upload(payload: Optional[bytes]):
            if payload is None:
                return [gr.skip()] * (3 + len(table_parts) + len(selectors))
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError:
                return [gr.skip(), world_summary_html(None, "The uploaded file is not text (UTF-8)."), gr.skip(),
                        *[gr.skip()] * (len(table_parts) + len(selectors))]
            return use_text(text)

        def on_apply_tables(current: str, name, business, days, storage, products, tasks_table, events):
            """Rebuild the world file from the tables, keeping what only the file can hold."""
            try:
                base = yaml.safe_load(current) if current else None
                data = forms.tables_to_world(base if isinstance(base, dict) else None, {
                    "name": name, "business": business, "days": days, "storage": storage, "products": products,
                    "tasks": tasks_table, "events": events})
                text = dump_world(data)
                world = load_world_text(text)
            except (forms.FormError, WorldError, yaml.YAMLError) as exc:
                return [gr.skip(), world_summary_html(None, str(exc)), gr.skip(), *[gr.skip()] * len(selectors)]
            return [text, world_summary_html(world), text, *_selectors(world)]

        def on_guided(name, business, days, table, shelf, fridge, scenarios, severity):
            """Build an inventory from the quick builder's plain-word answers."""
            try:
                data = wizard.build_guided_world(name, business, days, table, shelf, fridge, scenarios, severity)
                text = dump_world(data)
                world = load_world_text(text)
            except (forms.FormError, WorldError) as exc:
                # The inventory in use stays as it was; the problem is shown next to the button.
                return [gr.skip()] * (3 + len(table_parts) + len(selectors)) + [f"⚠️ {exc}"]
            note = (f"✅ **{world.name}** is ready: {len(world.products)} products, {len(world.storage)} storage area(s), "
                    f"{len(world.tasks)} practice scenario(s). Open **2 · Run an episode** to watch an agent try it, "
                    "or fine-tune it under **Tables** and **File**.")
            return [text, world_summary_html(world), text, *_tables_update(data), *_selectors(world), note]

        world_outputs = [world_text, summary, editor, *table_parts, *selectors]
        template.change(on_template, inputs=template, outputs=world_outputs)
        upload.upload(on_upload, inputs=upload, outputs=world_outputs)
        apply_file.click(use_text, inputs=editor, outputs=world_outputs)
        g_btn.click(on_guided, inputs=[g_name, g_business, g_days, g_products, g_shelf, g_fridge, g_scenarios, g_severity],
                    outputs=[*world_outputs, g_status], api_name="create_inventory")
        apply_tables.click(on_apply_tables, inputs=[world_text, *table_parts],
                           outputs=[world_text, summary, editor, *selectors])

        def on_preview(text: str, task_name: str, seed_value):
            try:
                world = parse_world(text)
                config = get_task_config(task_name, int(seed_value or 0), news=2, world=world)
            except (WorldError, ValueError) as exc:
                return go.Figure(), world_summary_html(None, str(exc))
            items = "".join(f'<div class="ig-news-item"><span class="ig-day">Day {e.announce_day}</span>'
                            f"{html.escape(e.message)}</div>" for e in sorted(config.news, key=lambda e: e.announce_day))
            news = f'<div class="ig-news">{items}</div>' if items else '<div class="ig-hint">No news: nothing is disrupted in this task.</div>'
            return preview_figure(config), news

        p_btn.click(on_preview, inputs=[world_text, p_task, p_seed], outputs=[p_plot, p_news])

        run_outputs = [status, results, kpis, verdict, score_plot, storage_plot, news_feed, focus, stock_plot, lead_plot,
                       orders_plot, table, heat_plot, log, trace_state]

        def _llm_token(api_token: str) -> str:
            return (api_token or "").strip() or os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""

        def on_run(text, task_name, seed_value, news_choice, agent_choice, pause, use_random, model_name, api_token):
            def message(note: str):
                return (note_html(note), *[gr.skip()] * (len(run_outputs) - 1))

            try:
                world = parse_world(text)
            except WorldError as exc:
                yield message(f"⚠️ The inventory in step 1 is not valid: {exc}")
                return
            if task_name not in world.tasks:
                yield message("⚠️ Pick a task of this inventory first.")
                return
            news = NEWS_LEVELS.get(news_choice, 0)
            scenario = "random" if use_random and is_builtin(world) else "fixed"
            seed_int = int(seed_value) if seed_value is not None else 42
            key = ""
            if agent_choice in LLM_AGENTS:
                key = _llm_token(api_token)
                if not key:
                    yield message("⚠️ The LLM agents need a token: paste one under **LLM settings**, "
                                  "or set the `HF_TOKEN` secret on the Space.")
                    return
                if agent_choice == AGENT_LLM_NEWS and news == 0:
                    yield message("⚠️ The hybrid agent only reads the news: switch **News** on.")
                    return
            try:
                runner = make_agent(agent_choice, (model_name or "").strip() or inference.MODEL_NAME, key)
                references = reference_curves(task_name, seed_int, world, news, scenario)
                frames = run_episode(task_name, seed_int, runner, world=world, news=news, scenario=scenario)
                sku_id = None
                for trace in frames:
                    trace.references = references
                    first = sku_id is None
                    sku_id = sku_id or focus_product(trace.config)
                    yield (
                        status_html(trace, agent_choice), gr.update(visible=True) if first else gr.skip(),
                        kpi_html(trace), verdict_html(trace, agent_choice),
                        score_figure(trace), storage_figure(trace), news_html(trace),
                        gr.update(choices=list(trace.config.skus), value=sku_id) if first else gr.skip(),
                        *render_detail(trace, sku_id), unmet_heatmap(trace), decisions_table(trace), trace,
                    )
                    if pause and trace.rows and not (trace.finished or trace.stopped_reason):
                        time.sleep(pause)
            except (WorldError, ValueError) as exc:
                yield message(f"⚠️ {exc}")

        def on_focus(trace, sku_id):
            if trace is None or not sku_id:
                return [gr.skip()] * 4
            return render_detail(trace, sku_id)

        run_event = run_btn.click(
            on_run, inputs=[world_text, task, seed, news_level, agent, delay, random_scenario, model, token],
            outputs=run_outputs,
        )
        stop_btn.click(lambda: note_html("⏹ Stopped.", "info"), outputs=status, cancels=[run_event])
        agent.change(agent_hint_html, inputs=agent, outputs=agent_hint)
        dice.click(lambda: random.randint(1, 99999), outputs=seed)

        def on_world_change(text: str):
            try:
                return current_chip_html(parse_world(text))
            except WorldError:
                return gr.skip()

        world_text.change(on_world_change, inputs=world_text, outputs=current)
        focus.input(on_focus, inputs=[trace_state, focus], outputs=[stock_plot, lead_plot, orders_plot, table])

        def on_compare(text, task_name, news_choice, n_seeds, chosen, model_name, api_token):
            empty = pd.DataFrame(columns=COMPARE_COLUMNS)
            try:
                world = parse_world(text)
            except WorldError as exc:
                yield f"⚠️ The inventory in step 1 is not valid: {exc}", gr.skip(), gr.skip()
                return
            if task_name not in world.tasks or not chosen:
                yield "⚠️ Pick a task and at least one agent.", gr.skip(), gr.skip()
                return
            news = NEWS_LEVELS.get(news_choice, 0)
            n = max(1, min(int(n_seeds or 1), MAX_COMPARE_SEEDS))
            key = _llm_token(api_token)
            scores: Dict[str, List[float]] = {}
            notes: List[str] = []
            yield f"⏳ Running {len(chosen)} agent(s) on {n} episode(s) of **{task_name}**...", comparison_figure(empty), comparison_html(empty)
            for label in [a for a in AGENT_CHOICES if a in chosen]:
                if label in LLM_AGENTS and not key:
                    notes.append(f"{label}: skipped, it needs a token (step 2, LLM settings).")
                    continue
                if label == AGENT_LLM_NEWS and news == 0:
                    notes.append(f"{label}: skipped, it only reads the news and news is off.")
                    continue
                seeds = list(TEST_SEEDS[:min(n, MAX_COMPARE_SEEDS_LLM) if label in LLM_AGENTS else n])
                values, errors = [], 0
                for s in seeds:
                    try:
                        value, failed = final_score(make_agent(label, (model_name or "").strip() or inference.MODEL_NAME, key),
                                                    task_name, s, world, news, "fixed")
                    except (WorldError, ValueError) as exc:
                        notes.append(f"{label}: {exc}")
                        break
                    values.append(value)
                    errors += failed
                    scores[label] = values
                    result = comparison_table(scores)
                    yield (f"⏳ {label}: episode {len(values)} of {len(seeds)}...", comparison_figure(result), comparison_html(result))
                if errors:
                    notes.append(f"{label}: {errors} model call(s) failed; episodes that stopped early score their unplayed days as 0.")
            if not scores:
                yield "⚠️ Nothing was run. " + " ".join(notes), comparison_figure(empty), comparison_html(empty)
                return
            result = comparison_table(scores)
            done = f"✅ Done: **{task_name}**, news {news_choice.lower()}, test seeds {TEST_SEEDS[0]} to {TEST_SEEDS[n - 1]}."
            yield done + ("\n\n" + "\n".join(f"- {x}" for x in notes) if notes else ""), comparison_figure(result), comparison_html(result)

        compare_event = c_btn.click(on_compare, inputs=[world_text, c_task, c_news, c_seeds, c_agents, model, token],
                                    outputs=[c_status, c_plot, c_table])
        c_stop.click(lambda: "⏹️ Stopped.", outputs=c_status, cancels=[compare_event])

    return demo
