"""
inference.py
============
Baseline inference script for the Pharma Inventory Management Environment.

Runs every task in tasks.FIXED_TASKS (supply_chain_broken, flu_season,
epidemic_two_wave) in sequence. Each task is a separate 60-day episode.

Environment variables
---------------------
API_BASE_URL       LLM API endpoint  (default: HuggingFace router)
MODEL_NAME         Model identifier  (default: Qwen2.5-72B-Instruct)
HF_TOKEN           API key
LOCAL_IMAGE_NAME   Docker image name (used by from_docker_image)
ENV_BASE_URL       Server URL when not using Docker (default: localhost:8000)
TASK_SEED          Seed for task generation and environment noise (default: 42)

STDOUT format (mandatory)
--------------------------
[START] task=<task> env=<benchmark> model=<model>
[STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Score formula (see tasks.compute_final_score)
---------------------------------------------
    final_score = clip(sum_days(step_reward) / no_of_days, 0, 1)
    step_reward = weighted_sku(demand_fulfilled_today) - waste_fraction_today
                  (fill rates averaged with stockout_penalty as the weight)

    waste_fraction_today is the waste_penalty-weighted share of the day's
    deliveries rejected because storage was full.

Success = score >= SUCCESS_THRESHOLD (0.80)

Action history
--------------
Every LLM call receives the recent (day, action, feedback) history from the
current episode, where feedback summarises the fill rate per SKU and any
overflow waste caused by the previous action.
"""

import asyncio
import json
import math
import re
import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from client import InventoryEnvClient
from models import EpisodeConfig, InventoryState, InventoryAction
from tasks import FIXED_TASKS, compute_final_score, get_task_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME        = os.getenv("LOCAL_IMAGE_NAME")
API_KEY           = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL      = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME        = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
TASK_SEED         = int(os.getenv("TASK_SEED", "42"))

BENCHMARK         = "pharma_coldchain_env"
TEMPERATURE       = 0.2
MAX_TOKENS        = 256         # orders can be multi-SKU JSON
SUCCESS_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# Who the policy is, the warehouse, and the report fields.
_PROMPT_INTRO = textwrap.dedent("""
    You are a pharmaceutical warehouse procurement agent managing a cold-chain inventory over a multi-day episode.

    OBJECTIVE
    ---------
    Every day you receive an inventory report and decide what to order.
    Maximise prescription fill rate while avoiding stockouts, waste, and capacity overflow.
    Shortages are weighted by stockout_penalty in your score: a day out of stock of a SKU with
    stockout_penalty 100 costs 20 times more than the same day for a SKU with 5.

    WAREHOUSE
    ---------
    Two storage pools:
      cold_storage  — refrigerated, for cold-chain SKUs only (cold_storage_required = True).
      ambient       — standard storage, for all other SKUs.
    cold_days and ambient_days in the report: how many days of current demand each pool can
    hold, for all its SKUs together (null until any demand has been observed).
    Do not order quantities that would exceed available capacity.
    
    SKUS
    ----
    Each SKU in the report shows:
      inventory_on_hand            — units physically on shelf right now.
      inbound_units                — units you have already ordered that have not arrived yet.
      days_of_cover                — (inventory_on_hand + inbound_units) / recent daily demand:
                                     how many days your stock plus open orders will last.
                                     null until any demand has been observed.
      avg_demand_per_day           — daily demand averaged over the whole episode so far.
      avg_demand_last_5_days       — recent daily demand. Reacts faster to spikes.
      avg_lead_time                — average days from order placement to arrival.
      lead_time_last3_orders       — actual lead times of the last 3 deliveries. Rising = supply stress.
      stockout_days_if_no_reorder  — days until the shelf is empty, ignoring open orders.
      stockout_penalty             — how critical this SKU is, and its weight in your score.
                                     Higher = protect it first.
      waste_penalty                — cost of this SKU's deliveries overflowing storage.
      cold_storage_required        — true means this SKU uses the cold storage pool.
    When there are insights, the report also has an "insights" list: announced disruptions (supplier
    delays, demand surges), each starting with the day it was published.
""").strip()

# How the model answers: an automatic rule orders# Action format "adjust": an automatic rule orders a sensible default for every SKU and
# the model only answers how many days to keep above or below it.
_ADJUST_RULES = textwrap.dedent("""
    YOUR DECISION: ADJUSTMENT DAYS
    ------------------------------
    You do not order units. An automatic rule already orders every SKU up to a sensible
    default: stock on hand plus inbound_units covering the SKU's lead time (the larger of
    avg_lead_time and lead_time_last3_orders) plus 3 safety days of recent demand.
    For EVERY SKU, choose ADJUSTMENT DAYS: how many EXTRA days of recent demand to keep on
    top of that default. 0 keeps the default, which is already a reasonable plan; add days
    only when the report gives you a reason.

    DECISION RULES
    --------------
    1. Answer for EVERY SKU in the report, every day. A SKU you leave out counts as 0.
    2. Demand rising: if avg_demand_last_5_days is clearly above avg_demand_per_day, add a
       few days; the faster it rises, the more days.
    3. Suppliers slowing: if the latest values in lead_time_last3_orders are clearly above
       avg_lead_time, add a few days.
    4. Recent shortages: if a SKU appears in short_skus in the recent results, add a few days.
    5. Critical SKUs: a shortage of a SKU with a high stockout_penalty costs the most, so
       when one of the reasons above applies to it, lean towards more days.
    6. Insights: for an announced supplier delay, add about (slowdown factor - 1) x avg_lead_time
       days from the day the insights appear until the delay is over; for an announced demand
       surge, add a few days until its peak. Products the insights do not mention: no change.
    7. Storage: each pool holds only cold_days / ambient_days days of its SKUs' demand,
       shared by all its SKUs. Large adjustments for many SKUs at once overflow storage, and deliveries that do
       not fit are wasted; wasted insulin costs the most. If cold_ratio or ambient_ratio is
       high, add days only where a shortage is likely, favouring a higher stockout_penalty.
    8. Adjustments run from 0 to +10 days.

    ACTION FORMAT
    -------------
    Respond with one JSON object mapping EVERY SKU name to its adjustment days. Use the exact
    SKU names from the report. The values are days, not units.

    Format:
        {"<sku_name>": <adjustment_days>, "<sku_name>": <adjustment_days>, ...}
""").strip()

SYSTEM_PROMPT = _PROMPT_INTRO + "\n\n" + _ADJUST_RULES
# Range of an answer, in days added to the default cover (lead time + 3 days).
# No negative values: keeping less than the default never helped (all -1: 0.853, all -3:
# 0.728, vs 0.869 for 0, on 20 test seeds per task), it only added stockouts.
ADJUST_DAYS_RANGE = (0.0, 10.0)



# ---------------------------------------------------------------------------
# Logging helpers (mandatory STDOUT format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = " ".join(error.split()) if error else "null"
    # The log format is line-based: collapse any newlines/indentation from the LLM.
    action = " ".join(action.split()) or "{}"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def inbound_units_by_sku(obs: InventoryState) -> Dict[str, float]:
    """Units ordered but not yet arrived, per SKU (from the policy-visible expected orders)."""
    inbound = {sku_id: 0.0 for sku_id in obs.skus}
    for sku_id, qty, _ in obs.expected_inbound_orders:
        inbound[sku_id] = inbound.get(sku_id, 0.0) + qty
    return inbound


def days_of_cover(on_hand: float, inbound: float, recent_demand: float) -> Optional[float]:
    """How many days stock plus open orders last at the recent demand rate; None if no demand seen yet."""
    if recent_demand <= 0:
        return None
    return round((on_hand + inbound) / recent_demand, 1)


def storage_days(obs: InventoryState, cold: bool) -> Optional[float]:
    """
    Days of current demand a storage pool can hold for all its SKUs together:
    capacity / sum of recent daily demand (the larger of the last-5-day and overall
    averages) of the SKUs stored in it. None until any demand has been observed.
    """
    total = obs.cold_storage_total_capacity if cold else obs.ambient_storage_total_capacity
    demand = sum(max(s.avg_demand_last_5_days, s.avg_demand_per_day)
                 for s in obs.skus.values() if bool(s.cold_storage_required) == cold)
    return round(total / demand, 1) if demand > 0 else None


def pool_days(obs: InventoryState, pool_name: str) -> Optional[float]:
    """storage_days() for one named pool: its capacity / recent demand of the SKUs in it."""
    demand = sum(max(s.avg_demand_last_5_days, s.avg_demand_per_day)
                 for s in obs.skus.values() if s.storage_pool == pool_name)
    return round(obs.storage[pool_name].total_capacity / demand, 1) if demand > 0 else None


def storage_report(obs: InventoryState) -> Dict[str, Any]:
    """
    The report's storage section. One cold-chain pool and one other pool (the pharma
    world, and anything built from cold/ambient capacities) keep the original flat
    cold_*/ambient_* keys, so prompts and trained models stay valid; any other layout
    lists every pool by name.
    """
    if not obs.storage or obs.has_classic_storage():
        return {
            "cold_total":    round(obs.cold_storage_total_capacity, 1),
            "cold_used":     round(obs.cold_storage_current_capacity, 1),
            "cold_ratio":    round(obs.cold_storage_ratio, 3),
            "ambient_total": round(obs.ambient_storage_total_capacity, 1),
            "ambient_used":  round(obs.ambient_storage_current_capacity, 1),
            "ambient_ratio": round(obs.ambient_storage_ratio, 3),
            "cold_days":     storage_days(obs, cold=True),
            "ambient_days":  storage_days(obs, cold=False),
        }
    return {
        name: {
            "total":      round(pool.total_capacity, 1),
            "used":       round(pool.current_capacity, 1),
            "ratio":      round(pool.ratio, 3),
            "days":       pool_days(obs, name),
            "cold_chain": pool.cold_chain,
        }
        for name, pool in obs.storage.items()
    }


def observation_to_dict(obs: InventoryState, include_cover: bool = True) -> Dict[str, Any]:
    """
    Convert InventoryState to a clean dict for the LLM prompt.
    Shows only what the policy may observe, plus two values computed from it
    (inbound_units, days_of_cover) so a model does not have to do that
    arithmetic itself. Hidden ground truth is never included.

    include_cover=False leaves days_of_cover out: in the "adjust" format small models
    copied it as their answer (Qwen 2.5 3B did so until its stock ran out).
    """
    inbound = inbound_units_by_sku(obs)
    report: Dict[str, Any] = {
        "day":  obs.current_date,
        "storage": storage_report(obs),
        "expected_inbound_orders": obs.expected_inbound_orders,
        "inventory": {
            sku_id: {
                "inventory_on_hand":            round(sku.inventory_on_hand, 1),
                "inbound_units":                round(inbound[sku_id], 1),
                "days_of_cover":                days_of_cover(
                    sku.inventory_on_hand, inbound[sku_id],
                    max(sku.avg_demand_last_5_days, sku.avg_demand_per_day),
                ),
                "avg_demand_per_day":           round(sku.avg_demand_per_day, 2),
                "avg_demand_last_5_days":       round(sku.avg_demand_last_5_days, 2),
                "avg_lead_time":                round(sku.avg_lead_time, 1),
                "lead_time_last3_orders":       sku.lead_time_last3_orders,
                "stockout_days_if_no_reorder":  round(sku.stockout_days_if_no_reorder, 1),
                "cold_storage_required":        sku.cold_storage_required,
                "stockout_penalty":             sku.stockout_penalty,
                "waste_penalty":                sku.waste_penalty,
            }
            for sku_id, sku in obs.skus.items()
        },
    }
    if obs.storage and not obs.has_classic_storage():
        # Pools are listed by name: say which one each SKU uses.
        for sku_id, sku in report["inventory"].items():
            sku["storage_pool"] = obs.skus[sku_id].storage_pool
    if not include_cover:
        for sku in report["inventory"].values():
            del sku["days_of_cover"]
    if obs.insights:
        report["insights"] = list(obs.insights)
    return report


def format_report(obs_dict: Dict[str, Any]) -> str:
    """
    Render the report as valid JSON with one line per section and per SKU, which
    reads well and uses far fewer tokens than fully indented JSON.
    """
    def compact(value: Any) -> str:
        return json.dumps(value, separators=(", ", ": "))

    lines = ["{"]
    keys = list(obs_dict)
    for i, key in enumerate(keys):
        comma = "," if i < len(keys) - 1 else ""
        value = obs_dict[key]
        if isinstance(value, dict) and value and all(isinstance(v, dict) for v in value.values()):
            lines.append(f'  "{key}": {{')
            items = list(value.items())
            for j, (name, entry) in enumerate(items):
                lines.append(f'    "{name}": {compact(entry)}' + ("," if j < len(items) - 1 else ""))
            lines.append("  }" + comma)
        else:
            lines.append(f'  "{key}": {compact(value)}{comma}')
    lines.append("}")
    return "\n".join(lines)


def build_feedback(obs: InventoryState) -> str:
    """Summarise the outcome of the last step for the policy's action history."""
    fill = {
        sku_id: round(sku.demand_fulfilled_today, 2)
        for sku_id, sku in obs.skus.items()
    }
    stockouts = [sku_id for sku_id, f in fill.items() if f < 1.0]
    return json.dumps({
        "fill_rate": fill,
        "short_skus": stockouts,
        "units_wasted_overflow": round(obs.inventory_excess_today, 1),
        "waste_fraction": round(obs.waste_fraction_today, 3),
        "reward": round(obs.reward or 0.0, 3),
    })


HISTORY_DAYS = 3


def build_action_history_str(
    action_history: List[Tuple[int, str, str]],
    max_entries: int = HISTORY_DAYS,
) -> str:
    """
    Build a compact JSON string of the results of the last few days.

    action_history holds (day, action_str, feedback_str) tuples. Only the day and
    its results (fill rates, short SKUs, waste, reward) are shown, never the
    model's own earlier replies: small models tend to copy those and repeat the
    same order every day. What is already on order is visible in the report
    (inbound_units), so the raw replies are not needed.

    Returns a formatted string block, or an empty string if there is no history yet.
    """
    if not action_history:
        return ""

    entries = []
    for day, _action, feedback in action_history[-max_entries:]:
        try:
            result = json.loads(feedback)
        except (TypeError, ValueError):
            result = feedback
        entries.append({"day": day, "result": result})
    return json.dumps(entries)


def build_user_prompt(
    obs_dict: Dict[str, Any],
    action_history: List[Tuple[int, str, str]],
    step: int,
) -> str:
    """
    Build the user-turn message from:
      1. The current inventory report (today's state)
      2. The action history (what has happened so far this episode)

    The history gives the policy memory across steps without requiring
    the LLM to maintain its own internal state.
    """
    parts = [f"Day {obs_dict['day']} — Step {step}"]
    history_str = build_action_history_str(action_history)
    if history_str:
        n_days = min(HISTORY_DAYS, len(action_history))
        parts.append(f"--- RESULTS OF THE LAST {n_days} DAY{'S' if n_days != 1 else ''} ---\n{history_str}")
    parts.append(f"--- TODAY'S INVENTORY REPORT ---\n{format_report(obs_dict)}")
    parts.append(
        "--- YOUR DECISION ---\n"
        "For every SKU, choose its adjustment days (see the rules; 0 keeps the default), then "
        "respond with a single JSON object on one line:\n"
        '{"<sku_name>": <adjustment_days>, ...}'
    )
    return "\n\n".join(parts)


def parse_json_object(text: str) -> Dict[str, Any]:
    """The first {...} block in a reply as a dict (also accepts {"orders": {...}}); {} if none."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group())
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    if isinstance(parsed.get("orders"), dict):
        parsed = parsed["orders"]
    return parsed



# ---------------------------------------------------------------------------
# Insights adviser (hybrid policy): the LLM reads the insights, the baseline does the maths
# ---------------------------------------------------------------------------

INSIGHTS_SYSTEM_PROMPT = textwrap.dedent("""
    You advise the procurement team of a pharmaceutical warehouse about disruptions announced in the insights.

    An automatic ordering rule already keeps, for every product, enough stock for its lead time plus
    3 safety days of recent demand. It only reacts after a problem shows up in deliveries or sales,
    which is too late. Your job is to read today's insights and decide how many EXTRA days of stock to keep
    for each product, so the warehouse is ready before the disruption hits.

    PRODUCTS (use these exact names in your answer)
      insulin             - insulin, kept in cold storage
      paracetamol         - paracetamol, fever and pain relief
      bp_medication       - blood-pressure medication
      vitamins            - vitamins
      hydroxychloroquine  - hydroxychloroquine

    GUIDELINES
    1. Supplier delay, deliveries about X times slower: extra days = about (X - 1) x avg_lead_time.
       Keep it from the day the insights appear until the delay is over.
    2. Demand surge, demand rising to about X times normal: extra days = about (avg_lead_time + 3) x (X - 1) / 2
       from the day the insights appear until the peak; 0 after the peak (recent sales already show the
       higher demand, and stock bought for the peak would overflow storage as demand falls).
    3. If both happen to one product, add the two.
    4. Products the insights do not affect, and events that are over: 0.
    5. Shortages are weighted by stockout_penalty in the score: for a product with a high
       stockout_penalty, lean towards the upper end of these estimates.
    6. Storage is limited, so never more than 15 extra days.

    Work out each product's extra days from today's insights and that product's numbers in the report.
    Respond with one JSON object on one line giving extra days for EVERY product, using the exact
    product names above:
    {"<product_name>": <extra_days>, "<product_name>": <extra_days>, ...}
""").strip()

# Names a model may use for a product instead of its sku_id.
SKU_ALIASES = {
    "blood-pressure medication": "bp_medication",
    "blood pressure medication": "bp_medication",
    "amlodipine": "bp_medication",
    "hcq": "hydroxychloroquine",
    "vitamin": "vitamins",
    "vitamin supplements": "vitamins",
}


# ---------------------------------------------------------------------------
# Prompts written from the world
# ---------------------------------------------------------------------------
# The constants above are the built-in pharma world's prompts and never change, so models
# trained on them stay valid. An episode of any other world carries a context
# (InventoryState.context: the business and the products' names); its prompts are the
# same texts with the pharma-specific sentences swapped for ones about that world.

def _swap(text: str, old: str, new: str) -> str:
    """text with `old` replaced by `new`; `old` must be there, so a reworded prompt cannot silently skip a swap."""
    if old not in text:
        raise ValueError(f"prompt text not found: {old[:60]!r}")
    return text.replace(old, new)


def _swap_if_present(text: str, swaps: List[Tuple[str, str]]) -> str:
    for old, new in swaps:
        text = text.replace(old, new)
    return text


def has_own_words(obs: Optional[InventoryState]) -> bool:
    """True when the episode's world supplies its own wording (every world but the built-in pharma one)."""
    return obs is not None and obs.context is not None


def has_classic_storage(obs: InventoryState) -> bool:
    """One cold-chain pool and one other pool: the report uses the cold_*/ambient_* keys."""
    return not obs.storage or obs.has_classic_storage()


def products_text(obs: InventoryState, cold_note: bool = False) -> str:
    """One line per SKU: its id, its plain name and description (and, optionally, that it is refrigerated)."""
    width = max(len(sku_id) for sku_id in obs.skus)
    lines = []
    for sku_id, sku in obs.skus.items():
        info = obs.context.products.get(sku_id) if obs.context else None
        text = info.name if info else sku_id.replace("_", " ")
        if info and info.description:
            text += f", {info.description}"
        if cold_note and sku.cold_storage_required:
            text += ", kept in cold storage"
        lines.append(f"  {sku_id:<{width}}  - {text}")
    return "\n".join(lines)


_POOLS_TEXT = textwrap.dedent("""
    The report's storage section lists every storage pool by name, with:
      total, used, ratio  — its capacity in units, the units it holds now, and used / total.
      days                — how many days of current demand the pool can hold, for all its
                            SKUs together (null until any demand has been observed).
      cold_chain          — true for refrigerated storage.
    Each SKU is kept in the pool named by its storage_pool and shares it with that pool's other SKUs.
""").strip()

# Sentences that name pharma products, swapped in every world with its own wording.
_GENERIC_SWAPS = [
    ("wasted insulin costs the most", "wasted deliveries of SKUs with a high waste_penalty cost the most"),
]
# Sentences about the cold/ambient report keys, swapped when the report lists pools by name.
_POOL_SWAPS = [
    ("cold_days / ambient_days days of its SKUs' demand",
     "as many days of its SKUs' demand as the days shown for it in the report"),
    ("If cold_ratio or ambient_ratio is", "If a pool's ratio is"),
]


def _intro(obs: InventoryState) -> str:
    text = _swap(_PROMPT_INTRO,
                 "You are a pharmaceutical warehouse procurement agent managing a cold-chain inventory over a multi-day episode.",
                 f"You are the procurement agent of {obs.context.business}, managing its inventory over a multi-day episode.")
    text = _swap(text, "Maximise prescription fill rate", "Maximise the demand fill rate")
    text = _swap(text, "SKUS\n----", f"PRODUCTS\n--------\n{products_text(obs)}\n\nSKUS\n----")
    if not has_classic_storage(obs):
        start, end = text.index("Two storage pools:"), text.index("Do not order quantities")
        text = text[:start] + _POOLS_TEXT + "\n" + text[end:]
        text = _swap(text, "  cold_storage_required        — true means this SKU uses the cold storage pool.",
                     "  cold_storage_required        — true means this SKU is kept in a refrigerated (cold-chain) pool.\n"
                     "  storage_pool                 — the storage pool this SKU is kept in.")
    return text


def system_prompt(obs: Optional[InventoryState] = None) -> str:
    """
    The system prompt. Without `obs`, or for an episode of the built-in pharma world, the
    original prompt; for any other world, the same prompt about that world: its business,
    its products and, when the report lists storage pools by name, its pools.
    """
    rules = _ADJUST_RULES
    if not has_own_words(obs):
        return _PROMPT_INTRO + "\n\n" + rules
    rules = _swap_if_present(rules, _GENERIC_SWAPS)
    if not has_classic_storage(obs):
        rules = _swap_if_present(rules, _POOL_SWAPS)
    return _intro(obs) + "\n\n" + rules


def insights_system_prompt(obs: Optional[InventoryState] = None) -> str:
    """INSIGHTS_SYSTEM_PROMPT, about the episode's own business and products when it has its own wording."""
    if not has_own_words(obs):
        return INSIGHTS_SYSTEM_PROMPT
    text = _swap(INSIGHTS_SYSTEM_PROMPT, "the procurement team of a pharmaceutical warehouse",
                 f"the procurement team of {obs.context.business}")
    start = text.index("  insulin ")
    end = text.index("\n\nGUIDELINES")
    return text[:start] + products_text(obs, cold_note=True) + text[end:]


def sku_aliases(obs: InventoryState) -> Dict[str, str]:
    """Lower-case names a model may use for a SKU instead of its id -> the sku_id."""
    if not has_own_words(obs):
        return SKU_ALIASES
    aliases: Dict[str, str] = {}
    for sku_id, info in obs.context.products.items():
        for label in (info.name, *info.aliases):
            aliases[label.strip().lower()] = sku_id
    return aliases


def build_insights_prompt(obs: InventoryState) -> str:
    """Today's insights and, per product, the numbers the guidelines use."""
    lines = [f"Today is day {obs.current_date}.", "", "INSIGHTS (each item starts with the day it was published):"]
    lines += [f"- {item}" for item in obs.insights] or ["- none"]
    lines += ["", "PRODUCTS:"]
    for sku_id, sku in obs.skus.items():
        recent = max(sku.avg_demand_last_5_days, sku.avg_demand_per_day)
        lines.append(f"- {sku_id}: " + json.dumps({
            "avg_lead_time": round(sku.avg_lead_time, 1),
            "lead_time_last3_orders": sku.lead_time_last3_orders,
            "avg_demand_per_day": round(sku.avg_demand_per_day, 1),
            "avg_demand_last_5_days": round(sku.avg_demand_last_5_days, 1),
            "days_of_cover": days_of_cover(sku.inventory_on_hand, inbound_units_by_sku(obs)[sku_id], recent),
            "stockout_penalty": sku.stockout_penalty,
        }, separators=(", ", ": ")))
    lines += ["", "Extra days of stock for every product, as one JSON object:"]
    return "\n".join(lines)


def parse_extra_days(obs: InventoryState, reply: Dict[str, Any]) -> Dict[str, float]:
    """
    Extra days per SKU from a parsed reply: SKU names (or aliases) in any case, finite
    numbers only; anything else is ignored. Clipping is left to policies.baseline_policy.
    """
    known = {sku_id.lower(): sku_id for sku_id in obs.skus}
    aliases = sku_aliases(obs)
    extra: Dict[str, float] = {}
    for name, value in reply.items():
        key = str(name).strip().lower()
        sku_id = known.get(key) or known.get(aliases.get(key, "").lower())
        if sku_id is None or isinstance(value, bool):
            continue
        try:
            days = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(days):
            extra[sku_id] = days
    return extra


def adjust_days(obs: InventoryState, reply: Dict[str, Any]) -> Dict[str, float]:
    """Adjustment days per SKU from a parsed "adjust" reply, clipped to ADJUST_DAYS_RANGE."""
    low, high = ADJUST_DAYS_RANGE
    return {sku_id: min(max(days, low), high) for sku_id, days in parse_extra_days(obs, reply).items()}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_llm_action(
    client: OpenAI,
    obs: InventoryState,
    action_history: List[Tuple[int, str, str]],
    step: int,
) -> str:
    """
    Ask the LLM for today's adjustment days and return the order to send, as JSON: the
    baseline rule's order with the model's extra days on top. If the call fails, the
    baseline rule's own order.
    """
    from policies import baseline_policy          # policies.py imports this module

    user_prompt = build_user_prompt(observation_to_dict(obs, include_cover=False), action_history, step)
    try:
        completion = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": system_prompt(obs)},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
            stream      = False,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        return json.dumps(baseline_policy(obs))
    extra = adjust_days(obs, parse_json_object(text))
    return json.dumps(baseline_policy(obs, extra_days=extra, min_extra=ADJUST_DAYS_RANGE[0]))

# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

async def run_episode(
    client: OpenAI,
    task_name: str,
    episode_config: EpisodeConfig,
    seed: int,
) -> None:
    """
    Run one full episode for a single task config.

    Parameters
    ----------
    client         : OpenAI        — initialised LLM client
    task_name      : str           — task name used in the log lines
    episode_config : EpisodeConfig — the task definition sent to the server
    seed           : int           — seed for the environment's demand/lead-time noise

    Emits mandatory [START] / [STEP] / [END] log lines to STDOUT.

    Action history
    --------------
    Maintained as a list of (day, action_str, feedback_str) tuples.
    Passed to every LLM call so the policy has full memory of the episode.
    Capped at last 10 entries in the prompt to keep context manageable,
    but the full list is retained internally for scoring and debugging.
    """
    if IMAGE_NAME:
        env = await InventoryEnvClient.from_docker_image(IMAGE_NAME)
    else:
        base_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
        env = InventoryEnvClient(base_url=base_url)

    # Per-episode state
    rewards:        List[float]                  = []
    action_history: List[Tuple[int, str, str]]   = []
    steps_taken  = 0
    score        = 0.01
    success      = False
    last_obs     = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # -- Reset with this task's config ----------------------------------
        # The config is sent as plain JSON; the server re-validates it.
        result   = await env.reset(
            episode_config=episode_config.model_dump(mode="json"),
            seed=seed,
        )
        last_obs = result.observation

        # -- Step loop ------------------------------------------------------
        for step in range(1, episode_config.no_of_days + 1):

            if result.done:
                break

            # Build observation dict (Layer 2 + 3 only — no hidden state)
            obs_dict = observation_to_dict(last_obs)

            # Get LLM action — passes full action history for context
            raw_action = get_llm_action(
                client         = client,
                obs            = last_obs,
                action_history = action_history,
                step           = step,
            )

            # Step the environment
            action = InventoryAction(message=raw_action)
            result = await env.step(action)

            reward   = result.reward if result.reward is not None else 0.0
            done     = result.done
            last_obs = result.observation

            # Record this step in action history
            # Stores: (day_index, what_policy_decided, what_environment_reported)
            action_history.append((
                obs_dict["day"],
                raw_action,
                build_feedback(last_obs),
            ))

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=raw_action, reward=reward, done=done, error=None)

            if done:
                break

        # -- Final score ----------------------------------------------------
        if last_obs is not None:
            score = compute_final_score( last_obs,episode_config)

        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error in {task_name}: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)

        log_end(
            success = success,
            steps   = steps_taken,
            score   = score,
            rewards = rewards,
        )

# ---------------------------------------------------------------------------
# Main — run all tasks in sequence
# ---------------------------------------------------------------------------

async def main() -> None:
    """
    Entry point.

    Runs every task in FIXED_TASKS in sequence:
        supply_chain_broken
        flu_season
        epidemic_two_wave

    Each task is a separate 60-day episode with its own demand curves,
    supplier stress schedule, and starting inventory. Tasks are built with
    TASK_SEED so runs are reproducible.
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_name in FIXED_TASKS:
        try:
            episode_config = get_task_config(task_name, TASK_SEED)
            await run_episode(
                client=client,
                task_name=task_name,
                episode_config=episode_config,
                seed=TASK_SEED,
            )
        except Exception as exc:
            print(f"[DEBUG] Task {task_name} failed entirely: {exc}", flush=True)
            log_end(success=False, steps=0, score=0.01, rewards=[])


if __name__ == "__main__":
    asyncio.run(main())