"""
The quick builder: an inventory described in plain words, turned into a world file.

Someone who has never seen a world file gives:

    a name and the kind of business,
    one row per product: what it sells per day, how many days a delivery takes,
        whether it needs a fridge, and how bad it is to run out of it,
    how much storage room there is,
    what should go wrong in the practice runs, and how badly.

`build_guided_world(...)` turns that into a world file (a dict, see world.py). It chooses
the ranges, the shortage and waste costs, the storage areas and a task for each kind of
trouble, with the disrupted product and the timing drawn per seed so that no fixed
timetable works. It raises FormError with a sentence a non-expert can act on, never a
traceback. No Gradio here, so everything is testable on its own.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence

try:
    from world_forms import FormError
except ModuleNotFoundError:
    from server.world_forms import FormError

GUIDED_COLUMNS = ["Product", "Sells per day", "Delivery takes (days)", "Needs a fridge", "If we run out"]
GUIDED_DATATYPES = ["str", "number", "number", "bool", "str"]
GUIDED_ROWS = 10
MIN_PRODUCTS = 2
MIN_DAYS, MAX_DAYS = 30, 120

# How bad it is to run out of a product: (shortage cost, waste cost). Only the ratios matter:
# a day without a critical product costs 20 times a day without a low one.
CRITICALITY = {"low": (5, 1), "medium": (20, 2), "high": (50, 5), "critical": (100, 10)}
CRITICALITY_HELP = (
    "**If we run out**: `low` (customers shrug), `medium` (they are annoyed), `high` (they go elsewhere) "
    "or `critical` (real harm, e.g. medicine). A shortage of a critical product costs 20 times a low one."
)

SC_CALM = "Calm weeks: nothing goes wrong"
SC_SUPPLIER = "A supplier slows down (a different product each time)"
SC_DEMAND = "Demand jumps for one product (a different one each time)"
SC_BOTH = "Both at once"
SCENARIO_CHOICES = [SC_CALM, SC_SUPPLIER, SC_DEMAND, SC_BOTH]

# Severity -> (supplier delay: lead times x this, demand jump: extra demand at the peak).
SEVERITIES = {"Mild": ([1.5, 2.5], [0.5, 1.0]), "Medium": ([2, 4], [1.0, 2.0]), "Severe": ([3, 5], [2.0, 3.5])}


def slug(text: Any) -> str:
    """A name as an identifier: 'Fresh Milk!' -> 'fresh_milk'. Empty if nothing is left."""
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")


def starter_rows() -> List[List[Any]]:
    """The builder's table at the start: three example products and empty rows to fill in."""
    rows: List[List[Any]] = [["Fresh milk", 100, 1, True, "high"], ["Rice", 40, 4, False, "medium"],
                             ["Soap", 25, 3, False, "low"]]
    return rows + [["", None, None, False, ""] for _ in range(GUIDED_ROWS - len(rows))]


# ---------------------------------------------------------------------------
# Reading the table
# ---------------------------------------------------------------------------

def _blank(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def _rows(table: Any) -> List[List[Any]]:
    """A Gradio table value (pandas DataFrame, dict or list of rows) as rows, without the empty ones."""
    if table is None:
        return []
    if hasattr(table, "values") and hasattr(table, "columns"):
        rows = table.values.tolist()
    elif isinstance(table, dict):
        rows = table.get("data", [])
    else:
        rows = list(table)
    return [list(r) + [None] * (len(GUIDED_COLUMNS) - len(r)) for r in rows if any(not _blank(c) for c in r)]


def _positive(value: Any, label: str, row: int, product: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float("nan")
    if not math.isfinite(number) or number <= 0:
        raise FormError(f"Row {row} ({product}): '{label}' needs a number above 0.")
    return number


def _yes(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("yes", "y", "true", "1", "x")
    return bool(value) and not (isinstance(value, float) and math.isnan(value))


def _tidy(number: float) -> Any:
    rounded = round(number, 1)
    return int(rounded) if rounded == int(rounded) else rounded


def _range(low: float, high: float) -> List[Any]:
    return [_tidy(low), _tidy(max(high, low))]


# ---------------------------------------------------------------------------
# Building the world
# ---------------------------------------------------------------------------

def _tasks(days: int, scenarios: Sequence[str], severity: str) -> Dict[str, Any]:
    delay_size, surge_size = SEVERITIES[severity]
    # Start days leave room for the news to come first and for the whole event to end inside the episode.
    delay_start = [max(5, days // 6), days - 14]
    peak = [max(10, days // 3), days - 12]

    def delay(product: str) -> Dict[str, Any]:
        return {"type": "supplier_delay", "product": product, "start_day": delay_start, "duration": [8, 12],
                "multiplier": delay_size}

    def surge(product: str) -> Dict[str, Any]:
        return {"type": "demand_wave", "product": product, "peak_day": peak, "amplitude": surge_size, "width": [3, 5]}

    tasks: Dict[str, Any] = {}
    if SC_CALM in scenarios:
        tasks["calm_weeks"] = {"description": "Steady demand and reliable suppliers."}
    if SC_SUPPLIER in scenarios:
        tasks["supplier_trouble"] = {
            "description": "One product's supplier slows down for a week or two; which product and when changes every seed.",
            "draw": {"struck": "any"}, "events": [delay("struck")]}
    if SC_DEMAND in scenarios:
        tasks["demand_jump"] = {
            "description": "Demand for one product jumps for a few weeks; which product and when changes every seed.",
            "draw": {"popular": "any"}, "events": [surge("popular")]}
    if SC_BOTH in scenarios:
        tasks["double_trouble"] = {
            "description": "A supplier slows down and demand jumps for a different product, at times that change every seed.",
            "draw": {"struck": "any", "popular": "any"}, "events": [delay("struck"), surge("popular")]}
    return tasks


def build_guided_world(
    name: Any,
    business: Any,
    days: Any,
    table: Any,
    shelf_days: Any = 21,
    fridge_days: Any = 14,
    scenarios: Optional[Sequence[str]] = None,
    severity: str = "Medium",
) -> Dict[str, Any]:
    """
    The world file (a dict) for the quick builder's answers. Raises FormError, in plain words,
    for anything that cannot be used: the message says which row or field.
    """
    world_name = slug(name)
    if not world_name:
        raise FormError("Give your inventory a name (letters and numbers).")
    try:
        episode_days = int(float(days))
    except (TypeError, ValueError):
        raise FormError("Days in an episode: give a whole number.") from None
    if not MIN_DAYS <= episode_days <= MAX_DAYS:
        raise FormError(f"Days in an episode must be between {MIN_DAYS} and {MAX_DAYS}.")
    scenarios = list(scenarios or [])
    unknown = [s for s in scenarios if s not in SCENARIO_CHOICES]
    if unknown:
        raise FormError(f"Unknown scenario '{unknown[0]}'.")
    if not scenarios:
        raise FormError("Pick at least one thing that should go wrong in the practice runs (calm weeks is fine).")
    if severity not in SEVERITIES:
        raise FormError(f"How bad should it get? Choose one of {', '.join(SEVERITIES)}.")

    pools: Dict[str, Any] = {}
    for label, pool, value, cold in (("Shelf", "shelf", shelf_days, False), ("Fridge", "fridge", fridge_days, True)):
        try:
            capacity = float(value)
        except (TypeError, ValueError):
            capacity = float("nan")
        if not math.isfinite(capacity) or capacity <= 0:
            raise FormError(f"Storage room: '{label}' needs a number of days above 0.")
        pools[pool] = {"capacity_days": _tidy(capacity), **({"cold_chain": True} if cold else {})}

    products: Dict[str, Any] = {}
    used_pools = set()
    for i, row in enumerate(_rows(table), start=1):
        typed, sells, delivery, fridge, urgency = row[:5]
        if _blank(typed):
            raise FormError(f"Row {i}: this row has numbers but no product name.")
        text = " ".join(str(typed).split())
        sku_id = slug(text)
        if not sku_id:
            raise FormError(f"Row {i} ({text}): the product name needs letters or numbers.")
        if sku_id in products:
            raise FormError(f"Row {i} ({text}): there is already a product called '{sku_id.replace('_', ' ')}'.")
        demand = _positive(sells, "Sells per day", i, text)
        lead = _positive(delivery, "Delivery takes (days)", i, text)
        if lead > episode_days // 3:
            raise FormError(f"Row {i} ({text}): a delivery that takes {lead:g} days is too long for a "
                            f"{episode_days}-day episode (at most {episode_days // 3}).")
        level = str(urgency or "").strip().lower()
        if level not in CRITICALITY:
            raise FormError(f"Row {i} ({text}): 'If we run out' must be one of {', '.join(CRITICALITY)}"
                            + (f", not '{urgency}'." if not _blank(urgency) else "."))
        pool = "fridge" if _yes(fridge) else "shelf"
        used_pools.add(pool)
        shortage, waste = CRITICALITY[level]
        products[sku_id] = {
            "name": text.lower(),
            "storage": pool,
            "base_demand": _range(demand * 0.8, demand * 1.2),
            "base_lead_time": _range(max(1.0, lead * 0.75), lead * 1.25),
            "starting_inventory": _range(demand * 2, demand * 4),
            "stockout_penalty": shortage,
            "waste_penalty": waste,
        }
    if len(products) < MIN_PRODUCTS:
        raise FormError(f"Add at least {MIN_PRODUCTS} products: the practice runs pick the product that is hit, "
                        "so they need something to choose from.")

    return {
        "name": world_name,
        "business": " ".join(str(business or "").split()) or "a small business",
        "days": episode_days,
        "storage": {pool: spec for pool, spec in pools.items() if pool in used_pools},
        "products": products,
        "tasks": _tasks(episode_days, scenarios, severity),
    }
