"""
A world as simple tables, for the dashboard's inventory form, and back.

    tables = world_to_tables(data)          # data: a world file as a dict
    data = tables_to_world(data, tables)    # the same file with the tables' values

The tables hold what most worlds need: storage pools, products, tasks and events.
Amounts and days are written as text: "100" for a number, "80-120" for a range drawn
per seed. A product cell in the events table is a product id, "any" (drawn per seed),
"a|b|c" (drawn from these) or the name of a draw defined in the file.

Whatever the tables cannot show stays as it is in the file: noise, insight causes,
product descriptions and aliases, a task's own storage sizes, and draws shared by
several events. No Gradio here, so everything is testable on its own.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any, Dict, List, Optional, Sequence

STORAGE_COLUMNS = ["Pool", "Days of demand it holds", "Refrigerated (yes/no)"]
PRODUCT_COLUMNS = ["Product id", "Name", "Storage pool", "Daily demand", "Lead time (days)", "Starting stock",
                   "Shortage cost", "Waste cost"]
TASK_COLUMNS = ["Task", "What happens in it"]
EVENT_COLUMNS = ["Task", "Event", "Product", "Start day (wave: peak day)", "End day", "Or lasts (days)",
                 "Size", "Wave width (days)"]
EVENT_LABELS = {"supplier_delay": "supplier delay", "demand_wave": "demand wave", "demand_step": "demand step"}
EVENT_TYPES = {label: kind for kind, label in EVENT_LABELS.items()}
EVENT_HELP = (
    "**Event**: `supplier delay` (orders placed in the window take *Size* × longer), `demand wave` (demand rises "
    "to 1 + *Size* × normal around the peak day, then falls) or `demand step` (demand is *Size* × normal in the "
    "window). **Product**: a product id, `any` (a different product each seed) or `milk|rice` (one of these). "
    "**Days and sizes**: a number, or a range like `8-25` drawn per seed."
)


class FormError(ValueError):
    """A table cell that cannot be understood; the message says which one."""


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def _text(value: Any) -> str:
    return "" if _blank(value) else str(value).strip()


def _number_text(value: float) -> str:
    value = float(value)
    return str(int(value)) if value == int(value) else repr(value)


def amount_text(value: Any) -> str:
    """A number or a [low, high] range as cell text: 100 -> "100", [80, 120] -> "80-120"."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "-".join(_number_text(v) for v in value)
    return _number_text(value)


_RANGE = re.compile(r"^\[?\s*(\d+(?:\.\d+)?)\s*(?:-|–|—|,|to)\s*(\d+(?:\.\d+)?)\s*\]?$")


def parse_amount(value: Any, where: str, whole: bool = False) -> Any:
    """Cell text back to a number or a [low, high] range; `whole` asks for whole numbers (days)."""
    text = _text(value)
    if not text:
        raise FormError(f"{where}: this cell is empty")

    def number(part: str) -> Any:
        try:
            parsed = float(part)
        except ValueError:
            raise FormError(f"{where}: '{text}' is not a number or a range like 8-25") from None
        if whole and parsed != int(parsed):
            raise FormError(f"{where}: '{text}' must be whole days")
        return int(parsed) if parsed == int(parsed) else parsed

    match = _RANGE.match(text)
    if match:
        return [number(match.group(1)), number(match.group(2))]
    return number(text)


def _yes(value: Any) -> bool:
    return _text(value).lower() in ("yes", "y", "true", "1", "x")


def _rows(table: Any) -> List[List[Any]]:
    """A Gradio Dataframe value (pandas DataFrame, dict or list of lists) as rows, without empty rows."""
    if table is None:
        return []
    if hasattr(table, "values") and hasattr(table, "columns"):       # pandas
        rows = table.values.tolist()
    elif isinstance(table, dict):
        rows = table.get("data", [])
    else:
        rows = list(table)
    return [list(r) for r in rows if any(not _blank(c) for c in r)]


# ---------------------------------------------------------------------------
# World -> tables
# ---------------------------------------------------------------------------

def world_to_tables(data: Dict[str, Any]) -> Dict[str, Any]:
    """The tables (and the three top fields) that show a world file given as a dict."""
    storage = [[name, amount_text(pool.get("capacity_days")), "yes" if pool.get("cold_chain") else "no"]
               for name, pool in (data.get("storage") or {}).items()]
    products = [[sku_id, p.get("name") or "", p.get("storage", ""), amount_text(p.get("base_demand")),
                 amount_text(p.get("base_lead_time")), amount_text(p.get("starting_inventory", 0)),
                 amount_text(p.get("stockout_penalty")), amount_text(p.get("waste_penalty"))]
                for sku_id, p in (data.get("products") or {}).items()]
    tasks, events = [], []
    for task_name, task in (data.get("tasks") or {}).items():
        task = task or {}
        tasks.append([task_name, " ".join(str(task.get("description") or "").split())])
        draws = task.get("draw") or {}
        uses = [e.get("product") for e in task.get("events") or []]
        for event in task.get("events") or []:
            product = event.get("product", "")
            # A draw used by one event shows as its choices; a shared draw keeps its name.
            if product in draws and uses.count(product) == 1:
                choices = draws[product]
                product = choices if isinstance(choices, str) else "|".join(choices)
            wave = event.get("type") == "demand_wave"
            events.append([
                task_name, EVENT_LABELS.get(event.get("type"), event.get("type", "")), product,
                amount_text(event.get("peak_day") if wave else event.get("start_day")),
                "" if wave else amount_text(event.get("end_day")),
                "" if wave else amount_text(event.get("duration")),
                amount_text(event.get("amplitude") if wave else event.get("multiplier")),
                amount_text(event.get("width")) if wave else "",
            ])
    return {"name": data.get("name", ""), "business": data.get("business", ""), "days": data.get("days", 60),
            "storage": storage, "products": products, "tasks": tasks, "events": events}


# ---------------------------------------------------------------------------
# Tables -> world
# ---------------------------------------------------------------------------

def tables_to_world(base: Optional[Dict[str, Any]], tables: Dict[str, Any]) -> Dict[str, Any]:
    """
    The world file `base` (a dict; None for a new world) with the tables' values. Raises
    FormError naming the table, row and column of a cell that cannot be read. The
    result still has to pass world.load_world, which checks that it all fits together.
    """
    base = copy.deepcopy(base) if base else {}
    data: Dict[str, Any] = {k: v for k, v in base.items() if k not in ("storage", "products", "tasks")}
    data["name"] = _text(tables.get("name")) or "my_world"
    business = _text(tables.get("business"))
    if business:
        data["business"] = business
    else:
        data.pop("business", None)
    try:
        data["days"] = int(float(tables.get("days")))
    except (TypeError, ValueError):
        raise FormError("Days in an episode: give a whole number") from None

    storage: Dict[str, Any] = {}
    for i, row in enumerate(_rows(tables.get("storage")), start=1):
        row += [""] * (len(STORAGE_COLUMNS) - len(row))
        name = _text(row[0])
        if not name:
            raise FormError(f"Storage, row {i}: the pool needs a name")
        pool = {"capacity_days": parse_amount(row[1], f"Storage, row {i} ({name}), '{STORAGE_COLUMNS[1]}'")}
        if _yes(row[2]):
            pool["cold_chain"] = True
        storage[name] = pool
    data["storage"] = storage

    old_products = base.get("products") or {}
    products: Dict[str, Any] = {}
    for i, row in enumerate(_rows(tables.get("products")), start=1):
        row += [""] * (len(PRODUCT_COLUMNS) - len(row))
        sku_id = _text(row[0])
        if not sku_id:
            raise FormError(f"Products, row {i}: the product needs an id")
        where = f"Products, row {i} ({sku_id})"
        product = {k: v for k, v in (old_products.get(sku_id) or {}).items() if k in ("description", "aliases")}
        if _text(row[1]):
            product["name"] = _text(row[1])
        product["storage"] = _text(row[2])
        for column, key in ((3, "base_demand"), (4, "base_lead_time"), (5, "starting_inventory"),
                            (6, "stockout_penalty"), (7, "waste_penalty")):
            product[key] = parse_amount(row[column], f"{where}, '{PRODUCT_COLUMNS[column]}'")
        products[sku_id] = product
    data["products"] = products

    old_tasks = base.get("tasks") or {}
    tasks: Dict[str, Any] = {}
    for i, row in enumerate(_rows(tables.get("tasks")), start=1):
        row += [""] * (len(TASK_COLUMNS) - len(row))
        name = _text(row[0])
        if not name:
            raise FormError(f"Tasks, row {i}: the task needs a name")
        tasks[name] = {"description": _text(row[1]), "events": []}

    event_rows: Dict[str, List[Any]] = {}
    for i, row in enumerate(_rows(tables.get("events")), start=1):
        row += [""] * (len(EVENT_COLUMNS) - len(row))
        task_name = _text(row[0])
        if not task_name:
            raise FormError(f"Events, row {i}: say which task the event belongs to")
        tasks.setdefault(task_name, {"description": "", "events": []})
        event_rows.setdefault(task_name, []).append((i, row))

    for task_name, task in tasks.items():
        old = old_tasks.get(task_name) or {}
        old_draws: Dict[str, Any] = dict(old.get("draw") or {})
        draws: Dict[str, Any] = {}
        for i, row in event_rows.get(task_name, []):
            where = f"Events, row {i} ({task_name})"
            kind = EVENT_TYPES.get(_text(row[1]).lower().replace("_", " "))
            if kind is None:
                raise FormError(f"{where}: '{_text(row[1])}' is not an event; use one of {sorted(EVENT_TYPES)}")
            product = _text(row[2])
            if not product:
                raise FormError(f"{where}: say which product the event hits")
            if product == "any" or "|" in product:
                choices: Any = "any" if product == "any" else [p.strip() for p in product.split("|") if p.strip()]
                # Keep the file's name for a draw with these choices; otherwise make one.
                name = next((n for n, c in old_draws.items() if c == choices and n not in draws), None)
                if name is None:
                    name = next(f"pick_{n}" for n in range(1, 1000)
                                if f"pick_{n}" not in draws and f"pick_{n}" not in old_draws)
                draws[name] = choices
                product = name
            elif product in old_draws:
                draws[product] = old_draws[product]
            event: Dict[str, Any] = {"type": kind, "product": product}
            if kind == "demand_wave":
                event["peak_day"] = parse_amount(row[3], f"{where}, '{EVENT_COLUMNS[3]}'", whole=True)
                event["amplitude"] = parse_amount(row[6], f"{where}, '{EVENT_COLUMNS[6]}'")
                event["width"] = parse_amount(row[7], f"{where}, '{EVENT_COLUMNS[7]}'")
            else:
                event["start_day"] = parse_amount(row[3], f"{where}, '{EVENT_COLUMNS[3]}'", whole=True)
                has_end, has_length = not _blank(row[4]), not _blank(row[5])
                if has_end == has_length:
                    raise FormError(f"{where}: fill in either '{EVENT_COLUMNS[4]}' or '{EVENT_COLUMNS[5]}', not both")
                if has_end:
                    event["end_day"] = parse_amount(row[4], f"{where}, '{EVENT_COLUMNS[4]}'", whole=True)
                else:
                    event["duration"] = parse_amount(row[5], f"{where}, '{EVENT_COLUMNS[5]}'", whole=True)
                event["multiplier"] = parse_amount(row[6], f"{where}, '{EVENT_COLUMNS[6]}'")
            task["events"].append(event)
        # Draws keep the file's order, so a seed still draws the same products.
        ordered = {n: draws[n] for n in old_draws if n in draws}
        ordered.update({n: c for n, c in draws.items() if n not in ordered})
        if ordered:
            task["draw"] = ordered
        if old.get("storage_days"):
            task["storage_days"] = old["storage_days"]
        # An unchanged description keeps the file's own line breaks.
        if task["description"] == " ".join(str(old.get("description") or "").split()) and old.get("description"):
            task["description"] = old["description"]
        if not task["description"]:
            del task["description"]
    data["tasks"] = tasks
    return data


def starter_world() -> Dict[str, Any]:
    """A small world to start a new inventory from: two pools, three products, two tasks."""
    return {
        "name": "my_store",
        "business": "a small shop",
        "days": 45,
        "storage": {"fridge": {"capacity_days": 10, "cold_chain": True}, "shelf": {"capacity_days": 21}},
        "products": {
            "milk": {"storage": "fridge", "base_demand": [80, 120], "base_lead_time": [1, 2],
                     "starting_inventory": [100, 200], "stockout_penalty": 50, "waste_penalty": 8},
            "rice": {"storage": "shelf", "base_demand": [30, 50], "base_lead_time": [3, 5],
                     "starting_inventory": [100, 200], "stockout_penalty": 30, "waste_penalty": 1},
            "soap": {"storage": "shelf", "base_demand": [20, 30], "base_lead_time": [2, 4],
                     "starting_inventory": [60, 100], "stockout_penalty": 10, "waste_penalty": 1},
        },
        "tasks": {
            "quiet_weeks": {"description": "Steady demand and reliable suppliers."},
            "surprise": {
                "description": "A supplier delay hits one product and demand for another rises; which and when changes with the seed.",
                "draw": {"pick_1": "any", "pick_2": "any"},
                "events": [
                    {"type": "supplier_delay", "product": "pick_1", "start_day": [8, 22], "duration": [8, 12], "multiplier": [2, 4]},
                    {"type": "demand_wave", "product": "pick_2", "peak_day": [15, 30], "amplitude": [0.8, 1.5], "width": [3, 5]},
                ],
            },
        },
    }
