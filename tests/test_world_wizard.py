"""Tests for the quick builder (server/world_wizard.py): plain-word answers in, a valid runnable world out."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

import eval as E
from server import world_forms as forms
from server import world_wizard as W
from world import load_world, load_world_text, build_episode

ALL = W.SCENARIO_CHOICES


def build(rows=None, **kwargs):
    args = dict(name="my_shop", business="a small shop", days=45, table=rows if rows is not None else W.starter_rows(),
                shelf_days=21, fridge_days=14, scenarios=ALL, severity="Medium")
    args.update(kwargs)
    return W.build_guided_world(**args)


def as_frame(rows):
    return pd.DataFrame(rows, columns=W.GUIDED_COLUMNS)


# ---------------------------------------------------------------------------
# What it builds
# ---------------------------------------------------------------------------

def test_the_starter_table_builds_a_valid_inventory_with_four_tasks():
    data = build()
    world = load_world(data)
    assert world.name == "my_shop" and world.business == "a small shop" and world.days == 45
    assert world.task_names == ["calm_weeks", "supplier_trouble", "demand_jump", "double_trouble"]
    assert list(world.products) == ["fresh_milk", "rice", "soap"]
    assert world.products["fresh_milk"].name == "fresh milk" and world.products["fresh_milk"].storage == "fridge"
    assert world.storage["fridge"].cold_chain and not world.storage["shelf"].cold_chain
    assert load_world_text(yaml.safe_dump(data, sort_keys=False)) == world       # survives being saved as text


def test_plain_words_become_ranges_and_costs():
    product = load_world(build()).products["fresh_milk"]
    assert product.base_demand == (80, 120)                                    # 100 a day, +-20%
    assert product.base_lead_time == (1, 1.2)                                  # a 1-day delivery, never below 1
    assert product.starting_inventory == (200, 400)                            # two to four days of sales
    assert (product.stockout_penalty, product.waste_penalty) == (50, 5)       # "high"
    soap = load_world(build()).products["soap"]
    assert (soap.stockout_penalty, soap.waste_penalty) == (5, 1) and soap.base_lead_time == (2.2, 3.8)
    for level, (shortage, waste) in W.CRITICALITY.items():
        rows = [["a", 10, 2, False, level], ["b", 10, 2, False, "low"]]
        built = load_world(build(rows)).products["a"]
        assert (built.stockout_penalty, built.waste_penalty) == (shortage, waste)


def test_storage_areas_exist_only_when_something_needs_them():
    only_shelf = load_world(build([["rice", 40, 4, False, "low"], ["soap", 20, 3, False, "low"]]))
    assert list(only_shelf.storage) == ["shelf"]
    only_fridge = load_world(build([["milk", 40, 1, True, "low"], ["fish", 20, 1, "yes", "low"]], fridge_days=9))
    assert list(only_fridge.storage) == ["fridge"] and only_fridge.storage["fridge"].capacity_days == 9


def test_blank_rows_and_all_input_shapes_are_accepted():
    rows = W.starter_rows()
    assert load_world(build(as_frame(rows))) == load_world(build(rows))
    assert load_world(build({"data": rows, "headers": W.GUIDED_COLUMNS})) == load_world(build(rows))
    assert len(rows) == W.GUIDED_ROWS and len(load_world(build(rows)).products) == 3


def test_names_become_identifiers_and_words():
    world = load_world(build([["Fresh Milk!", 100, 1, True, "High"], ["  Ice   cream ", 20, 2, True, " LOW "]]))
    assert list(world.products) == ["fresh_milk", "ice_cream"] and world.products["ice_cream"].name == "ice cream"
    assert W.slug("Ham & Cheese – 2") == "ham_cheese_2" and W.slug("!!!") == ""


# ---------------------------------------------------------------------------
# The practice scenarios
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("days", [30, 45, 90, 120])
@pytest.mark.parametrize("severity", list(W.SEVERITIES))
def test_every_scenario_builds_episodes_inside_the_episode_for_any_length_and_severity(days, severity):
    world = load_world(build(days=days, severity=severity))
    for task in world.task_names:
        for seed in range(8):
            config = build_episode(world, task, seed)
            assert config.no_of_days == days
            for sku in config.skus.values():
                assert len(sku.demand_curve) == len(sku.lead_time_curve) == days


def test_the_disrupted_products_and_times_change_with_the_seed():
    world = load_world(build())
    struck, popular, starts = set(), set(), set()
    for seed in range(120):
        config = build_episode(world, "double_trouble", seed)
        delayed = [k for k, s in config.skus.items() if max(s.lead_time_curve) > 1.0]
        surging = [k for k, s in config.skus.items() if max(s.demand_curve) > 1.0]
        assert len(delayed) == len(surging) == 1 and delayed != surging           # two different products
        struck.add(delayed[0]), popular.add(surging[0])
        starts.add(next(d for d, v in enumerate(config.skus[delayed[0]].lead_time_curve) if v > 1.0))
    assert struck == popular == {"fresh_milk", "rice", "soap"} and len(starts) > 8
    calm = build_episode(world, "calm_weeks", 3)
    assert all(max(s.lead_time_curve) == 1.0 and max(s.demand_curve) == 1.0 for s in calm.skus.values())


def test_severity_scales_the_trouble():
    def mean_peak(severity):
        world = load_world(build(severity=severity))
        peaks = [max(build_episode(world, "demand_jump", s).skus[k].demand_curve) for s in range(40)
                 for k in world.products if max(build_episode(world, "demand_jump", s).skus[k].demand_curve) > 1.0]
        return sum(peaks) / len(peaks)
    assert mean_peak("Mild") < mean_peak("Medium") < mean_peak("Severe")


def test_only_the_chosen_scenarios_are_built():
    assert list(load_world(build(scenarios=[W.SC_SUPPLIER])).tasks) == ["supplier_trouble"]
    assert list(load_world(build(scenarios=[W.SC_CALM, W.SC_BOTH])).tasks) == ["calm_weeks", "double_trouble"]


def test_agents_can_play_the_built_inventory_and_the_insights_pays():
    world = load_world(build())
    def mean(agent):
        return sum(E.run_episode(agent, "double_trouble", s, insights=True, world=world)["score"] for s in range(10000, 10008)) / 8
    nothing, baseline, plan = mean("nothing"), mean("baseline"), mean("insights_plan")
    assert nothing < 0.2 < baseline < plan


def test_the_result_opens_in_the_tables_and_survives_them():
    data = build()
    tables = forms.world_to_tables(data)
    assert load_world(forms.tables_to_world(data, tables)) == load_world(data)
    assert [r[0] for r in tables["products"]] == ["fresh_milk", "rice", "soap"]


# ---------------------------------------------------------------------------
# Mistakes are explained in plain words
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, message", [
    (dict(name="!!!"), "Give your inventory a name"),
    (dict(name=""), "Give your inventory a name"),
    (dict(days="many"), "Days in an episode: give a whole number"),
    (dict(days=10), "between 30 and 120"),
    (dict(days=400), "between 30 and 120"),
    (dict(scenarios=[]), "Pick at least one thing that should go wrong"),
    (dict(scenarios=["a meteor"]), "Unknown scenario 'a meteor'"),
    (dict(severity="Apocalyptic"), "Choose one of Mild, Medium, Severe"),
    (dict(shelf_days=0), "'Shelf' needs a number of days above 0"),
    (dict(fridge_days=None), "'Fridge' needs a number of days above 0"),
])
def test_field_mistakes(kwargs, message):
    with pytest.raises(forms.FormError, match=message):
        build(**kwargs)


@pytest.mark.parametrize("rows, message", [
    ([["milk", 10, 1, True, "low"]], "Add at least 2 products"),
    ([], "Add at least 2 products"),
    ([["milk", 10, 1, False, "low"], ["milk", 5, 2, False, "low"]], r"Row 2 \(milk\): there is already a product"),
    ([["milk", 10, 1, False, "low"], ["", 5, 2, False, "low"]], "Row 2: this row has numbers but no product name"),
    ([["milk", 0, 1, False, "low"], ["b", 5, 2, False, "low"]], r"Row 1 \(milk\): 'Sells per day' needs a number above 0"),
    ([["milk", "lots", 1, False, "low"], ["b", 5, 2, False, "low"]], r"'Sells per day' needs a number above 0"),
    ([["milk", 10, None, False, "low"], ["b", 5, 2, False, "low"]], r"'Delivery takes \(days\)' needs a number above 0"),
    ([["milk", 10, 40, False, "low"], ["b", 5, 2, False, "low"]], "too long for a 45-day episode \\(at most 15\\)"),
    ([["milk", 10, 1, False, "sort of"], ["b", 5, 2, False, "low"]], "must be one of low, medium, high, critical, not 'sort of'"),
    ([["milk", 10, 1, False, ""], ["b", 5, 2, False, "low"]], "must be one of low, medium, high, critical\\."),
    ([["***", 10, 1, False, "low"], ["b", 5, 2, False, "low"]], "needs letters or numbers"),
])
def test_table_mistakes_name_the_row(rows, message):
    with pytest.raises(forms.FormError, match=message):
        build(rows)
