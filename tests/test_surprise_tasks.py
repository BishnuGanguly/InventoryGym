"""The pharma world's surprise tasks: products and days drawn per seed, 14 days of storage."""

from __future__ import annotations

import pytest

import eval as E
import inference
from insights import PRODUCT_NAMES, find_events
from models import InventoryAction
from server.inventory_environment import InventoryEnvironment
from tasks import (FIXED_TASKS, PHARMA_TASKS, PHARMA_WORLD, SKU_IDS, SURPRISE_TASKS, TEST_SEEDS, TRAIN_SEEDS,
                   get_task_config)

SEEDS = list(TRAIN_SEEDS[:150]) + list(TEST_SEEDS[:150])


def _events(task, seed):
    return find_events(get_task_config(task, seed))


def _signature(task, seed):
    return tuple((e["kind"], e["sku_id"], e["start_day"]) for e in _events(task, seed))


def test_the_pharma_world_has_three_fixed_and_three_surprise_tasks():
    assert PHARMA_TASKS == FIXED_TASKS + SURPRISE_TASKS and len(FIXED_TASKS) == len(SURPRISE_TASKS) == 3
    assert PHARMA_WORLD.task_names == list(PHARMA_TASKS)


@pytest.mark.parametrize("task", FIXED_TASKS)
def test_fixed_tasks_are_the_same_for_every_seed(task):
    # The same SKUs and days for every seed (only the size of each disruption changes).
    assert len({_signature(task, s) for s in TEST_SEEDS[:20]}) == 1


@pytest.mark.parametrize("task", SURPRISE_TASKS)
def test_surprise_tasks_draw_skus_and_days_per_seed_reproducibly(task):
    assert get_task_config(task, 11) == get_task_config(task, 11)
    signatures = [_signature(task, s) for s in TEST_SEEDS[:60]]
    assert len({min(day for _, _, day in sig) for sig in signatures}) >= 6
    assert {sku for sig in signatures for _, sku, _ in sig} == set(SKU_IDS)


@pytest.mark.parametrize("task", SURPRISE_TASKS)
def test_surprise_tasks_have_14_days_of_storage(task):
    config = get_task_config(task, 3)
    for pool, cold in (("cold", True), ("ambient", False)):
        demand = sum(s.base_demand for s in config.skus.values() if s.cold_storage_required == cold)
        assert config.storage_pools[pool].capacity == pytest.approx(14 * demand)
    fixed = get_task_config(FIXED_TASKS[0], 3)
    assert fixed.storage_pools["ambient"].capacity == pytest.approx(
        21 * sum(s.base_demand for s in fixed.skus.values() if not s.cold_storage_required))


def test_surprise_days_and_skus_stay_in_their_ranges():
    for s in SEEDS:
        first, second = _events("surprise_delays", s)
        assert first["kind"] == second["kind"] == "supplier_delay" and first["sku_id"] != second["sku_id"]
        assert 10 <= first["start_day"] <= 20 and 26 <= second["start_day"] <= 45
        assert all(8 <= e["end_day"] - e["start_day"] <= 12 and e["end_day"] < 60 for e in (first, second))

        outbreak = {e["kind"]: e for e in _events("surprise_outbreak", s)}
        assert outbreak["demand_surge"]["sku_id"] == outbreak["supplier_delay"]["sku_id"]
        assert outbreak["demand_surge"]["start_day"] >= 8 and 30 <= outbreak["demand_surge"]["peak_day"] <= 38
        assert 28 <= outbreak["supplier_delay"]["start_day"] <= 38

        events = _events("surprise_epidemic", s)
        waves = [e for e in events if e["kind"] == "demand_surge"]
        assert len(waves) == 2 and waves[0]["sku_id"] == waves[1]["sku_id"]  # the waves never merge
        assert 13 <= waves[0]["peak_day"] <= 20 and 46 <= waves[1]["peak_day"] <= 52
        delays = [e for e in events if e["kind"] == "supplier_delay"]
        assert len(delays) == 2 and {e["sku_id"] for e in delays} != {waves[0]["sku_id"]}


def test_insights_follow_the_drawn_skus_and_days():
    cfg = get_task_config("surprise_delays", TEST_SEEDS[0], insights=True)
    assert len(cfg.insights) == 2
    for event in cfg.insights:
        curve = cfg.skus[event.sku_id].lead_time_curve
        assert PRODUCT_NAMES[event.sku_id] in event.message.lower()        # the insight names the drawn product
        assert curve[event.start_day] > 1 and curve[event.start_day - 1] == 1


def _report_after(days, task):
    env = InventoryEnvironment()
    obs = env.reset(task_name=task, seed=TEST_SEEDS[1])
    for _ in range(days):
        obs = env.step(InventoryAction(message="{}"))
    return inference.observation_to_dict(obs)["storage"]


def test_report_shows_how_many_days_each_pool_holds():
    assert _report_after(0, "supply_chain_broken")["ambient_days"] is None  # no demand observed yet
    fixed, surprise = _report_after(10, "supply_chain_broken"), _report_after(10, "surprise_delays")
    assert 17 <= fixed["ambient_days"] <= 25 and 17 <= fixed["cold_days"] <= 25
    assert 11 <= surprise["ambient_days"] <= 17 and 11 <= surprise["cold_days"] <= 17


def test_the_prompt_points_to_the_reported_storage_days():
    prompt = inference.SYSTEM_PROMPT
    assert "holds only about 21 days" not in prompt and "cold_days / ambient_days" in prompt


def test_environment_and_eval_run_a_surprise_task():
    env = InventoryEnvironment()
    env.reset(task_name="surprise_outbreak", seed=TEST_SEEDS[0])
    assert env._episode_config == get_task_config("surprise_outbreak", TEST_SEEDS[0])
    record = E.run_episode("baseline", "surprise_outbreak", TEST_SEEDS[0])
    assert "scenario" not in record and 0 < record["score"] <= 1
    oracle = E.run_episode("oracle", "surprise_outbreak", TEST_SEEDS[0])["score"]
    assert oracle > record["score"]
