"""Tests for insights (insights.py), the insights-aware agents (agents.py) and their prompt (inference.py)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import eval as E
import inference
from agents import MAX_COVER_DAYS, InsightsLLMAgent, InsightsRuleAgent, baseline_lead_time, baseline_policy
from models import InventoryAction
from insights import ANNOUNCE_DAYS_AHEAD, add_insights, find_events, visible_insights
from server.inventory_environment import InventoryEnvironment
from tasks import FIXED_TASKS, PHARMA_TASKS, TEST_SEEDS, get_task_config

SEED = TEST_SEEDS[0]
TASKS = sorted(FIXED_TASKS)


def fake_client(text="{}", fail=False):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if fail:
            raise TimeoutError("model too slow")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


# ---------------------------------------------------------------------------
# Events and text
# ---------------------------------------------------------------------------

def test_events_follow_the_task_curves():
    kinds = {t: sorted((e["kind"], e["sku_id"], e["start_day"], e["end_day"])
                       for e in find_events(get_task_config(t, SEED))) for t in TASKS}
    assert kinds["supply_chain_broken"] == [("supplier_delay", "insulin", 15, 25),
                                            ("supplier_delay", "paracetamol", 30, 40)]
    assert ("demand_surge", "paracetamol", 14, 46) in kinds["flu_season"]
    # The epidemic's two waves are two separate surges.
    waves = [k for k in kinds["epidemic_two_wave"] if k[:2] == ("demand_surge", "hydroxychloroquine")]
    assert len(waves) == 2 and waves[0][3] < waves[1][2]


@pytest.mark.parametrize("task_name", TASKS)
def test_insights_are_published_5_to_10_days_ahead(task_name):
    events = get_task_config(task_name, SEED, insights=True).insights
    assert events
    for e in events:
        ahead = e.start_day - e.announce_day
        assert e.announce_day == 0 or ANNOUNCE_DAYS_AHEAD[0] <= ahead <= ANNOUNCE_DAYS_AHEAD[1]


def test_the_wording_is_varied_but_deterministic():
    varied = [e.message for t in TASKS for s in TEST_SEEDS[:5] for e in get_task_config(t, s, insights=True).insights]
    assert not any("day 15" in m for m in varied)          # timing is relative, not exact
    assert len({m.split(":")[0].split(" ")[0] for m in varied}) > 2   # several templates
    assert varied == [e.message for t in TASKS for s in TEST_SEEDS[:5] for e in get_task_config(t, s, insights=True).insights]


def test_insights_are_off_by_default_and_only_on_or_off():
    config = get_task_config("flu_season", SEED)
    assert config.insights == [] and add_insights(config, False, SEED) is config
    for level in (0, 1, 2, "yes"):                       # the old levels are gone
        with pytest.raises(ValueError):
            add_insights(config, level, SEED)


@pytest.mark.parametrize("task_name", TASKS)
def test_insights_never_changes_the_world(task_name):
    plain = get_task_config(task_name, SEED)
    with_insights = get_task_config(task_name, SEED, insights=True)
    assert plain.model_dump(exclude={"insights"}) == with_insights.model_dump(exclude={"insights"})
    # The baseline ignores insights, so it scores exactly the same.
    assert (E.run_episode("baseline", task_name, SEED)["score"]
            == E.run_episode("baseline", task_name, SEED, insights=True)["score"])


def test_observation_shows_insights_from_publication_to_end():
    env = InventoryEnvironment()
    obs = env.reset(task_name="supply_chain_broken", seed=SEED, insights=True)
    event = env._episode_config.insights[0]
    seen = {}
    while not obs.done:
        seen[obs.current_date] = list(obs.insights)
        obs = env.step(InventoryAction(message="{}"))
    for day, items in seen.items():
        shown = any(item == f"Day {event.announce_day}: {event.message}" for item in items)
        assert shown == (event.announce_day <= day <= event.end_day)
    assert visible_insights(env._episode_config, 0) == []


def test_report_includes_insights_only_when_there_is_some():
    env = InventoryEnvironment()
    assert "insights" not in inference.observation_to_dict(env.reset(task_name="flu_season", seed=SEED))
    obs = env.reset(task_name="epidemic_two_wave", seed=SEED, insights=True)
    assert inference.observation_to_dict(obs)["insights"] == obs.insights


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def test_extra_days_are_capped_for_storage():
    env = InventoryEnvironment()
    obs = env.reset(task_name="flu_season", seed=SEED)
    for _ in range(5):
        obs = env.step(InventoryAction(message="{}"))
    plain = baseline_policy(obs)
    huge = baseline_policy(obs, extra_days={"paracetamol": 1000, "vitamins": -5})
    sku = obs.skus["paracetamol"]
    demand = max(sku.avg_demand_last_5_days, sku.avg_demand_per_day)
    cover = max(MAX_COVER_DAYS, baseline_lead_time(sku) + 3.0)
    assert huge["paracetamol"] == round(demand * cover - sku.inventory_on_hand)
    assert huge["vitamins"] == plain["vitamins"]       # negative extra is ignored


@pytest.mark.parametrize("task_name", TASKS)
def test_insights_rule_agent_beats_the_baseline(task_name):
    baseline = E.run_episode("baseline", task_name, SEED, insights=True)["score"]
    assert E.run_episode("baseline_insights", task_name, SEED, insights=True)["score"] > baseline
    # Without insights the agent is the baseline.
    assert E.run_episode("baseline_insights", task_name, SEED)["score"] == E.run_episode("baseline", task_name, SEED)["score"]


def test_insights_llm_agent_only_calls_the_model_on_insights_days(monkeypatch):
    client, calls = fake_client('{"insulin": 6, "Amlodipine": 2, "HCQ": "lots", "aspirin": 9}')
    monkeypatch.setattr(E, "make_agent", lambda name, llm=None: InsightsLLMAgent("stub", "x", "x", client=client))
    record = E.run_episode("custom", "supply_chain_broken", SEED, insights=True, save_steps=True)
    insights_days = [s["day"] for s in record["steps"] if s["insights"]]
    assert len(calls) == len(insights_days) > 0
    assert record["parse_failures"] == 0 and record["llm_errors"] == 0
    prompt = calls[0]["messages"][1]["content"]
    assert f"Today is day {insights_days[0]}" in prompt and "PRODUCTS:" in prompt
    assert calls[0]["messages"][0]["content"] == inference.INSIGHTS_SYSTEM_PROMPT


def test_insights_llm_agent_without_insights_is_the_baseline():
    client, calls = fake_client()
    env = InventoryEnvironment()
    obs = env.reset(task_name="flu_season", seed=SEED)
    for _ in range(4):
        obs = env.step(InventoryAction(message="{}"))
    agent = InsightsLLMAgent("stub", "x", "x", client=client)
    reply, error = agent.act(obs, [])
    assert json.loads(reply) == baseline_policy(obs) and error is None and calls == []


def test_insights_llm_agent_applies_the_reply_and_survives_failures():
    env = InventoryEnvironment()
    obs = env.reset(task_name="supply_chain_broken", seed=SEED, insights=True)
    while not obs.insights:
        obs = env.step(InventoryAction(message=json.dumps(baseline_policy(obs))))
    client, _ = fake_client('Insulin needs more: {"insulin": 5, "paracetamol": 0}')
    reply, error = InsightsLLMAgent("stub", "x", "x", client=client).act(obs, [])
    assert error is None and json.loads(reply) == baseline_policy(obs, extra_days={"insulin": 5})
    client, _ = fake_client(fail=True)
    reply, error = InsightsLLMAgent("stub", "x", "x", client=client).act(obs, [])
    assert "TimeoutError" in error and json.loads(reply) == baseline_policy(obs)


def test_parse_extra_days_accepts_aliases_and_skips_junk():
    obs = InventoryEnvironment().reset(task_name="flu_season", seed=SEED)
    reply = {"Insulin": 3, "amlodipine": "2.5", "HCQ": 4, "vitamins": True, "paracetamol": "nan", "x": 1}
    assert inference.parse_extra_days(obs, reply) == {"insulin": 3.0, "bp_medication": 2.5, "hydroxychloroquine": 4.0}


def test_rule_agent_reads_events_from_the_environment():
    env = InventoryEnvironment()
    env.reset(task_name="flu_season", seed=SEED, insights=True)
    agent = InsightsRuleAgent()
    agent.start_episode(env)
    assert agent._events == env._episode_config.insights


def test_command_line_insights_flag(tmp_path):
    out = tmp_path / "insights.jsonl"
    E.main(["--agent", "baseline_insights", "--insights", "--episodes", "1", "--tasks", "flu_season",
            "--workers", "1", "--out", str(out)])
    assert json.loads(out.read_text())["insights"] is True


def test_insights_prompt_has_no_example_answer():
    """Regression: the prompt ended with an all-zero example answer. A model that copies it
    plays exactly like the baseline every day, so every sampled answer gets the same
    reward and GRPO has nothing to learn from."""
    from test_prompt import SKU_WITH_NUMBER

    assert not SKU_WITH_NUMBER.search(inference.INSIGHTS_SYSTEM_PROMPT)
    assert '{"<product_name>": <extra_days>' in inference.INSIGHTS_SYSTEM_PROMPT
    env = InventoryEnvironment()
    obs = env.reset(task_name="flu_season", seed=TEST_SEEDS[0], insights=True)
    while not obs.insights:
        obs = env.step(InventoryAction(message="{}"))
    prompt = inference.build_insights_prompt(obs)
    assert not SKU_WITH_NUMBER.search(prompt[prompt.index("Extra days"):])


def test_insights_use_generic_product_names():
    """No drug or brand names: 'amlodipine' for bp_medication made models react to the wrong product."""
    from insights import PRODUCT_NAMES

    assert PRODUCT_NAMES["bp_medication"] == "blood-pressure medication"
    assert all("(" not in name for name in PRODUCT_NAMES.values())
    messages = [e.message.lower() for t in PHARMA_TASKS for s in TEST_SEEDS[:20]
                for e in get_task_config(t, s, insights=True).insights]
    assert messages and not any(w in m for m in messages for w in ("amlodipine", "hcq", "supplements"))
    # Every product an insight mentions is named exactly as in PRODUCT_NAMES.
    for t in PHARMA_TASKS:
        for e in get_task_config(t, TEST_SEEDS[0], insights=True).insights:
            assert PRODUCT_NAMES[e.sku_id] in e.message.lower()
