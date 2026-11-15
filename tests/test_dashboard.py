"""Tests for the live dashboard (server/dashboard.py). No server or network needed."""

from __future__ import annotations

from types import SimpleNamespace

import gradio as gr
import pytest

from server.dashboard import (
    MAX_CONSECUTIVE_LLM_ERRORS,
    BaselineAgent,
    LLMAgent,
    build_dashboard,
    decisions_table,
    kpi_html,
    render_detail,
    reward_figure,
    run_episode,
    score_so_far,
    unmet_heatmap,
)
from tasks import TASK_REGISTRY, compute_final_score


class StubCompletions:
    """Mimics client.chat.completions.create from the openai package."""

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.prompts = reply, error, []

    def create(self, model, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def stub_llm(reply=None, error=None, action="units") -> LLMAgent:
    client = SimpleNamespace(chat=SimpleNamespace(completions=StubCompletions(reply, error)))
    return LLMAgent(model="stub", api_key="unused", base_url="unused", client=client, action=action)


def last(frames):
    trace = None
    for trace in frames:
        pass
    return trace


@pytest.mark.parametrize("task_name", sorted(TASK_REGISTRY))
def test_baseline_plays_full_episode(task_name):
    trace = last(run_episode(task_name, 42, BaselineAgent()))
    assert trace.finished and not trace.stopped_reason
    assert len(trace.rows) == trace.config.no_of_days
    assert score_so_far(trace) == pytest.approx(compute_final_score(trace.state, trace.config))


def test_llm_free_text_reply_is_parsed():
    agent = stub_llm(reply='Stocking up.\n```json\n{"Insulin": 30, "paracetamol": "500"}\n```')
    trace = last(run_episode("flu_season", 1, agent))
    assert trace.finished and trace.llm_errors == 0
    assert trace.rows[0]["orders"] == {"insulin": 30.0, "paracetamol": 500.0}
    prompts = agent.client.chat.completions.prompts
    assert len(prompts) == trace.config.no_of_days
    # From day 2 the prompt carries the previous day's results.
    assert "RESULTS OF THE LAST 1 DAY ---" in prompts[1]


def test_llm_failures_stop_the_run():
    agent = stub_llm(error=RuntimeError("401 invalid token"))
    trace = last(run_episode("flu_season", 1, agent))
    assert not trace.finished
    assert len(trace.rows) == MAX_CONSECUTIVE_LLM_ERRORS
    assert "401 invalid token" in trace.stopped_reason
    assert all(r["orders"] == {} for r in trace.rows)


def test_rendering_works_before_and_after_a_run():
    frames = run_episode("epidemic_two_wave", 3, BaselineAgent())
    empty = next(frames)
    finished = last(frames)
    for trace in (empty, finished):
        for sku_id in trace.config.skus:
            stock, lead, orders, table = render_detail(trace, sku_id)
            assert len(table) == len(trace.rows)
        reward_figure(trace)
        unmet_heatmap(trace)
        assert len(decisions_table(trace)) == len(trace.rows)
        assert "Score so far" in kpi_html(trace)


def test_build_dashboard_returns_blocks():
    assert isinstance(build_dashboard(None, None, None, False, "pharma_env", ""), gr.Blocks)


# ---------------------------------------------------------------------------
# Any world, the form and the comparison
# ---------------------------------------------------------------------------

import yaml  # noqa: E402

from server import dashboard as D  # noqa: E402
from server import world_forms as forms  # noqa: E402
from world import load_world  # noqa: E402


@pytest.mark.parametrize("template", D.TEMPLATES)
def test_templates_are_valid_worlds_and_survive_the_tables(template):
    text = D.template_text(template)
    data = yaml.safe_load(text)
    world = D.parse_world(text)
    assert load_world(forms.tables_to_world(data, forms.world_to_tables(data))) == world
    assert "is ready" in D.world_summary_html(world)
    assert D.is_builtin(world) == (template == D.TEMPLATE_PHARMA)


def test_table_mistakes_are_explained():
    tables = forms.world_to_tables(forms.starter_world())
    tables["products"][0][3] = "lots"
    with pytest.raises(forms.FormError, match=r"Products, row 1 \(milk\), 'Daily demand': 'lots' is not a number"):
        forms.tables_to_world(None, tables)
    assert "cannot be used yet" in D.world_summary_html(None, "something is wrong")


def test_an_episode_of_another_world_draws_every_chart():
    world = D.parse_world(D.template_text(D.TEMPLATE_GROCERY))
    agent = D.make_agent(D.AGENT_NEWS_PLAN)
    trace = last(run_episode("surprise_disruptions", 10000, agent, world=world, news=2))
    trace.references = D.reference_curves("surprise_disruptions", 10000, world, 2, "fixed")
    assert trace.finished and set(trace.rows[0]["storage"]) == {"fridge", "freezer", "shelf"}
    assert any(r["new_news"] for r in trace.rows)
    for figure in (D.score_figure(trace), D.storage_figure(trace), D.preview_figure(trace.config), unmet_heatmap(trace)):
        assert figure.data
    render_detail(trace, D.focus_product(trace.config))
    assert "ig-news-item" in D.news_html(trace) and "Against the baseline" in kpi_html(trace)
    assert "Final score" in D.verdict_html(trace, D.AGENT_NEWS_PLAN)
    baseline, oracle = trace.references["Baseline"][-1], trace.references["Oracle"][-1]
    assert baseline < score_so_far(trace) < oracle


def test_comparison_table_pairs_agents_with_the_baseline():
    table = D.comparison_table({D.AGENT_BASELINE: [0.80, 0.84], D.AGENT_NEWS_PLAN: [0.85, 0.87], D.AGENT_ORACLE: [0.90, 0.94]})
    assert list(table["Agent"]) == [D.AGENT_ORACLE, D.AGENT_NEWS_PLAN, D.AGENT_BASELINE]
    plan = table[table["Agent"] == D.AGENT_NEWS_PLAN].iloc[0]
    assert plan["vs baseline"].startswith("+0.0400") and plan["Gap to oracle closed"] == "40%"
    assert D.comparison_figure(table).data


# ---------------------------------------------------------------------------
# The quick builder, as wired into the page
# ---------------------------------------------------------------------------

from server import world_wizard as W  # noqa: E402
from world import load_world_text  # noqa: E402


def _create_inventory():
    demo = build_dashboard(None, None, None, False, "x", "")
    return next(f for f in demo.fns.values() if getattr(f, "api_name", None) == "create_inventory").fn


def test_quick_builder_button_creates_an_inventory_for_the_whole_page():
    ok = _create_inventory()("Corner Bakery", "a neighbourhood bakery", 60, W.starter_rows(), 21, 14, W.SCENARIO_CHOICES, "Severe")
    world_text, summary, editor = ok[0], ok[1], ok[2]
    world = load_world_text(world_text)
    assert world.name == "corner_bakery" and world.days == 60 and editor == world_text
    assert "corner_bakery is ready" in summary and "is ready" in ok[-1]
    assert not D.is_builtin(world)


def test_quick_builder_mistakes_keep_the_current_inventory_and_say_what_to_fix():
    create = _create_inventory()
    ok = create("shop", "a shop", 45, W.starter_rows(), 21, 14, W.SCENARIO_CHOICES, "Medium")
    bad = create("shop", "a shop", 45, [["milk", 10, 1, True, "high"]], 21, 14, W.SCENARIO_CHOICES, "Medium")
    assert len(bad) == len(ok)
    assert "Add at least 2 products" in bad[-1] and bad[-1].startswith("⚠️")
    assert all(item == gr.skip() for item in bad[:-1])              # nothing else on the page changes
