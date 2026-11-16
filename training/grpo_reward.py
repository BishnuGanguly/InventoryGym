"""
GRPO reward for the `adjust` answer format. The rules, and why each was chosen, are in
training/REWARD.md.

    from training.grpo_reward import GRPOReward, insights_day_prompts

    prompts = insights_day_prompts(tasks, seeds=range(0, 300))   # train seeds, days with insights only
    reward = GRPOReward()
    reward(completions, task=[...], seed=[...], day=[...])   # one float per completion

GRPOReward follows the reward-function signature of TRL's GRPOTrainer: it receives the
completions plus the dataset's columns as keyword arguments, and returns one reward per
completion. Everything is computed with the simulator: no model calls.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import inference  # noqa: E402
from agents import Agent, BaselineAgent, baseline_policy  # noqa: E402
from models import InventoryState, InventoryAction  # noqa: E402
from server.inventory_environment import InventoryEnvironment  # noqa: E402
from training.teacher import TeacherAgent  # noqa: E402

HORIZON = 15             # days played forward per answer
FUTURES = 4              # possible futures per prompt, shared by every answer to it
POINTS = 100.0 / HORIZON  # reward units: hundredths of a day's reward, per day
PENALTY_NO_JSON = -1.0
PENALTY_OUT_OF_RANGE = -0.25
INSIGHTS = True
# Who can play the days around the scored answer: the baseline, or the teacher
# (training/teacher.py: the insights plan answering in the `adjust` format).
PLAYERS = ("baseline", "teacher")
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)  # the same check as eval.py's parse failures

History = List[Tuple[int, str, str]]


@dataclass(frozen=True)
class DayPrompt:
    """One training prompt: one day of one episode."""

    task: str
    seed: int
    day: int
    insights: bool = INSIGHTS
    states: str = "baseline"    # who played the earlier days (one of PLAYERS)


def player(name: str, env: InventoryEnvironment) -> Agent:
    """The rule agent called `name`, ready to play `env`'s episode."""
    if name not in PLAYERS:
        raise ValueError(f"player must be one of {PLAYERS}, not {name!r}")
    agent = TeacherAgent() if name == "teacher" else BaselineAgent()
    agent.start_episode(env)
    return agent


# ---------------------------------------------------------------------------
# Rebuilding a day's state and prompt
# ---------------------------------------------------------------------------

def build_state(prompt: DayPrompt) -> Tuple[InventoryEnvironment, InventoryState, History]:
    """
    The environment on `prompt.day`, reached by letting `prompt.states` (the baseline,
    or the teacher) play every earlier day, and the action history an LLM agent would
    have (as in eval.run_episode). Deterministic: the same prompt always gives the same state.
    """
    env = InventoryEnvironment()
    obs = env.reset(task_name=prompt.task, seed=prompt.seed, insights=prompt.insights)
    agent, history = player(prompt.states, env), []
    while obs.current_date < prompt.day:
        if obs.done:
            raise ValueError(f"day {prompt.day} is past the end of the episode")
        day = obs.current_date
        reply, _ = agent.act(obs, history)
        obs = env.step(InventoryAction(message=reply))
        history.append((day, reply, inference.build_feedback(obs)))
    return env, obs, history


def prompt_messages(prompt: DayPrompt) -> List[Dict[str, str]]:
    """The chat messages agents.LLMAgent sends on that day in the `adjust` format."""
    _, obs, history = build_state(prompt)
    report = inference.observation_to_dict(obs, include_cover=False)
    user = inference.build_user_prompt(report, history, step=obs.current_date + 1)
    return [{"role": "system", "content": inference.system_prompt(obs)},
            {"role": "user", "content": user}]


def insights_day_prompts(
    tasks: Iterable[str],
    seeds: Iterable[int],
    insights: bool = INSIGHTS,
    states: str = "baseline",
) -> List[DayPrompt]:
    """
    Every day with visible insights in the given episodes (days without insights teach nothing),
    with `states` (the baseline or the teacher) playing the episode.
    """
    prompts = []
    for task in tasks:
        for seed in seeds:
            env = InventoryEnvironment()
            obs = env.reset(task_name=task, seed=seed, insights=insights)
            agent = player(states, env)
            while not obs.done:
                if obs.insights:
                    prompts.append(DayPrompt(task, seed, obs.current_date, insights, states))
                obs = env.step(InventoryAction(message=agent.act(obs, [])[0]))
    return prompts


# ---------------------------------------------------------------------------
# Parsing an answer
# ---------------------------------------------------------------------------

def parse_answer(obs: InventoryState, text: str) -> Tuple[Dict[str, float], float]:
    """
    Extra days per SKU from a completion, and its format penalty.
    No JSON object: answer 0 and PENALTY_NO_JSON. A value outside 0..10: clipped and
    PENALTY_OUT_OF_RANGE (once). Missing, unknown or non-numeric entries: 0, no penalty.
    """
    if not JSON_OBJECT.search(text or ""):
        return {}, PENALTY_NO_JSON
    raw = inference.parse_extra_days(obs, inference.parse_json_object(text))
    low, high = inference.ADJUST_DAYS_RANGE
    penalty = PENALTY_OUT_OF_RANGE if any(not low <= v <= high for v in raw.values()) else 0.0
    return inference.adjust_days(obs, raw), penalty


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def future_rng(prompt: DayPrompt, index: int) -> Random:
    """Future `index` of a prompt: the same for every answer to that prompt."""
    return Random(f"future|{prompt.task}|{prompt.seed}|{prompt.day}|{prompt.insights}|{index}")


def play_forward(env: InventoryEnvironment, obs: InventoryState, extra: Dict[str, float], horizon: int = HORIZON,
                 continuation: str = "baseline") -> float:
    """
    Sum of the daily rewards: today's order with `extra`, then `continuation` (the
    baseline, or the teacher) orders on the following days. Changes `env`.
    """
    later = player(continuation, env)
    total = 0.0
    for i in range(horizon):
        if obs.done:
            break
        reply = json.dumps(baseline_policy(obs, extra_days=extra)) if i == 0 else later.act(obs, [])[0]
        obs = env.step(InventoryAction(message=reply))
        total += obs.reward or 0.0
    return total


def score(env: InventoryEnvironment, obs: InventoryState, prompt: DayPrompt, extra: Dict[str, float],
          futures: int = FUTURES, horizon: int = HORIZON, continuation: str = "baseline") -> float:
    """S(extra): the mean of play_forward over the prompt's futures. Leaves `env` unchanged."""
    totals = []
    for index in range(futures):
        clone = copy.deepcopy(env)
        clone.resample_future_noise(future_rng(prompt, index))
        totals.append(play_forward(clone, clone._inventory_state, extra, horizon, continuation))
    return sum(totals) / len(totals)


class GRPOReward:
    """
    reward = (S(answer) - S(answer 0)) x 100 / HORIZON + format penalty   (see REWARD.md)

    continuation: who orders on the days after the answer, "baseline" (the default) or
    "teacher". With the baseline, adding days early looks good because nobody will
    prepare later; with the teacher, an answer is judged as one step of good play, which
    is what a model warm-started on the teacher's answers needs (see REWARD.md).

    Keeps the rebuilt state and S(answer 0) of the prompts it has seen, so the 8 answers
    to one prompt rebuild the state once.
    """

    __name__ = "pharma_adjust_reward"   # TRL logs reward functions by name

    def __init__(self, futures: int = FUTURES, horizon: int = HORIZON, cache_size: int = 256,
                 continuation: str = "baseline") -> None:
        if continuation not in PLAYERS:
            raise ValueError(f"continuation must be one of {PLAYERS}, not {continuation!r}")
        self.futures, self.horizon, self.cache_size = futures, horizon, cache_size
        self.continuation = continuation
        self._cache: Dict[DayPrompt, Tuple[InventoryEnvironment, InventoryState, float]] = {}

    def _state(self, prompt: DayPrompt) -> Tuple[InventoryEnvironment, InventoryState, float]:
        if prompt not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            env, obs, _ = build_state(prompt)
            self._cache[prompt] = (env, obs, self._score(env, obs, prompt, {}))
        return self._cache[prompt]

    def _score(self, env: InventoryEnvironment, obs: InventoryState, prompt: DayPrompt, extra: Dict[str, float]) -> float:
        return score(env, obs, prompt, extra, self.futures, self.horizon, self.continuation)

    @staticmethod
    def prompts_of(task: Sequence[str], seed: Sequence[int], day: Sequence[int],
                   insights: Optional[Sequence[int]] = None,
                   states: Optional[Sequence[str]] = None) -> List[DayPrompt]:
        """The DayPrompt of each row, from the dataset columns TRL passes to a reward function."""
        return [DayPrompt(task[i], int(seed[i]), int(day[i]),
                          bool(insights[i]) if insights is not None else INSIGHTS,
                          states[i] if states is not None else "baseline")
                for i in range(len(task))]

    def score_answer(self, prompt: DayPrompt, text: str) -> Dict[str, Any]:
        """The reward of one answer, with its parts (for logging and tests)."""
        env, obs, zero = self._state(prompt)
        extra, penalty = parse_answer(obs, text)
        gain = 0.0 if not any(extra.values()) else (self._score(env, obs, prompt, extra) - zero)
        return {"reward": gain * 100.0 / self.horizon + penalty, "gain_points": gain * 100.0 / self.horizon,
                "penalty": penalty, "extra_days": extra}

    def __call__(self, completions: Sequence[Any], task: Sequence[str], seed: Sequence[int],
                 day: Sequence[int],
                 insights: Optional[Sequence[int]] = None, states: Optional[Sequence[str]] = None,
                 **_: Any) -> List[float]:
        prompts = self.prompts_of(task, seed, day, insights, states)
        return [self.score_answer(prompt, completion_text(completion))["reward"]
                for prompt, completion in zip(prompts, completions)]


def completion_text(completion: Any) -> str:
    """A completion as TRL passes it: a string, or a list of chat messages."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    return str(completion)


def dataset_rows(prompts: Iterable[DayPrompt]) -> List[Dict[str, Any]]:
    """Rows for a TRL dataset: the chat prompt plus the columns GRPOReward reads."""
    return [{"prompt": prompt_messages(p), **asdict(p)} for p in prompts]

