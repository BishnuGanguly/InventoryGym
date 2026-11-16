"""
The teacher: the insights plan (agents.insights_plan_targets) answering in the `adjust` format,
and supervised examples made from its answers.

GRPO starts from a model that has no idea how many days a delay or a surge is worth,
and its reward only compares the 8 answers the model happened to give. The teacher
knows the events behind the insights and answers every day with the extra days that bring
the baseline closest to an inventory plan. Fine-tuning on (prompt, teacher answer)
pairs first gives GRPO a model that already answers sensibly.

    from training.teacher import sft_examples, TeacherAgent

    rows = sft_examples(tasks, seeds=range(0, 300))        # train seeds only
    rows[0]["messages"]     # system, user (exactly the model's prompt), assistant (the teacher's answer)

    python -m training.teacher --seeds 0:300 --out data/teacher_adjust.jsonl   # the same, as a file

How good the teacher is, in the model's own answer format (50 test seeds per task,
insights): 0.893 on the surprise tasks (baseline 0.871, "+3 days on every SKU"
0.879, baseline_insights 0.888, the plan with exact orders 0.898, oracle 0.927) and 0.909
on the fixed tasks. No model is needed here: everything is the simulator.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from random import Random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import inference  # noqa: E402
from agents import Agent, BaselineAgent, insights_plan_adjust_days  # noqa: E402
from models import InventoryState, InventoryAction  # noqa: E402
from server.inventory_environment import InventoryEnvironment  # noqa: E402
from tasks import PHARMA_TASKS, SURPRISE_TASKS, TRAIN_SEEDS  # noqa: E402
from training.episodes import adjust_messages, answer_to_order  # noqa: E402

# The setting the model trains in (the same as training/grpo_reward.py, which imports
# this module): the surprise tasks (tasks.SURPRISE_TASKS), with insights.
INSIGHTS = True
# Train seeds the probe exam is drawn from (training/probe.py PROBE_SEEDS): never train on them.
PROBE_SEEDS = range(9000, 9100)

History = List[Tuple[int, str, str]]
# Whose play the example states come from: the teacher's own, the baseline's (as the
# GRPO prompts and the probe exam use), or one episode of each in turn.
STATES = ("teacher", "baseline", "mixed")
# Days without insights all have the answer 0. Keeping this share of them teaches the model
# to answer 0 when there is nothing to react to, without flooding the examples.
QUIET_SHARE = 0.15


# ---------------------------------------------------------------------------
# The teacher's answer
# ---------------------------------------------------------------------------

def normal_lead_times(env: InventoryEnvironment) -> Dict[str, float]:
    """
    Each SKU's normal lead time: the avg_lead_time of the first report, before any
    delivery (InsightsPlanAgent keeps that value; this reads the same number on any day).
    """
    return {sku_id: float(cfg.base_lead_time) for sku_id, cfg in env._episode_config.skus.items()}


def teacher_answer(obs: InventoryState, events: Sequence[Any], normal_lead_time: Dict[str, float]) -> Dict[str, int]:
    """
    Extra days for every SKU, whole numbers within the `adjust` range: the insights plan's
    answer for SKUs named by insights that are visible today (published, event not over),
    and 0 for every other SKU. A SKU whose event is over gets 0 like one without insights:
    its insights are no longer in the prompt, so nothing there could justify another answer.
    """
    today = obs.current_date
    visible = {e.sku_id for e in events if e.announce_day <= today <= e.end_day}
    plan = insights_plan_adjust_days(obs, events, normal_lead_time)
    return {sku_id: (days if sku_id in visible else 0) for sku_id, days in plan.items()}


def answer_text(answer: Dict[str, int]) -> str:
    """The answer as the model should write it: one JSON object with every SKU."""
    return json.dumps(answer)


class TeacherAgent(Agent):
    """
    Plays the teacher's answers through the model's own path (the answer text is parsed
    and turned into an order exactly as a model's would be), so its score is what a model
    that imitates the teacher perfectly would get. `last_model_reply` holds the answer.
    """

    def __init__(self) -> None:
        self._events: List[Any] = []
        self._normal_lead_time: Dict[str, float] = {}
        self.last_model_reply = ""

    def start_episode(self, env: Any) -> None:
        self._events = list(env._episode_config.insights)
        self._normal_lead_time = normal_lead_times(env)

    def act(self, obs: InventoryState, history: History) -> Tuple[str, Optional[str]]:
        self.last_model_reply = answer_text(teacher_answer(obs, self._events, self._normal_lead_time))
        return answer_to_order(obs, self.last_model_reply), None


# ---------------------------------------------------------------------------
# Supervised examples
# ---------------------------------------------------------------------------

def check_train_seeds(seeds: Iterable[int]) -> List[int]:
    """The seeds as a list; raises unless all are train seeds outside the probe exam's."""
    seeds = list(seeds)
    bad = [s for s in seeds if s not in TRAIN_SEEDS or s in PROBE_SEEDS]
    if bad:
        raise ValueError(f"teacher examples must come from train seeds outside the probe seeds "
                         f"({PROBE_SEEDS.start}-{PROBE_SEEDS.stop - 1}); not allowed: {bad[:5]}")
    return seeds


def sft_examples(
    tasks: Iterable[str],
    seeds: Iterable[int],
    insights: bool = INSIGHTS,
    states: str = "mixed",
    quiet_share: float = QUIET_SHARE,
) -> List[Dict[str, Any]]:
    """
    One example per kept day of every (task, seed) episode: the chat the model gets that
    day in the `adjust` format, with the teacher's answer as the assistant's reply.

    states: whose orders move the episode forward ("teacher", "baseline", or "mixed":
        the teacher on even seeds and the baseline on odd ones). The answer is always
        the teacher's for the state the day is in, whoever played up to it.
    quiet_share: the share of days without visible insights that is kept (their answer is
        all zeros); every day with insights are kept. The choice is fixed per day.

    Each row also has task, seed, day, insights, has_insights and states (who played).
    Deterministic: the same arguments give the same rows.
    """
    if states not in STATES:
        raise ValueError(f"states must be one of {STATES}, not {states!r}")
    if not 0.0 <= quiet_share <= 1.0:
        raise ValueError("quiet_share must be between 0 and 1")
    seeds = check_train_seeds(seeds)
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        for seed in seeds:
            played_by = states if states != "mixed" else ("teacher" if seed % 2 == 0 else "baseline")
            env = InventoryEnvironment()
            obs = env.reset(task_name=task, seed=seed, insights=insights)
            events = list(env._episode_config.insights)
            normal = normal_lead_times(env)
            baseline, history = BaselineAgent(), []
            while not obs.done:
                day = obs.current_date
                answer = answer_text(teacher_answer(obs, events, normal))
                has_insights = bool(obs.insights)
                if has_insights or Random(f"quiet|{task}|{seed}|{day}").random() < quiet_share:
                    rows.append({
                        "messages": adjust_messages(obs, history) + [{"role": "assistant", "content": answer}],
                        "task": task, "seed": seed, "day": day, "insights": insights,
                        "has_insights": has_insights, "states": played_by,
                    })
                reply = answer_to_order(obs, answer) if played_by == "teacher" else baseline.act(obs, history)[0]
                obs = env.step(InventoryAction(message=reply))
                history.append((day, reply, inference.build_feedback(obs)))
    return rows


IGNORE_INDEX = -100   # the label PyTorch's cross-entropy skips


def encode_example(tokenizer: Any, messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, List[int]]:
    """
    One example as token ids for supervised fine-tuning: the loss is on the assistant's
    answer (and the end-of-turn tokens after it) only; every prompt token is masked.

    The prompt is tokenised exactly as it is for generation (the chat template with the
    generation prompt, no extra special tokens), so the model is trained on the same
    token sequence it will later be asked to continue. Raises if the example is longer
    than max_tokens instead of cutting it: the insights are at the end of the prompt.
    """
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    if full.startswith(prompt):
        completion = full[len(prompt):]
    else:   # a template that does not simply append the answer: add it and the end token ourselves
        completion = messages[-1]["content"] + (tokenizer.eos_token or "")
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    completion_ids = list(tokenizer(completion, add_special_tokens=False)["input_ids"])
    if not completion_ids:
        raise ValueError("the answer has no tokens")
    if len(prompt_ids) + len(completion_ids) > max_tokens:
        raise ValueError(f"example has {len(prompt_ids) + len(completion_ids)} tokens, more than {max_tokens}")
    return {
        "input_ids": prompt_ids + completion_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(completion_ids)),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + completion_ids,
    }


def label_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """What the answers look like: how many are all zeros, and how often each value is given."""
    answers = [json.loads(r["messages"][-1]["content"]) for r in rows]
    insights_rows = [a for r, a in zip(rows, answers) if r["has_insights"]]
    values = Counter(v for a in insights_rows for v in a.values() if v)
    return {
        "examples": len(rows),
        "with_insights": len(insights_rows),
        "all_zero_with_insights": sum(not any(a.values()) for a in insights_rows),
        "nonzero_values": dict(sorted(values.items())),
        "by_task": dict(Counter(r["task"] for r in rows)),
        "by_states": dict(Counter(r["states"] for r in rows)),
    }


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="0:100", help="train seeds as START:STOP (default 0:100)")
    parser.add_argument("--tasks", default="surprise",
                        help="comma-separated pharma task names, or 'surprise' (the three tasks whose products "
                             "and days are drawn per seed; the default)")
    parser.add_argument("--states", choices=STATES, default="mixed")
    parser.add_argument("--quiet-share", type=float, default=QUIET_SHARE,
                        help="share of days without insights to keep (default %(default)s)")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "teacher_adjust.jsonl")
    args = parser.parse_args(argv)
    try:
        start, stop = (int(x) for x in args.seeds.split(":"))
    except ValueError:
        parser.error("--seeds must look like 0:100")
    tasks = list(SURPRISE_TASKS) if args.tasks == "surprise" else [t.strip() for t in args.tasks.split(",")]
    unknown = [t for t in tasks if t not in PHARMA_TASKS]
    if unknown:
        parser.error(f"unknown task(s) {unknown}; choose from {list(PHARMA_TASKS)}")
    try:
        rows = sft_examples(tasks, range(start, stop),
                            states=args.states, quiet_share=args.quiet_share)
    except ValueError as exc:
        parser.error(str(exc))
    write_jsonl(args.out, rows)
    print(json.dumps(label_stats(rows), indent=1))
    print(f"{len(rows)} examples written to {args.out}")


if __name__ == "__main__":
    main()
