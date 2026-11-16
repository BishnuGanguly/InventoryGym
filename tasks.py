from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union

from models import EpisodeConfig, InventoryState, SKUEpisodeConfig
from insights import add_insights
from world import World, build_episode, default_world, load_world


# ---------------------------------------------------------------------------
# SKU catalogue: read from the built-in pharma world (worlds/pharma.yaml)
# ---------------------------------------------------------------------------
# The world file is the single source of the catalogue. The names below keep the
# shape older code expects; ranges are (low, high), sampled once per task build, so
# every episode built with the same seed is identical. waste_penalty weights each
# SKU's share of overflow waste in the reward; only ratios matter.

PHARMA_WORLD: World = default_world()


def _as_range(amount: Any) -> Tuple[float, float]:
    return amount if isinstance(amount, tuple) else (amount, amount)


def _catalogue(world: World) -> Dict[str, Dict]:
    return {
        sku_id: {
            "cold_storage_required": world.storage[p.storage].cold_chain,
            "stockout_penalty":      p.stockout_penalty,
            "waste_penalty":         p.waste_penalty,
            "base_demand":           _as_range(p.base_demand),
            "base_lead_time":        _as_range(p.base_lead_time),
            "starting_inventory":    _as_range(p.starting_inventory),
        }
        for sku_id, p in world.products.items()
    }


SKU_CATALOGUE: Dict[str, Dict] = _catalogue(PHARMA_WORLD)

EPISODE_DAYS = PHARMA_WORLD.days
# Each storage pool holds this many days of its SKUs' combined base demand.
_POOL_DAYS = {pool.cold_chain: pool.capacity_days for pool in PHARMA_WORLD.storage.values()}
AMBIENT_STORAGE_DAYS = _POOL_DAYS[False]
COLD_STORAGE_DAYS = _POOL_DAYS[True]

# Day-to-day randomness. Daily demand varies with a standard deviation of
# DEMAND_NOISE_CV x base demand, and each order's lead time with LEAD_TIME_NOISE_STD
# days. The environment draws this noise once per episode from the seed, so it
# never depends on what the agent does.
DEMAND_NOISE_CV = PHARMA_WORLD.noise.demand_cv
LEAD_TIME_NOISE_STD = PHARMA_WORLD.noise.lead_time_std


# ---------------------------------------------------------------------------
# The pharma world's tasks (worlds/pharma.yaml)
# ---------------------------------------------------------------------------
# FIXED_TASKS disrupt the same products on the same days in every episode; only the
# size of each disruption changes with the seed. SURPRISE_TASKS draw the products and
# the days per seed and have less storage, so a fixed timetable or stockpiling
# everything no longer works: they are the tasks to train and to compare on.
FIXED_TASKS: Tuple[str, ...] = ("supply_chain_broken", "flu_season", "epidemic_two_wave")
SURPRISE_TASKS: Tuple[str, ...] = ("surprise_delays", "surprise_outbreak", "surprise_epidemic")
PHARMA_TASKS: Tuple[str, ...] = tuple(PHARMA_WORLD.tasks)
assert PHARMA_TASKS == FIXED_TASKS + SURPRISE_TASKS, "worlds/pharma.yaml and tasks.py list different tasks"
SKU_IDS = tuple(SKU_CATALOGUE)


# Seed split for evaluation and training. Train (and tune) only on TRAIN_SEEDS;
# report results on TEST_SEEDS, which a trained model must never have seen.
# eval.py always takes the first N seeds of a split, so runs are comparable.
TRAIN_SEEDS = range(0, 10_000)
TEST_SEEDS = range(10_000, 20_000)
SEED_SPLITS = {"train": TRAIN_SEEDS, "test": TEST_SEEDS}


def get_task_config(
    task_name: str, seed: Optional[int] = None, insights: bool = False,
    world: Union[World, str, Mapping[str, Any], None] = None,
) -> EpisodeConfig:
    """
    Build the EpisodeConfig for a task of a world.

    world: a World, a dict, or a built-in world's name (see world.load_world);
    default the built-in pharma world.
    insights: True announces the task's disruptions ahead of time (see insights.py); False
    (the default) gives the plain tasks. Insights never change the episode's world.
    """
    world = PHARMA_WORLD if world is None else load_world(world)
    if task_name not in world.tasks:
        raise ValueError(
            f"Unknown task '{task_name}' in world '{world.name}'. Available tasks: {sorted(world.tasks)}"
        )
    return add_insights(build_episode(world, task_name, seed), insights, seed)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def weighted_fill(skus: Dict[str, Any], attribute: str = "demand_fulfilled_today") -> float:
    """
    Mean of a fill value over SKUs, weighted by each SKU's stockout_penalty:

        Σ stockout_penalty_i × fill_i / Σ stockout_penalty_i

    so running out of insulin (penalty 100) costs 20x more than running out of
    vitamins (penalty 5). Falls back to the plain mean if no SKU has a penalty.
    """
    if not skus:
        return 0.0
    weights = {k: max(float(s.stockout_penalty), 0.0) for k, s in skus.items()}
    total = sum(weights.values())
    if total <= 0:
        return sum(getattr(s, attribute) for s in skus.values()) / len(skus)
    return sum(weights[k] * getattr(s, attribute) for k, s in skus.items()) / total


def compute_step_reward(inventory_state: InventoryState) -> float:
    """
    step_reward = weighted_fill(demand_fulfilled_today) - waste_fraction_today

    The fill rate is averaged over SKUs with stockout_penalty as the weight, just as
    waste_fraction_today is the waste_penalty-weighted share of today's deliveries
    rejected for lack of storage, in [0, 1]. Range is [-1, 1].
    """
    ts = inventory_state
    if not ts.skus:
        return 0.0
    return weighted_fill(ts.skus) - ts.waste_fraction_today


def average_step_reward(inventory_state: InventoryState, days: int) -> float:
    """
    Sum of the step rewards earned so far divided by `days`, clipped to [0, 1].

    Rebuilt from the environment's running totals, no extra state needed:
      sum of weighted fill over days = weighted mean over SKUs of demand_fulfilled_cumulative
                                       (the weights are fixed for the episode)
      sum of waste_fraction_today = days played - inventory_excess_cumulative
                                    (that total adds 1 - waste_fraction_today each day)
    """
    ts = inventory_state
    if days <= 0 or not ts.skus:
        return 0.0
    fulfilled = weighted_fill(ts.skus, "demand_fulfilled_cumulative")
    wasted = ts.current_date - ts.inventory_excess_cumulative
    return min(1.0, max(0.0, (fulfilled - wasted) / days))


def compute_final_score(
    inventory_state: InventoryState,
    episode_config: EpisodeConfig,
) -> float:
    """
    final_score = clip(sum of step rewards / no_of_days, 0, 1)
                = clip(mean over days of (weighted fill - waste_fraction_today), 0, 1)

    The episode score is the average daily reward, so an agent is evaluated on
    exactly what it is rewarded for each step. Ordering nothing scores ~0.
    Days not played (an episode cut short) count as 0.
    """
    return average_step_reward(inventory_state, episode_config.no_of_days)
