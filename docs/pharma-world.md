# The pharma world

The built-in world (`worlds/pharma.yaml`): a pharmaceutical warehouse with five drugs, two storage pools and three tasks. It is the world the published results and the trained models use.

## The problem

A warehouse stocks five drugs: insulin, BP medication, paracetamol, vitamins, and hydroxychloroquine. Every day the agent must decide:

- **What to order** — which SKUs need replenishment, and which can wait.
- **How much** — enough to cover demand without wasting capacity on overstock.

The hard parts are:

- You cannot see today's true demand — only recent observed history.
- Lead times are stochastic and can extend without warning during supply disruptions.
- Insulin has no substitute and a 100-point stockout penalty. Vitamins carry a 5-point penalty. The agent must learn this hierarchy.
- Over-ordering wastes storage capacity and is penalised — but under-ordering causes stockouts.


## Warehouse

### Storage Pools

| Pool | Capacity | Used for |
|---|:---:|---|
| Cold storage | 21 days of insulin demand (about 170–250 units) | Insulin |
| Ambient storage | 21 days of the other four SKUs' combined demand (about 6,300–8,600 units) | Paracetamol, BP medication, vitamins, HCQ |

Each pool's capacity is 21 times the summed base demand of the SKUs stored in it, so it scales with the demand drawn for each episode. That is about 25% more than the most a perfect-foresight plan ever needs on any task, so good play always fits, while over-ordering overflows and the excess is wasted (see [Scoring](scoring.md)). The sizes are set in [`worlds/pharma.yaml`](../worlds/pharma.yaml) (`capacity_days`).

These are the pharma world's two pools; a world file can declare any number, with any names (see [World files](world-files.md)). The observation lists every pool in `storage` (`total_capacity`, `current_capacity`, `cold_chain`), and each SKU names its pool in `storage_pool`. The `cold_storage_*` and `ambient_storage_*` fields are the totals over cold-chain pools and over the other pools, which for pharma are exactly the two pools.

### SKUs

Insulin is the cold-chain SKU; the other four use ambient storage.

| SKU | Drug | Stockout Penalty | Waste Penalty | Base Demand | Base Lead Time |
|---|---|:---:|:---:|---|---|
| `insulin` | Insulin | 100 | 10 | 8–12 units/day | 4–6 days |
| `bp_medication` | Amlodipine | 60 | 3 | 40–60 units/day | 2–4 days |
| `paracetamol` | Paracetamol | 20 | 1 | 180–220 units/day | 1–3 days |
| `vitamins` | Vitamins | 5 | 0.5 | 60–90 units/day | 1–2 days |
| `hydroxychloroquine` | Hydroxychloroquine (HCQ) | 35 | 4 | 20–40 units/day | 5–8 days |

Base demand, base lead time and starting inventory (1–5 units, so every episode starts nearly empty) are sampled when the task is built, from the ranges above. Tasks are seeded: the same `seed` always produces the same episode.

Every day is also a little random:

- **Demand:** true daily demand is `base_demand × demand_curve[day]` plus Gaussian noise with a standard deviation of 12% of base demand (`DEMAND_NOISE_CV` in `tasks.py`).
- **Lead times:** each order's lead time is `base_lead_time × lead_time_curve[day]` plus Gaussian noise with a standard deviation of 0.75 days (`LEAD_TIME_NOISE_STD`), rounded to whole days and at least 1.

The environment draws all of this noise once, at `reset`, from the seed. The same seed therefore gives the same demand every day, and the same lead time for an order placed on a given day, **whatever the agent does**. That makes comparisons between agents fair, and lets you replay one situation with different actions, e.g. to score several candidate answers for RL training.


## Tasks

Three fixed tasks, then three surprise tasks. All episodes run 60 days. Base demand and lead times are randomised at episode start within the ranges in the SKU table.

### Task 1 — Supply Chain Broken (`supply_chain_broken`)

Demand is flat across all SKUs. The challenge is entirely supply-side: lead times extend sharply for two SKUs in sequence, forcing the agent to build safety stock before the disruption hits.

- Insulin lead times step up during days 15–25.
- Paracetamol lead times step up during days 30–40.
- Step amplitudes are randomised at episode start (2×–5× baseline).

The agent that waits for stockout signals will order too late — orders placed after the lead time extends arrive after the stockout.

![Supply Chain Broken — Demand & Lead Times](../plots/task1_supply_chain.png)


### Task 2 — Flu Season (`flu_season`)

A flu outbreak drives a Gaussian demand surge for paracetamol, peaking at day 30 with amplitude 2× baseline. All other SKU demands remain flat.

- Paracetamol demand follows a bell curve (peak day 30, width 10).
- Paracetamol lead times also step up during days 30–40 (logistics stress during the peak).
- The agent must pre-stock before day 30 — ordering at peak demand while lead times are stretched leaves no time for delivery.

![Flu Season — Demand Curves](../plots/task2_flu_season.png)


### Task 3 — Epidemic, Two Waves (`epidemic_two_wave`)

A two-wave epidemic drives hydroxychloroquine (HCQ) demand through two sequential spikes. Insulin sees a mild secondary elevation following the first wave.

- HCQ wave 1: Gaussian peak at day 17 (amplitude 1.8×).
- HCQ wave 2: Gaussian peak at day 45 (amplitude 2.5×) — larger than wave 1.
- Insulin: mild Gaussian elevation peaking ~day 20.
- HCQ lead times step up during days 17–27 (amplitude 2×–5×).
- Insulin lead times step up during days 20–30 (amplitude 1×–3×).

The trough between waves (days 27–40) is a trap: demand is low but the agent must keep ordering because wave 2 is larger and HCQ lead times are long (5–8 day baseline).

![Epidemic Two Wave — HCQ Demand](../plots/task3_epidemic.png)

### Surprise tasks: products and days drawn per seed

The three tasks above disrupt the same SKUs on the same days for every seed; only the size of each disruption changes. Three more tasks draw, per seed, which SKUs are disrupted and when, and give each storage pool **14 days** of demand instead of 21. They are ordinary tasks of the world file ([`worlds/pharma.yaml`](../worlds/pharma.yaml)), written with `draw` and day ranges like any other world's ([World files](world-files.md)):

| Task | What happens |
|---|---|
| `surprise_delays` | two different SKUs drawn from all five each have a supplier delay of 9–13 days; the first starts on day 10–20, the second on day 26–45 |
| `surprise_outbreak` | one SKU drawn from all five: a demand wave peaking on day 30–38 at 2.5–3.5× normal (width 8–12), and a supplier delay of 11 days starting on day 28–38 |
| `surprise_epidemic` | two SKUs drawn from all five: for the first, waves peaking on day 13–20 and day 46–52 (the second larger) and a delay of 11 days starting on day 13–20; for the second, a mild rise and a delay around day 16–23 |

```bash
uv run python eval.py --agent insights_plan --insights --tasks surprise_delays,surprise_outbreak,surprise_epidemic --episodes 50
```

Insights follow the drawn SKUs and days. The report shows `cold_days` / `ambient_days`, how many days of current demand each pool can hold, and the prompt refers to them instead of a fixed 21 days. Training and the teacher use these tasks by default (`tasks.SURPRISE_TASKS`).

**Why they exist.** In the fixed tasks the right response to the insights is almost a fixed timetable. Averaging what a perfect reader does per task, day and SKU over 200 train seeds, and playing that calendar without reading anything, earned 94% of the reader's gain. The shortcut is mainly *which SKU* each task disrupts, and with 21 days of storage stockpiling it is nearly free. Drawing the SKUs and the days removes the calendar, and less storage makes stockpiling everything cost something. Measured on 50 test seeds per task with insights:

| Agent | Fixed tasks | Surprise tasks |
|---|:---:|:---:|
| Baseline | 0.867 | 0.871 |
| +3 days on every SKU (no reading) | 0.886 | 0.879 |
| Extra-days rule on the exact events (`baseline_insights`) | 0.903 | 0.888 |
| Inventory plan from the exact events (`insights_plan`) | 0.909 | 0.898 |
| Oracle | 0.931 | 0.927 |
| **Extra-days rule minus "+3 days on every SKU"** | +0.017 ± 0.003, but a calendar reproduces it | **+0.008 ± 0.003** |
| **Inventory plan minus "+3 days on every SKU"** | +0.023 ± 0.004 | **+0.018 ± 0.004** |

On the surprise tasks the inventory plan (`insights_plan`, see [Insights](results.md#insights-disruptions-announced-ahead-of-time)) closes 47% of the gap between the baseline and the oracle, and the extra-days rule 30%. The plan is the reference to aim for, and the better teacher.

These tasks replace an earlier "random scenario" option that was written in code for the pharma world only. They follow the same design with slightly different ranges, so their episodes, and the numbers above, are not the ones measured before.
