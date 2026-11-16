# What the agent sees and how it answers

The daily report an agent receives, and how an agent answers.

## What the agent observes

Each step the agent receives a full `InventoryState` rendered as a JSON report. Per-SKU signals:

| Field | Description |
|---|---|
| `inventory_on_hand` | Units on shelf right now |
| `stockout_days_if_no_reorder` | Days until stockout at current demand rate — most urgent signal |
| `avg_demand_per_day` | Rolling average demand over all history |
| `avg_demand_last_5_days` | Short-window average — reacts faster to spikes |
| `avg_lead_time` | Rolling average delivery time from all past orders |
| `lead_time_last3_orders` | Lead times of the last 3 delivered orders — rising values signal supply stress. A lead time only becomes visible once that order arrives |
| `expected_inbound_orders` | Open orders with estimated arrival day |
| `stockout_penalty` | Criticality weight for this SKU |

Global signals:

| Field | Description |
|---|---|
| `cold_storage_ratio` | Cold pool utilisation (0–1) |
| `ambient_storage_ratio` | Ambient pool utilisation (0–1) |
| `inventory_excess_today` | Units of today's deliveries rejected because storage was full |
| `waste_fraction_today` | Share of today's deliveries rejected, weighted by `waste_penalty` (0–1) |
| `inventory_excess_cumulative` | Sum of `1 - waste_fraction_today` so far (+1 per waste-free day) |


## How an agent answers

Each day the agent responds with a JSON object mapping SKU names to order quantities:

```json
{"insulin": 100, "paracetamol": 500}
```

Only include SKUs you want to order. Send `{}` to place no orders today.

The LLM prompt (`inference.py`, also used by `agents.LLMAgent`, `eval.py`, the dashboard and `InventoryEnvironment.to_llm_prompt()`) shows the report one SKU per line and adds two values computed from the observation: `inbound_units` (already ordered, not yet arrived) and `days_of_cover` ((on hand + inbound) ÷ recent daily demand). It shows the results of the last 3 days but not the model's own earlier replies, and its instructions contain no example quantities: small models tend to copy either one instead of reading the report.

**How an LLM agent answers.** An LLM does not write order quantities. The baseline rule does the ordering (lead time + 3 safety days of recent demand) and the model only says, per SKU, how many **extra days** of stock to keep on top of that, from 0 to +10:

```json
{"paracetamol": 4, "vitamins": 0, "insulin": 1}
```

The worst case is about the baseline: a copied number, a 0, a missing SKU or a reply without JSON keeps the default. On 20 test seeds per task, answering 0 for everything scores 0.869 (the baseline) and +3 for everything 0.886; negative values only added stockouts, so they are not accepted. The report leaves out `days_of_cover`, because small models copied it as their answer.

Two earlier formats, units to order and days of stock wanted, were removed: small models copied numbers from the prompt or the report and emptied or overflowed the warehouse ([LLM results](results.md#llm-results)).

The environment also accepts a raw LLM message string — it extracts the first valid JSON object from the text automatically. SKU names are matched case-insensitively and numeric strings are accepted; unknown SKUs, non-positive or non-numeric quantities are ignored instead of failing the step.
