# Server, playground and dashboard

Running the environment as an OpenEnv server, the web playground, the live dashboard and the environment variables.

## Web Playground

The Docker image sets `ENABLE_WEB_INTERFACE=true`, so the server also serves OpenEnv's Gradio Playground at `/web` (`/` redirects there). On the Hugging Face Space it is the page that opens.

- **Reset** starts a new episode, then **Step** advances one day.
- Type orders in either box: **Orders** takes a JSON object such as `{"insulin": 40}`, and **Message** takes free text containing one, as an LLM would reply.
- The raw JSON shows the full `InventoryState`, including ground-truth fields such as `actual_inbound_orders` that the agent's prompt in `inference.py` leaves out.
- The Reset button takes no parameters, so it loads the small 30-day, two-SKU default episode. To play one of the three tasks, reset through the API and keep using the page:

```bash
curl -X POST http://localhost:8000/web/reset \
     -H 'Content-Type: application/json' \
     -d '{"task_name": "flu_season", "seed": 42}'
```

To run it locally:

```bash
ENABLE_WEB_INTERFACE=true uv run uvicorn server.app:app --port 8000
# open http://localhost:8000/web
```


## Dashboard

The dashboard is served at `/dashboard` (the address the Hugging Face Space opens) and is also the **Custom** tab next to the Playground at `/web`. It has three steps:

1. **Your inventory.** Build your own, or pick a starting point (the pharma world, the grocery example or a blank starter) and change it:
   - **Quick builder** (no knowledge of world files needed). Give a name and the kind of business, then fill one row per product in plain words: *Sells per day*, *Delivery takes (days)*, *Needs a fridge* (a checkbox) and *If we run out* (`low`, `medium`, `high` or `critical`: how bad a shortage is). Choose what should go wrong in the practice runs (calm weeks, a supplier slowing down, demand jumping for one product, or both at once) and how bad (mild, medium or severe), then press **Create my inventory**. The builder turns your answers into a world file:
     - demand is ±20% around what you typed, delivery time ±25%, and starting stock is two to four days of sales;
     - a shortage costs 5, 20, 50 or 100 for `low` to `critical`;
     - a shelf and, if any product needs one, a fridge are created, with room (in days of sales) set under *Storage room*;
     - each chosen trouble becomes a task in which the product that is hit and the timing are drawn per seed, so no fixed timetable works and reading the news pays.
     A problem (such as fewer than two products, or a delivery time too long for the episode) is shown next to the button, in words, and the inventory in use stays as it was. The result opens in **Tables** and **File** for fine-tuning.
   - **Tables** for storage pools, products, tasks and events. Amounts and days are a number or a range such as `80-120`, drawn per seed. An event's product can be a product id, `any` (a different product each seed) or `milk|rice` (one of these).
   - **File** for everything a world file can hold (per-task storage, draws shared by several events, the words news uses), or **upload** a `.yaml` file.
   - A card says whether the inventory is valid and, if not, what is wrong and where. **Preview a task** draws what the task does to each product and lists the news it would produce.
2. **Run an episode.** Choose a task and an agent (a line under the agent says what it does), a seed (🎲 draws a new one) and the news level, then **Run episode** (**Stop** cancels). A progress bar shows the run, and the results appear when it starts. *More options* holds the pause per day, the random scenario (pharma world only) and the LLM settings.
   - **Agents:** the baseline rule, the two news references (`news_plan`, `baseline_news`), the oracle, "order nothing", and two LLM agents: one that adjusts the baseline's orders, and a hybrid that only reads the news.
   - **Charts, redrawn every simulated day:** the score so far against the baseline and the oracle on the same episode; how full each storage pool is, with wasted deliveries marked; the news feed; and for the chosen product, stock against demand, lead times and orders. **Orange is hidden truth the agent never sees**: shaded bands are the days of a disruption, and a dotted line marks the day it was announced.
   - At the end a line says how much of the gap between the baseline and the oracle the agent closed. In the score chart, the shaded zone between the baseline and the oracle is the room to improve.
3. **Compare agents.** Several agents on the same test seeds of one task: a chart and a table with the mean score and a 95% interval, the difference to the baseline seed by seed, and the share of the gap to the oracle closed. LLM agents run at most 3 episodes here.

**LLM agents.** On the Space, add your token as a **Secret** named `HF_TOKEN` (Space → Settings → Variables and secrets), or paste a token under *More options → LLM settings* for a single run. Calls go through [Inference Providers](https://huggingface.co/docs/inference-providers) and count against that account's credits. `MODEL_NAME` sets the default model and `API_BASE_URL` points it at any OpenAI-compatible endpoint. A token is used for the run only and is never stored.

**Limits.** A world typed or uploaded in the dashboard is parsed with the safe YAML loader and may have at most 30 products, 20 tasks, 180 days and 100 kB. The server never opens a file path sent from the browser. Each run builds its own environment, so viewers never share an episode.

## Choosing a task on reset

`reset()` accepts either a registered task name or a full config:

```python
await env.reset(task_name="flu_season", seed=42)
await env.reset(episode_config=get_task_config("flu_season", seed=42).model_dump(mode="json"))
```

With neither, a small 30-day two-SKU default episode is used.


## Environment variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | — | HuggingFace API key |
| `ENV_BASE_URL` | `http://localhost:8000` | Server URL |
| `LOCAL_IMAGE_NAME` | — | Run the server from this Docker image instead of `ENV_BASE_URL` |
| `TASK_SEED` | `42` | Seed for task generation and environment noise |
