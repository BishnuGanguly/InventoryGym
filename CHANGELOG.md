# Changelog

## 2.0.0

"News" is now "insights" everywhere, and it is simply on or off.

### Changed (breaking)
- **The word.** Every `news` name is now `insights`: the observation field (`obs.insights`),
  the `reset(insights=...)` parameter, the `--insights` flag, the `insights:` section of a world
  file, the module `insights.py`, and the agents `insights_plan`, `baseline_insights` and
  `llm_insights`. The prompts a model reads say "insights" too, so a model trained on the old
  prompts should be retrained.
- **No levels.** Insights are on or off (`insights=True`, `--insights`). The old level 1, one
  fixed template with exact days, is removed; "on" is what level 2 was.
- **A simpler dashboard.** One Insights switch replaces the Off / Exact / Natural choice. Four
  agents are offered instead of seven (baseline, insights plan, oracle, LLM), and the
  pharma-only "random scenario" checkbox is gone. The other agents remain in `eval.py`.

Scores are unchanged: with insights on, every seed has the same announcements, the same
wording and the same result as level 2 had.

## 1.0.0

The project is now InventoryGym in the code as well as in the name.

### Changed
- **Names.** `PharmaEnvironment` is `InventoryEnvironment`, `PharmaAction` is `InventoryAction`,
  `PharmaEnvClient` is `InventoryEnvClient`, and `server/my_env_environment.py` is
  `server/inventory_environment.py`. The old names are kept as aliases, so existing code and
  notebooks keep working.
- **Package.** The package is `inventorygym` (it was `openenv-my_env`), with one version number
  in `version.py`. `from inventorygym import InventoryEnvClient, InventoryAction` works after
  `pip install`.
- **OpenEnv 0.3.** At `/web` the dashboard is the first tab, named *Dashboard*, and the page is
  titled InventoryGym. `/metadata` describes the environment. `openenv.yaml` has the fields
  `openenv validate` and `openenv push` read.
- **Sessions.** Up to 8 agents can hold a session at the same time (it was 1); set
  `MAX_CONCURRENT_ENVS` to change it. Each session has its own environment and episode.

### Added
- **One command**, `inventorygym`: `serve`, `worlds`, `check <world file>`, `eval`, `teacher`.
- **Continuous integration** (`.github/workflows/tests.yml`): the tests, `openenv validate` and
  the example worlds, on Python 3.10 and 3.12.

### Security
- **The dashboard never uses the server's own token.** With the token box empty it used the
  `HF_TOKEN` / `API_KEY` secret of the server, so on a public Space any visitor could run any
  model on the owner's account. An LLM run now needs a token pasted for that run.

Scores are unchanged: every agent gives the same result on the same seed as before.
