"""
FastAPI application for InventoryGym.

Endpoints:
    POST /reset   — Reset the environment, returns initial InventoryState
    POST /step    — Execute a InventoryAction, returns updated InventoryState
    GET  /state   — Get current OpenEnv State (episode_id, step_count)
    GET  /schema  — Get action/observation JSON schemas
    WS   /ws      — WebSocket endpoint for persistent sessions

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Run directly:
    python -m server.app
"""

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent          # .../server
PACKAGE_ROOT = ROOT.parent                      # the project folder
sys.path.insert(0, str(ROOT))                   # finds inventory_environment.py
sys.path.insert(0, str(PACKAGE_ROOT))           # finds models.py, tasks.py, client.py

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with:\n    uv sync\n"
    ) from e

try:
    from models import InventoryState, InventoryAction
    from inventory_environment import InventoryEnvironment
except ModuleNotFoundError:
    from models import InventoryState, InventoryAction
    from server.inventory_environment import InventoryEnvironment


def build_dashboard(*args):
    """Live dashboard tab at /web; imported lazily so API-only servers skip it."""
    try:
        from dashboard import build_dashboard as _build
    except ModuleNotFoundError:
        from server.dashboard import build_dashboard as _build
    return _build(*args)


ENV_NAME = "inventorygym"
# How many policies can hold a session at the same time. Each session gets its own environment,
# so they never share an episode. A further session is refused until one of them closes.
MAX_CONCURRENT_ENVS = max(1, int(os.getenv("MAX_CONCURRENT_ENVS", "8")))

app = create_app(
    InventoryEnvironment,
    InventoryAction,
    InventoryState,
    env_name=ENV_NAME,
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
    gradio_builder=build_dashboard,
    # At /web the dashboard is the first tab, under its own name; the Playground comes second.
    custom_tab_name="Dashboard",
    custom_tab_primary=True,
    title_override="InventoryGym",
)


# The dashboard as a page of its own, at /dashboard: the address the Hugging Face Space
# opens (base_path in README.md). It is also the "Custom" tab next to the Playground at /web.
DASHBOARD_PATH = "/dashboard"
if os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes"):
    import gradio as gr
    from fastapi import Request
    from fastapi.responses import RedirectResponse

    @app.get(DASHBOARD_PATH, include_in_schema=False)
    async def dashboard_without_slash(request: Request):
        # A relative redirect that keeps the query string. The automatic one is an absolute
        # address built from the proxy's forwarded headers, and when a hop on the way says
        # "http" it points to http://..., which a page opened over https blocks inside its
        # iframe: the Space page then stays white although the dashboard itself works.
        query = f"?{request.url.query}" if request.url.query else ""
        return RedirectResponse(url=f"{DASHBOARD_PATH}/{query}")

    try:
        from dashboard import THEME
    except ModuleNotFoundError:
        from server.dashboard import THEME

    app = gr.mount_gradio_app(app, build_dashboard(), path=DASHBOARD_PATH, theme=THEME)


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()