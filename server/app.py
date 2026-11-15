"""
FastAPI application for the Pharmaceutical Warehouse Procurement Environment.

Endpoints:
    POST /reset   — Reset the environment, returns initial InventoryState
    POST /step    — Execute a PharmaAction, returns updated InventoryState
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

#from OpenEnv.envs.browsergym_env import models
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent          # .../pharma_coldstorage_env/server
PACKAGE_ROOT = ROOT.parent                      # .../pharma_coldstorage_env
sys.path.insert(0, str(ROOT))                   # finds my_env_environment.py
sys.path.insert(0, str(PACKAGE_ROOT))           # finds models.py, tasks.py, client.py

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with:\n    uv sync\n"
    ) from e

try:
   
    from models import InventoryState, PharmaAction
    from my_env_environment import PharmaEnvironment
except ModuleNotFoundError:
    from models import InventoryState, PharmaAction
    from server.my_env_environment import PharmaEnvironment


def build_dashboard(*args):
    """Live dashboard tab at /web; imported lazily so API-only servers skip it."""
    try:
        from dashboard import build_dashboard as _build
    except ModuleNotFoundError:
        from server.dashboard import build_dashboard as _build
    return _build(*args)


app = create_app(
    PharmaEnvironment,
    PharmaAction,
    InventoryState,
    env_name="pharma_env",
    max_concurrent_envs=1,
    gradio_builder=build_dashboard,
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