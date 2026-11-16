"""Tests for the names, the package and the `inventorygym` command."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

import cli
import client
import models
from server import inventory_environment, my_env_environment
from server.inventory_environment import InventoryEnvironment
from version import __version__

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def test_the_old_names_are_the_new_classes():
    assert models.PharmaAction is models.InventoryAction
    assert client.PharmaEnvClient is client.InventoryEnvClient
    assert inventory_environment.PharmaEnvironment is InventoryEnvironment
    assert my_env_environment.PharmaEnvironment is InventoryEnvironment        # the old module still imports


def test_an_episode_runs_through_the_old_names():
    env = my_env_environment.PharmaEnvironment()
    env.reset(seed=3, task_name="flu_season")
    obs = env.step(models.PharmaAction(orders={"paracetamol": 100}))
    assert obs.current_date == 1 and ("paracetamol", 100.0) == tuple(obs.expected_inbound_orders[0][:2])


def test_the_environment_describes_itself():
    meta = InventoryEnvironment().get_metadata()
    assert meta.name == "InventoryGym" and meta.version == __version__
    assert "world file" in meta.description and "Pharma" not in meta.description


# ---------------------------------------------------------------------------
# One version, one name
# ---------------------------------------------------------------------------

def test_the_version_is_written_once():
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in pyproject and 'name = "inventorygym"' in pyproject
    manifest = yaml.safe_load((ROOT / "openenv.yaml").read_text())
    assert manifest["version"] == __version__
    assert f"## {__version__}" in (ROOT / "CHANGELOG.md").read_text()


def test_the_manifest_has_what_openenv_reads():
    manifest = yaml.safe_load((ROOT / "openenv.yaml").read_text())
    assert manifest["spec_version"] == 1 and manifest["name"] == "inventorygym"
    assert manifest["app"] == "server.app:app" and manifest["port"] == 8000
    from world import default_world
    assert [t["id"] for t in manifest["tasks"]] == default_world().task_names       # all six, in the file's order


def test_nothing_of_the_template_is_left():
    for path in ("pyproject.toml", "openenv.yaml", "__init__.py", "server/__init__.py", "server/app.py"):
        text = (ROOT / path).read_text()
        assert "my_env" not in text.replace("my_env_environment", "") and "My Env" not in text, path


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

def test_worlds_lists_the_built_in_inventory(capsys):
    assert cli.main(["worlds"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("pharma: 5 products, 60 days") and "task flu_season:" in out and "cold chain" in out


def test_check_accepts_a_good_file_and_explains_a_bad_one(tmp_path, capsys):
    assert cli.main(["check", str(ROOT / "examples/worlds/grocery.yaml")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Valid.\ngrocery:") and "task holiday_rush:" in out
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\n")
    assert cli.main(["check", str(bad)]) == 1
    assert "products: Field required" in capsys.readouterr().err
    assert cli.main(["check", str(tmp_path / "missing.yaml")]) == 1
    assert "cannot read world file" in capsys.readouterr().err


def test_check_refuses_a_task_that_cannot_be_built(tmp_path, capsys):
    data = yaml.safe_load((ROOT / "examples/worlds/grocery.yaml").read_text())
    data["tasks"]["holiday_rush"]["events"][0]["product"] = "caviar"
    bad = tmp_path / "bad_task.yaml"
    bad.write_text(yaml.safe_dump(data))
    assert cli.main(["check", str(bad)]) == 1
    assert "caviar" in capsys.readouterr().err


def test_eval_is_passed_through(tmp_path, capsys):
    out = tmp_path / "run.jsonl"
    assert cli.main(["eval", "--policy", "baseline", "--episodes", "1", "--tasks", "flu_season",
                     "--workers", "1", "--out", str(out)]) == 0
    assert out.exists() and "flu_season" in capsys.readouterr().out


def test_no_command_prints_help_and_version_works(capsys):
    assert cli.main([]) == 2
    assert "inventorygym serve" in capsys.readouterr().out
    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0 and __version__ in capsys.readouterr().out


def test_the_server_allows_several_sessions(monkeypatch):
    from server import app as server_app
    assert server_app.ENV_NAME == "inventorygym" and server_app.MAX_CONCURRENT_ENVS >= 1
    assert InventoryEnvironment.SUPPORTS_CONCURRENT_SESSIONS
