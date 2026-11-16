"""
The `inventorygym` command: one entry point for the things people do with this project.

    inventorygym serve                      start the server and the dashboard
    inventorygym worlds                     list the built-in inventories and their tasks
    inventorygym check my_shop.yaml         check a world file and say what is in it
    inventorygym eval --policy baseline ...  score a policy (same options as eval.py)
    inventorygym teacher --seeds 0:300 ...  write teacher examples for the warm start

Run `inventorygym <command> --help` for a command's options.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from version import __version__ as VERSION  # noqa: E402

PASS_THROUGH = ("eval", "teacher")       # commands whose options are another module's


def describe_world(world) -> List[str]:
    """A world in a few lines: what it stocks, where, and which tasks it has."""
    lines = [f"{world.name}: {len(world.products)} products, {world.days} days per episode"]
    if world.description:
        lines.append(f"  {' '.join(world.description.split())}")
    for pool_name, pool in world.storage.items():
        kept = [p for p, product in world.products.items() if product.storage == pool_name]
        cold = ", cold chain" if pool.cold_chain else ""
        lines.append(f"  storage {pool_name} ({pool.capacity_days:g} days{cold}): {', '.join(kept) or 'empty'}")
    for task_name, task in world.tasks.items():
        lines.append(f"  task {task_name}: {' '.join((task.description or '').split()) or 'no description'}")
    return lines


def cmd_serve(args: argparse.Namespace) -> int:
    # Read by server/app.py when it is imported, so it is set first.
    os.environ["ENABLE_WEB_INTERFACE"] = "false" if args.api_only else "true"
    import uvicorn
    from server.app import app
    if not args.api_only:
        print(f"Dashboard: http://localhost:{args.port}/dashboard/   Playground: http://localhost:{args.port}/web")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_worlds(args: argparse.Namespace) -> int:
    from world import builtin_worlds, load_world
    for name in builtin_worlds():
        print("\n".join(describe_world(load_world(name))))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from world import WorldError, build_episode, load_world
    try:
        world = load_world(args.file)
        for task_name in world.tasks:                     # a file can be valid and still have a task that cannot be built
            build_episode(world, task_name, 0)
    except (WorldError, OSError) as exc:
        print(f"Not valid: {exc}", file=sys.stderr)
        return 1
    print("\n".join(["Valid."] + describe_world(world)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inventorygym", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"inventorygym {VERSION}")
    commands = parser.add_subparsers(dest="command", metavar="<command>")

    serve = commands.add_parser("serve", help="start the server and the dashboard")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--api-only", action="store_true", help="no dashboard and no Playground, only the API")
    serve.set_defaults(run=cmd_serve)

    worlds = commands.add_parser("worlds", help="list the built-in inventories and their tasks")
    worlds.set_defaults(run=cmd_worlds)

    check = commands.add_parser("check", help="check a world file and say what is in it")
    check.add_argument("file", help="path of a world file (.yaml or .json), or a built-in name")
    check.set_defaults(run=cmd_check)

    commands.add_parser("eval", help="score a policy (same options as eval.py)", add_help=False)
    commands.add_parser("teacher", help="write teacher examples for the warm start", add_help=False)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in PASS_THROUGH:
        if argv[0] == "eval":
            import eval as target
        else:
            from training import teacher as target
        target.main(argv[1:])
        return 0
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "run", None):
        parser.print_help()
        return 2
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
