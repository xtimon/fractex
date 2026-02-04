"""CLI for running Fractex interactive examples."""

from __future__ import annotations

import argparse
import runpy
import sys
from typing import List


def _examples() -> List[str]:
    return [
        "splash",
        "custom_pattern",
        "architecture_pattern",
        "composite_material",
        "crystal_cave",
        "integration",
        "terrain",
        "3d_integration_2d",
        "3d_integration",
        "3d",
        "underwater",
        "underwater_volkano",
        "game_texture",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fractex CLI")
    parser.add_argument("example", nargs="?", help="Example name")
    parser.add_argument("--list", action="store_true", help="List available examples")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--no-interactive", action="store_true", help="Disable interactive mode")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Extra args passed to example")
    args = parser.parse_args()

    if args.list or not args.example:
        print("Available examples:")
        for name in _examples():
            print(f"  - {name}")
        return

    if args.example not in _examples():
        print(f"Unknown example '{args.example}'. Use --list to see options.")
        sys.exit(1)

    module = f"fractex.examples.{args.example}"
    forwarded = []

    interactive = True
    if args.no_interactive:
        interactive = False
    if args.interactive:
        interactive = True

    if interactive:
        forwarded.append("--interactive")

    for name in ("scale", "fps", "width", "height", "preset", "speed"):
        value = getattr(args, name)
        if value is not None:
            forwarded.extend([f"--{name}", str(value)])

    if args.args:
        if args.args[0] == "--":
            forwarded.extend(args.args[1:])
        else:
            forwarded.extend(args.args)

    sys.argv = [module] + forwarded
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
