"""Example modules for Fractex."""

from __future__ import annotations

import runpy
import sys
from typing import List, Optional


def list_examples() -> List[str]:
    return [
        "splash",
        "fire_flame",
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


def run_example(name: str, args: Optional[List[str]] = None) -> None:
    """Run an example module by name."""
    if name not in list_examples():
        raise ValueError(f"Unknown example '{name}'.")
    module = f"fractex.examples.{name}"
    sys.argv = [module] + (args or [])
    runpy.run_module(module, run_name="__main__")
