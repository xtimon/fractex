"""Fractex public API."""

from .core import (
    FractalParams,
    FractalGenerator,
    InfiniteTexture,
    TextureStreamer,
    OptimizedNoise,
)
from .interactive import (
    InteractiveConfig,
    add_interactive_args,
    add_preset_arg,
    resolve_preset,
    run_interactive,
    get_screen_size,
    resize_nearest,
)
from .examples import list_examples, run_example

__all__ = [
    "FractalParams",
    "FractalGenerator",
    "InfiniteTexture",
    "TextureStreamer",
    "OptimizedNoise",
    "InteractiveConfig",
    "add_interactive_args",
    "add_preset_arg",
    "resolve_preset",
    "run_interactive",
    "get_screen_size",
    "resize_nearest",
    "list_examples",
    "run_example",
]

__version__ = "0.1.0"
