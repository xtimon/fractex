"""Compatibility wrapper for volume texture module."""

from importlib import import_module

_module = import_module(".3d", __package__)

__all__ = [name for name in dir(_module) if not name.startswith("_")]
globals().update({name: getattr(_module, name) for name in __all__})
