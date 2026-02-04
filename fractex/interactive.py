"""Helpers for interactive rendering with adaptive quality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import itertools
import time

import numpy as np


def add_interactive_args(parser) -> None:
    parser.add_argument("--interactive", action="store_true", help="Run interactive view")
    parser.add_argument("--scale", type=float, default=1.0, help="Display scale multiplier")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS")
    parser.add_argument("--width", type=int, default=None, help="Override width")
    parser.add_argument("--height", type=int, default=None, help="Override height")
    parser.add_argument("--speed", type=float, default=1.0, help="Animation speed multiplier")


def add_preset_arg(parser, presets, default: Optional[str] = None, dest: str = "preset") -> None:
    parser.add_argument(
        "--preset",
        choices=presets,
        default=default,
        dest=dest,
        help="Preset name",
    )


def resolve_preset(value: Optional[str], presets, fallback: Optional[str] = None) -> Optional[str]:
    if value is None:
        return fallback
    if value not in presets:
        raise ValueError(f"Unknown preset '{value}'. Available: {', '.join(presets)}")
    return value


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] >= 3:
        return image[:, :, :3]
    raise ValueError("Unsupported image shape for RGB conversion.")


def resize_nearest(image: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    if image.ndim == 2:
        image = image[:, :, None]
    in_h, in_w, channels = image.shape
    if in_w == out_w and in_h == out_h:
        return image
    scale_x = max(1, out_w // in_w)
    scale_y = max(1, out_h // in_h)
    up = np.repeat(np.repeat(image, scale_y, axis=0), scale_x, axis=1)
    up = up[:out_h, :out_w, :]
    return up if channels > 1 else up[:, :, 0]


def get_screen_size(default: Tuple[int, int] = (1920, 1080)) -> Tuple[int, int]:
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        return int(screen_width), int(screen_height)
    except Exception:
        return default


@dataclass
class InteractiveConfig:
    title: str = "Fractex"
    target_fps: float = 30.0
    scale: float = 1.0
    width: Optional[int] = None
    height: Optional[int] = None
    speed: float = 1.0
    min_scale: float = 0.4
    max_scale: float = 1.0
    min_render: int = 64

    @classmethod
    def from_args(cls, args, title: Optional[str] = None) -> "InteractiveConfig":
        return cls(
            title=title or "Fractex",
            target_fps=max(1.0, getattr(args, "fps", 30.0)),
            scale=max(0.1, getattr(args, "scale", 1.0)),
            width=getattr(args, "width", None),
            height=getattr(args, "height", None),
            speed=max(0.1, getattr(args, "speed", 1.0)),
        )


def run_interactive(
    render_frame: Callable[[float, int, int], np.ndarray],
    config: InteractiveConfig,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except Exception:
        print("matplotlib is not available; cannot display interactive output.")
        return

    screen_w, screen_h = get_screen_size()
    width = config.width or int(screen_w * config.scale)
    height = config.height or int(screen_h * config.scale)
    width = max(config.min_render, width)
    height = max(config.min_render, height)

    fig, ax = plt.subplots()
    dpi = fig.get_dpi()
    fig.set_size_inches(width / dpi, height / dpi)
    ax.axis("off")
    ax.set_title(config.title)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    target_ms = 1000.0 / max(1.0, config.target_fps)
    render_scale = 1.0
    last_time = time.perf_counter()
    ema_ms = target_ms

    frame0 = render_frame(0.0, width, height)
    im = ax.imshow(_ensure_rgb(frame0), animated=True, aspect="auto")

    def update(frame):
        nonlocal render_scale, last_time, ema_ms
        now = time.perf_counter()
        dt_ms = (now - last_time) * 1000.0
        last_time = now
        ema_ms = ema_ms * 0.9 + dt_ms * 0.1

        if ema_ms > target_ms * 1.1:
            render_scale = max(config.min_scale, render_scale * 0.9)
        elif ema_ms < target_ms * 0.9:
            render_scale = min(config.max_scale, render_scale * 1.05)

        render_w = max(config.min_render, int(width * render_scale))
        render_h = max(config.min_render, int(height * render_scale))
        t = (frame / 10.0) * config.speed
        frame_img = render_frame(t, render_w, render_h)
        frame_img = resize_nearest(frame_img, width, height)
        im.set_array(_ensure_rgb(frame_img))
        return (im,)

    anim = FuncAnimation(
        fig,
        update,
        frames=itertools.count(),
        interval=0,
        blit=True,
        repeat=True,
        cache_frame_data=False,
    )
    plt.show(block=True)
