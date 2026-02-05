import sys
from pathlib import Path
import numpy as np
import time
import itertools
import argparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fractex import FractalParams, FractalGenerator, InfiniteTexture
from fractex.interactive import add_interactive_args, add_preset_arg, resolve_preset


def main():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import matplotlib as mpl
    except Exception:
        print("matplotlib is not available; cannot display splash image.")
        return
    
    parser = argparse.ArgumentParser(description="Fractex splash animation")
    parser.add_argument("preset", nargs="?", help="Preset name")
    add_preset_arg(parser, ["marble", "clouds", "wood", "lava", "water"], dest="preset_flag")
    add_interactive_args(parser)
    args = parser.parse_args()
    
    params = FractalParams(seed=7, base_scale=0.01, detail_level=2.0)
    generator = FractalGenerator(params)
    presets = ["marble", "clouds", "wood", "lava", "water"]
    textures = {name: InfiniteTexture(generator, name) for name in presets}
    
    selected = None
    if args.preset_flag:
        selected = resolve_preset(args.preset_flag.strip().lower(), presets)
    elif args.preset:
        selected = resolve_preset(args.preset.strip().lower(), presets)
    
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
    except Exception:
        screen_width, screen_height = 1920, 1080
    
    mpl.rcParams["toolbar"] = "None"
    fig, ax = plt.subplots()
    try:
        fig.canvas.manager.toolbar.setVisible(False)
    except Exception:
        pass
    dpi = fig.get_dpi()
    fig.set_size_inches(screen_width / dpi, screen_height / dpi)
    ax.axis("off")
    ax.set_title("fractex")
    try:
        fig.canvas.manager.set_window_title("fractex")
    except Exception:
        pass
    
    texture = textures[presets[0]]
    width = max(64, int(screen_width * max(0.1, args.scale)))
    height = max(64, int(screen_height * max(0.1, args.scale)))
    image = texture.generate_tile(0, 0, width, height, zoom=1.0)
    rgb = image[..., :3]
    im = ax.imshow(rgb, animated=True, aspect="auto")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    target_fps = max(1.0, args.fps)
    target_ms = 1000.0 / target_fps
    render_scale = 1.0
    detail_level = params.detail_level
    last_time = time.perf_counter()
    ema_ms = target_ms
    phases = {name: 0.0 for name in presets}
    last_t_base = 0.0
    phase_wrap = 90.0
    
    def _upscale_nearest(image: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
        in_h, in_w, _ = image.shape
        if in_w == out_w and in_h == out_h:
            return image
        scale_x = max(1, out_w // in_w)
        scale_y = max(1, out_h // in_h)
        up = np.repeat(np.repeat(image, scale_y, axis=0), scale_x, axis=1)
        return up[:out_h, :out_w, :]
    
    def update(frame):
        nonlocal render_scale, detail_level, last_time, ema_ms, last_t_base
        t_base = frame / 10.0
        delta_t = t_base - last_t_base
        last_t_base = t_base
        
        if selected:
            base_name = selected
            detail_name = selected
        else:
            base_name = presets[(frame // 240) % len(presets)]
            detail_name = presets[(frame // 240 + 1) % len(presets)]
        
        phases[base_name] = (phases[base_name] + delta_t) % phase_wrap
        phases[detail_name] = (phases[detail_name] + delta_t) % phase_wrap
        def _render_layer(base_preset: str, detail_preset: str) -> np.ndarray:
            base_texture = textures[base_preset]
            detail_texture = textures[detail_preset]
            t = phases[base_preset]
            t2 = phases[detail_preset]
            
            base_zoom = 1.0 + 0.15 * np.sin(t * 0.25)
            detail_zoom = 2.4 + 0.25 * np.sin(t2 * 0.35)
            
            x0 = np.sin(t * 0.12) * 5.0
            y0 = np.cos(t * 0.10) * 5.0
            x1 = np.sin(t2 * 0.22 + 1.2) * 2.0
            y1 = np.cos(t2 * 0.18 + 0.7) * 2.0
            
            base_origin_x = -render_w / (2.0 * base_zoom)
            base_origin_y = -render_h / (2.0 * base_zoom)
            detail_origin_x = -render_w / (2.0 * detail_zoom)
            detail_origin_y = -render_h / (2.0 * detail_zoom)
            
            base = base_texture.generate_tile(
                base_origin_x + x0,
                base_origin_y + y0,
                render_w,
                render_h,
                zoom=base_zoom,
            )[..., :3]
            detail = detail_texture.generate_tile(
                detail_origin_x + x1,
                detail_origin_y + y1,
                render_w,
                render_h,
                zoom=detail_zoom,
            )[..., :3]
            
            depth = 0.35 + 0.15 * np.sin(t * 0.2)
            rgb = np.clip(base * (1.0 - depth) + detail * depth, 0, 1)
            tint = 0.6 + 0.4 * np.sin(t * 0.2)
            rgb = np.clip(rgb * np.array([1.0, tint, 0.9]), 0, 1)
            return rgb
        
        now = time.perf_counter()
        dt_ms = (now - last_time) * 1000.0
        last_time = now
        ema_ms = ema_ms * 0.9 + dt_ms * 0.1
        
        if ema_ms > target_ms * 1.1:
            render_scale = max(0.4, render_scale * 0.9)
            detail_level = max(0.6, detail_level * 0.9)
        elif ema_ms < target_ms * 0.9:
            render_scale = min(1.0, render_scale * 1.05)
            detail_level = min(3.0, detail_level * 1.05)
        
        generator.params.detail_level = detail_level
        render_w = max(64, int(width * render_scale))
        render_h = max(64, int(height * render_scale))
        
        rgb_frame = _render_layer(base_name, detail_name)
        if not selected:
            cycle_len = 240
            transition_len = 60
            cycle_pos = frame % cycle_len
            if cycle_pos >= cycle_len - transition_len:
                next_base = presets[(frame // cycle_len + 1) % len(presets)]
                next_detail = presets[(frame // cycle_len + 2) % len(presets)]
                phases[next_base] = (phases[next_base] + delta_t) % phase_wrap
                phases[next_detail] = (phases[next_detail] + delta_t) % phase_wrap
                next_rgb = _render_layer(next_base, next_detail)
                w = (cycle_pos - (cycle_len - transition_len)) / transition_len
                w = w * w * (3.0 - 2.0 * w)
                rgb_frame = np.clip(rgb_frame * (1.0 - w) + next_rgb * w, 0, 1)
        rgb_frame = _upscale_nearest(rgb_frame, width, height)
        im.set_array(rgb_frame)
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


if __name__ == "__main__":
    main()
