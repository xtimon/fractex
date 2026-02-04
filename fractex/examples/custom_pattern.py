# Создание собственных паттернов
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import argparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from _output import save_volume_slice
from fractex.interactive import add_interactive_args, InteractiveConfig, run_interactive


@dataclass
class CustomPatternParams:
    scale: float = 10.0
    surface_threshold: float = 0.5


class CustomPatternGenerator:
    def generate_custom_pattern(self, dimensions, params: CustomPatternParams):
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        # Пользовательская логика
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    value = self.custom_math_function(i, j, k, params)
                    
                    if self.is_on_surface(value, params):
                        color = self.calculate_color(i, j, k, value)
                        texture[i, j, k] = color
        
        return texture
    
    def custom_math_function(self, x, y, z, params: CustomPatternParams):
        # Пример: гиперболический параболоид
        return (x / params.scale) ** 2 - (y / params.scale) ** 2 - z / params.scale
    
    def is_on_surface(self, value: float, params: CustomPatternParams) -> bool:
        return abs(value) < params.surface_threshold
    
    def calculate_color(self, x, y, z, value: float) -> np.ndarray:
        return np.array([
            0.4 + 0.6 * np.clip(value, -1, 1),
            0.6,
            0.8,
            1.0,
        ], dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom pattern example")
    add_interactive_args(parser)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    
    generator = CustomPatternGenerator()
    params = CustomPatternParams(scale=8.0, surface_threshold=0.4)
    texture = generator.generate_custom_pattern((16, 16, 16), params)
    print("Custom pattern:", texture.shape, texture.min(), texture.max())
    save_volume_slice(
        texture,
        EXAMPLES_DIR / "output" / "custom_pattern_slice.ppm",
        stretch=True,
    )
    
    if args.interactive:
        config = InteractiveConfig.from_args(args, title="Custom Pattern (interactive)")
        
        def render_frame(t, w, h):
            speed = max(0.1, args.speed)
            tt = t * speed * 3.0
            zoom = np.exp(tt * 0.03)
            zoom = min(zoom, 200.0)
            depth = (tt * 1.5) % (params.scale * 6.0)
            drift = np.exp(-tt * 0.02)
            x0 = np.sin(tt * 0.4) * 3.0 * drift
            y0 = np.cos(tt * 0.35) * 3.0 * drift
            
            x = np.linspace(-1.0, 1.0, w) / zoom + x0
            y = np.linspace(-1.0, 1.0, h) / zoom + y0
            xx, yy = np.meshgrid(x, y)
            
            angle = t * 0.02
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            xr = xx * cos_a - yy * sin_a
            yr = xx * sin_a + yy * cos_a
            
            value = (xr / params.scale) ** 2 - (yr / params.scale) ** 2 - depth / params.scale
            value2 = np.sin(xr * 3.0 + t * 0.5) + np.cos(yr * 2.0 - t * 0.3)
            combined = value + 0.35 * value2
            mask = 0.1 + 0.9 * np.exp(-np.abs(combined) * 1.6)
            
            radius = np.sqrt(xr * xr + yr * yr)
            vignette = np.clip(1.2 - radius * 0.8, 0.0, 1.0)
            
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            color = np.clip(0.4 + 0.6 * np.tanh(combined), 0, 1)
            ripple = 0.5 + 0.5 * np.sin(t * 0.6 + (xr + yr) * 2.0)
            palette_shift = 0.5 + 0.5 * np.sin(t * 0.12)
            rgb[..., 0] = (color * 0.7 + ripple * 0.3) * mask
            rgb[..., 1] = (0.45 + 0.45 * ripple + 0.1 * palette_shift) * mask
            rgb[..., 2] = (0.65 + 0.3 * np.cos(t * 0.4) + 0.1 * palette_shift) * mask
            rgb = np.clip(rgb * vignette[..., None], 0, 1)
            return rgb
        
        run_interactive(render_frame, config)
