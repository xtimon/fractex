import sys
from pathlib import Path
import argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from fractex.simplex_noise import SimplexNoise
from fractex.interactive import add_interactive_args, InteractiveConfig, run_interactive
from _output import save_ppm


def _render_flame(t: float, w: int, h: int, noise: SimplexNoise) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, w)
    y = np.linspace(0.0, 2.0, h)
    xx, yy = np.meshgrid(x, y)
    taper = 1.0 - 0.5 * np.clip(yy / 2.0, 0, 1)
    xx_t = xx * taper
    zz = np.ones_like(xx) * (t * 0.15)
    
    base = noise.noise_3d(xx_t * 1.5, yy * 1.2, zz)
    detail = noise.noise_3d(xx_t * 5.0, yy * 4.0, zz * 1.8)
    micro = noise.noise_3d(xx_t * 14.0, yy * 10.0, zz * 5.0)
    warp = noise.noise_3d(xx_t * 1.2, yy * 1.0, zz * 0.8)
    turbulence = base * 0.45 + detail * 0.3 + micro * 0.2 + warp * 0.05
    turbulence = (turbulence + 1) * 0.5
    
    # Центральное ядро + вытянутая форма вверх (с повышенным контрастом)
    core = np.exp(-((xx_t * 1.8) ** 2)) * np.clip(2.6 - yy, 0, 1)
    # Языки с резкими краями (степень для контраста)
    tongues_raw = np.exp(-((xx_t * 5.0 + warp * 1.6) ** 2)) * np.clip(3.0 - yy, 0, 1)
    tongues = np.clip(tongues_raw * 1.5 - 0.2, 0, 1) ** 0.7
    lobe_raw = np.exp(-((xx_t * 6.0) ** 2)) * np.clip(3.2 - yy, 0, 1) * (0.6 + 0.4 * np.sin(t * 0.7 + yy * 4.0))
    lobe = np.clip(lobe_raw * 1.4 - 0.15, 0, 1) ** 0.8
    jets_raw = np.clip(turbulence - 0.15, 0, 1) * np.exp(-((xx_t * 7.0 + warp * 1.8) ** 2)) * np.clip(3.4 - yy, 0, 1)
    jets = np.clip(jets_raw * 1.6 - 0.1, 0, 1) ** 0.6
    rips = np.clip(turbulence * 1.4 - 0.35, 0, 1) * np.exp(-((xx_t * 4.2) ** 2)) * np.clip(2.2 - yy, 0, 1)
    
    # Усиленные языки у основания (резкие)
    base_tongues_raw = np.exp(-((xx_t * 3.5 + warp * 2.0) ** 2)) * np.clip(1.0 - yy * 0.6, 0, 1)
    base_tongues = np.clip(base_tongues_raw * 1.5 - 0.15, 0, 1) ** 0.7
    base_flicker = (0.7 + 0.3 * np.sin(t * 2.5 + xx * 8.0)) * np.clip(0.8 - yy, 0, 1)
    base_texture = np.clip(detail * 0.8 + micro * 0.4, 0, 1) * np.exp(-((xx_t * 2.0) ** 2)) * np.clip(0.6 - yy * 0.5, 0, 1)
    
    # Языки пламени веером от центра основания к краям
    yy_safe = np.maximum(yy, 0.05)  # избегаем деления на 0
    # Больше языков под меньшими углами (более вертикальные)
    tongue_angles = [-0.2, -0.13, -0.07, 0.0, 0.07, 0.13, 0.2]  # 7 языков, узкий веер
    tongues_radial = np.zeros_like(xx)
    for i, ta in enumerate(tongue_angles):
        # Ось языка с динамическим колебанием
        wobble = noise.noise_3d(np.full_like(xx, i * 0.7), yy * 1.5 - t * 0.6, zz) * 0.08
        tongue_axis = yy_safe * (ta + wobble)
        dist_from_axis = np.abs(xx_t - tongue_axis)
        # Ширина: больше у основания, сужается вверх
        width = 0.10 * np.clip(1.0 - yy * 0.35, 0.25, 1.0)
        tongue_shape = np.exp(-((dist_from_axis / width) ** 2))
        # Контрастность: резкие края
        tongue_shape = np.clip(tongue_shape * 1.5 - 0.2, 0, 1) ** 0.7
        # Яркость: сильнее у основания, центральный язык ярче
        center_boost = 1.0 + 0.4 * np.exp(-(ta * 8) ** 2)  # центр ярче
        brightness = np.clip(2.0 - yy * 1.0, 0, 1) * (0.8 + 0.2 * np.sin(t * 2.0 + i * 1.0)) * center_boost
        tongues_radial += tongue_shape * brightness
    tongues_radial = np.clip(tongues_radial * 0.6, 0, 1)
    
    flame = np.clip(core * (0.2 + 1.0 * turbulence) + tongues * 1.0 + lobe * 0.7 + jets * 0.7 + rips * 0.4 + base_tongues * 0.8 + base_flicker * 0.4 + base_texture * 0.5 + tongues_radial * 0.7, 0, 1)
    flame = np.clip((flame - 0.1) * 1.6, 0, 1)
    
    # Подсветка ядра
    core_glow = np.exp(-((xx_t * 3.2) ** 2 + (yy * 1.6) ** 2)) * np.clip(1.6 - yy, 0, 1)
    
    # Дым: только в верхней части (выше yy=1.0) - полностью чистый низ
    smoke_vertical = np.clip((yy - 1.0) * 2.0, 0, 1)  # маска: 0 внизу, 1 вверху
    smoke = np.clip((yy - 1.5) * 0.7, 0, 1) * (0.2 + 0.5 * turbulence)
    side_mask = np.clip((np.abs(xx) * 1.8 - 0.5), 0, 1)
    smoke = smoke * np.exp(-((xx_t * 1.5 + warp * 0.6) ** 2)) * side_mask
    trail = np.exp(-((xx_t * 0.8 + warp * 0.6) ** 2)) * np.clip(yy - 1.2, 0, 1)
    trail = trail * (0.3 + 0.5 * turbulence) * (0.5 + 0.5 * np.sin(t * 0.3 + yy * 1.5))
    smoke = np.clip((smoke + trail * 0.4) * smoke_vertical, 0, 1)
    
    # Искры: у основания без маски, выше - только по краям
    spark_mask = np.clip(micro - 0.7, 0, 1)
    edge = np.clip(np.abs(xx_t) * 2.0, 0, 1)
    bottom_zone = np.clip(1.0 - yy, 0, 1)
    sparks = spark_mask ** 2 * (bottom_zone * 1.2 + (1 - bottom_zone) * edge * 0.5)
    
    rgb = np.full((h, w, 3), np.array([0.55, 0.27, 0.20], dtype=np.float32), dtype=np.float32)
    heat = np.clip(flame * 1.2, 0, 1)
    
    # Температурный градиент: краснее/оранжевее у основания, белее к ядру
    vertical = np.clip(1.2 - yy, 0, 1)
    temp = np.clip(vertical * 0.6 + heat * 0.7, 0, 1)
    cold_color = np.array([1.0, 0.35, 0.02], dtype=np.float32)  # более насыщенный оранжево-красный
    hot_color = np.array([1.0, 0.92, 0.8], dtype=np.float32)
    temp_rgb = cold_color * (1 - temp)[..., None] + hot_color * temp[..., None]
    base_boost = np.clip(1.0 - yy * 0.3, 0.75, 1.0)
    base_heat = np.clip(1.0 - yy * 0.5, 0, 1) * np.exp(-((xx_t * 1.8) ** 2))
    # Дополнительная оранжевая текстура у основания (усилена)
    orange_base = np.clip(detail * 0.5 + 0.5, 0, 1) * np.clip(0.8 - yy * 0.45, 0, 1) * np.exp(-((xx_t * 1.5) ** 2))
    # Красное свечение у самого основания
    red_base = np.clip(0.5 - yy * 0.6, 0, 1) * np.exp(-((xx_t * 1.2) ** 2)) * (0.7 + 0.3 * turbulence)
    
    core_boost = 1.0 + 0.5 * core_glow
    core_ripple = 0.85 + 0.15 * np.sin(t * 1.4 + (xx_t + yy) * 3.5) * core_glow
    rgb[..., 0] = np.clip(temp_rgb[..., 0] * heat * core_boost * core_ripple * (1.15 * base_boost) + base_heat * 0.55 + orange_base * 0.7 + red_base * 0.5 + sparks * 1.6 + core_glow * 0.35, 0, 1)
    rgb[..., 1] = np.clip(temp_rgb[..., 1] * heat * core_boost * core_ripple * (1.08 * base_boost) + base_heat * 0.35 + orange_base * 0.35 + red_base * 0.15 + sparks * 0.8 + core_glow * 0.2, 0, 1)
    rgb[..., 2] = np.clip(temp_rgb[..., 2] * heat * (0.9 + 0.3 * core_glow) * core_ripple * (0.95 * base_boost) + base_heat * 0.03 + core_glow * 0.06, 0, 1)
    
    # Добавляем дым как серо-синий слой
    rgb[..., 0] = np.clip(rgb[..., 0] * (1 - smoke) + smoke * 0.3, 0, 1)
    rgb[..., 1] = np.clip(rgb[..., 1] * (1 - smoke) + smoke * 0.35, 0, 1)
    rgb[..., 2] = np.clip(rgb[..., 2] * (1 - smoke) + smoke * 0.4, 0, 1)
    return rgb


def main():
    parser = argparse.ArgumentParser(description="Fire flame example")
    add_interactive_args(parser)
    args = parser.parse_args()
    
    noise = SimplexNoise(seed=42)
    
    if args.interactive:
        config = InteractiveConfig.from_args(args, title="fractex")
        
        def render_frame(t, w, h):
            return _render_flame(t, w, h, noise)
        
        run_interactive(render_frame, config)
    else:
        image = _render_flame(0.0, 512, 512, noise)
        save_ppm(image, EXAMPLES_DIR / "output" / "fire_flame.ppm", stretch=True)


if __name__ == "__main__":
    main()
