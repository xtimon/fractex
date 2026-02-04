# Полный пайплайн: текстура + рассеяние
import sys
from pathlib import Path
import numpy as np
import argparse
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from fractex.volume_textures import VolumeTextureGenerator3D
from fractex.volume_scattering import VolumeScatteringRenderer, MediumProperties, LightSource
from _output import save_ppm, save_mp4
from fractex.interactive import add_interactive_args, InteractiveConfig, run_interactive


@dataclass
class Player:
    position: np.ndarray
    look_at: np.ndarray


def blend_volume_with_scene(scene_image: np.ndarray, volume_image: np.ndarray, weight: float = 0.6) -> np.ndarray:
    return np.clip(scene_image * (1.0 - weight) + volume_image * weight, 0, 1)


def main():
    parser = argparse.ArgumentParser(description="3D integration example")
    add_interactive_args(parser)
    args = parser.parse_args()
    # Генерация объемной текстуры облаков
    generator = VolumeTextureGenerator3D(seed=42)
    clouds_3d = generator.generate_clouds_3d(
        width=64, height=32, depth=64,
        scale=0.02, density=0.3, animated=False
    )
    
    # Настройка атмосферного рассеяния
    atmosphere = MediumProperties(
        scattering_coefficient=0.08,
        absorption_coefficient=0.02,
        phase_function_g=0.7,
        density=1.0,
        color=(1.0, 0.95, 0.9)
    )
    
    # Солнечный свет
    sun = LightSource(
        direction=np.array([0.2, 0.9, 0.1]),
        color=(1.0, 0.95, 0.9),
        intensity=1.0,
        light_type="directional"
    )
    
    player = Player(
        position=np.array([0.5, 0.5, -1.0]),
        look_at=np.array([0.5, 0.5, 0.5])
    )
    
    frames = []
    directions = [
        np.array([0.3, 0.9, 0.1]),
        np.array([0.1, 0.9, 0.3]),
        np.array([-0.1, 0.9, 0.3]),
        np.array([-0.3, 0.9, 0.1]),
        np.array([0.0, 0.7, 0.7]),
        np.array([0.0, 0.6, 0.8]),
    ]
    for direction in directions:
        sun.direction = direction
        renderer = VolumeScatteringRenderer(
            volume=clouds_3d,
            medium=atmosphere,
            light_sources=[sun],
            use_multiple_scattering=False
        )
        image = renderer.render_volumetric_light(
            camera_pos=player.position,
            camera_target=player.look_at,
            image_size=(320, 180),
            max_steps=48
        )
        scene_image = np.zeros_like(image)
        final_image = blend_volume_with_scene(scene_image, image)
        frames.append(final_image)
    
    final_image = frames[0]
    print("Rendered image:", final_image.shape, final_image.min(), final_image.max())
    if float(final_image.max()) == 0.0:
        density = clouds_3d.data[clouds_3d.data.shape[0] // 2, :, :, 3]
        save_ppm(density, EXAMPLES_DIR / "output" / "3d_integration.ppm", stretch=True)
    else:
        save_ppm(final_image, EXAMPLES_DIR / "output" / "3d_integration.ppm", stretch=True)
    save_mp4(frames, EXAMPLES_DIR / "output" / "3d_integration.mp4", fps=8, stretch=True)
    
    if args.interactive:
        config = InteractiveConfig.from_args(args, title="3D Integration (interactive)")
        depth = clouds_3d.data.shape[0]
        
        def render_frame(t, w, h):
            idx = int(abs(np.sin(t * 0.2)) * (depth - 1))
            density = clouds_3d.data[idx, :, :, 3]
            return density
        
        run_interactive(render_frame, config)


if __name__ == "__main__":
    main()
