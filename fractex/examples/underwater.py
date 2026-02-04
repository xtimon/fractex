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

from fractex.volume_scattering import UnderwaterScattering
from fractex.volume_textures import VolumeTextureGenerator3D
from _output import save_ppm, save_mp4
from fractex.interactive import add_interactive_args, InteractiveConfig, run_interactive


@dataclass
class Player:
    position: np.ndarray
    look_at: np.ndarray


def compute_bioluminescence(plankton, position, time_of_day: float) -> np.ndarray:
    strength = 1.0 if time_of_day > 0.8 or time_of_day < 0.2 else 0.0
    glow = plankton.data[..., 3].mean() if plankton.data is not None else 0.0
    return np.ones((64, 96, 3), dtype=np.float32) * glow * strength


def main():
    parser = argparse.ArgumentParser(description="Underwater example")
    add_interactive_args(parser)
    args = parser.parse_args()
    underwater_renderer = UnderwaterScattering()
    
    particle_generator = VolumeTextureGenerator3D(seed=42)
    plankton = particle_generator.generate_clouds_3d(
        width=32, height=32, depth=32,
        scale=0.1, density=0.1, detail=2
    )
    
    player = Player(
        position=np.array([0.0, -5.0, 0.0]),
        look_at=np.array([0.0, -5.0, 1.0])
    )
    water_level = 0.0
    time_of_day = 0.9
    is_night = time_of_day > 0.8 or time_of_day < 0.2
    
    frames = []
    for t in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.75]:
        is_night = t > 0.8 or t < 0.2
        underwater_image = underwater_renderer.render_underwater(
            camera_pos=player.position,
            view_direction=player.look_at - player.position,
            water_surface_height=water_level,
            image_size=(192, 128),
            max_depth=30.0
        )
        if is_night:
            bioluminescence = compute_bioluminescence(
                plankton, player.position, t
            )
            underwater_image = np.clip(underwater_image + bioluminescence * 0.3, 0, 1)
        frames.append(underwater_image)
    
    underwater_image = frames[-1]
    print("Underwater image:", underwater_image.shape)
    save_ppm(underwater_image, EXAMPLES_DIR / "output" / "underwater.ppm")
    save_mp4(frames, EXAMPLES_DIR / "output" / "underwater.mp4", fps=8, stretch=True)
    
    if args.interactive:
        config = InteractiveConfig.from_args(args, title="Underwater (interactive)")
        
        def render_frame(t, w, h):
            underwater_image = underwater_renderer.render_underwater(
                camera_pos=player.position,
                view_direction=player.look_at - player.position,
                water_surface_height=water_level,
                image_size=(w, h),
                max_depth=20.0
            )
            if t % 10 > 5:
                bioluminescence = compute_bioluminescence(plankton, player.position, 0.9)
                underwater_image = np.clip(underwater_image + bioluminescence * 0.3, 0, 1)
            return underwater_image
        
        run_interactive(render_frame, config)


if __name__ == "__main__":
    main()
