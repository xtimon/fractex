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

from fractex.volume_textures import (
    VolumeTextureGenerator3D,
    VolumeTextureBlender3D,
    VolumeTextureRenderer,
    VolumeTexture3D,
    VolumeFormat,
)
from _output import save_ppm, save_mp4
from fractex.interactive import add_interactive_args, InteractiveConfig, run_interactive


@dataclass
class Player:
    position: np.ndarray
    look_at: np.ndarray


def main():
    parser = argparse.ArgumentParser(description="3D volume example")
    add_interactive_args(parser)
    args = parser.parse_args()
    game_time = 0.0
    player = Player(
        position=np.array([0.5, 0.5, 2.0]),
        look_at=np.array([0.5, 0.5, 0.5])
    )
    
    # Создание объемного облачного неба
    generator = VolumeTextureGenerator3D(seed=42)
    cloud_volume = generator.generate_clouds_3d(
        width=64, height=32, depth=64,
        scale=0.02, density=0.4, animated=False, time=game_time
    )
    
    # Создание подземной пещеры с лавой
    cave_noise = generator.generate_perlin_3d(
        width=48, height=48, depth=48,
        scale=0.03, octaves=3
    )
    
    lava_pockets = generator.generate_lava_3d(
        width=48, height=48, depth=48,
        scale=0.01, temperature=0.8, animated=False, time=game_time
    )
    
    # Смешивание пещеры и лавы
    blender = VolumeTextureBlender3D()
    cave_mask = (cave_noise.data > 0.7).astype(np.float32)
    cave_rgba = VolumeTexture3D(
        data=np.repeat(cave_noise.data, 4, axis=3),
        format=VolumeFormat.RGBA_FLOAT,
        voxel_size=cave_noise.voxel_size,
    )
    cave_with_lava = blender.blend(
        cave_rgba, lava_pockets,
        blend_mode="add",
        blend_mask=cave_mask
    )
    
    # Рендеринг
    renderer = VolumeTextureRenderer(cave_with_lava)
    frames = []
    for z in [2.2, 2.0, 1.8, 1.6, 1.4, 1.2]:
        camera_pos = np.array([0.5, 0.5, z])
        frame = renderer.render_raycast(
            camera_pos=camera_pos,
            camera_target=player.look_at,
            image_size=(256, 144),
            max_steps=64
        )
        frames.append(frame)
    
    print("Cloud volume:", cloud_volume.data.shape)
    print("Cave with lava:", cave_with_lava.data.shape)
    print("Rendered frame:", frames[0].shape)
    if float(frames[0].max()) == 0.0:
        projection = cave_with_lava.data[..., 3].max(axis=0)
        save_ppm(projection, EXAMPLES_DIR / "output" / "3d_raycast.ppm", stretch=True)
    else:
        save_ppm(frames[0], EXAMPLES_DIR / "output" / "3d_raycast.ppm", stretch=True)
    save_mp4(frames, EXAMPLES_DIR / "output" / "3d_raycast.mp4", fps=8, stretch=True)
    
    if args.interactive:
        config = InteractiveConfig.from_args(args, title="3D Raycast (interactive)")
        depth = cave_with_lava.data.shape[0]
        
        def render_frame(t, w, h):
            idx = int(abs(np.sin(t * 0.2)) * (depth - 1))
            slice_img = cave_with_lava.data[idx]
            return slice_img[..., :3]
        
        run_interactive(render_frame, config)


if __name__ == "__main__":
    main()
