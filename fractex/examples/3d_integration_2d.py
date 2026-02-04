# Создание hybrid материала: 2D текстура + 3D детали
import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from fractex.simplex_noise import SimplexTextureGenerator
from fractex.volume_textures import VolumeTextureGenerator3D
from _output import save_ppm
from fractex.interactive import add_interactive_args, InteractiveConfig, run_interactive


def main():
    parser = argparse.ArgumentParser(description="3D integration 2D example")
    add_interactive_args(parser)
    args = parser.parse_args()
    
    # 2D terrain текстура
    tex_gen_2d = SimplexTextureGenerator(seed=42)
    terrain_2d = tex_gen_2d.generate_terrain(64, 64)
    
    # 3D детали (камни, трава)
    tex_gen_3d = VolumeTextureGenerator3D(seed=42)
    rocks_3d = tex_gen_3d.generate_rocks_3d(64, 64, 32)
    grass_3d = tex_gen_3d.generate_grass_3d(64, 64, 16)
    
    print("Terrain 2D:", terrain_2d.shape)
    print("Rocks 3D:", rocks_3d.data.shape)
    print("Grass 3D:", grass_3d.data.shape)
    
    # Проекция 3D деталей на 2D terrain
    # (демо — используем средний срез по глубине)
    rocks_slice = rocks_3d.data[rocks_3d.data.shape[0] // 2, :, :, 3]
    grass_slice = grass_3d.data[grass_3d.data.shape[0] // 2, :, :, 3]
    combined = terrain_2d.copy()
    combined[..., 1] = (combined[..., 1] * 0.7 + grass_slice * 0.3)
    combined[..., 2] = (combined[..., 2] * 0.7 + rocks_slice * 0.3)
    
    print("Combined terrain sample:", combined[0, 0])
    save_ppm(combined, EXAMPLES_DIR / "output" / "3d_integration_2d.ppm")
    
    if args.interactive:
        config = InteractiveConfig.from_args(args, title="3D Integration 2D (interactive)")

        def render_frame(t, w, h):
            terrain = tex_gen_2d.generate_terrain(w, h)
            return terrain[..., :3]
        
        run_interactive(render_frame, config)


if __name__ == "__main__":
    main()
