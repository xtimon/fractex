import sys
from pathlib import Path
import numpy as np
import argparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from fractex.texture_blending import TerrainTextureBlender
from fractex.simplex_noise import SimplexTextureGenerator
from _output import save_ppm
from fractex.interactive import add_interactive_args, InteractiveConfig, run_interactive


def main():
    parser = argparse.ArgumentParser(description="Terrain example")
    add_interactive_args(parser)
    args = parser.parse_args()
    terrain_blender = TerrainTextureBlender(seed=42)
    tex_gen = SimplexTextureGenerator(seed=42)
    
    size = 256
    terrain_base = tex_gen.generate_terrain(size, size)
    height_map = terrain_base[..., 0]
    
    grass_texture = tex_gen.generate_grass(size, size)
    dirt_texture = tex_gen.generate_wood(size, size)
    rock_texture = tex_gen.generate_marble(size, size)
    snow_texture = tex_gen.generate_clouds(size, size)
    
    terrain = terrain_blender.create_terrain_material(
        height_map=height_map,
        texture_layers={
            "grass": grass_texture,
            "dirt": dirt_texture,
            "rock": rock_texture,
            "snow": snow_texture,
        },
        biome="mountain",
        custom_params={
            "height_ranges": [(0.0, 0.3), (0.2, 0.5), (0.4, 0.8), (0.7, 1.0)],
            "slope_thresholds": [0.4, 0.6, 0.8],
        },
    )
    
    grass_detail = tex_gen.generate_grass(size, size)
    rock_detail = tex_gen.generate_marble(size, size)
    moss_detail = tex_gen.generate_clouds(size, size)
    
    terrain_detailed = terrain_blender.add_detail_layers(
        terrain,
        detail_textures=[grass_detail, rock_detail, moss_detail],
        scale_factors=[2.0, 1.5, 3.0],
        blend_modes=["overlay", "multiply", "screen"],
    )
    
    print("Terrain material:", terrain.shape, terrain.min(), terrain.max())
    print("Terrain detailed:", terrain_detailed.shape)
    save_ppm(terrain_detailed, EXAMPLES_DIR / "output" / "terrain_detailed.ppm")
    
    if args.interactive:
        config = InteractiveConfig.from_args(args, title="Terrain (interactive)")
        
        def render_frame(t, w, h):
            tex = SimplexTextureGenerator(seed=42)
            return tex.generate_terrain(w, h, scale=0.004 + 0.002 * np.sin(t * 0.1))[..., :3]
        
        run_interactive(render_frame, config)


if __name__ == "__main__":
    main()
