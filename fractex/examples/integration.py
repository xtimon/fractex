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

from fractex.simplex_noise import SimplexTextureGenerator
from fractex.texture_blending import TextureBlender
from _output import save_ppm
from fractex.interactive import add_interactive_args, InteractiveConfig, run_interactive


def main():
    parser = argparse.ArgumentParser(description="Texture blending example")
    add_interactive_args(parser)
    args = parser.parse_args()
    # Генерация текстур
    tex_gen = SimplexTextureGenerator(seed=42)
    size = 256
    clouds = tex_gen.generate_clouds(size, size)
    rock = tex_gen.generate_marble(size, size)
    grass = tex_gen.generate_grass(size, size)
    
    height_map = rock[..., 0]
    cloud_mask = clouds[..., 3]
    
    blender = TextureBlender()
    result = blender.blend_layer_stack(
        base_texture=rock,
        layers=[
            {
                "texture": grass,
                "blend_mode": "overlay",
                "opacity": 0.7,
                "mask_params": {
                    "mask_type": "height_based",
                    "parameters": {
                        "height_map": height_map,
                        "min_height": 0.3,
                        "max_height": 0.6,
                    },
                },
            },
            {
                "texture": clouds,
                "blend_mode": "screen",
                "opacity": 0.3,
                "mask": cloud_mask,
            },
        ],
    )
    
    print("Blended texture:", result.shape, result.min(), result.max())
    save_ppm(result, EXAMPLES_DIR / "output" / "integration_blend.ppm")
    
    if args.interactive:
        config = InteractiveConfig.from_args(args, title="Integration Blend (interactive)")
        
        def render_frame(t, w, h):
            tex_gen = SimplexTextureGenerator(seed=42)
            clouds = tex_gen.generate_clouds(w, h, scale=0.01 + 0.003 * np.sin(t * 0.1))
            rock = tex_gen.generate_marble(w, h, scale=0.005 + 0.002 * np.cos(t * 0.07))
            grass = tex_gen.generate_grass(w, h, scale=0.02 + 0.01 * np.sin(t * 0.08))
            height_map = rock[..., 0]
            cloud_mask = clouds[..., 3]
            blender = TextureBlender()
            return blender.blend_layer_stack(
                base_texture=rock,
                layers=[
                    {
                        "texture": grass,
                        "blend_mode": "overlay",
                        "opacity": 0.7,
                        "mask_params": {
                            "mask_type": "height_based",
                            "parameters": {
                                "height_map": height_map,
                                "min_height": 0.3,
                                "max_height": 0.6,
                            },
                        },
                    },
                    {
                        "texture": clouds,
                        "blend_mode": "screen",
                        "opacity": 0.3,
                        "mask": cloud_mask,
                    },
                ],
            )[..., :3]
        
        run_interactive(render_frame, config)


if __name__ == "__main__":
    main()
