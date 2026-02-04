# Кристаллическая пещера
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

from fractex.geometric_patterns_3d import (
    CompositePatternGenerator3D,
    GeometricPattern3D,
    PatternParameters,
)
from _output import save_volume_slice
from fractex.interactive import add_interactive_args, InteractiveConfig, run_interactive


def main():
    parser = argparse.ArgumentParser(description="Crystal cave example")
    add_interactive_args(parser)
    args = parser.parse_args()
    world_seed = 123
    crystal_cave = CompositePatternGenerator3D(seed=world_seed)
    
    cave_pattern = crystal_cave.generate_layered_pattern(
        pattern_layers=[
            (GeometricPattern3D.CRYSTAL_LATTICE,
             PatternParameters(crystal_type="hexagonal", scale=1.5), 0.8),
            (GeometricPattern3D.DIAMOND_STRUCTURE,
             PatternParameters(scale=3.0, thickness=0.02), 0.4),
            (GeometricPattern3D.LAVA_LAMPS,
             PatternParameters(surface_isolevel=0.3), 0.6),
        ],
        dimensions=(64, 64, 64),
    )
    
    print("Cave pattern:", cave_pattern.shape)
    save_volume_slice(
        cave_pattern,
        EXAMPLES_DIR / "output" / "crystal_cave_slice.ppm",
        stretch=True,
    )
    
    if args.interactive:
        config = InteractiveConfig.from_args(args, title="Crystal Cave (interactive)")
        depth = cave_pattern.shape[0]
        
        def render_frame(t, w, h):
            idx = int(abs(np.sin(t * 0.2)) * (depth - 1))
            return cave_pattern[idx, :, :, :3]
        
        run_interactive(render_frame, config)


if __name__ == "__main__":
    main()
