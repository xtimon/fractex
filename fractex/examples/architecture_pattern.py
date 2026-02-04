# Готические витражи
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
    parser = argparse.ArgumentParser(description="Architecture pattern example")
    add_interactive_args(parser)
    args = parser.parse_args()
    stained_glass = CompositePatternGenerator3D(seed=42)
    
    gothic_pattern = stained_glass.generate_composite(
        pattern_types=[
            GeometricPattern3D.HONEYCOMB,
            GeometricPattern3D.DIAMOND_STRUCTURE,
            GeometricPattern3D.CRYSTAL_LATTICE,
        ],
        dimensions=(64, 64, 64),
        params_list=[
            PatternParameters(cell_size=0.2, wall_thickness=0.05),
            PatternParameters(scale=2.0, thickness=0.03),
            PatternParameters(crystal_type="cubic", scale=0.5, thickness=0.01),
        ],
    )
    
    print("Gothic pattern:", gothic_pattern.shape)
    save_volume_slice(
        gothic_pattern,
        EXAMPLES_DIR / "output" / "architecture_pattern_slice.ppm",
        stretch=True,
    )
    
    if args.interactive:
        config = InteractiveConfig.from_args(args, title="Architecture Pattern (interactive)")
        depth = gothic_pattern.shape[0]
        
        def render_frame(t, w, h):
            idx = int(abs(np.sin(t * 0.2)) * (depth - 1))
            return gothic_pattern[idx, :, :, :3]
        
        run_interactive(render_frame, config)


if __name__ == "__main__":
    main()
