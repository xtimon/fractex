# Создание сложных гибридных материалов
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
    parser = argparse.ArgumentParser(description="Composite material example")
    add_interactive_args(parser)
    args = parser.parse_args()
    composite = CompositePatternGenerator3D(seed=42)
    
    # Кристаллическая основа с гироидными каналами
    hybrid_material = composite.generate_composite(
        pattern_types=[
            GeometricPattern3D.CRYSTAL_LATTICE,
            GeometricPattern3D.GYROID,
            GeometricPattern3D.DIAMOND_STRUCTURE,
        ],
        dimensions=(64, 64, 64),
        params_list=[
            PatternParameters(crystal_type="diamond", thickness=0.05),
            PatternParameters(scale=4.0, surface_threshold=0.2),
            PatternParameters(scale=2.0, thickness=0.02),
        ],
        blend_mode="add",
    )
    
    print("Hybrid material:", hybrid_material.shape)
    save_volume_slice(
        hybrid_material,
        EXAMPLES_DIR / "output" / "composite_material_slice.ppm",
        stretch=True,
    )
    
    if args.interactive:
        config = InteractiveConfig.from_args(args, title="Composite Material (interactive)")
        depth = hybrid_material.shape[0]
        
        def render_frame(t, w, h):
            idx = int(abs(np.sin(t * 0.2)) * (depth - 1))
            return hybrid_material[idx, :, :, :3]
        
        run_interactive(render_frame, config)


if __name__ == "__main__":
    main()
