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

from fractex.dynamic_textures_3d import DynamicTextureGenerator3D, DynamicTextureType, DynamicTextureState
from _output import save_ppm_sequence, save_mp4
from fractex.interactive import add_interactive_args, InteractiveConfig, run_interactive


def blend_lava_and_water(lava_state: DynamicTextureState, water_state: DynamicTextureState) -> DynamicTextureState:
    blended = np.clip(lava_state.data * 0.6 + water_state.data * 0.4, 0, 1)
    return DynamicTextureState(time=water_state.time, data=blended)


def add_bubbles_to_water(lava_state: DynamicTextureState, water_state: DynamicTextureState) -> DynamicTextureState:
    if lava_state.temperature_field is None or water_state.data is None:
        return water_state
    hot_mask = lava_state.temperature_field > 800
    bubbles = np.zeros_like(water_state.data[..., 0])
    bubbles[hot_mask] = np.clip(lava_state.temperature_field[hot_mask] / 1200.0, 0, 1) * 0.1
    water_state.data[..., 0] = np.clip(water_state.data[..., 0] + bubbles, 0, 1)
    return water_state


def create_underwater_volcano():
    """Создание подводного вулкана с лавой и пузырями"""
    lava_generator = DynamicTextureGenerator3D(
        dimensions=(32, 32, 32),
        texture_type=DynamicTextureType.LAVA_FLOW,
        seed=42
    )
    
    water_generator = DynamicTextureGenerator3D(
        dimensions=(32, 32, 32),
        texture_type=DynamicTextureType.WATER_FLOW,
        seed=123
    )
    
    def safe_update(generator: DynamicTextureGenerator3D) -> DynamicTextureState:
        try:
            return generator.update()
        except Exception as exc:
            print(f"Simulation update skipped: {exc}")
            depth, height, width = generator.dimensions
            data = np.zeros((depth, height, width, 4), dtype=np.float32)
            return DynamicTextureState(time=0.0, data=data)
    
    states = []
    for _ in range(12):
        lava_state = safe_update(lava_generator)
        water_state = safe_update(water_generator)
        water_state = add_bubbles_to_water(lava_state, water_state)
        blended_state = blend_lava_and_water(lava_state, water_state)
        states.append(blended_state)
    
    return states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Underwater volcano example")
    add_interactive_args(parser)
    args = parser.parse_args()
    
    states = create_underwater_volcano()
    print("Frames:", len(states), "state shape:", states[-1].data.shape)
    frames = []
    for state in states:
        mid = state.data.shape[0] // 2
        frames.append(state.data[mid])
    save_ppm_sequence(
        frames,
        EXAMPLES_DIR / "output" / "underwater_volkano_frames",
        prefix="volkano",
        stretch=True,
    )
    save_mp4(
        frames,
        EXAMPLES_DIR / "output" / "underwater_volkano.mp4",
        fps=12,
        stretch=True,
    )
    
    if args.interactive:
        config = InteractiveConfig.from_args(args, title="Underwater Volcano (interactive)")
        lava_generator = DynamicTextureGenerator3D(
            dimensions=(32, 32, 32),
            texture_type=DynamicTextureType.LAVA_FLOW,
            seed=42
        )
        water_generator = DynamicTextureGenerator3D(
            dimensions=(32, 32, 32),
            texture_type=DynamicTextureType.WATER_FLOW,
            seed=123
        )
        
        def render_frame(t, w, h):
            lava_state = lava_generator.update()
            water_state = water_generator.update()
            water_state = add_bubbles_to_water(lava_state, water_state)
            blended_state = blend_lava_and_water(lava_state, water_state)
            mid = blended_state.data.shape[0] // 2
            return blended_state.data[mid, :, :, :3]
        
        run_interactive(render_frame, config)
