# Геометрические паттерны для разрушаемых материалов
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fractex.geometric_patterns_3d import GeometricPatternGenerator3D, GeometricPattern3D, PatternParameters


class DestructibleMaterial:
    def __init__(self, pattern_type, params):
        self.pattern_generator = GeometricPatternGenerator3D()
        self.base_pattern = self.pattern_generator.generate_pattern(
            pattern_type, (32, 32, 32), params
        )
        self.stress_field = np.zeros((32, 32, 32), dtype=np.float32)
    
    def apply_stress(self, position, force):
        stress_point = self.world_to_voxel(position)
        self.propagate_stress(stress_point, force)
        fracture_points = self.find_fracture_points()
        self.modify_pattern_for_fractures(fracture_points)
    
    def propagate_stress(self, point, force, depth: int = 0):
        if depth > 6 or force < 0.5:
            return
        for direction in self.get_structure_directions(point):
            self.stress_field[tuple(point)] += force
            next_point = point + direction
            if self.is_connected(point, next_point):
                self.propagate_stress(next_point, force * 0.7, depth + 1)
    
    def world_to_voxel(self, position):
        pos = np.clip(np.array(position, dtype=int), 0, 31)
        return pos
    
    def get_structure_directions(self, point):
        return [
            np.array([1, 0, 0]),
            np.array([-1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, -1, 0]),
            np.array([0, 0, 1]),
            np.array([0, 0, -1]),
        ]
    
    def is_connected(self, point_a, point_b):
        if np.any(point_b < 0) or np.any(point_b >= 32):
            return False
        return True
    
    def find_fracture_points(self):
        threshold = self.stress_field.mean() + self.stress_field.std()
        return np.argwhere(self.stress_field > threshold)
    
    def modify_pattern_for_fractures(self, fracture_points):
        for p in fracture_points[:50]:
            self.base_pattern[tuple(p)] = 0


if __name__ == "__main__":
    material = DestructibleMaterial(
        GeometricPattern3D.CRYSTAL_LATTICE,
        PatternParameters(scale=1.0, thickness=0.05),
    )
    material.apply_stress((10, 10, 10), force=5.0)
    print("Fracture points:", material.find_fracture_points().shape[0])
