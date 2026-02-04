# Геометрические паттерны для разрушаемых материалов
class DestructibleMaterial:
    def __init__(self, pattern_type, params):
        self.pattern_generator = GeometricPatternGenerator3D()
        self.base_pattern = self.pattern_generator.generate_pattern(
            pattern_type, (64, 64, 64), params
        )
        self.stress_field = np.zeros((64, 64, 64))
    
    def apply_stress(self, position, force):
        # Локализация напряжения в паттерне
        stress_point = self.world_to_voxel(position)
        
        # Распространение напряжения по структурным линиям
        self.propagate_stress(stress_point, force)
        
        # Определение точек разрушения
        fracture_points = self.find_fracture_points()
        
        # Модификация паттерна на основе разрушений
        self.modify_pattern_for_fractures(fracture_points)
    
    def propagate_stress(self, point, force):
        # Напряжение распространяется вдоль линий паттерна
        # (например, по ребрам кристаллической решетки)
        for direction in self.get_structure_directions(point):
            self.stress_field[point] += force
            next_point = point + direction
            
            if self.is_connected(point, next_point):
                # Рекурсивное распространение
                self.propagate_stress(next_point, force * 0.7)
