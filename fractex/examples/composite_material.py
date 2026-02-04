# Создание сложных гибридных материалов
composite = CompositePatternGenerator3D(seed=42)

# Кристаллическая основа с гироидными каналами
hybrid_material = composite.generate_composite(
    pattern_types=[
        GeometricPattern3D.CRYSTAL_LATTICE,      # Основа
        GeometricPattern3D.GYROID,               # Каналы
        GeometricPattern3D.SPHERE_PACKING        # Включения
    ],
    dimensions=(128, 128, 128),
    params_list=[
        PatternParameters(crystal_type="diamond", thickness=0.05),
        PatternParameters(scale=4.0, surface_threshold=0.2),
        PatternParameters(sphere_radius=0.1, packing_density=0.2)
    ],
    blend_mode="add"
)
