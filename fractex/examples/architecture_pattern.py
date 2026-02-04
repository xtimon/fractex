# Готические витражи
stained_glass = CompositePatternGenerator3D(seed=42)

gothic_pattern = stained_glass.generate_composite(
    pattern_types=[
        GeometricPattern3D.HONEYCOMB,      # Основная структура
        GeometricPattern3D.VORONOI_CELLS,  # Цветные сегменты
        GeometricPattern3D.CRYSTAL_LATTICE # Декоративные элементы
    ],
    params_list=[
        PatternParameters(cell_size=0.2, wall_thickness=0.05),
        PatternParameters(packing_density=0.4),
        PatternParameters(crystal_type="cubic", scale=0.5, thickness=0.01)
    ]
)
