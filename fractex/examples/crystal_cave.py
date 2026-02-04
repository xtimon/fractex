# Кристаллическая пещера
crystal_cave = CompositePatternGenerator3D(seed=world_seed)

cave_pattern = crystal_cave.generate_layered_pattern(
    pattern_layers=[
        (GeometricPattern3D.CRYSTAL_LATTICE,  # Основная порода
         PatternParameters(crystal_type="hexagonal", scale=1.5), 0.8),
        
        (GeometricPattern3D.DIAMOND_STRUCTURE,  # Драгоценные кристаллы
         PatternParameters(scale=3.0, thickness=0.02), 0.4),
        
        (GeometricPattern3D.LAVA_LAMPS,  # Светящиеся включения
         PatternParameters(surface_isolevel=0.3), 0.6)
    ],
    dimensions=(96, 96, 96)
)
