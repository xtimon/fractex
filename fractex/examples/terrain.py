# Создание сложного terrain материала
terrain_blender = TerrainTextureBlender(seed=42)

terrain = terrain_blender.create_terrain_material(
    height_map=height_map,
    texture_layers={
        "grass": grass_texture,
        "dirt": dirt_texture, 
        "rock": rock_texture,
        "snow": snow_texture
    },
    biome="mountain",
    custom_params={
        "height_ranges": [(0.0, 0.3), (0.2, 0.5), (0.4, 0.8), (0.7, 1.0)],
        "slope_thresholds": [0.4, 0.6, 0.8]
    }
)

# Добавление деталей
terrain_detailed = terrain_blender.add_detail_layers(
    terrain,
    detail_textures=[grass_detail, rock_detail, moss_detail],
    scale_factors=[2.0, 1.5, 3.0],
    blend_modes=["overlay", "multiply", "screen"]
)
