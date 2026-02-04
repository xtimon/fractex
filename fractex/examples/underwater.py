# Полный подводный рендеринг
underwater_renderer = UnderwaterScattering()

# Добавляем подводные частицы
from fractex.volume_textures import VolumeTextureGenerator3D
particle_generator = VolumeTextureGenerator3D(seed=42)
plankton = particle_generator.generate_clouds_3d(
    width=64, height=64, depth=64,
    scale=0.1, density=0.1, detail=2
)

# Рендеринг подводной сцены
underwater_image = underwater_renderer.render_underwater(
    camera_pos=player.position,
    view_direction=player.look_at - player.position,
    water_surface_height=water_level,
    image_size=(1920, 1080),
    max_depth=50.0
)

# Добавляем биолюминесценцию
if is_night:
    # Свечение планктона
    bioluminescence = compute_bioluminescence(
        plankton, player.position, time_of_day
    )
    underwater_image += bioluminescence * 0.3
