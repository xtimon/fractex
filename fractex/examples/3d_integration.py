# Полный пайплайн: текстура + рассеяние
from fractex.volume_textures import VolumeTextureGenerator3D
from fractex.volume_scattering import VolumeScatteringRenderer, MediumProperties, LightSource

# Генерация объемной текстуры облаков
generator = VolumeTextureGenerator3D(seed=42)
clouds_3d = generator.generate_clouds_3d(
    width=128, height=64, depth=128,
    scale=0.02, density=0.3, animated=True, time=game_time
)

# Настройка атмосферного рассеяния
atmosphere = MediumProperties(
    scattering_coefficient=0.08,  # Рассеяние в облаках
    absorption_coefficient=0.02,  # Небольшое поглощение
    phase_function_g=0.7,  # Рассеяние вперед (облака)
    density=1.0,
    color=(1.0, 0.95, 0.9)  # Теплый свет заката
)

# Солнечный свет
sun = LightSource(
    direction=sun_direction,  # Меняется в течение дня
    color=sun_color,  # От белого до красного на закате
    intensity=sun_intensity,
    light_type="directional"
)

# Рендеринг с рассеянием
renderer = VolumeScatteringRenderer(
    volume=clouds_3d,
    medium=atmosphere,
    light_sources=[sun],
    use_multiple_scattering=True
)

# Камера игрока
image = renderer.render_volumetric_light(
    camera_pos=player.position,
    camera_target=player.look_at,
    image_size=(1920, 1080),
    max_steps=128
)

# Смешивание с 3D сценой
final_image = blend_volume_with_scene(scene_image, image, clouds_3d)
