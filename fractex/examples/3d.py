# Создание объемного облачного неба
generator = VolumeTextureGenerator3D(seed=42)
cloud_volume = generator.generate_clouds_3d(
    width=128, height=64, depth=128,
    scale=0.02, density=0.4, animated=True, time=game_time
)

# Создание подземной пещеры с лавой
cave_noise = generator.generate_perlin_3d(
    width=96, height=96, depth=96,
    scale=0.03, octaves=4
)

lava_pockets = generator.generate_lava_3d(
    width=96, height=96, depth=96,
    scale=0.01, temperature=0.8, animated=True, time=game_time
)

# Смешивание пещеры и лавы
blender = VolumeTextureBlender3D()
cave_mask = (cave_noise.data > 0.7).astype(np.float32)  # Порог для лавы
cave_with_lava = blender.blend(
    cave_noise, lava_pockets,
    blend_mode="add",
    blend_mask=cave_mask
)

# Рендеринг в реальном времени
renderer = VolumeTextureRenderer(cave_with_lava)
frame = renderer.render_raycast(
    camera_pos=player.position,
    camera_target=player.look_at,
    image_size=(1920, 1080),
    max_steps=128
)
