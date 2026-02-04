# Создание hybrid материала: 2D текстура + 3D детали
from fractex.simplex_noise import SimplexTextureGenerator
from fractex.texture_blending import TextureBlender
from fractex.volume_textures import VolumeTextureGenerator3D

# 2D terrain текстура
tex_gen_2d = SimplexTextureGenerator(seed=42)
terrain_2d = tex_gen_2d.generate_terrain(1024, 1024)

# 3D детали (камни, трава)
tex_gen_3d = VolumeTextureGenerator3D(seed=42)
rocks_3d = tex_gen_3d.generate_rocks_3d(128, 128, 32)
grass_3d = tex_gen_3d.generate_grass_3d(128, 128, 16)

# Проекция 3D деталей на 2D terrain
# (здесь нужна дополнительная логика проекции)
