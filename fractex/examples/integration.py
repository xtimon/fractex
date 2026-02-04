from fractex.simplex_noise import SimplexTextureGenerator
from fractex.texture_blending import TextureBlender, TerrainTextureBlender

# Генерация текстур
tex_gen = SimplexTextureGenerator(seed=42)
clouds = tex_gen.generate_clouds(1024, 1024)
rock = tex_gen.generate_marble(1024, 1024)
grass = tex_gen.generate_grass(1024, 1024)  # hypothetical

# Смешивание
blender = TextureBlender()
result = blender.blend_layer_stack(
    base_texture=rock,
    layers=[
        {
            'texture': grass,
            'blend_mode': 'overlay',
            'opacity': 0.7,
            'mask_params': {
                'mask_type': 'height_based',
                'height_map': height_map,
                'min_height': 0.3,
                'max_height': 0.6
            }
        },
        {
            'texture': clouds,
            'blend_mode': 'screen',
            'opacity': 0.3,
            'mask': cloud_mask
        }
    ]
)
