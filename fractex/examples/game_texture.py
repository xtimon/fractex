# fractex/examples/game_texture.py
"""
Пример интеграции в игровой движок
"""

import sys
from pathlib import Path
from queue import Queue
import threading
import argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from fractex import FractalParams, FractalGenerator, InfiniteTexture, TextureStreamer
from _output import save_ppm
from fractex.interactive import add_interactive_args, InteractiveConfig, run_interactive


class GameTextureSystem:
    """Система текстур для игры с бесконечной детализацией"""
    
    def __init__(self):
        self.textures = {}
        self.streamers = {}
        self.worker_threads = []
        self.task_queue = Queue()
        
        # Запускаем рабочие потоки для генерации текстур
        for i in range(4):  # 4 потока
            thread = threading.Thread(target=self._texture_worker)
            thread.daemon = True
            thread.start()
            self.worker_threads.append(thread)
    
    def register_texture(self, name, params, texture_type="procedural"):
        """Регистрация новой текстуры"""
        generator = FractalGenerator(params)
        texture = InfiniteTexture(generator, texture_type)
        streamer = TextureStreamer(texture)
        
        self.textures[name] = texture
        self.streamers[name] = streamer
        
        return texture
    
    def request_texture_tiles(self, name, viewport, camera_zoom):
        """Запрос тайлов текстуры для вьюпорта"""
        streamer = self.streamers[name]
        
        # Определяем необходимые тайлы и уровни детализации
        tiles_needed = self._calculate_tiles_needed(viewport, camera_zoom)
        
        # Запрашиваем все необходимые тайлы
        results = {}
        for tile_info in tiles_needed:
            tile_x, tile_y, lod = tile_info
            
            # Проверяем кэш
            tile = streamer.request_tile(tile_x, tile_y, lod)
            results[(tile_x, tile_y, lod)] = tile
        
        return results
    
    def _calculate_tiles_needed(self, viewport, zoom):
        """Расчет необходимых тайлов на основе положения камеры"""
        tiles = []
        
        # Преобразуем вьюпорт в тайловые координаты
        min_x, min_y = viewport['min']
        max_x, max_y = viewport['max']
        
        # Разные LOD для разных расстояний (MIP-маппинг)
        base_tile_size = 256
        for lod in range(4):  # 4 уровня детализации
            tile_scale = 2 ** lod
            effective_tile_size = base_tile_size / tile_scale
            
            start_tile_x = int(min_x // effective_tile_size)
            start_tile_y = int(min_y // effective_tile_size)
            end_tile_x = int(max_x // effective_tile_size) + 1
            end_tile_y = int(max_y // effective_tile_size) + 1
            
            for tx in range(start_tile_x, end_tile_x):
                for ty in range(start_tile_y, end_tile_y):
                    tiles.append((tx, ty, lod))
        
        return tiles
    
    def _texture_worker(self):
        """Рабочий поток для генерации текстур"""
        while True:
            try:
                task = self.task_queue.get()
                if task is None:
                    break
                
                texture_name, tile_x, tile_y, lod = task
                streamer = self.streamers[texture_name]
                streamer.request_tile(tile_x, tile_y, lod)
                
                self.task_queue.task_done()
            except:
                pass

# Демонстрация
def demo_terrain_texture():
    """Создание бесконечно детализируемой текстуры terrain"""
    
    # Параметры для terrain текстуры
    terrain_params = FractalParams(
        seed=42,
        base_scale=0.005,
        detail_level=4.0,
        persistence=0.55,
        lacunarity=2.1,
        octaves=16,
        fractal_dimension=2.7
    )
    
    # Создаем систему текстур
    texture_system = GameTextureSystem()
    
    # Регистрируем различные текстуры
    texture_system.register_texture("terrain", terrain_params, "stone")
    texture_system.register_texture("clouds", 
        FractalParams(seed=123, persistence=0.7, octaves=8), "clouds")
    texture_system.register_texture("water",
        FractalParams(seed=456, base_scale=0.002, persistence=0.4), "water")
    
    # Имитация запросов от игрового движка
    viewport = {'min': (0, 0), 'max': (1024, 1024)}
    zoom_levels = [1.0, 2.0, 4.0, 8.0]
    
    for zoom in zoom_levels:
        print(f"\nGenerating texture tiles at zoom {zoom}x...")
        
        terrain_tiles = texture_system.request_texture_tiles(
            "terrain", viewport, zoom
        )
        
        print(f"Generated {len(terrain_tiles)} terrain tiles")
        
        # Здесь можно визуализировать или сохранить тайлы
        for (tx, ty, lod), tile in list(terrain_tiles.items())[:3]:
            print(f"  Tile ({tx},{ty}) LOD {lod}: {tile.shape}")
        
        if terrain_tiles:
            first_tile = next(iter(terrain_tiles.values()))
            save_ppm(first_tile, EXAMPLES_DIR / "output" / f"game_texture_tile_{zoom:.1f}x.ppm")
    
    return texture_system

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Game texture example")
    add_interactive_args(parser)
    args = parser.parse_args()
    
    # Запускаем демо
    system = demo_terrain_texture()
    print("\nTexture system ready!")
    
    if args.interactive:
        config = InteractiveConfig.from_args(args, title="Game Texture (interactive)")
        terrain_params = FractalParams(seed=42, base_scale=0.005, detail_level=3.0)
        generator = FractalGenerator(terrain_params)
        texture = InfiniteTexture(generator, "stone")
        
        def render_frame(t, w, h):
            zoom = 1.0 + 0.5 * (1 + np.sin(t * 0.2))
            x0 = -w / (2.0 * zoom)
            y0 = -h / (2.0 * zoom)
            return texture.generate_tile(x0, y0, w, h, zoom=zoom)[..., :3]
        
        run_interactive(render_frame, config)
