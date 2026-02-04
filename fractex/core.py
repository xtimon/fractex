# fractex/core.py
"""
Ядро для генерации бесконечно детализируемых фрактальных текстур
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from functools import lru_cache
import hashlib

@dataclass
class FractalParams:
    """Параметры фрактальной текстуры"""
    seed: int = 42
    base_scale: float = 1.0
    detail_level: float = 1.0  # Множитель детализации (1.0 = стандарт)
    persistence: float = 0.5   # Сохранение энергии между октавами
    lacunarity: float = 2.0    # Множитель частоты между октавами
    octaves: int = 8           # Базовое количество октав
    fractal_dimension: float = 2.3  # Фрактальная размерность
    color_gradient: Optional[np.ndarray] = None

class FractalGenerator:
    """Базовый генератор фрактальных текстур"""
    
    def __init__(self, params: FractalParams = None):
        self.params = params or FractalParams()
        self._gradient_cache = {}
        self._init_gradients()
    
    def _init_gradients(self):
        """Инициализация градиентов для шума"""
        np.random.seed(self.params.seed)
        # Градиенты для 2D шума
        angles = np.random.rand(256) * 2 * np.pi
        self.gradients = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
        # Таблица перестановок для шума Перлина
        self.perm = np.random.permutation(512)
    
    def fractal_noise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Генерация фрактального шума (fBm - fractional Brownian motion)
        с адаптивным количеством октав для бесконечной детализации
        """
        # Динамическое определение октав на основе требуемой детализации
        effective_octaves = max(1, int(self.params.octaves * self.params.detail_level))
        
        value = np.zeros_like(x)
        amplitude = 1.0
        frequency = self.params.base_scale
        
        for i in range(effective_octaves):
            # Добавляем октаву шума
            noise = self._octave_noise(x * frequency, y * frequency, i)
            value += amplitude * noise
            
            # Обновляем параметры для следующей октавы
            amplitude *= self.params.persistence
            frequency *= self.params.lacunarity
            
            # Если амплитуда слишком мала - останавливаемся
            if amplitude < 0.001:
                break
        
        # Нормализация
        max_val = (1 - self.params.persistence ** effective_octaves) / (1 - self.params.persistence)
        return value / max_val
    
    def _octave_noise(self, x: np.ndarray, y: np.ndarray, octave: int) -> np.ndarray:
        """Генерация одной октавы шума с уникальным seed для октавы"""
        # Используем хеш seed + octave для уникальности каждой октавы
        octave_seed = hashlib.md5(f"{self.params.seed}_{octave}".encode()).hexdigest()
        octave_seed_int = int(octave_seed, 16) % (2**32)
        
        # Временное сохранение seed
        original_state = np.random.get_state()
        np.random.seed(octave_seed_int)
        
        # Генерация симплекс-подобного шума
        noise = self._simplex_noise_2d(x, y + octave * 100)  # Сдвиг для уникальности
        
        # Восстановление seed
        np.random.set_state(original_state)
        
        return noise
    
    def _simplex_noise_2d(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Упрощенный 2D симплекс-шум для производительности"""
        # Преобразуем в float для точности
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        
        # Константы для симплекс-шума
        F2 = 0.5 * (np.sqrt(3.0) - 1.0)
        G2 = (3.0 - np.sqrt(3.0)) / 6.0
        
        # Скалярное произведение с градиентами
        noise = np.zeros_like(x)
        
        # Упрощенная реализация (полная реализация слишком длинная для примера)
        # Здесь используем упрощенный шум для демонстрации
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # Упрощенный псевдо-шум
                nx = x[i, j] * 0.01
                ny = y[i, j] * 0.01
                noise[i, j] = np.sin(nx * 12.9898 + ny * 78.233) * 43758.5453
                noise[i, j] = noise[i, j] - np.floor(noise[i, j])
        
        return noise * 2 - 1  # Нормализация к [-1, 1]

class InfiniteTexture:
    """Класс бесконечно детализируемой текстуры"""
    
    def __init__(self, generator: FractalGenerator, 
                 texture_type: str = "procedural"):
        self.generator = generator
        self.texture_type = texture_type
        self._cache = {}
        
        # Предопределенные типы текстур
        self.texture_presets = {
            "clouds": self._cloud_preset,
            "marble": self._marble_preset,
            "wood": self._wood_preset,
            "stone": self._stone_preset,
            "lava": self._lava_preset,
            "water": self._water_preset,
        }
    
    def sample(self, x: float, y: float, zoom: float = 1.0, 
               channel: str = "all") -> np.ndarray:
        """
        Сэмплирование текстуры в точке (x, y) с заданным зумом
        
        Args:
            x, y: Координаты (могут быть дробными любого масштаба)
            zoom: Уровень приближения (>1 = увеличение, <1 = уменьшение)
            channel: 'r', 'g', 'b', 'a' или 'all'
        """
        # Адаптивная детализация на основе зума
        detail_factor = max(0.1, zoom ** 0.7)
        
        # Создаем небольшую область вокруг точки для сглаживания
        eps = 0.001 / zoom
        xx = np.array([[x - eps, x + eps]])
        yy = np.array([[y - eps, y + eps]])
        
        # Получаем базовый шум
        base_noise = self.generator.fractal_noise(xx, yy)
        
        # Применяем пресет текстуры
        if self.texture_type in self.texture_presets:
            texture_func = self.texture_presets[self.texture_type]
            color = texture_func(base_noise, x, y, zoom)
        else:
            # По умолчанию: градиент от шума
            color = self._default_texture(base_noise)
        
        # Выбор канала
        if channel != "all":
            channel_idx = {"r": 0, "g": 1, "b": 2, "a": 3}[channel]
            return color[..., channel_idx]
        
        return color
    
    def generate_tile(self, x0: float, y0: float, width: int, height: int,
                     zoom: float = 1.0) -> np.ndarray:
        """Генерация тайла текстуры для заданной области"""
        # Создаем координатную сетку
        x = np.linspace(x0, x0 + width / zoom, width)
        y = np.linspace(y0, y0 + height / zoom, height)
        xx, yy = np.meshgrid(x, y)
        
        # Генерация текстуры
        texture = np.zeros((height, width, 4), dtype=np.float32)
        
        # Используем стратифицированную выборку для больших тайлов
        if width * height > 1000000:  # Больше 1М пикселей
            texture = self._generate_large_tile(xx, yy, zoom)
        else:
            noise = self.generator.fractal_noise(xx, yy)
            if self.texture_type in self.texture_presets:
                texture_func = self.texture_presets[self.texture_type]
                texture = texture_func(noise, xx, yy, zoom)
            else:
                texture = self._default_texture(noise)
        
        return np.clip(texture, 0, 1)
    
    def _generate_large_tile(self, xx, yy, zoom):
        """Оптимизированная генерация больших тайлов"""
        height, width = xx.shape
        texture = np.zeros((height, width, 4), dtype=np.float32)
        
        # Разбиваем на блоки для кэширования
        block_size = 64
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                i_end = min(i + block_size, height)
                j_end = min(j + block_size, width)
                
                block_key = f"{xx[i,j]:.6f}_{yy[i,j]:.6f}_{zoom:.3f}"
                if block_key in self._cache:
                    texture[i:i_end, j:j_end] = self._cache[block_key]
                else:
                    block_xx = xx[i:i_end, j:j_end]
                    block_yy = yy[i:i_end, j:j_end]
                    noise = self.generator.fractal_noise(block_xx, block_yy)
                    
                    if self.texture_type in self.texture_presets:
                        texture_func = self.texture_presets[self.texture_type]
                        block_texture = texture_func(noise, block_xx, block_yy, zoom)
                    else:
                        block_texture = self._default_texture(noise)
                    
                    texture[i:i_end, j:j_end] = block_texture
                    self._cache[block_key] = block_texture
                    
                    # Ограничение размера кэша
                    if len(self._cache) > 100:
                        self._cache.pop(next(iter(self._cache)))
        
        return texture
    
    def _default_texture(self, noise):
        """Текстура по умолчанию (градиент от шума)"""
        noise_normalized = (noise + 1) / 2  # [0, 1]
        height, width = noise.shape
        texture = np.zeros((height, width, 4), dtype=np.float32)
        
        # RGB каналы на основе шума
        texture[..., 0] = noise_normalized  # R
        texture[..., 1] = np.abs(noise)     # G
        texture[..., 2] = 1 - noise_normalized  # B
        texture[..., 3] = 1.0  # Alpha
        
        return texture
    
    def _cloud_preset(self, noise, x, y, zoom):
        """Пресет облачной текстуры"""
        height, width = noise.shape
        texture = np.zeros((height, width, 4), dtype=np.float32)
        
        # Основной цвет облаков
        cloud_base = (noise + 1) * 0.5
        cloud_detail = np.sin(x * 0.1 + y * 0.1) * 0.2 + 0.5
        
        # Смешивание слоев
        clouds = np.clip(cloud_base * 0.7 + cloud_detail * 0.3, 0, 1)
        
        texture[..., 0] = clouds  # R
        texture[..., 1] = clouds  # G
        texture[..., 2] = clouds * 0.9 + 0.1  # B (легкая голубизна)
        texture[..., 3] = clouds * 0.8 + 0.2  # Alpha (полупрозрачность)
        
        return texture
    
    def _marble_preset(self, noise, x, y, zoom):
        """Мраморная текстура"""
        height, width = noise.shape
        texture = np.zeros((height, width, 4), dtype=np.float32)
        
        # Вены мрамора
        veins = np.sin(x * 2 + noise * 5) * 0.5 + 0.5
        
        # Базовый цвет мрамора
        base_color = np.array([0.9, 0.85, 0.8])  # Светлый мрамор
        vein_color = np.array([0.7, 0.6, 0.5])   # Темные вены
        
        # Смешивание
        mix_factor = veins ** 2
        for i in range(3):
            texture[..., i] = base_color[i] * (1 - mix_factor) + vein_color[i] * mix_factor
        
        texture[..., 3] = 1.0  # Непрозрачный
        
        return texture

# ------------------------------------------------------------
# Утилиты для работы с текстурами
# ------------------------------------------------------------

class TextureStreamer:
    """Потоковая загрузка текстур с бесконечной детализацией"""
    
    def __init__(self, base_texture: InfiniteTexture, cache_size_mb: int = 512):
        self.base_texture = base_texture
        self.cache_size_mb = cache_size_mb
        self.tile_cache = {}
        self.requests_queue = []
        
    def request_tile(self, tile_x: int, tile_y: int, lod: int) -> np.ndarray:
        """Запрос тайла текстуры определенного уровня детализации"""
        tile_key = (tile_x, tile_y, lod)
        
        if tile_key in self.tile_cache:
            return self.tile_cache[tile_key]
        
        # Генерация тайла
        tile_size = 256  # Размер тайла в пикселях
        scale = 2 ** lod  # Масштаб на уровне детализации
        
        x0 = tile_x * tile_size / scale
        y0 = tile_y * tile_size / scale
        
        tile = self.base_texture.generate_tile(
            x0, y0, tile_size, tile_size, zoom=scale
        )
        
        # Кэширование
        self.tile_cache[tile_key] = tile
        self._manage_cache()
        
        return tile
    
    def _manage_cache(self):
        """Управление кэшем (LRU)"""
        max_tiles = (self.cache_size_mb * 1024 * 1024) // (256 * 256 * 4 * 4)  # Примерный расчет
        if len(self.tile_cache) > max_tiles:
            # Удаляем самые старые тайлы
            keys_to_remove = list(self.tile_cache.keys())[:len(self.tile_cache) - max_tiles]
            for key in keys_to_remove:
                del self.tile_cache[key]

# ------------------------------------------------------------
# Пример использования
# ------------------------------------------------------------

def create_example_scene():
    """Создание демонстрационной сцены с бесконечно детализируемыми текстурами"""
    
    # 1. Создаем генератор с параметрами
    params = FractalParams(
        seed=42,
        base_scale=0.01,
        detail_level=2.0,
        persistence=0.6,
        lacunarity=1.8,
        octaves=12,
        fractal_dimension=2.5
    )
    
    generator = FractalGenerator(params)
    
    # 2. Создаем разные типы текстур
    textures = {
        "clouds": InfiniteTexture(generator, "clouds"),
        "marble": InfiniteTexture(generator, "marble"),
        "wood": InfiniteTexture(generator, "wood"),
        "lava": InfiniteTexture(generator, "lava"),
    }
    
    # 3. Генерация тайлов на разных уровнях детализации
    streamer = TextureStreamer(textures["clouds"])
    
    # Пример: получение тайлов для квадранта
    tiles = {}
    for lod in range(4):  # 4 уровня детализации
        for tx in range(2):
            for ty in range(2):
                tile = streamer.request_tile(tx, ty, lod)
                tiles[(tx, ty, lod)] = tile
    
    return textures, tiles

# ------------------------------------------------------------
# Оптимизированный шум для производительности
# ------------------------------------------------------------

class OptimizedNoise:
    """Оптимизированные алгоритмы шума для реального времени"""
    
    @staticmethod
    def fast_perlin(x, y, seed=0):
        """Быстрая реализация шума Перлина"""
        # Упрощенная версия для производительности
        X = np.floor(x).astype(int) & 255
        Y = np.floor(y).astype(int) & 255
        
        xf = x - np.floor(x)
        yf = y - np.floor(y)
        
        # Функция сглаживания
        u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
        v = yf * yf * yf * (yf * (yf * 6 - 15) + 10)
        
        # Градиенты в углах
        np.random.seed(seed)
        grad = np.random.randn(256, 256, 2)
        
        # Скалярные произведения
        dot00 = (xf) * grad[X, Y, 0] + (yf) * grad[X, Y, 1]
        dot01 = (xf) * grad[X, Y+1, 0] + (yf-1) * grad[X, Y+1, 1]
        dot10 = (xf-1) * grad[X+1, Y, 0] + (yf) * grad[X+1, Y, 1]
        dot11 = (xf-1) * grad[X+1, Y+1, 0] + (yf-1) * grad[X+1, Y+1, 1]
        
        # Интерполяция
        x1 = dot00 + u * (dot10 - dot00)
        x2 = dot01 + u * (dot11 - dot01)
        
        return x1 + v * (x2 - x1)

if __name__ == "__main__":
    # Демонстрация работы
    print("Fractal Texture Engine Demo")
    print("=" * 50)
    
    # Создаем текстуру облаков
    params = FractalParams(seed=123, detail_level=3.0)
    generator = FractalGenerator(params)
    cloud_texture = InfiniteTexture(generator, "clouds")
    
    # Сэмплируем в одной точке с разным зумом
    point = (10.5, 20.3)
    for zoom in [1.0, 10.0, 100.0, 1000.0]:
        color = cloud_texture.sample(point[0], point[1], zoom=zoom)
        print(f"Zoom {zoom:6.1f}: Color = {color[0,0]}")
    
    # Генерация тайла 512x512
    print("\nGenerating 512x512 cloud tile...")
    tile = cloud_texture.generate_tile(0, 0, 512, 512, zoom=1.0)
    print(f"Tile shape: {tile.shape}, dtype: {tile.dtype}")
    print(f"Min/Max: {tile.min():.3f}/{tile.max():.3f}")

# fractex/core.py - обновленный FractalGenerator

from .simplex_noise import SimplexNoise

class FractalGenerator:
    def __init__(self, params=None):
        self.params = params or FractalParams()
        self.simplex = SimplexNoise(self.params.seed)
        
    def fractal_noise(self, x, y):
        """Используем симплекс-шум вместо упрощенного"""
        effective_octaves = max(1, int(self.params.octaves * self.params.detail_level))
        
        value = np.zeros_like(x)
        amplitude = 1.0
        frequency = self.params.base_scale
        
        for i in range(effective_octaves):
            # Используем симплекс-шум
            noise = self.simplex.noise_2d(x * frequency, y * frequency)
            value += amplitude * noise
            
            amplitude *= self.params.persistence
            frequency *= self.params.lacunarity
            
            if amplitude < 0.001:
                break
        
        # Нормализация
        max_val = (1 - self.params.persistence ** effective_octaves) / (1 - self.params.persistence)
        return value / max_val if max_val > 0 else value
