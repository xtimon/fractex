# fractex/volume_textures.py
"""
Полная система для работы с 3D текстурами (объемными текстурами)
Поддержка генерации, смешивания, кэширования и рендеринга объемных материалов
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from numba import jit, prange, vectorize, float32, float64, int32, int64
import warnings
import time
import threading
from queue import Queue, PriorityQueue
from dataclasses import dataclass
from enum import Enum
import hashlib
import zlib

# ----------------------------------------------------------------------
# Базовые структуры данных для 3D текстур
# ----------------------------------------------------------------------

class VolumeFormat(Enum):
    """Форматы объемных текстур"""
    GRAYSCALE = 1     # 1 канал: плотность/прозрачность
    GRAYSCALE_ALPHA = 2  # 2 канала: плотность + альфа
    RGB = 3           # 3 канала: цвет
    RGBA = 4          # 4 канала: цвет + альфа
    RGBA_FLOAT = 5    # 4 канала с float32 точностью
    DENSITY = 6       # 1 канал плотности (для облаков, дыма)
    MATERIAL_ID = 7   # 1 канал ID материала

@dataclass
class VolumeTexture3D:
    """Структура для хранения 3D текстуры"""
    data: np.ndarray           # Форма: (depth, height, width, channels)
    format: VolumeFormat       # Формат данных
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Размер вокселя
    world_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Начало координат
    compression: Optional[str] = None  # Тип сжатия
    
    def __post_init__(self):
        """Проверка и нормализация данных"""
        if self.data.ndim != 4:
            raise ValueError(f"Volume data must be 4D (depth, height, width, channels), got {self.data.ndim}D")
        
        # Нормализация в [0, 1] если данные не float
        if self.data.dtype != np.float32 and self.data.dtype != np.float64:
            if self.data.dtype in [np.uint8, np.uint16, np.uint32]:
                max_val = np.iinfo(self.data.dtype).max
                self.data = self.data.astype(np.float32) / max_val
            else:
                self.data = self.data.astype(np.float32)
    
    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Возвращает форму данных (D, H, W, C)"""
        return self.data.shape
    
    @property
    def size_bytes(self) -> int:
        """Размер в байтах"""
        return self.data.nbytes
    
    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Возвращает 3D размеры (без каналов)"""
        return self.shape[:3]
    
    @property
    def channels(self) -> int:
        """Количество каналов"""
        return self.shape[3]
    
    def sample(self, x: float, y: float, z: float, 
               wrap_mode: str = "repeat") -> np.ndarray:
        """
        Трилинейная интерполяция в точке (x, y, z)
        Координаты нормализованы к [0, 1]
        """
        return sample_volume_trilinear(self.data, x, y, z, wrap_mode)
    
    def get_slice(self, axis: str, index: int) -> np.ndarray:
        """
        Получение 2D среза из 3D текстуры
        
        Args:
            axis: 'x', 'y', или 'z'
            index: Индекс среза (0 - размер-1)
        """
        if axis == 'x':
            if index < 0 or index >= self.shape[2]:
                raise IndexError(f"X index {index} out of bounds [0, {self.shape[2]-1}]")
            return self.data[:, :, index, :]
        elif axis == 'y':
            if index < 0 or index >= self.shape[1]:
                raise IndexError(f"Y index {index} out of bounds [0, {self.shape[1]-1}]")
            return self.data[:, index, :, :]
        elif axis == 'z':
            if index < 0 or index >= self.shape[0]:
                raise IndexError(f"Z index {index} out of bounds [0, {self.shape[0]-1}]")
            return self.data[index, :, :, :]
        else:
            raise ValueError(f"Axis must be 'x', 'y', or 'z', got {axis}")
    
    def compress(self, method: str = "zlib", level: int = 6) -> bytes:
        """Сжатие данных текстуры"""
        if method == "zlib":
            return zlib.compress(self.data.tobytes(), level=level)
        elif method == "lz4":
            try:
                import lz4.frame
                return lz4.frame.compress(self.data.tobytes())
            except ImportError:
                warnings.warn("lz4 not installed, falling back to zlib")
                return zlib.compress(self.data.tobytes(), level=level)
        else:
            raise ValueError(f"Unknown compression method: {method}")
    
    @classmethod
    def decompress(cls, compressed_data: bytes, shape: Tuple[int, int, int, int],
                   dtype: np.dtype = np.float32, method: str = "zlib") -> 'VolumeTexture3D':
        """Восстановление из сжатых данных"""
        if method == "zlib":
            data_bytes = zlib.decompress(compressed_data)
        elif method == "lz4":
            try:
                import lz4.frame
                data_bytes = lz4.frame.decompress(compressed_data)
            except ImportError:
                raise ImportError("lz4 required for lz4 decompression")
        else:
            raise ValueError(f"Unknown compression method: {method}")
        
        data = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
        return cls(data=data, format=VolumeFormat.RGBA_FLOAT)

# ----------------------------------------------------------------------
# Функции трилинейной интерполяции (оптимизированные с Numba)
# ----------------------------------------------------------------------

@jit(nopython=True, cache=True)
def sample_volume_trilinear(volume: np.ndarray, 
                           x: float, y: float, z: float,
                           wrap_mode: str = "repeat") -> np.ndarray:
    """
    Трилинейная интерполяция в 3D текстуре
    
    Args:
        volume: 4D массив (D, H, W, C)
        x, y, z: Нормализованные координаты [0, 1]
        wrap_mode: Режим заворачивания координат: "repeat", "clamp", "mirror"
    
    Returns:
        Интерполированное значение (C,)
    """
    depth, height, width, channels = volume.shape
    x = np.float32(x)
    y = np.float32(y)
    z = np.float32(z)
    
    # Обрабатываем режим заворачивания
    if wrap_mode == "repeat":
        x = x - np.floor(x)
        y = y - np.floor(y)
        z = z - np.floor(z)
    elif wrap_mode == "clamp":
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        if y < 0.0:
            y = 0.0
        elif y > 1.0:
            y = 1.0
        if z < 0.0:
            z = 0.0
        elif z > 1.0:
            z = 1.0
    elif wrap_mode == "mirror":
        x = np.abs(1.0 - np.abs(1.0 - x * 2.0)) / 2.0
        y = np.abs(1.0 - np.abs(1.0 - y * 2.0)) / 2.0
        z = np.abs(1.0 - np.abs(1.0 - z * 2.0)) / 2.0
    else:
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        if y < 0.0:
            y = 0.0
        elif y > 1.0:
            y = 1.0
        if z < 0.0:
            z = 0.0
        elif z > 1.0:
            z = 1.0
    
    # Переводим в координаты текстуры
    fx = x * (width - 1)
    fy = y * (height - 1)
    fz = z * (depth - 1)
    
    # Целочисленные координаты
    ix0 = int(np.floor(fx))
    iy0 = int(np.floor(fy))
    iz0 = int(np.floor(fz))
    
    # Проверяем границы
    if ix0 < 0 or ix0 >= width or iy0 < 0 or iy0 >= height or iz0 < 0 or iz0 >= depth:
        return np.zeros(channels, dtype=volume.dtype)
    
    # Дробные части
    dx = fx - ix0
    dy = fy - iy0
    dz = fz - iz0
    
    # Следующие координаты с проверкой границ
    ix1 = min(ix0 + 1, width - 1)
    iy1 = min(iy0 + 1, height - 1)
    iz1 = min(iz0 + 1, depth - 1)
    
    # Интерполяция по X
    c000 = volume[iz0, iy0, ix0, :]
    c001 = volume[iz0, iy0, ix1, :]
    c010 = volume[iz0, iy1, ix0, :]
    c011 = volume[iz0, iy1, ix1, :]
    c100 = volume[iz1, iy0, ix0, :]
    c101 = volume[iz1, iy0, ix1, :]
    c110 = volume[iz1, iy1, ix0, :]
    c111 = volume[iz1, iy1, ix1, :]
    
    # Линейная интерполяция по X
    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx
    
    # Линейная интерполяция по Y
    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy
    
    # Линейная интерполяция по Z
    result = c0 * (1 - dz) + c1 * dz
    
    return result.astype(volume.dtype)

@jit(nopython=True, parallel=True, cache=True)
def sample_volume_trilinear_batch(volume: np.ndarray,
                                 coords: np.ndarray,
                                 wrap_mode: str = "repeat") -> np.ndarray:
    """
    Пакетная трилинейная интерполяция для множества точек
    
    Args:
        volume: 4D массив (D, H, W, C)
        coords: Массив координат (N, 3) в [0, 1]
        wrap_mode: Режим заворачивания координат
    
    Returns:
        Массив интерполированных значений (N, C)
    """
    depth, height, width, channels = volume.shape
    n_points = coords.shape[0]
    result = np.zeros((n_points, channels), dtype=volume.dtype)
    
    for i in prange(n_points):
        x, y, z = coords[i, 0], coords[i, 1], coords[i, 2]
        
        # Обрабатываем режим заворачивания
        if wrap_mode == "repeat":
            x = x - np.floor(x)
            y = y - np.floor(y)
            z = z - np.floor(z)
        elif wrap_mode == "clamp":
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            z = max(0.0, min(1.0, z))
        
        # Переводим в координаты текстуры
        fx = x * (width - 1)
        fy = y * (height - 1)
        fz = z * (depth - 1)
        
        # Целочисленные координаты
        ix0 = int(np.floor(fx))
        iy0 = int(np.floor(fy))
        iz0 = int(np.floor(fz))
        
        # Проверяем границы
        if ix0 < 0 or ix0 >= width or iy0 < 0 or iy0 >= height or iz0 < 0 or iz0 >= depth:
            continue
        
        # Дробные части
        dx = fx - ix0
        dy = fy - iy0
        dz = fz - iz0
        
        # Следующие координаты с проверкой границ
        ix1 = ix0 + 1 if ix0 < width - 1 else ix0
        iy1 = iy0 + 1 if iy0 < height - 1 else iy0
        iz1 = iz0 + 1 if iz0 < depth - 1 else iz0
        
        # Берем значения из текстуры
        c000 = volume[iz0, iy0, ix0, :]
        c001 = volume[iz0, iy0, ix1, :]
        c010 = volume[iz0, iy1, ix0, :]
        c011 = volume[iz0, iy1, ix1, :]
        c100 = volume[iz1, iy0, ix0, :]
        c101 = volume[iz1, iy0, ix1, :]
        c110 = volume[iz1, iy1, ix0, :]
        c111 = volume[iz1, iy1, ix1, :]
        
        # Линейная интерполяция по X
        c00 = c000 * (1 - dx) + c001 * dx
        c01 = c010 * (1 - dx) + c011 * dx
        c10 = c100 * (1 - dx) + c101 * dx
        c11 = c110 * (1 - dx) + c111 * dx
        
        # Линейная интерполяция по Y
        c0 = c00 * (1 - dy) + c01 * dy
        c1 = c10 * (1 - dy) + c11 * dy
        
        # Линейная интерполяция по Z
        result[i, :] = c0 * (1 - dz) + c1 * dz
    
    return result

# ----------------------------------------------------------------------
# Генераторы 3D текстур на основе шума
# ----------------------------------------------------------------------

class VolumeTextureGenerator3D:
    """Генератор 3D текстур на основе процедурного шума"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.cache = {}
        try:
            from .simplex_noise import SimplexNoise
            self.noise = SimplexNoise(seed)
        except ImportError:
            warnings.warn("SimplexNoise not available, using fallback")
            self.noise = None
    
    def generate_clouds_3d(self, 
                          width: int = 64,
                          height: int = 64, 
                          depth: int = 64,
                          scale: float = 0.05,
                          density: float = 0.5,
                          detail: int = 3,
                          animated: bool = False,
                          time: float = 0.0) -> VolumeTexture3D:
        """
        Генерация 3D текстуры облаков
        
        Args:
            width, height, depth: Размеры 3D текстуры
            scale: Масштаб шума
            density: Плотность облаков (0-1)
            detail: Детализация (количество октав)
            animated: Анимированная текстура (использует 4D шум)
            time: Время для анимации
            
        Returns:
            VolumeTexture3D с облачной текстурой
        """
        print(f"Generating 3D cloud texture {width}x{height}x{depth}...")
        
        # Создаем координатную сетку
        x = np.linspace(0, width * scale, width)
        y = np.linspace(0, height * scale, height)
        z = np.linspace(0, depth * scale, depth)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        # Генерируем шум
        if self.noise is None:
            # Fallback: простой шум
            noise_data = np.sin(xx * 0.1) * np.cos(yy * 0.1) * np.sin(zz * 0.1)
        else:
            if animated and time > 0:
                # Анимированные облака с 4D шумом
                noise_data = np.zeros((depth, height, width), dtype=np.float32)
                for i in range(depth):
                    # Используем z-координату + время как четвертое измерение
                    w = zz[i, :, :] * 0.1 + time * 0.1
                    noise_slice = self.noise.noise_4d(xx[i, :, :], yy[i, :, :], 
                                                     zz[i, :, :], w)
                    noise_data[i, :, :] = noise_slice
            else:
                # Статичные облаки с 3D шумом
                noise_data = self.noise.noise_3d(xx, yy, zz)
        
        # Фрактальный шум для детализации
        if detail > 1 and self.noise is not None:
            fractal = np.zeros_like(noise_data)
            amplitude = 1.0
            frequency = scale * 2
            
            for i in range(detail):
                nx = xx * frequency
                ny = yy * frequency
                nz = zz * frequency
                
                if animated and time > 0:
                    # Анимированный детальный шум
                    octave_noise = np.zeros_like(noise_data)
                    for j in range(depth):
                        w = nz[j, :, :] * 0.1 + time * 0.1 + i * 10
                        octave_slice = self.noise.noise_4d(nx[j, :, :], ny[j, :, :], 
                                                          nz[j, :, :], w)
                        octave_noise[j, :, :] = octave_slice
                else:
                    octave_noise = self.noise.noise_3d(nx, ny, nz)
                
                fractal += amplitude * octave_noise
                amplitude *= 0.5
                frequency *= 2.0
            
            noise_data = noise_data * 0.7 + fractal * 0.3
        
        # Нормализация и применение плотности
        noise_data = (noise_data - noise_data.min()) / (noise_data.max() - noise_data.min())
        
        # Формируем плотность облаков
        density_map = np.clip(noise_data * 2.0 - (1.0 - density), 0, 1)
        
        # Создаем RGBA текстуру
        texture_data = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        # Белый цвет для облаков
        texture_data[..., 0] = 1.0  # R
        texture_data[..., 1] = 1.0  # G
        texture_data[..., 2] = 1.0  # B
        texture_data[..., 3] = density_map  # Альфа = плотность
        
        return VolumeTexture3D(
            data=texture_data,
            format=VolumeFormat.RGBA_FLOAT,
            voxel_size=(1.0/width, 1.0/height, 1.0/depth)
        )
    
    def generate_marble_3d(self,
                          width: int = 64,
                          height: int = 64,
                          depth: int = 64,
                          scale: float = 0.02,
                          vein_strength: float = 0.8,
                          vein_frequency: float = 5.0) -> VolumeTexture3D:
        """
        Генерация 3D мраморной текстуры
        
        Args:
            width, height, depth: Размеры 3D текстуры
            scale: Масштаб шума
            vein_strength: Сила прожилок (0-1)
            vein_frequency: Частота прожилок
            
        Returns:
            VolumeTexture3D с мраморной текстурой
        """
        print(f"Generating 3D marble texture {width}x{height}x{depth}...")
        
        # Создаем координатную сетку
        x = np.linspace(0, width * scale, width)
        y = np.linspace(0, height * scale, height)
        z = np.linspace(0, depth * scale, depth)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        # Базовый шум для структуры
        if self.noise is None:
            base_noise = np.sin(xx * 0.5) * np.cos(yy * 0.5) * np.sin(zz * 0.5)
        else:
            base_noise = self.noise.noise_3d(xx, yy, zz)
        
        # Создаем синусоидальные прожилки в 3D
        # Используем синус от расстояния до центра по разным осям
        marble_pattern = np.zeros_like(base_noise)
        
        # Несколько направлений прожилок
        directions = [
            (1.0, 0.5, 0.3),
            (0.3, 1.0, 0.5),
            (0.5, 0.3, 1.0)
        ]
        
        for dir_x, dir_y, dir_z in directions:
            # Проекция на направление
            projection = xx * dir_x + yy * dir_y + zz * dir_z
            
            # Синусоидальные прожилки
            veins = np.sin(projection * vein_frequency + base_noise * 3) * 0.5 + 0.5
            marble_pattern += veins
        
        marble_pattern /= len(directions)
        
        # Добавляем детали
        if self.noise is not None:
            detail = self.noise.fractal_noise_3d(
                xx, yy, zz, octaves=3, persistence=0.5,
                lacunarity=2.0, base_scale=scale * 5
            )
            marble_pattern = marble_pattern * 0.8 + detail * 0.2
        
        # Нормализация
        marble_pattern = (marble_pattern - marble_pattern.min()) / \
                        (marble_pattern.max() - marble_pattern.min())
        
        # Создаем RGBA текстуру
        texture_data = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        # Цвета мрамора
        base_color = np.array([0.92, 0.87, 0.82])  # Светлый мрамор
        vein_color = np.array([0.75, 0.65, 0.55])  # Темные прожилки
        
        # Интерполяция между цветами
        for i in range(3):
            texture_data[..., i] = base_color[i] * (1 - marble_pattern) + \
                                  vein_color[i] * marble_pattern
        
        # Непрозрачный
        texture_data[..., 3] = 1.0
        
        return VolumeTexture3D(
            data=texture_data,
            format=VolumeFormat.RGBA_FLOAT,
            voxel_size=(1.0/width, 1.0/height, 1.0/depth)
        )
    
    def generate_wood_3d(self,
                        width: int = 64,
                        height: int = 64,
                        depth: int = 64,
                        scale: float = 0.03,
                        ring_frequency: float = 10.0) -> VolumeTexture3D:
        """
        Генерация 3D деревянной текстуры
        
        Args:
            width, height, depth: Размеры 3D текстуры
            scale: Масштаб текстуры
            ring_frequency: Частота годичных колец
            
        Returns:
            VolumeTexture3D с деревянной текстурой
        """
        print(f"Generating 3D wood texture {width}x{height}x{depth}...")
        
        # Создаем координатную сетку от -1 до 1
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        z = np.linspace(-1, 1, depth)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        # Создаем цилиндрические координаты для древесных колец
        # Используем расстояние от центральной оси (например, ось Z)
        radius = np.sqrt(xx*xx + yy*yy) * ring_frequency
        
        # Добавляем шум для реалистичности
        if self.noise is None:
            noise = np.sin(xx * 5) * np.cos(yy * 5) * np.sin(zz * 3) * 0.2
        else:
            noise = self.noise.noise_3d(xx * 5, yy * 5, zz * 3) * 0.2
        
        # Кольца плюс шум
        wood_pattern = np.sin(radius * 2 * np.pi + noise * 2) * 0.5 + 0.5
        
        # Добавляем волокна вдоль оси Z
        fiber = np.sin(zz * 20 + self.noise.noise_3d(xx * 10, yy * 10, zz) * 3) * 0.1
        wood_pattern += fiber
        
        # Нормализация
        wood_pattern = np.clip(wood_pattern, 0, 1)
        
        # Создаем RGBA текстуру
        texture_data = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        # Цвета дерева
        light_wood = np.array([0.7, 0.5, 0.3])  # Светлая древесина
        dark_wood = np.array([0.4, 0.25, 0.1])  # Темная древесина
        
        # Интерполяция между цветами
        for i in range(3):
            texture_data[..., i] = light_wood[i] * wood_pattern + \
                                  dark_wood[i] * (1 - wood_pattern)
        
        # Добавляем вариации цвета на основе шума
        color_variation = np.zeros_like(texture_data[..., :3])
        if self.noise is not None:
            for i in range(3):
                color_noise = self.noise.noise_3d(xx * 2, yy * 2, zz * 2)
                color_variation[..., i] = color_noise * 0.1
        
        texture_data[..., :3] += color_variation
        texture_data[..., 3] = 1.0  # Непрозрачный
        
        return VolumeTexture3D(
            data=np.clip(texture_data, 0, 1),
            format=VolumeFormat.RGBA_FLOAT,
            voxel_size=(1.0/width, 1.0/height, 1.0/depth)
        )
    
    def generate_perlin_3d(self,
                          width: int = 64,
                          height: int = 64,
                          depth: int = 64,
                          scale: float = 0.05,
                          octaves: int = 4) -> VolumeTexture3D:
        """
        Генерация 3D текстуры на основе фрактального шума Перлина
        
        Args:
            width, height, depth: Размеры 3D текстуры
            scale: Масштаб шума
            octaves: Количество октав
            
        Returns:
            VolumeTexture3D с текстурой шума
        """
        if self.noise is None:
            raise ValueError("Noise generator not available")
        
        print(f"Generating 3D Perlin noise texture {width}x{height}x{depth}...")
        
        # Создаем координатную сетку
        x = np.linspace(0, width * scale, width)
        y = np.linspace(0, height * scale, height)
        z = np.linspace(0, depth * scale, depth)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        # Генерируем фрактальный шум
        noise_data = np.zeros_like(xx, dtype=np.float32)
        amplitude = 1.0
        frequency = scale
        
        for i in range(octaves):
            nx = xx * frequency
            ny = yy * frequency
            nz = zz * frequency
            
            octave_noise = self.noise.noise_3d(nx, ny, nz)
            noise_data += amplitude * octave_noise
            
            amplitude *= 0.5
            frequency *= 2.0
        
        # Нормализация к [0, 1]
        noise_data = (noise_data - noise_data.min()) / (noise_data.max() - noise_data.min())
        noise_data = np.transpose(noise_data, (2, 1, 0))
        
        # Создаем grayscale текстуру
        texture_data = np.zeros((depth, height, width, 1), dtype=np.float32)
        texture_data[..., 0] = noise_data
        
        return VolumeTexture3D(
            data=texture_data,
            format=VolumeFormat.GRAYSCALE,
            voxel_size=(1.0/width, 1.0/height, 1.0/depth)
        )
    
    def generate_lava_3d(self,
                        width: int = 64,
                        height: int = 64,
                        depth: int = 64,
                        scale: float = 0.03,
                        temperature: float = 0.7,
                        animated: bool = False,
                        time: float = 0.0) -> VolumeTexture3D:
        """
        Генерация 3D текстуры лавы
        
        Args:
            width, height, depth: Размеры 3D текстуры
            scale: Масштаб шума
            temperature: Температура (влияет на цвета)
            animated: Анимированная текстура
            time: Время для анимации
            
        Returns:
            VolumeTexture3D с текстурой лавы
        """
        print(f"Generating 3D lava texture {width}x{height}x{depth}...")
        
        # Создаем координатную сетку
        x = np.linspace(0, width * scale, width)
        y = np.linspace(0, height * scale, height)
        z = np.linspace(0, depth * scale, depth)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        # Базовый шум
        if self.noise is None:
            base_noise = np.sin(xx * 0.5) * np.cos(yy * 0.5) * np.sin(zz * 0.5)
        else:
            if animated and time > 0:
                # Анимированная лава с 4D шумом
                base_noise = np.zeros_like(xx, dtype=np.float32)
                for i in range(depth):
                    w = zz[:, :, i] * 0.1 + time * 0.2
                    noise_slice = self.noise.noise_4d(
                        xx[:, :, i],
                        yy[:, :, i],
                        zz[:, :, i],
                        w,
                    )
                    base_noise[:, :, i] = noise_slice
            else:
                base_noise = self.noise.noise_3d(xx, yy, zz)
        
        # Детализированный шум для текстуры
        if self.noise is not None:
            detail = self.noise.fractal_noise_3d(
                xx, yy, zz, octaves=5, persistence=0.6,
                lacunarity=1.8, base_scale=scale * 3
            )
            lava = base_noise * 0.6 + detail * 0.4
        else:
            lava = base_noise
        
        # Нормализация
        lava = (lava - lava.min()) / (lava.max() - lava.min())
        
        # Создаем RGBA текстуру
        texture_data = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        # Цвета лавы на основе температуры
        # Холодная лава: темно-красная, горячая: ярко-желтая
        hot_color = np.array([1.0, 0.8, 0.1])  # Ярко-желтый
        cold_color = np.array([0.6, 0.1, 0.0])  # Темно-красный
        
        # Смешиваем цвета на основе значения шума и температуры
        for i in range(3):
            texture_data[..., i] = cold_color[i] * (1 - lava * temperature) + \
                                  hot_color[i] * lava * temperature
        
        # Яркость для эффекта свечения
        brightness = np.power(lava, 2) * temperature
        
        # Альфа-канал с вариациями
        texture_data[..., 3] = 0.9 + brightness * 0.2
        
        # Добавляем шум к альфа-каналу для эффекта пузырей
        if self.noise is not None:
            bubble_noise = self.noise.noise_3d(xx * 10, yy * 10, zz * 10) * 0.1
            texture_data[..., 3] += bubble_noise
        
        return VolumeTexture3D(
            data=np.clip(texture_data, 0, 1),
            format=VolumeFormat.RGBA_FLOAT,
            voxel_size=(1.0/width, 1.0/height, 1.0/depth)
        )

    def generate_rocks_3d(self,
                          width: int = 64,
                          height: int = 64,
                          depth: int = 64,
                          scale: float = 0.08,
                          hardness: float = 0.6) -> VolumeTexture3D:
        """Генерация 3D текстуры камня"""
        print(f"Generating 3D rock texture {width}x{height}x{depth}...")
        
        x = np.linspace(0, width * scale, width)
        y = np.linspace(0, height * scale, height)
        z = np.linspace(0, depth * scale, depth)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        if self.noise is None:
            noise_data = np.sin(xx * 0.3) * np.cos(yy * 0.3) * np.sin(zz * 0.3)
        else:
            noise_data = np.zeros_like(xx, dtype=np.float32)
            amplitude = 1.0
            frequency = 1.0
            for _ in range(4):
                noise_data += amplitude * self.noise.noise_3d(
                    xx * frequency, yy * frequency, zz * frequency
                )
                amplitude *= 0.5
                frequency *= 2.0
        
        noise_data = (noise_data - noise_data.min()) / (noise_data.max() - noise_data.min())
        noise_data = np.transpose(noise_data, (2, 1, 0))
        
        density_map = np.clip((noise_data - (1.0 - hardness)) * 3.0, 0.0, 1.0)
        
        texture_data = np.zeros((depth, height, width, 4), dtype=np.float32)
        rock_color = np.array([0.45, 0.45, 0.48])
        texture_data[..., :3] = rock_color
        texture_data[..., 3] = density_map
        
        return VolumeTexture3D(
            data=np.clip(texture_data, 0, 1),
            format=VolumeFormat.RGBA_FLOAT,
            voxel_size=(1.0/width, 1.0/height, 1.0/depth)
        )

    def generate_grass_3d(self,
                          width: int = 64,
                          height: int = 64,
                          depth: int = 64,
                          scale: float = 0.1,
                          density: float = 0.5) -> VolumeTexture3D:
        """Генерация 3D текстуры травы"""
        print(f"Generating 3D grass texture {width}x{height}x{depth}...")
        
        x = np.linspace(0, width * scale, width)
        y = np.linspace(0, height * scale, height)
        z = np.linspace(0, depth * scale, depth)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        if self.noise is None:
            noise_data = np.sin(xx * 0.5) * np.cos(yy * 0.5) * np.sin(zz * 0.5)
        else:
            noise_data = self.noise.noise_3d(xx, yy, zz)
        
        noise_data = (noise_data - noise_data.min()) / (noise_data.max() - noise_data.min())
        noise_data = np.transpose(noise_data, (2, 1, 0))
        
        density_map = np.clip(noise_data * 1.5 - (1.0 - density), 0.0, 1.0)
        
        texture_data = np.zeros((depth, height, width, 4), dtype=np.float32)
        grass_color = np.array([0.15, 0.50, 0.20])
        texture_data[..., :3] = grass_color
        texture_data[..., 3] = density_map
        
        return VolumeTexture3D(
            data=np.clip(texture_data, 0, 1),
            format=VolumeFormat.RGBA_FLOAT,
            voxel_size=(1.0/width, 1.0/height, 1.0/depth)
        )
    
    def generate_material_id_3d(self,
                               width: int = 64,
                               height: int = 64,
                               depth: int = 64,
                               num_materials: int = 5) -> VolumeTexture3D:
        """
        Генерация 3D текстуры с ID материалов (для воксельных миров)
        
        Args:
            width, height, depth: Размеры 3D текстуры
            num_materials: Количество различных материалов
            
        Returns:
            VolumeTexture3D с ID материалов
        """
        print(f"Generating 3D material ID texture {width}x{height}x{depth}...")
        
        # Создаем случайное распределение материалов
        np.random.seed(self.seed)
        
        # Простой шум для распределения
        x = np.linspace(0, 10, width)
        y = np.linspace(0, 10, height)
        z = np.linspace(0, 10, depth)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        if self.noise is None:
            # Простой паттерн
            noise = np.sin(xx) * np.cos(yy) * np.sin(zz)
        else:
            # Используем фрактальный шум для естественного распределения
            noise = self.noise.fractal_noise_3d(
                xx, yy, zz, octaves=3, persistence=0.5,
                lacunarity=2.0, base_scale=0.1
            )
        
        # Нормализуем и квантуем на num_materials материалов
        noise_normalized = (noise - noise.min()) / (noise.max() - noise.min())
        material_ids = (noise_normalized * (num_materials - 1)).astype(np.uint8)
        
        # Создаем текстуру с одним каналом
        texture_data = np.zeros((depth, height, width, 1), dtype=np.uint8)
        texture_data[..., 0] = material_ids
        
        return VolumeTexture3D(
            data=texture_data,
            format=VolumeFormat.MATERIAL_ID,
            voxel_size=(1.0/width, 1.0/height, 1.0/depth)
        )

# ----------------------------------------------------------------------
# Система кэширования и потоковой загрузки 3D текстур
# ----------------------------------------------------------------------

class VolumeTextureCache:
    """Кэш для 3D текстур с LRU политикой и сжатием"""
    
    def __init__(self, max_size_mb: int = 1024, use_compression: bool = True):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.use_compression = use_compression
        self.cache = {}  # key -> (texture, timestamp, size_bytes)
        self.lru_queue = []
        self.current_size_bytes = 0
        self.hits = 0
        self.misses = 0
        self.compression_ratio = 0.0
        
    def get(self, key: str) -> Optional[VolumeTexture3D]:
        """Получение текстуры из кэша"""
        if key in self.cache:
            # Обновляем LRU
            self.lru_queue.remove(key)
            self.lru_queue.append(key)
            
            texture, _, _ = self.cache[key]
            self.hits += 1
            return texture
        
        self.misses += 1
        return None
    
    def put(self, key: str, texture: VolumeTexture3D):
        """Добавление текстуры в кэш"""
        size_bytes = texture.size_bytes
        
        # Сжатие если нужно
        if self.use_compression and texture.compression is None:
            try:
                compressed_data = texture.compress(method="zlib")
                compressed_size = len(compressed_data)
                
                # Создаем сжатую версию
                compressed_texture = VolumeTexture3D(
                    data=texture.data,  # Оригинальные данные все еще в памяти
                    format=texture.format,
                    voxel_size=texture.voxel_size,
                    world_origin=texture.world_origin,
                    compression="zlib"
                )
                texture = compressed_texture
                size_bytes = compressed_size
                
                if texture.size_bytes > 0:
                    self.compression_ratio = compressed_size / texture.size_bytes
            except Exception as e:
                warnings.warn(f"Compression failed: {e}")
        
        # Проверяем, поместится ли
        while self.current_size_bytes + size_bytes > self.max_size_bytes and self.lru_queue:
            # Удаляем самый старый элемент
            oldest_key = self.lru_queue.pop(0)
            _, _, old_size = self.cache[oldest_key]
            self.current_size_bytes -= old_size
            del self.cache[oldest_key]
        
        # Добавляем
        self.cache[key] = (texture, time.time(), size_bytes)
        self.lru_queue.append(key)
        self.current_size_bytes += size_bytes
    
    def clear(self):
        """Очистка кэша"""
        self.cache.clear()
        self.lru_queue.clear()
        self.current_size_bytes = 0
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict:
        """Статистика кэша"""
        return {
            "size_bytes": self.current_size_bytes,
            "size_mb": self.current_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "items": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / max(self.hits + self.misses, 1),
            "compression_ratio": self.compression_ratio
        }

# ----------------------------------------------------------------------
# Система смешивания 3D текстур
# ----------------------------------------------------------------------

class VolumeTextureBlender3D:
    """Система смешивания 3D текстур"""
    
    def __init__(self):
        self.blend_modes_3d = {
            "add": self._blend_3d_add,
            "multiply": self._blend_3d_multiply,
            "overlay": self._blend_3d_overlay,
            "screen": self._blend_3d_screen,
            "max": self._blend_3d_max,
            "min": self._blend_3d_min,
            "lerp": self._blend_3d_lerp,
            "height_blend": self._blend_3d_height_based,
        }
    
    def blend(self, 
              volume_a: VolumeTexture3D,
              volume_b: VolumeTexture3D,
              blend_mode: str = "lerp",
              blend_mask: Optional[np.ndarray] = None,
              **kwargs) -> VolumeTexture3D:
        """
        Смешивание двух 3D текстур
        
        Args:
            volume_a, volume_b: 3D текстуры для смешивания
            blend_mode: Режим смешивания
            blend_mask: 3D маска смешивания (D, H, W) или (D, H, W, 1)
            **kwargs: Дополнительные параметры
            
        Returns:
            Новая смешанная 3D текстура
        """
        # Проверка размеров
        if volume_a.shape != volume_b.shape:
            raise ValueError(f"Volume shapes must match: {volume_a.shape} != {volume_b.shape}")
        
        # Проверка маски
        if blend_mask is not None:
            if blend_mask.ndim == 3:
                blend_mask = blend_mask[..., np.newaxis]
            if blend_mask.shape[:3] != volume_a.shape[:3]:
                raise ValueError(f"Mask shape {blend_mask.shape[:3]} doesn't match volume shape {volume_a.shape[:3]}")
        
        # Выбор режима смешивания
        if blend_mode not in self.blend_modes_3d:
            raise ValueError(f"Unknown blend mode: {blend_mode}")
        
        blend_func = self.blend_modes_3d[blend_mode]
        
        # Смешивание
        result_data = blend_func(volume_a.data, volume_b.data, **kwargs)
        
        # Применение маски если есть
        if blend_mask is not None:
            # Линейная интерполяция на основе маски
            result_data = volume_a.data * (1 - blend_mask) + result_data * blend_mask
        
        return VolumeTexture3D(
            data=result_data,
            format=volume_a.format,
            voxel_size=volume_a.voxel_size,
            world_origin=volume_a.world_origin
        )
    
    def blend_multiple(self,
                      volumes: List[VolumeTexture3D],
                      blend_modes: List[str],
                      opacities: List[float]) -> VolumeTexture3D:
        """
        Последовательное смешивание нескольких 3D текстур
        """
        if len(volumes) < 2:
            return volumes[0] if volumes else None
        
        result = volumes[0]
        
        for i in range(1, len(volumes)):
            # Создаем маску на основе прозрачности
            opacity = opacities[i-1] if i-1 < len(opacities) else 1.0
            blend_mode = blend_modes[i-1] if i-1 < len(blend_modes) else "lerp"
            
            if opacity < 1.0:
                mask = np.full(volumes[i].shape[:3], opacity, dtype=np.float32)
                result = self.blend(result, volumes[i], blend_mode, mask)
            else:
                result = self.blend(result, volumes[i], blend_mode)
        
        return result
    
    def _blend_3d_add(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Сложение"""
        return np.clip(a + b, 0, 1)
    
    def _blend_3d_multiply(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Умножение"""
        return a * b
    
    def _blend_3d_overlay(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Overlay для 3D"""
        # Аналогично 2D overlay, но для каждого вокселя
        result = np.zeros_like(a)
        
        # Для каждого канала
        for c in range(a.shape[3]):
            a_channel = a[..., c]
            b_channel = b[..., c]
            
            mask = a_channel < 0.5
            result_channel = np.zeros_like(a_channel)
            
            result_channel[mask] = 2 * a_channel[mask] * b_channel[mask]
            result_channel[~mask] = 1 - 2 * (1 - a_channel[~mask]) * (1 - b_channel[~mask])
            
            result[..., c] = result_channel
        
        return result
    
    def _blend_3d_screen(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Screen для 3D"""
        return 1 - (1 - a) * (1 - b)
    
    def _blend_3d_max(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Максимум"""
        return np.maximum(a, b)
    
    def _blend_3d_min(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Минимум"""
        return np.minimum(a, b)
    
    def _blend_3d_lerp(self, a: np.ndarray, b: np.ndarray, t: float = 0.5, **kwargs) -> np.ndarray:
        """Линейная интерполяция"""
        return a * (1 - t) + b * t
    
    def _blend_3d_height_based(self, a: np.ndarray, b: np.ndarray, 
                              height_map: np.ndarray, **kwargs) -> np.ndarray:
        """Смешивание на основе высоты (по оси Y)"""
        if height_map.ndim == 3:
            height_map = height_map[..., np.newaxis]
        
        # Нормализуем карту высот
        height_norm = (height_map - height_map.min()) / (height_map.max() - height_map.min())
        
        # Параметры перехода
        low_threshold = kwargs.get('low_threshold', 0.3)
        high_threshold = kwargs.get('high_threshold', 0.7)
        transition = kwargs.get('transition', 0.1)
        
        # Создаем маску смешивания
        blend_mask = np.zeros_like(height_norm)
        
        # Для каждого вокселя
        for i in range(a.shape[0]):  # По глубине
            for j in range(a.shape[1]):  # По высоте
                for k in range(a.shape[2]):  # По ширине
                    height = height_norm[i, j, k, 0]
                    
                    if height <= low_threshold - transition/2:
                        blend_mask[i, j, k, 0] = 0
                    elif height >= high_threshold + transition/2:
                        blend_mask[i, j, k, 0] = 1
                    elif height <= low_threshold + transition/2:
                        t = (height - (low_threshold - transition/2)) / transition
                        blend_mask[i, j, k, 0] = t
                    elif height >= high_threshold - transition/2:
                        t = 1 - (height - (high_threshold - transition/2)) / transition
                        blend_mask[i, j, k, 0] = t
                    else:
                        blend_mask[i, j, k, 0] = 1
        
        return a * (1 - blend_mask) + b * blend_mask

# ----------------------------------------------------------------------
# Система потоковой генерации 3D текстур (chunk-based)
# ----------------------------------------------------------------------

class VolumeTextureStreamer:
    """Потоковая генерация больших 3D текстур по чанкам"""
    
    def __init__(self, generator: VolumeTextureGenerator3D, 
                 chunk_size: Tuple[int, int, int] = (32, 32, 32),
                 cache_size_mb: int = 512):
        self.generator = generator
        self.chunk_size = chunk_size
        self.cache = VolumeTextureCache(max_size_mb=cache_size_mb)
        self.worker_threads = []
        self.task_queue = PriorityQueue()
        self.result_queue = Queue()
        self.running = False
        
    def start_workers(self, num_workers: int = 4):
        """Запуск рабочих потоков"""
        self.running = True
        for i in range(num_workers):
            thread = threading.Thread(target=self._worker_loop, daemon=True)
            thread.start()
            self.worker_threads.append(thread)
    
    def stop_workers(self):
        """Остановка рабочих потоков"""
        self.running = False
        for _ in range(len(self.worker_threads)):
            self.task_queue.put((999, None))  # Sentinel
        
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        self.worker_threads.clear()
    
    def request_chunk(self, chunk_coords: Tuple[int, int, int],
                      texture_type: str = "clouds",
                      priority: int = 0,
                      **kwargs) -> Optional[VolumeTexture3D]:
        """
        Запрос чанка 3D текстуры
        
        Args:
            chunk_coords: Координаты чанка (cx, cy, cz)
            texture_type: Тип текстуры
            priority: Приоритет (меньше = выше приоритет)
            **kwargs: Параметры генерации
            
        Returns:
            Чанк текстуры или None если еще не сгенерирован
        """
        chunk_key = self._get_chunk_key(chunk_coords, texture_type, kwargs)
        
        # Проверяем кэш
        cached = self.cache.get(chunk_key)
        if cached is not None:
            return cached
        
        # Добавляем в очередь задач если не в процессе генерации
        task = (priority, (chunk_coords, texture_type, kwargs, chunk_key))
        self.task_queue.put(task)
        
        return None
    
    def get_available_chunks(self) -> List[Tuple[int, int, int]]:
        """Получение списка готовых чанков"""
        # Проверяем очередь результатов
        available_chunks = []
        while not self.result_queue.empty():
            chunk_coords, texture = self.result_queue.get()
            available_chunks.append(chunk_coords)
        
        return available_chunks
    
    def _worker_loop(self):
        """Цикл рабочего потока"""
        while self.running:
            try:
                priority, task = self.task_queue.get(timeout=0.1)
                if task is None:  # Sentinel
                    break
                
                chunk_coords, texture_type, kwargs, chunk_key = task
                
                # Генерация чанка
                chunk = self._generate_chunk(chunk_coords, texture_type, **kwargs)
                
                # Кэширование
                self.cache.put(chunk_key, chunk)
                
                # Добавление в очередь результатов
                self.result_queue.put((chunk_coords, chunk))
                
                self.task_queue.task_done()
                
            except Exception as e:
                warnings.warn(f"Worker thread error: {e}")
    
    def _generate_chunk(self, chunk_coords: Tuple[int, int, int],
                       texture_type: str, **kwargs) -> VolumeTexture3D:
        """Генерация одного чанка"""
        cx, cy, cz = chunk_coords
        chunk_w, chunk_h, chunk_d = self.chunk_size
        
        # Вычисляем мировые координаты чанка
        world_x = cx * chunk_w
        world_y = cy * chunk_h
        world_z = cz * chunk_d
        
        # Вызываем соответствующий генератор
        gen_method = getattr(self.generator, f"generate_{texture_type}_3d", None)
        if gen_method is None:
            # По умолчанию облака
            gen_method = self.generator.generate_clouds_3d
        
        # Генерируем чанк
        chunk = gen_method(
            width=chunk_w,
            height=chunk_h,
            depth=chunk_d,
            **kwargs
        )
        
        # Обновляем мировые координаты
        chunk.world_origin = (world_x, world_y, world_z)
        
        return chunk
    
    def _get_chunk_key(self, chunk_coords: Tuple[int, int, int],
                      texture_type: str, params: Dict) -> str:
        """Создание уникального ключа для чанка"""
        # Хешируем параметры для создания ключа
        params_str = str(sorted(params.items()))
        hash_obj = hashlib.md5(f"{chunk_coords}_{texture_type}_{params_str}".encode())
        return hash_obj.hexdigest()

# ----------------------------------------------------------------------
# Рендерер для визуализации 3D текстур
# ----------------------------------------------------------------------

class VolumeTextureRenderer:
    """Простой рейкастинг-рендерер для 3D текстур"""
    
    def __init__(self, 
                 volume: VolumeTexture3D,
                 light_direction: Tuple[float, float, float] = (0.5, 1.0, 0.5)):
        self.volume = volume
        self.light_direction = np.array(light_direction, dtype=np.float32)
        self.light_direction = self.light_direction / np.linalg.norm(self.light_direction)
        self.transfer_function = self._default_transfer_function()
    
    def _default_transfer_function(self) -> Callable[[np.ndarray], np.ndarray]:
        """Функция передачи по умолчанию (значение плотности -> цвет)"""
        def transfer(density: np.ndarray) -> np.ndarray:
            # Простая функция: плотность -> оттенок серого
            color = np.zeros((*density.shape, 4), dtype=np.float32)
            color[..., 0] = density  # R
            color[..., 1] = density  # G
            color[..., 2] = density  # B
            color[..., 3] = density  # A
            return color
        
        return transfer
    
    def render_slice(self, axis: str = 'z', index: int = 0) -> np.ndarray:
        """Рендеринг 2D среза из 3D текстуры"""
        slice_data = self.volume.get_slice(axis, index)
        
        # Если нужно, применяем функцию передачи
        if self.volume.format in [VolumeFormat.GRAYSCALE, VolumeFormat.DENSITY]:
            # Одноканальные данные -> RGB
            if slice_data.shape[-1] == 1:
                slice_rgb = np.zeros((*slice_data.shape[:2], 4), dtype=np.float32)
                slice_rgb[..., 0] = slice_data[..., 0]  # R
                slice_rgb[..., 1] = slice_data[..., 0]  # G
                slice_rgb[..., 2] = slice_data[..., 0]  # B
                slice_rgb[..., 3] = 1.0  # A
                return slice_rgb
        
        return slice_data
    
    def render_raycast(self, 
                      camera_pos: Tuple[float, float, float] = (0.5, 0.5, 2.0),
                      camera_target: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                      image_size: Tuple[int, int] = (256, 256),
                      max_steps: int = 256,
                      step_size: float = 0.005) -> np.ndarray:
        """
        Простой рейкастинг для визуализации объема
        
        Args:
            camera_pos: Позиция камеры
            camera_target: Цель камеры
            image_size: Размер выходного изображения
            max_steps: Максимальное количество шагов луча
            step_size: Размер шага
            
        Returns:
            2D изображение (H, W, 4) RGBA
        """
        print(f"Raycasting volume {self.volume.shape}...")
        
        width, height = image_size
        image = np.zeros((height, width, 4), dtype=np.float32)
        
        # Вычисляем направление взгляда
        camera_dir = np.array(camera_target) - np.array(camera_pos)
        camera_dir = camera_dir / np.linalg.norm(camera_dir)
        
        # Вычисляем базис камеры
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(camera_dir, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, camera_dir)
        
        # FOV
        fov = 60.0
        aspect = width / height
        half_height = np.tan(np.radians(fov) / 2.0)
        half_width = aspect * half_height
        
        # Для каждого пикселя
        for y in range(height):
            for x in range(width):
                # Вычисляем направление луча
                u = (2.0 * x / width - 1.0) * half_width
                v = (1.0 - 2.0 * y / height) * half_height
                
                ray_dir = camera_dir + u * right + v * up
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # Стартовая позиция
                ray_pos = np.array(camera_pos, dtype=np.float32)
                
                # Интегрирование вдоль луча
                color = np.zeros(4, dtype=np.float32)
                
                for step in range(max_steps):
                    # Проверяем границы объема
                    if (ray_pos[0] < 0 or ray_pos[0] >= 1 or
                        ray_pos[1] < 0 or ray_pos[1] >= 1 or
                        ray_pos[2] < 0 or ray_pos[2] >= 1):
                        break
                    
                    # Выборка из объема
                    sample = self.volume.sample(ray_pos[0], ray_pos[1], ray_pos[2])
                    
                    # Фронтально-заднее смешивание
                    alpha = sample[3] if len(sample) >= 4 else sample[0]
                    color = color + (1.0 - color[3]) * alpha * np.append(sample[:3], alpha)
                    
                    # Если полностью непрозрачный, останавливаемся
                    if color[3] >= 0.99:
                        break
                    
                    # Двигаем луч
                    ray_pos += ray_dir * step_size
                
                image[y, x] = color
        
        return np.clip(image, 0, 1)
    
    def render_mip(self, axis: str = 'z') -> np.ndarray:
        """
        Рендеринг MIP (Maximum Intensity Projection) - используется в медицине
        
        Args:
            axis: Ось проекции ('x', 'y', или 'z')
            
        Returns:
            2D изображение MIP
        """
        volume_data = self.volume.data
        
        if axis == 'x':
            # Проекция вдоль оси X
            mip = np.max(volume_data, axis=2)
        elif axis == 'y':
            # Проекция вдоль оси Y
            mip = np.max(volume_data, axis=1)
        elif axis == 'z':
            # Проекция вдоль оси Z
            mip = np.max(volume_data, axis=0)
        else:
            raise ValueError(f"Invalid axis: {axis}")
        
        # Если одноканальный, конвертируем в RGB
        if mip.ndim == 2 or mip.shape[-1] == 1:
            mip_rgb = np.zeros((*mip.shape[:2], 4), dtype=np.float32)
            if mip.ndim == 3:
                mip_rgb[..., 0] = mip[..., 0]
                mip_rgb[..., 1] = mip[..., 0]
                mip_rgb[..., 2] = mip[..., 0]
            else:
                mip_rgb[..., 0] = mip
                mip_rgb[..., 1] = mip
                mip_rgb[..., 2] = mip
            mip_rgb[..., 3] = 1.0
            return mip_rgb
        
        return mip

# ----------------------------------------------------------------------
# Примеры использования
# ----------------------------------------------------------------------

def example_3d_clouds():
    """Пример создания и визуализации 3D облаков"""
    
    print("Generating 3D cloud texture example...")
    
    # Создаем генератор
    generator = VolumeTextureGenerator3D(seed=42)
    
    # Генерируем 3D облака
    clouds_3d = generator.generate_clouds_3d(
        width=64, height=64, depth=64,
        scale=0.05, density=0.4, detail=3
    )
    
    print(f"3D Cloud texture shape: {clouds_3d.shape}")
    print(f"Size: {clouds_3d.size_bytes / (1024*1024):.2f} MB")
    
    # Рендерим срезы
    renderer = VolumeTextureRenderer(clouds_3d)
    
    # Срез по оси Z
    slice_z = renderer.render_slice('z', 32)
    print(f"Slice shape: {slice_z.shape}")
    
    # MIP проекция
    mip = renderer.render_mip('z')
    print(f"MIP shape: {mip.shape}")
    
    return clouds_3d, slice_z, mip

def example_3d_marble():
    """Пример создания 3D мраморной текстуры"""
    
    print("\nGenerating 3D marble texture example...")
    
    generator = VolumeTextureGenerator3D(seed=123)
    
    marble_3d = generator.generate_marble_3d(
        width=48, height=48, depth=48,
        scale=0.03, vein_strength=0.7, vein_frequency=8.0
    )
    
    print(f"3D Marble texture shape: {marble_3d.shape}")
    
    # Создаем еще один слой для смешивания
    clouds_3d = generator.generate_clouds_3d(
        width=48, height=48, depth=48,
        scale=0.02, density=0.3, detail=2
    )
    
    # Смешиваем текстуры
    blender = VolumeTextureBlender3D()
    
    # Создаем маску смешивания (градиент по высоте)
    height, width, depth = marble_3d.shape[:3]
    y_coords = np.linspace(0, 1, height)
    mask_3d = np.zeros((depth, height, width, 1), dtype=np.float32)
    
    for i in range(depth):
        for j in range(height):
            mask_3d[i, j, :, 0] = y_coords[j]  # Градиент по Y
    
    blended = blender.blend(
        marble_3d, clouds_3d,
        blend_mode="lerp",
        blend_mask=mask_3d
    )
    
    print(f"Blended texture shape: {blended.shape}")
    
    return marble_3d, blended

def example_streaming_3d_textures():
    """Пример потоковой генерации больших 3D текстур"""
    
    print("\nStreaming 3D texture generation example...")
    
    generator = VolumeTextureGenerator3D(seed=42)
    streamer = VolumeTextureStreamer(
        generator=generator,
        chunk_size=(16, 16, 16),
        cache_size_mb=256
    )
    
    # Запускаем рабочие потоки
    streamer.start_workers(num_workers=2)
    
    # Запрашиваем несколько чанков
    chunks = []
    for cz in range(2):
        for cy in range(2):
            for cx in range(2):
                chunk = streamer.request_chunk(
                    chunk_coords=(cx, cy, cz),
                    texture_type="clouds",
                    priority=cx + cy + cz,
                    scale=0.1, density=0.5
                )
                if chunk is not None:
                    chunks.append(((cx, cy, cz), chunk))
    
    # Ждем генерации
    time.sleep(2.0)
    
    # Проверяем готовые чанки
    available = streamer.get_available_chunks()
    print(f"Available chunks: {len(available)}")
    
    # Останавливаем workers
    streamer.stop_workers()
    
    # Статистика кэша
    stats = streamer.cache.stats()
    print(f"Cache stats: {stats}")
    
    return streamer

if __name__ == "__main__":
    print("3D Texture System")
    print("=" * 60)
    
    # Пример 1: 3D облака
    clouds_3d, cloud_slice, cloud_mip = example_3d_clouds()
    
    # Пример 2: 3D мрамор и смешивание
    marble_3d, blended_3d = example_3d_marble()
    
    # Пример 3: Потоковая генерация
    streamer = example_streaming_3d_textures()
    
    print("\n" + "=" * 60)
    print("3D Texture System Features:")
    print("-" * 40)
    print("1. Multiple 3D texture types: clouds, marble, wood, lava, perlin noise")
    print("2. Volume texture cache with compression")
    print("3. 3D texture blending with multiple blend modes")
    print("4. Streaming generation for large volumes")
    print("5. Raycasting and MIP rendering")
    print("6. Trilinear interpolation for smooth sampling")
    
    print("\nPerformance tips:")
    print("- Use smaller volumes for real-time applications (64^3 or less)")
    print("- Enable compression for texture cache")
    print("- Use streaming for large worlds")
    print("- Consider GPU acceleration for raycasting")
    
    print("\n3D texture system ready for volumetric rendering!")
