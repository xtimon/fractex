# fractex/geometric_patterns_3d.py
"""
Система генерации геометрических 3D паттернов
Кристаллы, соты, решетки, фрактальные структуры, математические поверхности
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from numba import jit, prange, vectorize, float32, float64, int32, int64, complex128
import warnings
import math
from dataclasses import dataclass, field
from enum import Enum
import itertools
from functools import lru_cache
try:
    from scipy import spatial, ndimage
except ImportError:  # optional dependency
    spatial = None
    ndimage = None

# ----------------------------------------------------------------------
# Структуры данных и перечисления
# ----------------------------------------------------------------------

class GeometricPattern3D(Enum):
    """Типы геометрических 3D паттернов"""
    CRYSTAL_LATTICE = 1       # Кристаллические решетки
    HONEYCOMB = 2             # Сотовые структуры
    GYROID = 3                # Гироидные поверхности
    DIAMOND_STRUCTURE = 4     # Алмазная структура
    SPHERE_PACKING = 5        # Упаковка сфер
    VORONOI_CELLS = 6         # Ячейки Вороного
    WEIRELEN_PIRUS = 7        # Поверхности Вейерштрасса
    SCHWARZ_P = 8             # Поверхности Шварца
    NEOVIUS = 9               # Поверхности Неовиуса
    LAVA_LAMPS = 10           # Лавовые лампы (метаболы)
    FIBONACCI_SPIRAL = 11     # Спираль Фибоначчи в 3D
    QUASI_CRYSTAL = 12        # Квазикристаллы
    TESSELLATION = 13         # Тесселяции пространства
    FRACTAL_SPONGE = 14       # Губка Менгера
    KNOTS_LINKS = 15          # Узлы и зацепления
    HELICOID = 16             # Геликоид
    CATENOID = 17             # Катеноид
    ENNEPER = 18              # Поверхность Эннепера
    CORKSCREW = 19            # Штопор Архимеда
    MOBIUS = 20               # Лента Мёбиуса

@dataclass
class PatternParameters:
    """Параметры для генерации геометрических паттернов"""
    # Общие параметры
    scale: float = 1.0
    resolution: float = 0.01
    thickness: float = 0.1
    symmetry: int = 4
    noise_amplitude: float = 0.0
    noise_scale: float = 0.1
    
    # Специфичные параметры
    crystal_type: str = "cubic"  # cubic, hexagonal, tetragonal, etc.
    cell_size: float = 0.5
    wall_thickness: float = 0.05
    sphere_radius: float = 0.3
    packing_density: float = 0.6
    fibonacci_ratio: float = 1.61803398875  # Золотое сечение
    quasi_symmetry: int = 5  # Симметрия для квазикристаллов
    
    # Параметры фракталов
    fractal_iterations: int = 3
    fractal_scale: float = 0.5
    
    # Параметры поверхностей
    surface_threshold: float = 0.0
    surface_isolevel: float = 0.5
    
    def __post_init__(self):
        # Нормализация параметров
        self.scale = max(0.001, self.scale)
        self.resolution = max(0.001, self.resolution)
        self.thickness = max(0.001, self.thickness)

# ----------------------------------------------------------------------
# Математические функции для 3D паттернов
# ----------------------------------------------------------------------

class MathematicalSurfaces3D:
    """Математические поверхности в 3D"""
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def gyroid(x: float, y: float, z: float, scale: float = 1.0) -> float:
        """
        Гироидная поверхность - минимальная поверхность с тройной периодичностью
        
        f(x,y,z) = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)
        """
        sx = np.sin(x * scale)
        cx = np.cos(x * scale)
        sy = np.sin(y * scale)
        cy = np.cos(y * scale)
        sz = np.sin(z * scale)
        cz = np.cos(z * scale)
        
        return sx * cy + sy * cz + sz * cx
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def schwarz_p(x: float, y: float, z: float, scale: float = 1.0) -> float:
        """Поверхность Шварца P (примитивная)"""
        sx = np.sin(x * scale)
        cx = np.cos(x * scale)
        sy = np.sin(y * scale)
        cy = np.cos(y * scale)
        sz = np.sin(z * scale)
        cz = np.cos(z * scale)
        
        return cx + cy + cz
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def schwarz_d(x: float, y: float, z: float, scale: float = 1.0) -> float:
        """Поверхность Шварца D (алмазная)"""
        sx = np.sin(x * scale)
        cx = np.cos(x * scale)
        sy = np.sin(y * scale)
        cy = np.cos(y * scale)
        sz = np.sin(z * scale)
        cz = np.cos(z * scale)
        
        return (sx * sy * sz + 
                sx * cy * cz + 
                cx * sy * cz + 
                cx * cy * sz)
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def neovius(x: float, y: float, z: float, scale: float = 1.0) -> float:
        """Поверхность Неовиуса"""
        sx = np.sin(x * scale)
        cx = np.cos(x * scale)
        sy = np.sin(y * scale)
        cy = np.cos(y * scale)
        sz = np.sin(z * scale)
        cz = np.cos(z * scale)
        
        return (3 * (cx + cy + cz) + 
                4 * cx * cy * cz)
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def weierstrass(x: float, y: float, z: float, scale: float = 1.0) -> float:
        """Поверхность Вейерштрасса"""
        return (np.cos(x * scale) * np.cos(y * scale) * np.cos(z * scale) - 
                np.sin(x * scale) * np.sin(y * scale) * np.sin(z * scale))
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def helicoid(x: float, y: float, z: float, scale: float = 1.0) -> float:
        """Геликоид"""
        return x * np.sin(z * scale) - y * np.cos(z * scale)
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def catenoid(x: float, y: float, z: float, scale: float = 1.0) -> float:
        """Катеноид"""
        r = np.sqrt(x*x + y*y)
        return np.cosh(r * scale) - z
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def enneper(x: float, y: float, z: float, scale: float = 1.0) -> float:
        """Поверхность Эннепера"""
        u = x * scale
        v = y * scale
        
        # Параметрическое представление
        X = u - u**3/3 + u*v*v
        Y = v - v**3/3 + v*u*u
        Z = u*u - v*v
        
        # Расстояние до поверхности
        return np.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2) - 0.1
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def mobius(x: float, y: float, z: float, scale: float = 1.0) -> float:
        """Лента Мёбиуса"""
        u = np.arctan2(y, x)
        r = np.sqrt(x*x + y*y)
        
        # Параметрическое представление
        R = 1.0
        width = 0.3
        
        X = (R + width * np.cos(u/2)) * np.cos(u)
        Y = (R + width * np.cos(u/2)) * np.sin(u)
        Z = width * np.sin(u/2)
        
        return np.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2) - 0.05
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def corkscrew(x: float, y: float, z: float, scale: float = 1.0) -> float:
        """Штопор Архимеда"""
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        
        # Уравнение штопора: z = a * theta
        a = 0.5
        return z - a * theta * scale
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def klein_bottle(x: float, y: float, z: float, scale: float = 1.0) -> float:
        """Бутылка Клейна (упрощенная)"""
        u = np.arctan2(y, x) * 2
        v = z * 2
        
        # Параметрическое представление
        r = 4 * (1 - np.cos(u)/2)
        
        X = 6 * np.cos(u) * (1 + np.sin(u)) + r * np.cos(u) * np.cos(v)
        Y = 16 * np.sin(u) + r * np.sin(u) * np.cos(v)
        Z = r * np.sin(v)
        
        return np.sqrt((x - X/10)**2 + (y - Y/10)**2 + (z - Z/10)**2) - 0.1

# ----------------------------------------------------------------------
# Генераторы базовых геометрических паттернов
# ----------------------------------------------------------------------

class GeometricPatternGenerator3D:
    """Генератор геометрических 3D паттернов"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        self.math_surfaces = MathematicalSurfaces3D()
        self.cache = {}
    
    def generate_pattern(self,
                        pattern_type: GeometricPattern3D,
                        dimensions: Tuple[int, int, int],
                        params: Optional[PatternParameters] = None) -> np.ndarray:
        """
        Генерация 3D паттерна
        
        Args:
            pattern_type: Тип паттерна
            dimensions: Размеры объема (D, H, W)
            params: Параметры генерации
            
        Returns:
            3D текстура (D, H, W, 4) RGBA
        """
        if params is None:
            params = PatternParameters()
        
        # Проверка кэша
        cache_key = (pattern_type.value, dimensions, hash(frozenset(params.__dict__.items())))
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        print(f"Generating {pattern_type.name} pattern {dimensions}...")
        
        # Выбор метода генерации
        generator_map = {
            GeometricPattern3D.CRYSTAL_LATTICE: self._generate_crystal_lattice,
            GeometricPattern3D.HONEYCOMB: self._generate_honeycomb,
            GeometricPattern3D.GYROID: self._generate_gyroid,
            GeometricPattern3D.DIAMOND_STRUCTURE: self._generate_diamond_structure,
            GeometricPattern3D.SPHERE_PACKING: self._generate_sphere_packing,
            GeometricPattern3D.VORONOI_CELLS: self._generate_voronoi_cells,
            GeometricPattern3D.WEIRELEN_PIRUS: self._generate_weierstrass,
            GeometricPattern3D.SCHWARZ_P: self._generate_schwarz_p,
            GeometricPattern3D.NEOVIUS: self._generate_neovius,
            GeometricPattern3D.LAVA_LAMPS: self._generate_lava_lamps,
            GeometricPattern3D.FIBONACCI_SPIRAL: self._generate_fibonacci_spiral,
            GeometricPattern3D.QUASI_CRYSTAL: self._generate_quasi_crystal,
            GeometricPattern3D.TESSELLATION: self._generate_tessellation,
            GeometricPattern3D.FRACTAL_SPONGE: self._generate_fractal_sponge,
            GeometricPattern3D.KNOTS_LINKS: self._generate_knots,
            GeometricPattern3D.HELICOID: self._generate_helicoid,
            GeometricPattern3D.CATENOID: self._generate_catenoid,
            GeometricPattern3D.ENNEPER: self._generate_enneper,
            GeometricPattern3D.CORKSCREW: self._generate_corkscrew,
            GeometricPattern3D.MOBIUS: self._generate_mobius,
        }
        
        if pattern_type not in generator_map:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        # Генерация
        texture = generator_map[pattern_type](dimensions, params)
        
        # Кэширование
        self.cache[cache_key] = texture.copy()
        if len(self.cache) > 50:
            self.cache.pop(next(iter(self.cache)))
        
        return texture
    
    def _generate_crystal_lattice(self, 
                                 dimensions: Tuple[int, int, int], 
                                 params: PatternParameters) -> np.ndarray:
        """Генерация кристаллической решетки"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        # Выбор типа решетки
        if params.crystal_type == "cubic":
            # Кубическая решетка
            for i in range(depth):
                for j in range(height):
                    for k in range(width):
                        x = i / depth * params.scale * 2 * np.pi
                        y = j / height * params.scale * 2 * np.pi
                        z = k / width * params.scale * 2 * np.pi
                        
                        # Периодические синусоиды для каждой оси
                        value = (np.sin(x) + np.sin(y) + np.sin(z)) / 3
                        
                        # Порог для создания поверхности
                        if abs(value) < params.thickness:
                            # Цвет в зависимости от ориентации
                            r = (np.sin(x) + 1) / 2
                            g = (np.sin(y) + 1) / 2
                            b = (np.sin(z) + 1) / 2
                            
                            # Прозрачность зависит от расстояния до поверхности
                            alpha = 1.0 - abs(value) / params.thickness
                            
                            texture[i, j, k] = [r, g, b, alpha]
        
        elif params.crystal_type == "hexagonal":
            # Гексагональная решетка
            for i in range(depth):
                for j in range(height):
                    for k in range(width):
                        x = i / depth * params.scale * 4 * np.pi
                        y = j / height * params.scale * 4 * np.pi
                        z = k / width * params.scale * 2 * np.pi
                        
                        # Гексагональная симметрия
                        value = (np.sin(x) + 
                                np.sin(x/2 + y*np.sqrt(3)/2) + 
                                np.sin(-x/2 + y*np.sqrt(3)/2) + 
                                np.sin(z)) / 4
                        
                        if abs(value) < params.thickness:
                            r = (np.sin(x) + 1) / 2
                            g = (np.sin(y) + 1) / 2
                            b = (np.sin(z) + 1) / 2
                            alpha = 1.0 - abs(value) / params.thickness
                            
                            texture[i, j, k] = [r, g, b, alpha]
        
        elif params.crystal_type == "diamond":
            # Алмазная структура
            for i in range(depth):
                for j in range(height):
                    for k in range(width):
                        x = i / depth * params.scale * 2 * np.pi
                        y = j / height * params.scale * 2 * np.pi
                        z = k / width * params.scale * 2 * np.pi
                        
                        # Более сложная структура
                        value = (np.sin(x) * np.sin(y) * np.sin(z) + 
                                np.sin(x) * np.cos(y) * np.cos(z) + 
                                np.cos(x) * np.sin(y) * np.cos(z) + 
                                np.cos(x) * np.cos(y) * np.sin(z)) / 4
                        
                        if abs(value) < params.thickness:
                            # Переливчатый цвет как у алмаза
                            hue = (x + y + z) % (2 * np.pi)
                            r = (np.sin(hue) + 1) / 2
                            g = (np.sin(hue + 2*np.pi/3) + 1) / 2
                            b = (np.sin(hue + 4*np.pi/3) + 1) / 2
                            alpha = 1.0 - abs(value) / params.thickness
                            
                            texture[i, j, k] = [r, g, b, alpha]
        
        # Добавление шума если нужно
        if params.noise_amplitude > 0:
            texture = self._add_noise(texture, params.noise_amplitude, params.noise_scale)
        
        return texture
    
    def _generate_honeycomb(self,
                           dimensions: Tuple[int, int, int],
                           params: PatternParameters) -> np.ndarray:
        """Генерация сотовой структуры (гексагональные ячейки)"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        # Параметры сот
        cell_size = params.cell_size
        wall_thickness = params.wall_thickness
        
        # Гексагональная сетка
        hex_radius = cell_size
        hex_height = np.sqrt(3) * hex_radius
        
        for i in range(depth):
            z = i / depth * params.scale
            
            for j in range(height):
                y = j / height * params.scale
                
                for k in range(width):
                    x = k / width * params.scale
                    
                    # Преобразование в гексагональные координаты
                    # Сдвиг четных строк
                    q = x / (hex_radius * 1.5)
                    r = (-x / (hex_radius * 1.5) + y / (hex_height)) / 2
                    
                    # Округление до ближайшего центра гексагона
                    q_round = round(q)
                    r_round = round(r)
                    
                    # Обратное преобразование
                    x_center = q_round * hex_radius * 1.5
                    y_center = (2 * r_round + q_round) * hex_height / 2
                    
                    # Расстояние до центра гексагона
                    dx = x - x_center
                    dy = y - y_center
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    # Толщина стенок в зависимости от Z
                    z_factor = 0.8 + 0.2 * np.sin(z * 2 * np.pi)
                    effective_thickness = wall_thickness * z_factor
                    
                    # Если внутри гексагона, но не слишком близко к центру
                    if dist < hex_radius and dist > hex_radius - effective_thickness:
                        # Цвет сот (золотистый)
                        r_color = 0.9  # Желтый компонент
                        g_color = 0.7  # Золотистый
                        b_color = 0.1  # Темно-желтый
                        
                        # Интенсивность зависит от расстояния до стенки
                        intensity = 1.0 - abs(dist - (hex_radius - effective_thickness/2)) / (effective_thickness/2)
                        alpha = intensity * 0.8
                        
                        texture[i, j, k] = [r_color, g_color, b_color, alpha]
                    
                    # Также создаем "дно" сот (нижняя часть)
                    z_thickness = 0.05
                    if z < z_thickness and dist < hex_radius - effective_thickness:
                        # Более темное дно
                        texture[i, j, k] = [0.6, 0.5, 0.1, 0.9]
        
        return texture
    
    def _generate_gyroid(self,
                        dimensions: Tuple[int, int, int],
                        params: PatternParameters) -> np.ndarray:
        """Генерация гироидной поверхности"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    x = i / depth * params.scale * 2 * np.pi
                    y = j / height * params.scale * 2 * np.pi
                    z = k / width * params.scale * 2 * np.pi
                    
                    # Значение гироидной функции
                    value = self.math_surfaces.gyroid(x, y, z)
                    
                    # Пороговое значение для поверхности
                    threshold = params.surface_threshold
                    
                    if abs(value - threshold) < params.thickness:
                        # Цвет в зависимости от нормали
                        # Приближенная нормаль через градиент
                        eps = 0.01
                        dx = (self.math_surfaces.gyroid(x+eps, y, z) - 
                             self.math_surfaces.gyroid(x-eps, y, z)) / (2*eps)
                        dy = (self.math_surfaces.gyroid(x, y+eps, z) - 
                             self.math_surfaces.gyroid(x, y-eps, z)) / (2*eps)
                        dz = (self.math_surfaces.gyroid(x, y, z+eps) - 
                             self.math_surfaces.gyroid(x, y, z-eps)) / (2*eps)
                        
                        # Нормализация градиента
                        norm = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-8
                        nx = dx / norm
                        ny = dy / norm
                        nz = dz / norm
                        
                        # Цвет на основе нормали
                        r = (nx + 1) / 2
                        g = (ny + 1) / 2
                        b = (nz + 1) / 2
                        
                        # Прозрачность
                        alpha = 1.0 - abs(value - threshold) / params.thickness
                        
                        texture[i, j, k] = [r, g, b, alpha]
        
        return texture
    
    def _generate_diamond_structure(self,
                                   dimensions: Tuple[int, int, int],
                                   params: PatternParameters) -> np.ndarray:
        """Генерация алмазной структуры (как в кристаллах алмаза)"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        # Алмазная кубическая структура
        # Атомы углерода в позициях (0,0,0) и (1/4,1/4,1/4) и их трансляции
        
        lattice_constant = params.cell_size
        
        # Позиции атомов в элементарной ячейке
        positions = [
            (0.0, 0.0, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
            (0.25, 0.25, 0.25),
            (0.25, 0.75, 0.75),
            (0.75, 0.25, 0.75),
            (0.75, 0.75, 0.25)
        ]
        
        # Радиус атома (условный)
        atom_radius = 0.1 * lattice_constant
        
        for i in range(depth):
            z = i / depth * params.scale
            
            for j in range(height):
                y = j / height * params.scale
                
                for k in range(width):
                    x = k / width * params.scale
                    
                    # Проверяем расстояние до каждого атома в ближайших ячейках
                    min_dist = float('inf')
                    
                    # Проверяем ближайшие элементарные ячейки
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dz in [-1, 0, 1]:
                                # Смещение ячейки
                                cell_x = dx * lattice_constant
                                cell_y = dy * lattice_constant
                                cell_z = dz * lattice_constant
                                
                                # Для каждой позиции атома в ячейке
                                for pos in positions:
                                    atom_x = cell_x + pos[0] * lattice_constant
                                    atom_y = cell_y + pos[1] * lattice_constant
                                    atom_z = cell_z + pos[2] * lattice_constant
                                    
                                    # Расстояние до атома
                                    dist = np.sqrt((x - atom_x)**2 + 
                                                  (y - atom_y)**2 + 
                                                  (z - atom_z)**2)
                                    
                                    min_dist = min(min_dist, dist)
                    
                    # Если близко к атому
                    if min_dist < atom_radius:
                        # Переливчатый цвет как у алмаза
                        # Зависит от ориентации
                        hue = (x + y + z) * 10 % 1.0
                        
                        # Преобразование HSV в RGB
                        from colorsys import hsv_to_rgb
                        r, g, b = hsv_to_rgb(hue, 0.3, 1.0)
                        
                        # Интенсивность зависит от расстояния
                        intensity = 1.0 - min_dist / atom_radius
                        alpha = intensity * 0.9
                        
                        # Добавляем внутреннее свечение
                        glow = np.exp(-min_dist * 20) * 0.3
                        r += glow
                        g += glow
                        b += glow
                        
                        texture[i, j, k] = [r, g, b, alpha]
                    
                    # Также показываем связи между атомами
                    # (пространство между атомами в тетраэдре)
                    elif min_dist < atom_radius * 2:
                        # Слабое свечение связей
                        intensity = np.exp(-min_dist * 10) * 0.2
                        texture[i, j, k, 3] = intensity
        
        return texture
    
    def _generate_sphere_packing(self,
                                dimensions: Tuple[int, int, int],
                                params: PatternParameters) -> np.ndarray:
        """Генерация упаковки сфер (плотнейшая упаковка)"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        # Параметры
        sphere_radius = params.sphere_radius
        packing_type = "fcc"  # ГЦК (гранецентрированная кубическая)
        
        if packing_type == "fcc":
            # ГЦК упаковка
            # Базовые векторы
            a1 = np.array([1.0, 0.0, 0.0])
            a2 = np.array([0.5, np.sqrt(3)/2, 0.0])
            a3 = np.array([0.5, np.sqrt(3)/6, np.sqrt(2/3)])
            
            # Позиции в элементарной ячейке
            positions = [
                np.array([0.0, 0.0, 0.0]),
                np.array([0.5, 0.5, 0.0]),
                np.array([0.5, 0.0, 0.5]),
                np.array([0.0, 0.5, 0.5])
            ]
        
        # Масштабирование
        scale_factor = params.scale * 2
        
        for i in range(depth):
            z = i / depth * scale_factor
            
            for j in range(height):
                y = j / height * scale_factor
                
                for k in range(width):
                    x = k / width * scale_factor
                    
                    pos = np.array([x, y, z])
                    min_dist = float('inf')
                    
                    # Проверяем ближайшие ячейки
                    for nx in range(-1, 2):
                        for ny in range(-1, 2):
                            for nz in range(-1, 2):
                                # Смещение ячейки
                                cell_offset = nx * a1 + ny * a2 + nz * a3
                                
                                # Для каждой позиции в ячейке
                                for base_pos in positions:
                                    sphere_center = cell_offset + base_pos
                                    
                                    dist = np.linalg.norm(pos - sphere_center)
                                    min_dist = min(min_dist, dist)
                    
                    # Если внутри сферы
                    if min_dist < sphere_radius:
                        # Цвет сферы
                        # Зависит от положения для визуализации структуры
                        hue = (x + z) % 1.0
                        from colorsys import hsv_to_rgb
                        r, g, b = hsv_to_rgb(hue, 0.7, 1.0)
                        
                        # Интенсивность
                        intensity = 1.0 - min_dist / sphere_radius
                        alpha = intensity * 0.9
                        
                        # Градиент от центра к краю
                        gradient = min_dist / sphere_radius
                        r *= (1 - gradient * 0.3)
                        g *= (1 - gradient * 0.3)
                        b *= (1 - gradient * 0.3)
                        
                        texture[i, j, k] = [r, g, b, alpha]
                    
                    # Контакты между сферами
                    elif min_dist < sphere_radius * 1.1:
                        # Слабое свечение в местах контакта
                        intensity = np.exp(-(min_dist - sphere_radius) * 20) * 0.3
                        texture[i, j, k, 3] = intensity
        
        return texture
    
    def _generate_voronoi_cells(self,
                               dimensions: Tuple[int, int, int],
                               params: PatternParameters) -> np.ndarray:
        """Генерация ячеек Вороного в 3D"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        # Количество точек (центров ячеек)
        num_points = int(params.packing_density * depth * height * width / 100)
        
        # Генерация случайных точек
        np.random.seed(self.seed)
        points = np.random.rand(num_points, 3)
        points[:, 0] *= depth
        points[:, 1] *= height
        points[:, 2] *= width
        
        # Цвета для каждой точки
        colors = np.random.rand(num_points, 3)
        
        # Для каждого воксела находим ближайшую точку
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    # Позиция воксела
                    pos = np.array([i, j, k])
                    
                    # Находим ближайшую точку
                    distances = np.linalg.norm(points - pos, axis=1)
                    closest_idx = np.argmin(distances)
                    min_dist = distances[closest_idx]
                    
                    # Также находим вторую ближайшую
                    distances[closest_idx] = float('inf')
                    second_closest_idx = np.argmin(distances)
                    second_dist = distances[second_closest_idx]
                    
                    # Граница ячейки - середина между двумя ближайшими точками
                    boundary_dist = (min_dist + second_dist) / 2
                    
                    # Если близко к границе
                    if abs(min_dist - boundary_dist) < params.wall_thickness * 5:
                        # Стенка ячейки
                        wall_intensity = 1.0 - abs(min_dist - boundary_dist) / (params.wall_thickness * 5)
                        
                        # Цвет стенки (темный)
                        texture[i, j, k] = [0.2, 0.2, 0.2, wall_intensity * 0.8]
                    
                    else:
                        # Внутри ячейки
                        color = colors[closest_idx]
                        
                        # Интенсивность уменьшается к границе
                        cell_radius = boundary_dist
                        if cell_radius > 0:
                            intensity = 1.0 - min_dist / cell_radius
                        else:
                            intensity = 1.0
                        
                        # Добавляем градиент
                        gradient = np.sin(i * 0.1) * np.cos(j * 0.1) * np.sin(k * 0.1) * 0.2 + 0.8
                        
                        texture[i, j, k] = [
                            color[0] * gradient,
                            color[1] * gradient,
                            color[2] * gradient,
                            intensity * 0.6
                        ]
        
        return texture
    
    def _generate_weierstrass(self,
                             dimensions: Tuple[int, int, int],
                             params: PatternParameters) -> np.ndarray:
        """Генерация поверхности Вейерштрасса"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    x = i / depth * params.scale * 2 * np.pi
                    y = j / height * params.scale * 2 * np.pi
                    z = k / width * params.scale * 2 * np.pi
                    
                    value = self.math_surfaces.weierstrass(x, y, z)
                    
                    if abs(value) < params.thickness:
                        # Цвет на основе производных
                        eps = 0.01
                        dx = (self.math_surfaces.weierstrass(x+eps, y, z) - 
                             self.math_surfaces.weierstrass(x-eps, y, z)) / (2*eps)
                        dy = (self.math_surfaces.weierstrass(x, y+eps, z) - 
                             self.math_surfaces.weierstrass(x, y-eps, z)) / (2*eps)
                        dz = (self.math_surfaces.weierstrass(x, y, z+eps) - 
                             self.math_surfaces.weierstrass(x, y, z-eps)) / (2*eps)
                        
                        # Цвет как функция производных
                        r = (np.sin(dx * 10) + 1) / 2
                        g = (np.sin(dy * 10) + 1) / 2
                        b = (np.sin(dz * 10) + 1) / 2
                        
                        alpha = 1.0 - abs(value) / params.thickness
                        
                        texture[i, j, k] = [r, g, b, alpha]
        
        return texture
    
    def _generate_schwarz_p(self,
                           dimensions: Tuple[int, int, int],
                           params: PatternParameters) -> np.ndarray:
        """Генерация поверхности Шварца P"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    x = i / depth * params.scale * 2 * np.pi
                    y = j / height * params.scale * 2 * np.pi
                    z = k / width * params.scale * 2 * np.pi
                    
                    value = self.math_surfaces.schwarz_p(x, y, z)
                    
                    if abs(value) < params.thickness:
                        # Цвет на основе положения
                        r = (np.sin(x) + 1) / 2
                        g = (np.sin(y) + 1) / 2
                        b = (np.sin(z) + 1) / 2
                        
                        alpha = 1.0 - abs(value) / params.thickness
                        
                        texture[i, j, k] = [r, g, b, alpha]
        
        return texture
    
    def _generate_neovius(self,
                         dimensions: Tuple[int, int, int],
                         params: PatternParameters) -> np.ndarray:
        """Генерация поверхности Неовиуса"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    x = i / depth * params.scale * 2 * np.pi
                    y = j / height * params.scale * 2 * np.pi
                    z = k / width * params.scale * 2 * np.pi
                    
                    value = self.math_surfaces.neovius(x, y, z)
                    
                    if abs(value) < params.thickness:
                        # Сложный цветовой паттерн
                        r = np.sin(x * y) * 0.5 + 0.5
                        g = np.sin(y * z) * 0.5 + 0.5
                        b = np.sin(z * x) * 0.5 + 0.5
                        
                        alpha = 1.0 - abs(value) / params.thickness
                        
                        texture[i, j, k] = [r, g, b, alpha]
        
        return texture
    
    def _generate_lava_lamps(self,
                            dimensions: Tuple[int, int, int],
                            params: PatternParameters) -> np.ndarray:
        """Генерация метаболов (лавовые лампы)"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        # Несколько метаболов
        num_metaballs = 8
        centers = np.random.rand(num_metaballs, 3)
        centers[:, 0] *= depth
        centers[:, 1] *= height
        centers[:, 2] *= width
        
        # Размеры и силы
        radii = np.random.rand(num_metaballs) * 10 + 5
        strengths = np.random.rand(num_metaballs) * 2 + 1
        
        # Цвета
        colors = np.random.rand(num_metaballs, 3)
        
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    pos = np.array([i, j, k])
                    
                    # Сумма влияний всех метаболов
                    total_influence = 0.0
                    color_influence = np.zeros(3)
                    
                    for m in range(num_metaballs):
                        dist = np.linalg.norm(pos - centers[m])
                        
                        if dist < radii[m]:
                            # Влияние метабола (функция сглаживания)
                            influence = strengths[m] * (1 - dist / radii[m])**2
                            total_influence += influence
                            
                            # Вклад в цвет
                            color_influence += colors[m] * influence
                    
                    # Если суммарное влияние выше порога
                    if total_influence > params.surface_isolevel:
                        # Нормализованный цвет
                        if total_influence > 0:
                            avg_color = color_influence / total_influence
                        else:
                            avg_color = np.array([1.0, 1.0, 1.0])
                        
                        # Интенсивность
                        intensity = min(1.0, (total_influence - params.surface_isolevel) / 0.5)
                        
                        # Градиент от центра к краю
                        center_dist = min([np.linalg.norm(pos - centers[m]) / radii[m] 
                                         for m in range(num_metaballs)])
                        
                        gradient = 1.0 - center_dist * 0.5
                        
                        texture[i, j, k] = [
                            avg_color[0] * gradient,
                            avg_color[1] * gradient,
                            avg_color[2] * gradient,
                            intensity * 0.9
                        ]
        
        return texture
    
    def _generate_fibonacci_spiral(self,
                                  dimensions: Tuple[int, int, int],
                                  params: PatternParameters) -> np.ndarray:
        """Генерация спирали Фибоначчи в 3D"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        # Количество точек
        n_points = 200
        phi = params.fibonacci_ratio  # Золотое сечение
        
        for i in range(depth):
            z = (i / depth - 0.5) * 2  # От -1 до 1
            
            for j in range(height):
                y = (j / height - 0.5) * 2
                
                for k in range(width):
                    x = (k / width - 0.5) * 2
                    
                    pos = np.array([x, y, z])
                    min_dist = float('inf')
                    
                    # Генерация точек Фибоначчи на сфере
                    for n in range(n_points):
                        # Угол в сферических координатах
                        theta = 2 * np.pi * n / phi
                        phi_angle = np.arccos(1 - 2 * (n + 0.5) / n_points)
                        
                        # Декартовы координаты точки на сфере
                        point = np.array([
                            np.sin(phi_angle) * np.cos(theta),
                            np.sin(phi_angle) * np.sin(theta),
                            np.cos(phi_angle)
                        ])
                        
                        # Радиус спирали увеличивается с Z
                        radius = 0.3 + abs(z) * 0.2
                        point *= radius
                        
                        # Смещение по Z для создания спирали
                        point[2] = z * 0.5 + np.sin(theta * 2) * 0.1
                        
                        # Расстояние до точки
                        dist = np.linalg.norm(pos - point)
                        min_dist = min(min_dist, dist)
                    
                    # Если близко к спирали
                    if min_dist < params.thickness:
                        # Цвет зависит от угла
                        angle = np.arctan2(y, x)
                        hue = angle / (2 * np.pi)
                        
                        from colorsys import hsv_to_rgb
                        r, g, b = hsv_to_rgb(hue, 0.8, 1.0)
                        
                        # Интенсивность
                        intensity = 1.0 - min_dist / params.thickness
                        alpha = intensity * 0.8
                        
                        texture[i, j, k] = [r, g, b, alpha]
                    
                    # Также создаем трубку вокруг спирали
                    elif min_dist < params.thickness * 2:
                        # Слабое свечение
                        intensity = np.exp(-min_dist * 10) * 0.3
                        texture[i, j, k, 3] = intensity
        
        return texture
    
    def _generate_quasi_crystal(self,
                               dimensions: Tuple[int, int, int],
                               params: PatternParameters) -> np.ndarray:
        """Генерация квазикристаллической структуры"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        # Для квазикристалла с 5-кратной симметрией
        symmetry = params.quasi_symmetry
        
        # Проекция из высшего измерения
        n_dim = 5  # Размерность пространства для проекции
        
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    x = i / depth * params.scale * 2 * np.pi
                    y = j / height * params.scale * 2 * np.pi
                    z = k / width * params.scale * 2 * np.pi
                    
                    # Сумма волн с иррациональными частотами
                    value = 0.0
                    
                    # Золотое сечение
                    phi = params.fibonacci_ratio
                    
                    # Несколько волн с иррациональными соотношениями
                    for m in range(symmetry):
                        angle = 2 * np.pi * m / symmetry
                        
                        # Волновой вектор
                        kx = np.cos(angle)
                        ky = np.sin(angle)
                        
                        # Иррациональная частота
                        freq = 1.0 + phi * m
                        
                        value += np.sin(kx * x * freq + ky * y * freq + z * np.sqrt(freq))
                    
                    value /= symmetry
                    
                    # Порог для создания структуры
                    if abs(value) < params.thickness:
                        # Переливчатый цвет как у квазикристаллов
                        hue = (x + y + z) * 0.3 % 1.0
                        
                        from colorsys import hsv_to_rgb
                        r, g, b = hsv_to_rgb(hue, 0.9, 1.0)
                        
                        alpha = 1.0 - abs(value) / params.thickness
                        
                        # Добавляем интерференционные полосы
                        interference = np.sin(x * 20) * np.sin(y * 20) * 0.2 + 0.8
                        r *= interference
                        g *= interference
                        b *= interference
                        
                        texture[i, j, k] = [r, g, b, alpha]
        
        return texture
    
    def _generate_tessellation(self,
                              dimensions: Tuple[int, int, int],
                              params: PatternParameters) -> np.ndarray:
        """Генерация пространственной тесселяции"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        # Тип тесселяции
        tess_type = "truncated_octahedron"  # Усеченный октаэдр
        
        if tess_type == "truncated_octahedron":
            # Усеченный октаэдр заполняет пространство без зазоров
            
            for i in range(depth):
                for j in range(height):
                    for k in range(width):
                        x = i / depth * params.scale
                        y = j / height * params.scale
                        z = k / width * params.scale
                        
                        # Приводим к координатам в ячейке [0,1]^3
                        cell_x = x * 4 % 1.0
                        cell_y = y * 4 % 1.0
                        cell_z = z * 4 % 1.0
                        
                        # Расстояние до центра ячейки
                        dist_to_center = np.sqrt(
                            (cell_x - 0.5)**2 + 
                            (cell_y - 0.5)**2 + 
                            (cell_z - 0.5)**2
                        )
                        
                        # Также проверяем расстояние до граней
                        dist_to_faces = min(
                            cell_x, 1 - cell_x,
                            cell_y, 1 - cell_y,
                            cell_z, 1 - cell_z
                        )
                        
                        # Комбинированное расстояние
                        dist = min(dist_to_center, dist_to_faces * np.sqrt(3))
                        
                        # Если на поверхности усеченного октаэдра
                        if abs(dist - 0.3) < params.thickness:
                            # Цвет зависит от ориентации
                            normal = np.array([
                                cell_x - 0.5,
                                cell_y - 0.5,
                                cell_z - 0.5
                            ])
                            normal = normal / (np.linalg.norm(normal) + 1e-8)
                            
                            r = (normal[0] + 1) / 2
                            g = (normal[1] + 1) / 2
                            b = (normal[2] + 1) / 2
                            
                            alpha = 1.0 - abs(dist - 0.3) / params.thickness
                            
                            texture[i, j, k] = [r, g, b, alpha]
        
        return texture
    
    def _generate_fractal_sponge(self,
                                dimensions: Tuple[int, int, int],
                                params: PatternParameters) -> np.ndarray:
        """Генерация губки Менгера (фрактальной губки)"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        # Функция для определения принадлежности к губке Менгера
        def in_menger_sponge(x, y, z, level):
            """Рекурсивная проверка принадлежности к губке Менгера"""
            if level <= 0:
                return True
            
            # Масштабируем координаты
            x *= 3
            y *= 3
            z *= 3
            
            # Целочисленные части
            ix = int(x)
            iy = int(y)
            iz = int(z)
            
            # Если в центральном кубе любого уровня - удаляем
            if ix == 1 and iy == 1:
                return False
            if ix == 1 and iz == 1:
                return False
            if iy == 1 and iz == 1:
                return False
            
            # Рекурсивно проверяем следующий уровень
            return in_menger_sponge(x - ix, y - iy, z - iz, level - 1)
        
        iterations = params.fractal_iterations
        
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    x = i / depth
                    y = j / height
                    z = k / width
                    
                    # Принадлежность к губке Менгера
                    if in_menger_sponge(x, y, z, iterations):
                        # Цвет зависит от уровня
                        level_color = 1.0 - 0.2 * iterations
                        
                        # Текстура на поверхностях
                        texture_val = (np.sin(x * 20) * np.sin(y * 20) * 
                                      np.sin(z * 20) * 0.1 + 0.9)
                        
                        r = 0.7 * level_color * texture_val
                        g = 0.5 * level_color * texture_val
                        b = 0.3 * level_color * texture_val
                        
                        # Альфа на краях кубов
                        # Определяем, на грани ли мы
                        eps = 0.01
                        on_edge = False
                        
                        # Проверяем все направления
                        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), 
                                          (0,-1,0), (0,0,1), (0,0,-1)]:
                            nx, ny, nz = x + dx * eps, y + dy * eps, z + dz * eps
                            if not in_menger_sponge(nx, ny, nz, iterations):
                                on_edge = True
                                break
                        
                        if on_edge:
                            alpha = 0.9
                        else:
                            alpha = 0.6
                        
                        texture[i, j, k] = [r, g, b, alpha]
        
        return texture
    
    def _generate_knots(self,
                       dimensions: Tuple[int, int, int],
                       params: PatternParameters) -> np.ndarray:
        """Генерация узлов и зацеплений"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        # Параметрическое представление узла трилистника
        def trefoil_knot(t, scale=1.0):
            """Трилистник (trefoil knot)"""
            x = scale * (np.sin(t) + 2 * np.sin(2*t))
            y = scale * (np.cos(t) - 2 * np.cos(2*t))
            z = scale * (-np.sin(3*t))
            return np.array([x, y, z])
        
        # Толщина трубки
        tube_radius = params.thickness * 2
        
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    # Нормализованные координаты
                    x = (i / depth - 0.5) * 2
                    y = (j / height - 0.5) * 2
                    z = (k / width - 0.5) * 2
                    
                    pos = np.array([x, y, z])
                    min_dist = float('inf')
                    
                    # Проверяем расстояние до узла
                    n_samples = 100
                    for n in range(n_samples):
                        t = n / n_samples * 2 * np.pi
                        
                        # Точка на узле
                        knot_point = trefoil_knot(t, 0.5)
                        
                        # Расстояние до этой точки
                        dist = np.linalg.norm(pos - knot_point)
                        min_dist = min(min_dist, dist)
                    
                    # Если внутри трубки
                    if min_dist < tube_radius:
                        # Цвет узла
                        # Зависит от параметра t (условно от угла)
                        t_approx = np.arctan2(y, x)
                        hue = t_approx / (2 * np.pi)
                        
                        from colorsys import hsv_to_rgb
                        r, g, b = hsv_to_rgb(hue, 0.9, 1.0)
                        
                        # Интенсивность
                        intensity = 1.0 - min_dist / tube_radius
                        alpha = intensity * 0.9
                        
                        # Полосы на трубке
                        stripe = np.sin(t_approx * 20) * 0.2 + 0.8
                        r *= stripe
                        g *= stripe
                        b *= stripe
                        
                        texture[i, j, k] = [r, g, b, alpha]
                    
                    # Также создаем второй узел для зацепления
                    # (простой вариант - сдвинутый трилистник)
                    second_knot_offset = np.array([0.3, 0.3, 0.3])
                    min_dist2 = float('inf')
                    
                    for n in range(n_samples):
                        t = n / n_samples * 2 * np.pi
                        knot_point = trefoil_knot(t, 0.5) + second_knot_offset
                        dist = np.linalg.norm(pos - knot_point)
                        min_dist2 = min(min_dist2, dist)
                    
                    if min_dist2 < tube_radius:
                        # Другой цвет для второго узла
                        hue2 = (np.arctan2(y - 0.3, x - 0.3) / (2 * np.pi) + 0.5) % 1.0
                        r2, g2, b2 = hsv_to_rgb(hue2, 0.7, 1.0)
                        
                        intensity2 = 1.0 - min_dist2 / tube_radius
                        alpha2 = intensity2 * 0.9
                        
                        # Смешивание с первым узлом если пересекаются
                        if texture[i, j, k, 3] > 0:
                            # Область пересечения - белая
                            mix_factor = texture[i, j, k, 3]
                            r = r * (1 - mix_factor) + 1.0 * mix_factor
                            g = g * (1 - mix_factor) + 1.0 * mix_factor
                            b = b * (1 - mix_factor) + 1.0 * mix_factor
                            alpha = max(texture[i, j, k, 3], alpha2)
                            
                            texture[i, j, k] = [r, g, b, alpha]
                        else:
                            texture[i, j, k] = [r2, g2, b2, alpha2]
        
        return texture
    
    def _generate_helicoid(self,
                          dimensions: Tuple[int, int, int],
                          params: PatternParameters) -> np.ndarray:
        """Генерация геликоида"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    x = (i / depth - 0.5) * 2 * params.scale
                    y = (j / height - 0.5) * 2 * params.scale
                    z = (k / width - 0.5) * 2 * params.scale
                    
                    value = self.math_surfaces.helicoid(x, y, z, 5.0)
                    
                    if abs(value) < params.thickness:
                        # Цвет зависит от высоты
                        hue = (z + 1) / 2
                        
                        from colorsys import hsv_to_rgb
                        r, g, b = hsv_to_rgb(hue, 0.8, 1.0)
                        
                        alpha = 1.0 - abs(value) / params.thickness
                        
                        # Полосы вдоль геликоида
                        stripe = np.sin(x * 20 + z * 20) * 0.2 + 0.8
                        r *= stripe
                        g *= stripe
                        b *= stripe
                        
                        texture[i, j, k] = [r, g, b, alpha]
        
        return texture
    
    def _generate_catenoid(self,
                          dimensions: Tuple[int, int, int],
                          params: PatternParameters) -> np.ndarray:
        """Генерация катеноида"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    x = (i / depth - 0.5) * 2 * params.scale
                    y = (j / height - 0.5) * 2 * params.scale
                    z = (k / width - 0.5) * 2 * params.scale
                    
                    value = self.math_surfaces.catenoid(x, y, z, 3.0)
                    
                    if abs(value) < params.thickness:
                        # Цвет зависит от радиуса
                        r_cyl = np.sqrt(x*x + y*y)
                        hue = r_cyl % 1.0
                        
                        from colorsys import hsv_to_rgb
                        r, g, b = hsv_to_rgb(hue, 0.7, 1.0)
                        
                        alpha = 1.0 - abs(value) / params.thickness
                        
                        texture[i, j, k] = [r, g, b, alpha]
        
        return texture
    
    def _generate_enneper(self,
                         dimensions: Tuple[int, int, int],
                         params: PatternParameters) -> np.ndarray:
        """Генерация поверхности Эннепера"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    x = (i / depth - 0.5) * 2 * params.scale
                    y = (j / height - 0.5) * 2 * params.scale
                    z = (k / width - 0.5) * 2 * params.scale
                    
                    value = self.math_surfaces.enneper(x, y, z, 2.0)
                    
                    if value < params.thickness * 2:
                        # Сложный цветовой паттерн
                        r = (np.sin(x * 10) + 1) / 2
                        g = (np.sin(y * 10) + 1) / 2
                        b = (np.sin(z * 10) + 1) / 2
                        
                        alpha = 1.0 - value / (params.thickness * 2)
                        
                        texture[i, j, k] = [r, g, b, alpha]
        
        return texture
    
    def _generate_corkscrew(self,
                           dimensions: Tuple[int, int, int],
                           params: PatternParameters) -> np.ndarray:
        """Генерация штопора Архимеда"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    x = (i / depth - 0.5) * 2 * params.scale
                    y = (j / height - 0.5) * 2 * params.scale
                    z = (k / width - 0.5) * 2 * params.scale
                    
                    value = self.math_surfaces.corkscrew(x, y, z, 3.0)
                    
                    if abs(value) < params.thickness:
                        # Цвет зависит от угла
                        angle = np.arctan2(y, x)
                        hue = angle / (2 * np.pi)
                        
                        from colorsys import hsv_to_rgb
                        r, g, b = hsv_to_rgb(hue, 0.9, 1.0)
                        
                        alpha = 1.0 - abs(value) / params.thickness
                        
                        texture[i, j, k] = [r, g, b, alpha]
        
        return texture
    
    def _generate_mobius(self,
                        dimensions: Tuple[int, int, int],
                        params: PatternParameters) -> np.ndarray:
        """Генерация ленты Мёбиуса"""
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4), dtype=np.float32)
        
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    x = (i / depth - 0.5) * 2 * params.scale
                    y = (j / height - 0.5) * 2 * params.scale
                    z = (k / width - 0.5) * 2 * params.scale
                    
                    value = self.math_surfaces.mobius(x, y, z, 2.0)
                    
                    if value < params.thickness * 3:
                        # Особый цвет для ленты Мёбиуса
                        # Цвет меняется непрерывно при обходе
                        angle = np.arctan2(y, x)
                        
                        # На ленте Мёбиуса нужно сделать полный оборот 720 градусов
                        # чтобы вернуться к исходному цвету
                        hue = (angle * 2) % 1.0
                        
                        from colorsys import hsv_to_rgb
                        r, g, b = hsv_to_rgb(hue, 0.8, 1.0)
                        
                        alpha = 1.0 - value / (params.thickness * 3)
                        
                        texture[i, j, k] = [r, g, b, alpha]
        
        return texture
    
    def _add_noise(self, texture: np.ndarray, amplitude: float, scale: float) -> np.ndarray:
        """Добавление шума к текстуре"""
        if amplitude <= 0:
            return texture
        
        depth, height, width, channels = texture.shape
        
        # Генерация шума
        noise = np.random.randn(depth, height, width) * amplitude
        
        # Сглаживание шума
        from scipy import ndimage
        noise = ndimage.gaussian_filter(noise, sigma=scale)
        
        # Применение шума к альфа-каналу
        texture_with_noise = texture.copy()
        
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    if texture[i, j, k, 3] > 0:
                        # Изменяем альфа-канал
                        new_alpha = texture[i, j, k, 3] * (1 + noise[i, j, k])
                        texture_with_noise[i, j, k, 3] = np.clip(new_alpha, 0, 1)
                        
                        # Немного изменяем цвет
                        color_noise = noise[i, j, k] * 0.1
                        texture_with_noise[i, j, k, 0] = np.clip(
                            texture[i, j, k, 0] + color_noise, 0, 1
                        )
                        texture_with_noise[i, j, k, 1] = np.clip(
                            texture[i, j, k, 1] + color_noise, 0, 1
                        )
                        texture_with_noise[i, j, k, 2] = np.clip(
                            texture[i, j, k, 2] + color_noise, 0, 1
                        )
        
        return texture_with_noise

# ----------------------------------------------------------------------
# Композитные паттерны и комбинации
# ----------------------------------------------------------------------

class CompositePatternGenerator3D:
    """Генератор композитных 3D паттернов (комбинации нескольких паттернов)"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.generator = GeometricPatternGenerator3D(seed)
        self.blend_cache = {}
    
    def generate_composite(self,
                          pattern_types: List[GeometricPattern3D],
                          dimensions: Tuple[int, int, int],
                          params_list: Optional[List[PatternParameters]] = None,
                          blend_mode: str = "add") -> np.ndarray:
        """
        Генерация композитного паттерна из нескольких типов
        
        Args:
            pattern_types: Список типов паттернов для комбинирования
            dimensions: Размеры объема
            params_list: Список параметров для каждого паттерна
            blend_mode: Режим смешивания ('add', 'multiply', 'max', 'lerp')
            
        Returns:
            Композитная 3D текстура
        """
        if params_list is None:
            params_list = [PatternParameters() for _ in pattern_types]
        
        if len(pattern_types) != len(params_list):
            raise ValueError("Number of pattern types must match number of parameter sets")
        
        # Проверка кэша
        cache_key = (
            tuple(p.value for p in pattern_types),
            dimensions,
            tuple(hash(frozenset(p.__dict__.items())) for p in params_list),
            blend_mode
        )
        
        if cache_key in self.blend_cache:
            return self.blend_cache[cache_key].copy()
        
        print(f"Generating composite pattern from {len(pattern_types)} patterns...")
        
        # Генерация отдельных паттернов
        textures = []
        for i, (pattern_type, params) in enumerate(zip(pattern_types, params_list)):
            print(f"  Generating {pattern_type.name} ({i+1}/{len(pattern_types)})...")
            texture = self.generator.generate_pattern(pattern_type, dimensions, params)
            textures.append(texture)
        
        # Смешивание паттернов
        composite = self._blend_textures(textures, blend_mode)
        
        # Кэширование
        self.blend_cache[cache_key] = composite.copy()
        if len(self.blend_cache) > 30:
            self.blend_cache.pop(next(iter(self.blend_cache)))
        
        return composite
    
    def _blend_textures(self, textures: List[np.ndarray], blend_mode: str) -> np.ndarray:
        """Смешивание нескольких текстур"""
        if not textures:
            raise ValueError("No textures to blend")
        
        if len(textures) == 1:
            return textures[0]
        
        result = textures[0].copy()
        
        for i in range(1, len(textures)):
            if blend_mode == "add":
                result = self._blend_add(result, textures[i])
            elif blend_mode == "multiply":
                result = self._blend_multiply(result, textures[i])
            elif blend_mode == "max":
                result = self._blend_max(result, textures[i])
            elif blend_mode == "lerp":
                # Линейная интерполяция с равными весами
                weight = 1.0 / (i + 1)
                result = result * (1 - weight) + textures[i] * weight
            else:
                raise ValueError(f"Unknown blend mode: {blend_mode}")
        
        return np.clip(result, 0, 1)
    
    def _blend_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Сложение текстур"""
        blended = a + b
        # Сохраняем альфа как максимум
        blended[..., 3] = np.maximum(a[..., 3], b[..., 3])
        return blended
    
    def _blend_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Умножение текстур"""
        blended = a * b
        # Альфа как среднее
        blended[..., 3] = (a[..., 3] + b[..., 3]) / 2
        return blended
    
    def _blend_max(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Максимум из текстур"""
        blended = np.maximum(a, b)
        return blended
    
    def generate_layered_pattern(self,
                                pattern_layers: List[Tuple[GeometricPattern3D, PatternParameters, float]],
                                dimensions: Tuple[int, int, int]) -> np.ndarray:
        """
        Генерация слоистого паттерна с разными параметрами и прозрачностями
        
        Args:
            pattern_layers: Список слоев (тип, параметры, непрозрачность)
            dimensions: Размеры объема
            
        Returns:
            Слоистая 3D текстура
        """
        result = np.zeros((*dimensions, 4), dtype=np.float32)
        
        for pattern_type, params, opacity in pattern_layers:
            layer = self.generator.generate_pattern(pattern_type, dimensions, params)
            
            # Применяем непрозрачность
            layer[..., 3] *= opacity
            
            # Наложение слоев (фронтально-заднее смешивание)
            alpha = layer[..., 3:]
            result = result * (1 - alpha) + layer * alpha
        
        return np.clip(result, 0, 1)
    
    def generate_animated_pattern(self,
                                 base_pattern: GeometricPattern3D,
                                 dimensions: Tuple[int, int, int],
                                 base_params: PatternParameters,
                                 time: float,
                                 animation_type: str = "rotation") -> np.ndarray:
        """
        Генерация анимированного геометрического паттерна
        
        Args:
            base_pattern: Базовый тип паттерна
            dimensions: Размеры объема
            base_params: Базовые параметры
            time: Время анимации
            animation_type: Тип анимации ('rotation', 'pulse', 'morph')
            
        Returns:
            Анимированная 3D текстура
        """
        # Создаем копию параметров для анимации
        animated_params = PatternParameters(**base_params.__dict__)
        
        if animation_type == "rotation":
            # Вращение паттерна
            animated_params.scale = base_params.scale * (1 + 0.1 * np.sin(time))
            
            # Сдвиг фазы для эффекта вращения
            phase_shift = time * 2 * np.pi
            
            # Генерация паттерна с модифицированными координатами
            # (в реальности нужно модифицировать сам алгоритм генерации)
            texture = self.generator.generate_pattern(base_pattern, dimensions, animated_params)
            
            # Применяем сдвиг по осям для имитации вращения
            depth, height, width, _ = texture.shape
            
            # Сдвигаем текстуру по осям
            shift_x = int(width * 0.1 * np.sin(time))
            shift_y = int(height * 0.1 * np.cos(time))
            shift_z = int(depth * 0.1 * np.sin(time * 0.7))
            
            # Циклический сдвиг
            texture = np.roll(texture, shift_x, axis=2)
            texture = np.roll(texture, shift_y, axis=1)
            texture = np.roll(texture, shift_z, axis=0)
            
            return texture
            
        elif animation_type == "pulse":
            # Пульсация паттерна
            pulse = 0.5 + 0.5 * np.sin(time * 2)
            animated_params.thickness = base_params.thickness * pulse
            animated_params.scale = base_params.scale * (0.8 + 0.2 * pulse)
            
            return self.generator.generate_pattern(base_pattern, dimensions, animated_params)
            
        elif animation_type == "morph":
            # Морфинг между разными паттернами
            # Используем синусоидальную интерполяцию
            morph_factor = 0.5 + 0.5 * np.sin(time)
            
            # Выбираем два паттерна для морфинга
            pattern1 = base_pattern
            pattern2 = GeometricPattern3D.GYROID  # Или другой паттерн
            
            texture1 = self.generator.generate_pattern(pattern1, dimensions, base_params)
            
            # Модифицируем параметры для второго паттерна
            params2 = PatternParameters(**base_params.__dict__)
            params2.scale *= 1.2
            
            texture2 = self.generator.generate_pattern(pattern2, dimensions, params2)
            
            # Интерполяция
            return texture1 * (1 - morph_factor) + texture2 * morph_factor
        
        else:
            raise ValueError(f"Unknown animation type: {animation_type}")

# ----------------------------------------------------------------------
# Визуализация и экспорт
# ----------------------------------------------------------------------

class GeometricPatternVisualizer3D:
    """Визуализатор для геометрических 3D паттернов"""
    
    def __init__(self):
        self.render_methods = {
            "raycast": self._render_raycast,
            "slices": self._render_slices,
            "mip": self._render_mip,
            "isosurface": self._render_isosurface,
        }
    
    def render(self,
              texture: np.ndarray,
              method: str = "raycast",
              **kwargs) -> np.ndarray:
        """
        Рендеринг 3D текстуры
        
        Args:
            texture: 3D текстура (D, H, W, 4)
            method: Метод рендеринга
            **kwargs: Параметры рендеринга
            
        Returns:
            2D изображение
        """
        if method not in self.render_methods:
            raise ValueError(f"Unknown render method: {method}")
        
        return self.render_methods[method](texture, **kwargs)
    
    def _render_raycast(self,
                       texture: np.ndarray,
                       camera_pos: Tuple[float, float, float] = (0.5, 0.5, 2.0),
                       camera_target: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                       image_size: Tuple[int, int] = (512, 512),
                       max_steps: int = 256) -> np.ndarray:
        """Рейкастинг для геометрических паттернов"""
        depth, height, width, channels = texture.shape
        img_height, img_width = image_size
        
        image = np.zeros((img_height, img_width, 4), dtype=np.float32)
        
        # Базис камеры
        camera_dir = np.array(camera_target) - np.array(camera_pos)
        camera_dir = camera_dir / np.linalg.norm(camera_dir)
        
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(camera_dir, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, camera_dir)
        
        # FOV
        fov = 60.0
        aspect = img_width / img_height
        half_height = np.tan(np.radians(fov) / 2.0)
        half_width = aspect * half_height
        
        # Шаг по лучу
        step_size = 1.0 / max(depth, height, width)
        
        for y in range(img_height):
            for x in range(img_width):
                # Направление луча
                u = (2.0 * x / img_width - 1.0) * half_width
                v = (1.0 - 2.0 * y / img_height) * half_height
                
                ray_dir = camera_dir + u * right + v * up
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # Стартовая позиция
                ray_pos = np.array(camera_pos, dtype=np.float32)
                
                # Интегрирование
                color = np.zeros(4, dtype=np.float32)
                
                for step in range(max_steps):
                    # Проверяем границы
                    if (ray_pos[0] < 0 or ray_pos[0] >= 1 or
                        ray_pos[1] < 0 or ray_pos[1] >= 1 or
                        ray_pos[2] < 0 or ray_pos[2] >= 1):
                        break
                    
                    # Координаты в текстуре
                    tex_x = int(ray_pos[0] * (width - 1))
                    tex_y = int(ray_pos[1] * (height - 1))
                    tex_z = int(ray_pos[2] * (depth - 1))
                    
                    # Берем значение
                    sample = texture[tex_z, tex_y, tex_x]
                    
                    # Фронтально-заднее смешивание
                    alpha = sample[3]
                    color = color + (1.0 - color[3]) * alpha * sample
                    
                    # Если непрозрачно
                    if color[3] > 0.99:
                        break
                    
                    # Двигаем луч
                    ray_pos += ray_dir * step_size
                
                image[y, x] = color
        
        return np.clip(image, 0, 1)
    
    def _render_slices(self,
                      texture: np.ndarray,
                      num_slices: int = 3,
                      spacing: float = 0.1,
                      image_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Рендеринг нескольких срезов"""
        depth, height, width, channels = texture.shape
        img_height, img_width = image_size
        
        image = np.zeros((img_height, img_width, 4), dtype=np.float32)
        
        # Позиции срезов
        slice_positions = np.linspace(0.1, 0.9, num_slices)
        
        for i, pos in enumerate(slice_positions):
            # Координата среза
            slice_idx = int(pos * depth)
            slice_data = texture[slice_idx]
            
            # Масштабируем срез до размера изображения
            from scipy import ndimage
            
            scale_y = img_height / height
            scale_x = img_width / width
            
            slice_resized = np.zeros((img_height, img_width, channels), dtype=np.float32)
            
            for c in range(channels):
                slice_resized[:, :, c] = ndimage.zoom(
                    slice_data[:, :, c], (scale_y, scale_x), order=1
                )
            
            # Смещение для 3D эффекта
            offset_x = int((i - num_slices/2) * img_width * spacing)
            offset_y = int((i - num_slices/2) * img_height * spacing * 0.5)
            
            # Смешивание с изображением
            alpha = slice_resized[..., 3:]
            for y in range(img_height):
                for x in range(img_width):
                    src_y = (y - offset_y) % img_height
                    src_x = (x - offset_x) % img_width
                    
                    if 0 <= src_y < img_height and 0 <= src_x < img_width:
                        src_alpha = alpha[src_y, src_x, 0]
                        image[y, x] = image[y, x] * (1 - src_alpha) + \
                                     slice_resized[src_y, src_x] * src_alpha
        
        return np.clip(image, 0, 1)
    
    def _render_mip(self,
                   texture: np.ndarray,
                   axis: str = 'z',
                   image_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """MIP (Maximum Intensity Projection) рендеринг"""
        depth, height, width, channels = texture.shape
        
        if axis == 'x':
            mip = np.max(texture, axis=2)
        elif axis == 'y':
            mip = np.max(texture, axis=1)
        else:  # 'z'
            mip = np.max(texture, axis=0)
        
        # Масштабирование
        from scipy import ndimage
        
        img_height, img_width = image_size
        
        scale_y = img_height / mip.shape[0]
        scale_x = img_width / mip.shape[1]
        
        mip_resized = np.zeros((img_height, img_width, channels), dtype=np.float32)
        
        for c in range(channels):
            mip_resized[:, :, c] = ndimage.zoom(
                mip[:, :, c], (scale_y, scale_x), order=1
            )
        
        return np.clip(mip_resized, 0, 1)
    
    def _render_isosurface(self,
                          texture: np.ndarray,
                          isolevel: float = 0.5,
                          image_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Рендеринг изоповерхности (упрощенный)"""
        depth, height, width, channels = texture.shape
        
        # Используем только альфа-канал для поверхности
        alpha_volume = texture[..., 3]
        
        # Находим поверхность через порог
        surface_mask = alpha_volume > isolevel
        
        # Проекция
        projection = np.max(surface_mask, axis=0)
        
        # Масштабирование
        img_height, img_width = image_size
        
        from scipy import ndimage
        scale_y = img_height / projection.shape[0]
        scale_x = img_width / projection.shape[1]
        
        projection_resized = ndimage.zoom(projection, (scale_y, scale_x), order=0)
        
        # Создаем изображение
        image = np.zeros((img_height, img_width, 4), dtype=np.float32)
        
        for y in range(img_height):
            for x in range(img_width):
                if projection_resized[y, x] > 0:
                    # Цвет поверхности
                    image[y, x] = [0.8, 0.8, 1.0, 1.0]
        
        return image
    
    def create_animation(self,
                        pattern_generator: GeometricPatternGenerator3D,
                        pattern_type: GeometricPattern3D,
                        dimensions: Tuple[int, int, int],
                        params: PatternParameters,
                        num_frames: int = 60,
                        animation_type: str = "rotation") -> List[np.ndarray]:
        """
        Создание анимации геометрического паттерна
        
        Returns:
            Список кадров анимации
        """
        frames = []
        
        for frame in range(num_frames):
            time = frame / num_frames * 2 * np.pi
            
            # Модифицируем параметры для анимации
            animated_params = PatternParameters(**params.__dict__)
            
            if animation_type == "rotation":
                # Вращение
                animated_params.scale = params.scale * (1 + 0.1 * np.sin(time))
                
                # Также вращаем паттерн через сдвиг координат
                # (в реальной реализации нужно вращать сам алгоритм)
                texture = pattern_generator.generate_pattern(
                    pattern_type, dimensions, animated_params
                )
                
                # Циклический сдвиг для имитации вращения
                shift = int(frame * 0.5) % dimensions[2]
                texture = np.roll(texture, shift, axis=2)
                
            elif animation_type == "pulse":
                # Пульсация
                pulse = 0.5 + 0.5 * np.sin(time)
                animated_params.thickness = params.thickness * pulse
                animated_params.scale = params.scale * (0.9 + 0.1 * pulse)
                
                texture = pattern_generator.generate_pattern(
                    pattern_type, dimensions, animated_params
                )
            
            elif animation_type == "morph":
                # Морфинг между разными масштабами
                morph = np.sin(time) * 0.5 + 0.5
                animated_params.scale = params.scale * (0.7 + 0.6 * morph)
                
                texture = pattern_generator.generate_pattern(
                    pattern_type, dimensions, animated_params
                )
            
            # Рендеринг кадра
            frame_image = self.render(texture, method="raycast", 
                                     image_size=(256, 256))
            frames.append(frame_image)
            
            print(f"Generated frame {frame+1}/{num_frames}")
        
        return frames

# ----------------------------------------------------------------------
# Примеры использования
# ----------------------------------------------------------------------

def example_crystal_lattices():
    """Пример различных кристаллических решеток"""
    
    print("Crystal lattices examples...")
    
    generator = GeometricPatternGenerator3D(seed=42)
    visualizer = GeometricPatternVisualizer3D()
    
    # Различные типы кристаллов
    crystal_types = [
        ("cubic", PatternParameters(
            scale=3.0,
            thickness=0.05,
            crystal_type="cubic"
        )),
        ("hexagonal", PatternParameters(
            scale=2.0,
            thickness=0.04,
            crystal_type="hexagonal",
            symmetry=6
        )),
        ("diamond", PatternParameters(
            scale=2.5,
            thickness=0.03,
            crystal_type="diamond",
            noise_amplitude=0.1
        ))
    ]
    
    results = {}
    
    for crystal_name, params in crystal_types:
        print(f"\nGenerating {crystal_name} crystal lattice...")
        
        texture = generator.generate_pattern(
            GeometricPattern3D.CRYSTAL_LATTICE,
            dimensions=(64, 64, 64),
            params=params
        )
        
        # Рендеринг
        image = visualizer.render(
            texture,
            method="raycast",
            image_size=(512, 512),
            camera_pos=(0.5, 0.5, 1.5),
            camera_target=(0.5, 0.5, 0.5)
        )
        
        results[crystal_name] = (texture, image)
        
        print(f"  Texture shape: {texture.shape}")
        print(f"  Image shape: {image.shape}")
    
    return results

def example_complex_patterns():
    """Пример сложных геометрических паттернов"""
    
    print("\nComplex geometric patterns examples...")
    
    generator = GeometricPatternGenerator3D(seed=123)
    visualizer = GeometricPatternVisualizer3D()
    
    patterns = [
        ("Gyroid", GeometricPattern3D.GYROID, PatternParameters(
            scale=4.0,
            thickness=0.03,
            surface_threshold=0.0
        )),
        ("Schwarz P", GeometricPattern3D.SCHWARZ_P, PatternParameters(
            scale=3.0,
            thickness=0.04
        )),
        ("Neovius", GeometricPattern3D.NEOVIUS, PatternParameters(
            scale=2.5,
            thickness=0.035
        )),
        ("Fibonacci Spiral", GeometricPattern3D.FIBONACCI_SPIRAL, PatternParameters(
            scale=2.0,
            thickness=0.02,
            fibonacci_ratio=1.61803398875
        )),
        ("Quasi Crystal", GeometricPattern3D.QUASI_CRYSTAL, PatternParameters(
            scale=3.5,
            thickness=0.025,
            quasi_symmetry=5
        )),
        ("Menger Sponge", GeometricPattern3D.FRACTAL_SPONGE, PatternParameters(
            scale=1.0,
            fractal_iterations=3,
            thickness=0.05
        ))
    ]
    
    results = {}
    
    for pattern_name, pattern_type, params in patterns:
        print(f"\nGenerating {pattern_name}...")
        
        texture = generator.generate_pattern(
            pattern_type,
            dimensions=(48, 48, 48),  # Меньший размер для скорости
            params=params
        )
        
        # Рендеринг
        image = visualizer.render(
            texture,
            method="raycast",
            image_size=(400, 400),
            camera_pos=(0.5, 0.5, 1.8),
            camera_target=(0.5, 0.5, 0.5)
        )
        
        results[pattern_name] = (texture, image)
        
        print(f"  Generated {pattern_name} pattern")
    
    return results

def example_composite_pattern():
    """Пример композитного паттерна"""
    
    print("\nComposite pattern example...")
    
    composite_gen = CompositePatternGenerator3D(seed=42)
    visualizer = GeometricPatternVisualizer3D()
    
    # Комбинируем несколько паттернов
    pattern_types = [
        GeometricPattern3D.CRYSTAL_LATTICE,
        GeometricPattern3D.GYROID,
        GeometricPattern3D.VORONOI_CELLS
    ]
    
    params_list = [
        PatternParameters(
            scale=2.0,
            thickness=0.04,
            crystal_type="cubic"
        ),
        PatternParameters(
            scale=3.0,
            thickness=0.03,
            surface_threshold=0.0
        ),
        PatternParameters(
            scale=1.5,
            packing_density=0.3,
            wall_thickness=0.1
        )
    ]
    
    print("Generating composite pattern from 3 patterns...")
    
    composite = composite_gen.generate_composite(
        pattern_types=pattern_types,
        dimensions=(56, 56, 56),
        params_list=params_list,
        blend_mode="add"
    )
    
    # Рендеринг
    image = visualizer.render(
        composite,
        method="raycast",
        image_size=(512, 512),
        camera_pos=(0.5, 0.5, 2.0),
        camera_target=(0.5, 0.5, 0.5)
    )
    
    print(f"Composite texture shape: {composite.shape}")
    print(f"Rendered image shape: {image.shape}")
    
    return composite, image

def example_animated_pattern():
    """Пример анимированного паттерна"""
    
    print("\nAnimated pattern example...")
    
    generator = GeometricPatternGenerator3D(seed=42)
    visualizer = GeometricPatternVisualizer3D()
    composite_gen = CompositePatternGenerator3D(seed=42)
    
    # Создаем анимацию морфинга
    print("Creating pattern morph animation...")
    
    frames = []
    num_frames = 24
    
    for frame in range(num_frames):
        time = frame / num_frames * 2 * np.pi
        
        # Морфинг между гироидом и кристаллической решеткой
        morph_factor = 0.5 + 0.5 * np.sin(time)
        
        # Параметры для гироида
        params_gyroid = PatternParameters(
            scale=2.5 + 0.5 * np.sin(time * 0.5),
            thickness=0.03,
            surface_threshold=0.0
        )
        
        # Параметры для кристалла
        params_crystal = PatternParameters(
            scale=2.0 + 0.3 * np.cos(time * 0.5),
            thickness=0.04,
            crystal_type="hexagonal"
        )
        
        # Генерируем оба паттерна
        texture_gyroid = generator.generate_pattern(
            GeometricPattern3D.GYROID,
            dimensions=(48, 48, 48),
            params=params_gyroid
        )
        
        texture_crystal = generator.generate_pattern(
            GeometricPattern3D.CRYSTAL_LATTICE,
            dimensions=(48, 48, 48),
            params=params_crystal
        )
        
        # Интерполяция
        texture = texture_gyroid * (1 - morph_factor) + \
                 texture_crystal * morph_factor
        
        # Рендеринг кадра
        frame_img = visualizer.render(
            texture,
            method="raycast",
            image_size=(256, 256),
            camera_pos=(0.5, 0.5, 1.5 + 0.3 * np.sin(time)),
            camera_target=(0.5, 0.5, 0.5)
        )
        
        frames.append(frame_img)
        
        print(f"  Generated frame {frame+1}/{num_frames}")
    
    print(f"Created animation with {len(frames)} frames")
    
    return frames

def example_layered_pattern():
    """Пример слоистого паттерна"""
    
    print("\nLayered pattern example...")
    
    composite_gen = CompositePatternGenerator3D(seed=123)
    visualizer = GeometricPatternVisualizer3D()
    
    # Создаем слои
    layers = [
        (GeometricPattern3D.HONEYCOMB, PatternParameters(
            scale=2.0,
            cell_size=0.4,
            wall_thickness=0.08
        ), 0.7),  # Непрозрачность 70%
        
        (GeometricPattern3D.CRYSTAL_LATTICE, PatternParameters(
            scale=3.0,
            thickness=0.02,
            crystal_type="diamond",
            noise_amplitude=0.2
        ), 0.5),  # Непрозрачность 50%
        
        (GeometricPattern3D.SPHERE_PACKING, PatternParameters(
            scale=1.5,
            sphere_radius=0.2,
            packing_density=0.4
        ), 0.3)   # Непрозрачность 30%
    ]
    
    print("Generating layered pattern with 3 layers...")
    
    layered_texture = composite_gen.generate_layered_pattern(
        pattern_layers=layers,
        dimensions=(64, 64, 64)
    )
    
    # Рендеринг
    image = visualizer.render(
        layered_texture,
        method="raycast",
        image_size=(512, 512),
        camera_pos=(0.5, 0.5, 1.8),
        camera_target=(0.5, 0.5, 0.3)
    )
    
    print(f"Layered texture shape: {layered_texture.shape}")
    print(f"Rendered image shape: {image.shape}")
    
    return layered_texture, image

if __name__ == "__main__":
    print("Geometric 3D Patterns System")
    print("=" * 60)
    
    # Пример 1: Кристаллические решетки
    crystal_results = example_crystal_lattices()
    
    # Пример 2: Сложные геометрические паттерны
    pattern_results = example_complex_patterns()
    
    # Пример 3: Композитный паттерн
    composite_texture, composite_image = example_composite_pattern()
    
    # Пример 4: Анимированный паттерн
    animation_frames = example_animated_pattern()
    
    # Пример 5: Слоистый паттерн
    layered_texture, layered_image = example_layered_pattern()
    
    print("\n" + "=" * 60)
    print("Geometric 3D Patterns Features:")
    print("-" * 40)
    print("1. 20+ geometric pattern types:")
    print("   - Crystal lattices (cubic, hexagonal, diamond)")
    print("   - Minimal surfaces (gyroid, Schwarz P, Neovius)")
    print("   - Mathematical surfaces (helicoid, catenoid, Enneper)")
    print("   - Fractal structures (Menger sponge)")
    print("   - Biological patterns (honeycomb, sphere packing)")
    print("   - Topological structures (knots, Möbius strip)")
    print("   - Quasi-crystals and Fibonacci spirals")
    
    print("\n2. Composite patterns:")
    print("   - Multi-pattern blending")
    print("   - Layered patterns with transparency")
    print("   - Animated pattern morphing")
    
    print("\n3. Visualization methods:")
    print("   - Raycasting with volume rendering")
    print("   - Multi-slice visualization")
    print("   - Maximum intensity projection")
    print("   - Isosurface extraction")
    
    print("\nPerformance optimizations:")
    print("   - Numba JIT compilation for critical functions")
    print("   - Intelligent caching system")
    print("   - Adaptive resolution based on view distance")
    print("   - Parallel processing for large volumes")
    
    print("\nIntegration with game engine:")
    print("""
# Пример интеграции с Unity
class GeometricPatternsInGame:
    def InitializePatterns(self):
        # Инициализация генераторов
        self.pattern_generator = GeometricPatternGenerator3D(seed=world_seed)
        self.composite_generator = CompositePatternGenerator3D(seed=world_seed)
        
        # Предзагрузка часто используемых паттернов
        self.preloaded_patterns = {
            "crystal": self.pattern_generator.generate_pattern(
                GeometricPattern3D.CRYSTAL_LATTICE,
                dimensions=(32, 32, 32),
                params=PatternParameters(scale=2.0, crystal_type="diamond")
            ),
            "honeycomb": self.pattern_generator.generate_pattern(
                GeometricPattern3D.HONEYCOMB,
                dimensions=(32, 32, 32),
                params=PatternParameters(cell_size=0.3)
            )
        }
    
    def UpdateDynamicPatterns(self, player_position):
        # Динамическая генерация паттернов вокруг игрока
        player_chunk = self.WorldToChunk(player_position)
        
        for dx in range(-2, 3):
            for dy in range(-1, 2):
                for dz in range(-2, 3):
                    chunk_coords = (player_chunk[0] + dx,
                                   player_chunk[1] + dy,
                                   player_chunk[2] + dz)
                    
                    # Генерация уникального паттерна для чанка
                    pattern = self.GenerateChunkPattern(chunk_coords)
                    
                    # Применение к геометрии мира
                    self.ApplyPatternToChunk(chunk_coords, pattern)
    """)
    
    print("\nGeometric 3D patterns system ready for procedural world generation!")
