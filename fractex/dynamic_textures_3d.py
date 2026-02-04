# fractex/dynamic_textures_3d.py
"""
Система динамических 3D текстур с физическим моделированием
Потоки лавы, течение воды, дым, огонь, деформации материалов
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from numba import jit, prange, vectorize, float32, float64, int32, int64, complex128
import warnings
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import hashlib

# ----------------------------------------------------------------------
# Структуры данных для динамических текстур
# ----------------------------------------------------------------------

class DynamicTextureType(Enum):
    """Типы динамических 3D текстур"""
    LAVA_FLOW = 1         # Лавовые потоки
    WATER_FLOW = 2        # Течение воды
    SMOKE_PLUME = 3       # Дымовые шлейфы
    FIRE = 4             # Огонь и пламя
    CLOUD_DRIFT = 5      # Дрейф облаков
    SEDIMENT = 6         # Осаждение/эрозия
    DEFORMATION = 7      # Деформации материала
    CHEMICAL_REACTION = 8 # Химические реакции
    BIOLUMINESCENCE = 9  # Биолюминесценция
    MAGMA_CHAMBER = 10   # Движение магмы

@dataclass
class DynamicTextureState:
    """Состояние динамической текстуры в момент времени"""
    time: float = 0.0
    data: np.ndarray = None  # (D, H, W, C)
    velocity_field: Optional[np.ndarray] = None  # Поле скоростей (D, H, W, 3)
    temperature_field: Optional[np.ndarray] = None  # Температурное поле
    pressure_field: Optional[np.ndarray] = None  # Поле давления
    divergence_field: Optional[np.ndarray] = None  # Дивергенция
    
    def __post_init__(self):
        if self.data is None:
            raise ValueError("Data must be provided")
    
    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return self.data.shape
    
    @property
    def dimensions(self) -> Tuple[int, int, int]:
        return self.shape[:3]

@dataclass
class PhysicsParameters:
    """Физические параметры для симуляции"""
    # Общие параметры
    density: float = 1.0          # Плотность
    viscosity: float = 0.1        # Вязкость
    diffusion_rate: float = 0.01  # Коэффициент диффузии
    time_step: float = 0.01       # Шаг по времени
    gravity: Tuple[float, float, float] = (0.0, -9.8, 0.0)
    
    # Температурные параметры
    thermal_conductivity: float = 0.01
    specific_heat: float = 1.0
    temperature_decay: float = 0.99
    
    # Параметры для конкретных типов
    lava_viscosity: float = 100.0
    water_viscosity: float = 0.001
    smoke_buoyancy: float = 2.0
    fire_temperature: float = 1000.0
    
    # Пределы стабильности
    max_velocity: float = 10.0
    max_temperature: float = 2000.0
    max_pressure: float = 100.0
    
    def __post_init__(self):
        self.gravity = np.array(self.gravity, dtype=np.float32)

# ----------------------------------------------------------------------
# Решатели физических уравнений (оптимизированные с Numba)
# ----------------------------------------------------------------------

class NavierStokesSolver3D:
    """Решатель уравнений Навье-Стокса для жидкостей и газов"""
    
    def __init__(self, dimensions: Tuple[int, int, int], params: PhysicsParameters):
        self.dimensions = dimensions
        self.params = params
        
        # Поля
        self.velocity = np.zeros((*dimensions, 3), dtype=np.float32)  # (D, H, W, 3)
        self.velocity_prev = np.zeros_like(self.velocity)
        self.pressure = np.zeros(dimensions, dtype=np.float32)
        self.divergence = np.zeros(dimensions, dtype=np.float32)
        
        # Временные поля
        self.temp_field = np.zeros(dimensions, dtype=np.float32)
        
        # Предварительные вычисления
        self._precompute_laplacian_kernel()
    
    def _precompute_laplacian_kernel(self):
        """Предварительное вычисление ядра лапласиана"""
        self.laplacian_kernel = np.array([
            [[0, 1, 0],
             [1, -6, 1],
             [0, 1, 0]],
            
            [[1, 1, 1],
             [1, -6, 1],
             [1, 1, 1]],
            
            [[0, 1, 0],
             [1, -6, 1],
             [0, 1, 0]]
        ], dtype=np.float32) / 26.0
    
    def step(self, 
             external_forces: Optional[np.ndarray] = None,
             obstacles: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Один шаг симуляции
        
        Алгоритм:
        1. Добавление внешних сил
        2. Адвекция скорости
        3. Диффузия вязкости
        4. Проецирование (соленоидальность)
        """
        # 1. Добавление внешних сил (гравитация и т.д.)
        self._add_external_forces(external_forces)
        
        # 2. Адвекция скорости (перенос скорости полем скорости)
        self._advect_velocity()
        
        # 3. Диффузия вязкости
        self._diffuse_viscosity()
        
        # 4. Проецирование для обеспечения соленоидальности (div(u) = 0)
        self._project()
        
        # 5. Обработка препятствий
        if obstacles is not None:
            self._apply_obstacles(obstacles)
        
        return self.velocity.copy()
    
    def _add_external_forces(self, forces: Optional[np.ndarray]):
        """Добавление внешних сил (гравитация, ветер, etc.)"""
        if forces is not None:
            self.velocity += forces * self.params.time_step
        else:
            # Добавляем гравитацию по умолчанию
            for i in range(self.dimensions[0]):
                for j in range(self.dimensions[1]):
                    for k in range(self.dimensions[2]):
                        self.velocity[i, j, k] += self.params.gravity * self.params.time_step
    
    @jit(nopython=True, parallel=True, cache=True)
    def _advect_velocity(self):
        """Адвекция скорости (полулагранжевым методом)"""
        dim_z, dim_y, dim_x = self.dimensions
        vel_new = np.zeros_like(self.velocity)
        
        for i in prange(dim_z):
            for j in range(dim_y):
                for k in range(dim_x):
                    # Текущая скорость
                    vx, vy, vz = self.velocity[i, j, k]
                    
                    # Координата предыдущего шага (обратное течение)
                    prev_i = i - vz * self.params.time_step * dim_z
                    prev_j = j - vy * self.params.time_step * dim_y
                    prev_k = k - vx * self.params.time_step * dim_x
                    
                    # Обеспечиваем граничные условия (повторение)
                    prev_i = prev_i % dim_z
                    prev_j = prev_j % dim_y
                    prev_k = prev_k % dim_x
                    
                    # Трилинейная интерполяция
                    i0 = int(np.floor(prev_i))
                    j0 = int(np.floor(prev_j))
                    k0 = int(np.floor(prev_k))
                    
                    i1 = (i0 + 1) % dim_z
                    j1 = (j0 + 1) % dim_y
                    k1 = (k0 + 1) % dim_x
                    
                    di = prev_i - i0
                    dj = prev_j - j0
                    dk = prev_k - k0
                    
                    # Интерполяция по всем трем компонентам скорости
                    for comp in range(3):
                        c000 = self.velocity_prev[i0, j0, k0, comp]
                        c001 = self.velocity_prev[i0, j0, k1, comp]
                        c010 = self.velocity_prev[i0, j1, k0, comp]
                        c011 = self.velocity_prev[i0, j1, k1, comp]
                        c100 = self.velocity_prev[i1, j0, k0, comp]
                        c101 = self.velocity_prev[i1, j0, k1, comp]
                        c110 = self.velocity_prev[i1, j1, k0, comp]
                        c111 = self.velocity_prev[i1, j1, k1, comp]
                        
                        c00 = c000 * (1 - dk) + c001 * dk
                        c01 = c010 * (1 - dk) + c011 * dk
                        c10 = c100 * (1 - dk) + c101 * dk
                        c11 = c110 * (1 - dk) + c111 * dk
                        
                        c0 = c00 * (1 - dj) + c01 * dj
                        c1 = c10 * (1 - dj) + c11 * dj
                        
                        vel_new[i, j, k, comp] = c0 * (1 - di) + c1 * di
        
        self.velocity = vel_new
    
    @jit(nopython=True, parallel=True, cache=True)
    def _diffuse_viscosity(self):
        """Диффузия вязкости (явная схема)"""
        if self.params.viscosity <= 0:
            return
        
        dim_z, dim_y, dim_x = self.dimensions
        dt = self.params.time_step
        viscosity = self.params.viscosity
        alpha = dt * viscosity * dim_x * dim_y * dim_z
        
        vel_new = np.zeros_like(self.velocity)
        
        for comp in range(3):
            for i in prange(dim_z):
                for j in range(dim_y):
                    for k in range(dim_x):
                        # 7-точечный шаблон лапласиана
                        center = self.velocity[i, j, k, comp]
                        
                        # Соседи
                        left = self.velocity[i, j, (k-1)%dim_x, comp]
                        right = self.velocity[i, j, (k+1)%dim_x, comp]
                        down = self.velocity[i, (j-1)%dim_y, k, comp]
                        up = self.velocity[i, (j+1)%dim_y, k, comp]
                        back = self.velocity[(i-1)%dim_z, j, k, comp]
                        front = self.velocity[(i+1)%dim_z, j, k, comp]
                        
                        # Лапласиан
                        laplacian = (left + right + down + up + back + front - 6 * center)
                        
                        # Обновление
                        vel_new[i, j, k, comp] = center + alpha * laplacian
        
        self.velocity = vel_new
    
    def _project(self):
        """Проецирование для обеспечения соленоидальности"""
        # Вычисление дивергенции
        self._compute_divergence()
        
        # Решение уравнения Пуассона для давления
        self._solve_pressure_poisson()
        
        # Вычитание градиента давления
        self._subtract_pressure_gradient()
    
    @jit(nopython=True, parallel=True, cache=True)
    def _compute_divergence(self):
        """Вычисление дивергенции поля скоростей"""
        dim_z, dim_y, dim_x = self.dimensions
        
        for i in prange(dim_z):
            for j in range(dim_y):
                for k in range(dim_x):
                    # Градиенты скорости
                    du_dx = (self.velocity[i, j, (k+1)%dim_x, 0] - 
                            self.velocity[i, j, (k-1)%dim_x, 0]) / 2.0
                    dv_dy = (self.velocity[i, (j+1)%dim_y, k, 1] - 
                            self.velocity[i, (j-1)%dim_y, k, 1]) / 2.0
                    dw_dz = (self.velocity[(i+1)%dim_z, j, k, 2] - 
                            self.velocity[(i-1)%dim_z, j, k, 2]) / 2.0
                    
                    self.divergence[i, j, k] = du_dx + dv_dy + dw_dz
    
    @jit(nopython=True, parallel=True, cache=True)
    def _solve_pressure_poisson(self, iterations: int = 20):
        """Решение уравнения Пуассона методом Якоби"""
        dim_z, dim_y, dim_x = self.dimensions
        pressure_new = np.zeros_like(self.pressure)
        
        for _ in range(iterations):
            for i in prange(dim_z):
                for j in range(dim_y):
                    for k in range(dim_x):
                        # Соседи давления
                        p_left = self.pressure[i, j, (k-1)%dim_x]
                        p_right = self.pressure[i, j, (k+1)%dim_x]
                        p_down = self.pressure[i, (j-1)%dim_y, k]
                        p_up = self.pressure[i, (j+1)%dim_y, k]
                        p_back = self.pressure[(i-1)%dim_z, j, k]
                        p_front = self.pressure[(i+1)%dim_z, j, k]
                        
                        # Новое значение давления
                        pressure_new[i, j, k] = (p_left + p_right + p_down + 
                                                p_up + p_back + p_front - 
                                                self.divergence[i, j, k]) / 6.0
            
            self.pressure = pressure_new.copy()
    
    @jit(nopython=True, parallel=True, cache=True)
    def _subtract_pressure_gradient(self):
        """Вычитание градиента давления из скорости"""
        dim_z, dim_y, dim_x = self.dimensions
        
        for i in prange(dim_z):
            for j in range(dim_y):
                for k in range(dim_x):
                    # Градиенты давления
                    dp_dx = (self.pressure[i, j, (k+1)%dim_x] - 
                            self.pressure[i, j, (k-1)%dim_x]) / 2.0
                    dp_dy = (self.pressure[i, (j+1)%dim_y] - 
                            self.pressure[i, (j-1)%dim_y]) / 2.0
                    dp_dz = (self.pressure[(i+1)%dim_z, j, k] - 
                            self.pressure[(i-1)%dim_z, j, k]) / 2.0
                    
                    # Вычитаем градиент давления
                    self.velocity[i, j, k, 0] -= dp_dx
                    self.velocity[i, j, k, 1] -= dp_dy
                    self.velocity[i, j, k, 2] -= dp_dz
    
    @jit(nopython=True, cache=True)
    def _apply_obstacles(self, obstacles: np.ndarray):
        """Применение граничных условий на препятствиях"""
        dim_z, dim_y, dim_x = self.dimensions
        
        for i in range(dim_z):
            for j in range(dim_y):
                for k in range(dim_x):
                    if obstacles[i, j, k] > 0.5:
                        # Обнуляем скорость в препятствиях
                        self.velocity[i, j, k] = 0.0
                        
                        # Отражение скорости от соседей
                        if k > 0 and obstacles[i, j, k-1] < 0.5:
                            self.velocity[i, j, k-1, 0] = 0
                        if k < dim_x-1 and obstacles[i, j, k+1] < 0.5:
                            self.velocity[i, j, k+1, 0] = 0
                        if j > 0 and obstacles[i, j-1, k] < 0.5:
                            self.velocity[i, j-1, k, 1] = 0
                        if j < dim_y-1 and obstacles[i, j+1, k] < 0.5:
                            self.velocity[i, j+1, k, 1] = 0
                        if i > 0 and obstacles[i-1, j, k] < 0.5:
                            self.velocity[i-1, j, k, 2] = 0
                        if i < dim_z-1 and obstacles[i+1, j, k] < 0.5:
                            self.velocity[i+1, j, k, 2] = 0

# ----------------------------------------------------------------------
# Генераторы динамических текстур
# ----------------------------------------------------------------------

class DynamicTextureGenerator3D:
    """Генератор динамических 3D текстур с физическим моделированием"""
    
    def __init__(self, 
                 dimensions: Tuple[int, int, int],
                 texture_type: DynamicTextureType,
                 physics_params: Optional[PhysicsParameters] = None,
                 seed: int = 42):
        
        self.dimensions = dimensions
        self.texture_type = texture_type
        self.seed = seed
        np.random.seed(seed)
        
        # Параметры физики
        if physics_params is None:
            self.params = self._get_default_params(texture_type)
        else:
            self.params = physics_params
        
        # Инициализация симуляторов
        self.navier_stokes = NavierStokesSolver3D(dimensions, self.params)
        self._init_fields()
        
        # История состояний для оптимизации
        self.state_history = deque(maxlen=10)
        self.time = 0.0
        
        # Кэш для оптимизации
        self.cache = {}
    
    def _get_default_params(self, texture_type: DynamicTextureType) -> PhysicsParameters:
        """Получение параметров по умолчанию для типа текстуры"""
        if texture_type == DynamicTextureType.LAVA_FLOW:
            return PhysicsParameters(
                density=2.8,
                viscosity=100.0,  # Высокая вязкость лавы
                diffusion_rate=0.005,
                time_step=0.005,
                gravity=(0.0, -9.8, 0.0),
                thermal_conductivity=0.02,
                temperature_decay=0.995
            )
        elif texture_type == DynamicTextureType.WATER_FLOW:
            return PhysicsParameters(
                density=1.0,
                viscosity=0.001,  # Низкая вязкость воды
                diffusion_rate=0.01,
                time_step=0.01,
                gravity=(0.0, -9.8, 0.0),
                thermal_conductivity=0.001,
                temperature_decay=0.99
            )
        elif texture_type == DynamicTextureType.SMOKE_PLUME:
            return PhysicsParameters(
                density=0.3,  # Легкий дым
                viscosity=0.01,
                diffusion_rate=0.05,
                time_step=0.02,
                gravity=(0.0, 2.0, 0.0),  # Подъемная сила
                thermal_conductivity=0.1,
                temperature_decay=0.95
            )
        elif texture_type == DynamicTextureType.FIRE:
            return PhysicsParameters(
                density=0.5,
                viscosity=0.005,
                diffusion_rate=0.1,
                time_step=0.015,
                gravity=(0.0, 1.5, 0.0),  # Пламя поднимается
                thermal_conductivity=0.2,
                temperature_decay=0.9,
                fire_temperature=1000.0
            )
        elif texture_type == DynamicTextureType.CLOUD_DRIFT:
            return PhysicsParameters(
                density=0.8,
                viscosity=0.02,
                diffusion_rate=0.02,
                time_step=0.01,
                gravity=(0.0, -0.5, 0.0),  # Слабая гравитация
                thermal_conductivity=0.01,
                temperature_decay=0.98
            )
        else:
            return PhysicsParameters()
    
    def _init_fields(self):
        """Инициализация полей в зависимости от типа текстуры"""
        dim_z, dim_y, dim_x = self.dimensions
        
        # Поле плотности/температуры
        self.density_field = np.zeros(self.dimensions, dtype=np.float32)
        
        # Поле цвета
        self.color_field = np.zeros((*self.dimensions, 4), dtype=np.float32)
        
        # Дополнительные поля в зависимости от типа
        if self.texture_type == DynamicTextureType.LAVA_FLOW:
            self._init_lava_fields()
        elif self.texture_type == DynamicTextureType.WATER_FLOW:
            self._init_water_fields()
        elif self.texture_type == DynamicTextureType.SMOKE_PLUME:
            self._init_smoke_fields()
        elif self.texture_type == DynamicTextureType.FIRE:
            self._init_fire_fields()
        elif self.texture_type == DynamicTextureType.CLOUD_DRIFT:
            self._init_cloud_fields()
        
        # Инициализация препятствий
        self.obstacles = self._generate_obstacles()
    
    def _init_lava_fields(self):
        """Инициализация полей для лавовых потоков"""
        dim_z, dim_y, dim_x = self.dimensions
        
        # Создаем источник лавы (горячая точка)
        source_center = (dim_z//2, dim_y//4, dim_x//2)
        source_radius = min(dim_x, dim_z) // 4
        
        # Температурное поле
        self.temperature_field = np.zeros(self.dimensions, dtype=np.float32)
        
        for i in range(dim_z):
            for j in range(dim_y):
                for k in range(dim_x):
                    # Расстояние до центра источника
                    dz = i - source_center[0]
                    dy = j - source_center[1]
                    dx = k - source_center[2]
                    dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    if dist < source_radius:
                        # Горячая лава в источнике
                        temperature = 1200.0 * (1.0 - dist / source_radius)
                        self.temperature_field[i, j, k] = temperature
                        self.density_field[i, j, k] = 0.8
                        
                        # Начальная скорость (вытекание из источника)
                        if dist > 0:
                            self.navier_stokes.velocity[i, j, k, 0] = dx / dist * 0.5
                            self.navier_stokes.velocity[i, j, k, 2] = dz / dist * 0.5
        
        # Цветовое поле (лава)
        self._update_lava_color()
    
    def _init_water_fields(self):
        """Инициализация полей для течения воды"""
        dim_z, dim_y, dim_x = self.dimensions
        
        # Водный резервуар (верхняя часть)
        water_level = dim_y * 3 // 4
        
        for i in range(dim_z):
            for j in range(dim_y):
                for k in range(dim_x):
                    if j > water_level:
                        self.density_field[i, j, k] = 0.9
        
        # Течение (слева направо)
        flow_strength = 1.0
        for i in range(dim_z):
            for j in range(dim_y):
                for k in range(dim_x):
                    if self.density_field[i, j, k] > 0:
                        # Случайные возмущения
                        self.navier_stokes.velocity[i, j, k, 0] = flow_strength + np.random.randn() * 0.1
                        self.navier_stokes.velocity[i, j, k, 1] = np.random.randn() * 0.05
        
        # Цвет воды
        self._update_water_color()
    
    def _init_smoke_fields(self):
        """Инициализация полей для дымового шлейфа"""
        dim_z, dim_y, dim_x = self.dimensions
        
        # Источник дыма (внизу)
        source_x = dim_x // 2
        source_z = dim_z // 2
        
        for i in range(dim_z):
            for j in range(dim_y):
                for k in range(dim_x):
                    # Расстояние до источника
                    dz = i - source_z
                    dx = k - source_x
                    dist = np.sqrt(dx*dx + dz*dz)
                    
                    if dist < 5 and j < 10:
                        # Горячий дым
                        self.density_field[i, j, k] = 0.8
                        
                        # Восходящий поток
                        self.navier_stokes.velocity[i, j, k, 1] = 2.0 + np.random.rand() * 0.5
        
        # Температурное поле
        self.temperature_field = np.zeros(self.dimensions, dtype=np.float32)
        self.temperature_field[:, :10, :] = 500.0  # Горячий источник
        
        # Цвет дыма
        self._update_smoke_color()
    
    def _init_fire_fields(self):
        """Инициализация полей для огня"""
        dim_z, dim_y, dim_x = self.dimensions
        
        # Несколько источников огня
        num_sources = 3
        for _ in range(num_sources):
            source_x = np.random.randint(dim_x//4, 3*dim_x//4)
            source_z = np.random.randint(dim_z//4, 3*dim_z//4)
            
            for i in range(max(0, source_z-2), min(dim_z, source_z+3)):
                for j in range(5):
                    for k in range(max(0, source_x-2), min(dim_x, source_x+3)):
                        # Огонь
                        intensity = np.random.rand() * 0.8 + 0.2
                        self.density_field[i, j, k] = intensity
                        
                        # Турбулентность
                        self.navier_stokes.velocity[i, j, k, 0] = np.random.randn() * 0.2
                        self.navier_stokes.velocity[i, j, k, 1] = 3.0 + np.random.rand() * 1.0
                        self.navier_stokes.velocity[i, j, k, 2] = np.random.randn() * 0.2
        
        # Температурное поле
        self.temperature_field = np.zeros(self.dimensions, dtype=np.float32)
        self.temperature_field[:, :10, :] = self.params.fire_temperature
        
        # Цвет огня
        self._update_fire_color()
    
    def _init_cloud_fields(self):
        """Инициализация полей для облаков"""
        dim_z, dim_y, dim_x = self.dimensions
        
        # Создаем несколько облачных слоев
        for layer in range(3):
            layer_height = dim_y * (layer + 1) // 4
            
            # Генерация облачной текстуры шумом
            for i in range(dim_z):
                for j in range(max(0, layer_height-5), min(dim_y, layer_height+5)):
                    for k in range(dim_x):
                        # Шум Перлина для облаков
                        noise = (np.sin(i*0.1 + layer*10) * 
                                np.cos(k*0.1) * 
                                np.sin(j*0.05 + layer*5))
                        noise = (noise + 1) * 0.5
                        
                        if noise > 0.6:
                            self.density_field[i, j, k] = noise * 0.7
        
        # Легкий ветер
        wind_strength = 0.5
        for i in range(dim_z):
            for j in range(dim_y):
                for k in range(dim_x):
                    if self.density_field[i, j, k] > 0:
                        self.navier_stokes.velocity[i, j, k, 0] = wind_strength
        
        # Цвет облаков
        self._update_cloud_color()
    
    def _generate_obstacles(self) -> np.ndarray:
        """Генерация препятствий для симуляции"""
        dim_z, dim_y, dim_x = self.dimensions
        obstacles = np.zeros(self.dimensions, dtype=np.float32)
        
        if self.texture_type == DynamicTextureType.LAVA_FLOW:
            # Камни и неровности для лавы
            for _ in range(10):
                center_x = np.random.randint(dim_x)
                center_z = np.random.randint(dim_z)
                center_y = np.random.randint(dim_y//2, dim_y)
                radius = np.random.randint(3, 8)
                
                for i in range(max(0, center_z-radius), min(dim_z, center_z+radius)):
                    for j in range(max(0, center_y-radius), min(dim_y, center_y+radius)):
                        for k in range(max(0, center_x-radius), min(dim_x, center_x+radius)):
                            dz = i - center_z
                            dy = j - center_y
                            dx = k - center_x
                            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                            
                            if dist < radius:
                                obstacles[i, j, k] = 1.0
        
        elif self.texture_type == DynamicTextureType.WATER_FLOW:
            # Камни на дне реки
            for _ in range(15):
                center_x = np.random.randint(dim_x)
                center_z = np.random.randint(dim_z)
                center_y = dim_y - np.random.randint(5, 15)
                radius = np.random.randint(2, 5)
                
                for i in range(max(0, center_z-radius), min(dim_z, center_z+radius)):
                    for j in range(max(0, center_y-radius), min(dim_y, center_y+radius)):
                        for k in range(max(0, center_x-radius), min(dim_x, center_x+radius)):
                            dz = i - center_z
                            dy = j - center_y
                            dx = k - center_x
                            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                            
                            if dist < radius:
                                obstacles[i, j, k] = 1.0
        
        return obstacles
    
    def _update_lava_color(self):
        """Обновление цветового поля для лавы"""
        dim_z, dim_y, dim_x = self.dimensions
        
        for i in range(dim_z):
            for j in range(dim_y):
                for k in range(dim_x):
                    density = self.density_field[i, j, k]
                    temperature = self.temperature_field[i, j, k] if hasattr(self, 'temperature_field') else 1000.0
                    
                    if density > 0:
                        # Цвет лавы в зависимости от температуры
                        # Холодная лава: темно-красная, горячая: ярко-желтая
                        t_norm = min(temperature / 1200.0, 1.0)
                        
                        # Интерполяция между цветами
                        cold_color = np.array([0.6, 0.1, 0.0, 1.0])  # Темно-красный
                        hot_color = np.array([1.0, 0.8, 0.1, 1.0])   # Ярко-желтый
                        
                        color = cold_color * (1 - t_norm) + hot_color * t_norm
                        
                        # Яркость в зависимости от плотности
                        color[:3] *= density
                        
                        self.color_field[i, j, k] = color
                    else:
                        self.color_field[i, j, k] = np.array([0.0, 0.0, 0.0, 0.0])
    
    def _update_water_color(self):
        """Обновление цветового поля для воды"""
        dim_z, dim_y, dim_x = self.dimensions
        
        # Цвет воды с глубиной
        for i in range(dim_z):
            for j in range(dim_y):
                for k in range(dim_x):
                    density = self.density_field[i, j, k]
                    
                    if density > 0:
                        # Глубина (чем ниже, тем темнее)
                        depth_factor = 1.0 - (j / dim_y)
                        
                        # Базовый цвет воды
                        base_color = np.array([0.1, 0.3, 0.6, 0.8])  # Синий
                        
                        # Темнее с глубиной
                        color = base_color * (0.7 + 0.3 * depth_factor)
                        
                        # Прозрачность зависит от плотности
                        color[3] = 0.6 + density * 0.3
                        
                        # Пузырьки (случайные яркие точки)
                        if np.random.rand() < 0.001:
                            color[:3] = np.array([1.0, 1.0, 1.0])
                            color[3] = 0.9
                        
                        self.color_field[i, j, k] = color
                    else:
                        self.color_field[i, j, k] = np.array([0.0, 0.0, 0.0, 0.0])
    
    def _update_smoke_color(self):
        """Обновление цветового поля для дыма"""
        dim_z, dim_y, dim_x = self.dimensions
        
        for i in range(dim_z):
            for j in range(dim_y):
                for k in range(dim_x):
                    density = self.density_field[i, j, k]
                    
                    if density > 0:
                        # Цвет дыма: от черного у источника к серому/белому
                        age_factor = min(j / dim_y, 1.0)  # Чем выше, тем "старше" дым
                        
                        # Интерполяция между черным и серым
                        young_color = np.array([0.1, 0.1, 0.1, 0.9])  # Черный дым
                        old_color = np.array([0.7, 0.7, 0.7, 0.3])    # Серый/белый дым
                        
                        color = young_color * (1 - age_factor) + old_color * age_factor
                        
                        # Интенсивность зависит от плотности
                        color[:3] *= density
                        color[3] *= density * 0.8
                        
                        self.color_field[i, j, k] = color
                    else:
                        self.color_field[i, j, k] = np.array([0.0, 0.0, 0.0, 0.0])
    
    def _update_fire_color(self):
        """Обновление цветового поля для огня"""
        dim_z, dim_y, dim_x = self.dimensions
        
        for i in range(dim_z):
            for j in range(dim_y):
                for k in range(dim_x):
                    density = self.density_field[i, j, k]
                    
                    if density > 0:
                        # Цвет огня: ядро - белое/желтое, края - красные
                        temperature = 1.0
                        if hasattr(self, 'temperature_field'):
                            temperature = min(self.temperature_field[i, j, k] / self.params.fire_temperature, 1.0)
                        
                        # Три зоны цвета
                        if temperature > 0.8:
                            color = np.array([1.0, 1.0, 0.7, 0.9])  # Бело-желтый
                        elif temperature > 0.5:
                            color = np.array([1.0, 0.6, 0.1, 0.8])  # Оранжевый
                        else:
                            color = np.array([0.8, 0.2, 0.0, 0.7])  # Красный
                        
                        # Интенсивность
                        intensity = density * temperature
                        color[:3] *= intensity
                        color[3] *= density
                        
                        # Случайные мерцания
                        if np.random.rand() < 0.05:
                            color[:3] *= 1.2
                        
                        self.color_field[i, j, k] = color
                    else:
                        self.color_field[i, j, k] = np.array([0.0, 0.0, 0.0, 0.0])
    
    def _update_cloud_color(self):
        """Обновление цветового поля для облаков"""
        dim_z, dim_y, dim_x = self.dimensions
        
        for i in range(dim_z):
            for j in range(dim_y):
                for k in range(dim_x):
                    density = self.density_field[i, j, k]
                    
                    if density > 0:
                        # Цвет облаков: белый с легкими тенями
                        # Тени на нижней стороне облаков
                        shadow_factor = 1.0
                        
                        # Проверяем плотность ниже (для теней)
                        if j > 0 and self.density_field[i, j-1, k] < density:
                            shadow_factor = 0.85
                        
                        # Базовый белый цвет
                        color = np.array([0.95, 0.95, 0.98, density * 0.8])
                        
                        # Применяем тени
                        color[:3] *= shadow_factor
                        
                        # Легкий синий оттенок для высоких облаков
                        altitude_factor = j / dim_y
                        blue_tint = np.array([0.9, 0.95, 1.0, 1.0])
                        color = color * (1 - altitude_factor*0.3) + blue_tint * (altitude_factor*0.3)
                        
                        self.color_field[i, j, k] = color
                    else:
                        self.color_field[i, j, k] = np.array([0.0, 0.0, 0.0, 0.0])
    
    def update(self, dt: Optional[float] = None) -> DynamicTextureState:
        """
        Обновление динамической текстуры на один шаг
        
        Args:
            dt: Шаг по времени (если None, используется params.time_step)
            
        Returns:
            Состояние текстуры после обновления
        """
        if dt is None:
            dt = self.params.time_step
        
        self.time += dt
        
        # 1. Обновление поля скоростей (Навье-Стокс)
        external_forces = self._compute_external_forces()
        velocity = self.navier_stokes.step(external_forces, self.obstacles)
        
        # 2. Адвекция плотности
        self._advect_density(velocity)
        
        # 3. Диффузия плотности
        self._diffuse_density()
        
        # 4. Источники/стоки (в зависимости от типа)
        self._apply_sources_and_sinks()
        
        # 5. Обновление температуры (если есть)
        if hasattr(self, 'temperature_field'):
            self._update_temperature(velocity)
        
        # 6. Обновление цвета
        self._update_color_field()
        
        # 7. Создание состояния
        state = DynamicTextureState(
            time=self.time,
            data=self.color_field.copy(),
            velocity_field=velocity.copy(),
            temperature_field=self.temperature_field.copy() if hasattr(self, 'temperature_field') else None,
            divergence_field=self.navier_stokes.divergence.copy()
        )
        
        # Сохраняем в историю
        self.state_history.append(state)
        
        return state
    
    def _compute_external_forces(self) -> np.ndarray:
        """Вычисление внешних сил в зависимости от типа текстуры"""
        dim_z, dim_y, dim_x = self.dimensions
        forces = np.zeros((dim_z, dim_y, dim_x, 3), dtype=np.float32)
        
        if self.texture_type == DynamicTextureType.LAVA_FLOW:
            # Гравитация + термическая конвекция
            for i in range(dim_z):
                for j in range(dim_y):
                    for k in range(dim_x):
                        # Горячая лава поднимается
                        if hasattr(self, 'temperature_field'):
                            temp = self.temperature_field[i, j, k]
                            buoyancy = (temp / 1000.0 - 1.0) * 2.0  # Подъемная сила
                            forces[i, j, k, 1] = self.params.gravity[1] + buoyancy
                        else:
                            forces[i, j, k] = self.params.gravity
        
        elif self.texture_type == DynamicTextureType.SMOKE_PLUME:
            # Сильная подъемная сила для дыма
            for i in range(dim_z):
                for j in range(dim_y):
                    for k in range(dim_x):
                        if self.density_field[i, j, k] > 0:
                            # Дым поднимается
                            forces[i, j, k, 1] = self.params.smoke_buoyancy
                            
                            # Случайные турбулентности
                            forces[i, j, k, 0] += np.random.randn() * 0.1
                            forces[i, j, k, 2] += np.random.randn() * 0.1
        
        elif self.texture_type == DynamicTextureType.FIRE:
            # Огонь сильно поднимается + турбулентность
            for i in range(dim_z):
                for j in range(dim_y):
                    for k in range(dim_x):
                        if self.density_field[i, j, k] > 0:
                            # Интенсивная подъемная сила
                            buoyancy = 3.0 + np.random.rand() * 2.0
                            forces[i, j, k, 1] = buoyancy
                            
                            # Вихревые движения
                            angle = self.time * 5.0 + i * 0.1 + k * 0.1
                            forces[i, j, k, 0] += np.sin(angle) * 0.5
                            forces[i, j, k, 2] += np.cos(angle) * 0.5
        
        elif self.texture_type == DynamicTextureType.CLOUD_DRIFT:
            # Легкий ветер + случайные движения
            wind_strength = 0.3
            for i in range(dim_z):
                for j in range(dim_y):
                    for k in range(dim_x):
                        forces[i, j, k, 0] = wind_strength
                        
                        # Слабые вертикальные движения
                        if self.density_field[i, j, k] > 0:
                            forces[i, j, k, 1] = np.random.randn() * 0.05
        
        else:
            # По умолчанию только гравитация
            for i in range(dim_z):
                for j in range(dim_y):
                    for k in range(dim_x):
                        forces[i, j, k] = self.params.gravity
        
        return forces
    
    @jit(nopython=True, parallel=True, cache=True)
    def _advect_density(self, velocity: np.ndarray):
        """Адвекция плотности полем скоростей (полулагранжевым методом)"""
        dim_z, dim_y, dim_x = self.dimensions
        density_new = np.zeros_like(self.density_field)
        
        for i in prange(dim_z):
            for j in range(dim_y):
                for k in range(dim_x):
                    # Текущая скорость
                    vx, vy, vz = velocity[i, j, k]
                    
                    # Координата предыдущего шага
                    prev_i = i - vz * self.params.time_step * dim_z
                    prev_j = j - vy * self.params.time_step * dim_y
                    prev_k = k - vx * self.params.time_step * dim_x
                    
                    # Граничные условия
                    prev_i = max(0, min(dim_z-1, prev_i))
                    prev_j = max(0, min(dim_y-1, prev_j))
                    prev_k = max(0, min(dim_x-1, prev_k))
                    
                    # Трилинейная интерполяция плотности
                    i0 = int(np.floor(prev_i))
                    j0 = int(np.floor(prev_j))
                    k0 = int(np.floor(prev_k))
                    
                    i1 = min(i0 + 1, dim_z - 1)
                    j1 = min(j0 + 1, dim_y - 1)
                    k1 = min(k0 + 1, dim_x - 1)
                    
                    di = prev_i - i0
                    dj = prev_j - j0
                    dk = prev_k - k0
                    
                    c000 = self.density_field[i0, j0, k0]
                    c001 = self.density_field[i0, j0, k1]
                    c010 = self.density_field[i0, j1, k0]
                    c011 = self.density_field[i0, j1, k1]
                    c100 = self.density_field[i1, j0, k0]
                    c101 = self.density_field[i1, j0, k1]
                    c110 = self.density_field[i1, j1, k0]
                    c111 = self.density_field[i1, j1, k1]
                    
                    c00 = c000 * (1 - dk) + c001 * dk
                    c01 = c010 * (1 - dk) + c011 * dk
                    c10 = c100 * (1 - dk) + c101 * dk
                    c11 = c110 * (1 - dk) + c111 * dk
                    
                    c0 = c00 * (1 - dj) + c01 * dj
                    c1 = c10 * (1 - dj) + c11 * dj
                    
                    density_new[i, j, k] = c0 * (1 - di) + c1 * di
        
        self.density_field = density_new
    
    @jit(nopython=True, parallel=True, cache=True)
    def _diffuse_density(self):
        """Диффузия плотности"""
        if self.params.diffusion_rate <= 0:
            return
        
        dim_z, dim_y, dim_x = self.dimensions
        diffusion = self.params.diffusion_rate
        dt = self.params.time_step
        alpha = dt * diffusion * dim_x * dim_y * dim_z
        
        density_new = np.zeros_like(self.density_field)
        
        for i in prange(dim_z):
            for j in range(dim_y):
                for k in range(dim_x):
                    center = self.density_field[i, j, k]
                    
                    # Соседи
                    left = self.density_field[i, j, (k-1)%dim_x]
                    right = self.density_field[i, j, (k+1)%dim_x]
                    down = self.density_field[i, (j-1)%dim_y, k]
                    up = self.density_field[i, (j+1)%dim_y, k]
                    back = self.density_field[(i-1)%dim_z, j, k]
                    front = self.density_field[(i+1)%dim_z, j, k]
                    
                    # Лапласиан
                    laplacian = (left + right + down + up + back + front - 6 * center)
                    
                    # Обновление
                    density_new[i, j, k] = center + alpha * laplacian
        
        self.density_field = np.clip(density_new, 0, 1)
    
    def _apply_sources_and_sinks(self):
        """Применение источников и стоков в зависимости от типа"""
        dim_z, dim_y, dim_x = self.dimensions
        
        if self.texture_type == DynamicTextureType.LAVA_FLOW:
            # Источник лавы продолжает извергаться
            source_center = (dim_z//2, dim_y//4, dim_x//2)
            source_radius = min(dim_x, dim_z) // 6
            
            for i in range(max(0, source_center[0]-source_radius), 
                          min(dim_z, source_center[0]+source_radius)):
                for j in range(max(0, source_center[1]-source_radius),
                              min(dim_y, source_center[1]+source_radius)):
                    for k in range(max(0, source_center[2]-source_radius),
                                  min(dim_x, source_center[2]+source_radius)):
                        dz = i - source_center[0]
                        dy = j - source_center[1]
                        dx = k - source_center[2]
                        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                        
                        if dist < source_radius:
                            # Добавляем новую лаву
                            self.density_field[i, j, k] = min(1.0, self.density_field[i, j, k] + 0.1)
                            
                            if hasattr(self, 'temperature_field'):
                                self.temperature_field[i, j, k] = 1200.0
        
        elif self.texture_type == DynamicTextureType.SMOKE_PLUME:
            # Постоянный источник дыма
            source_x = dim_x // 2
            source_z = dim_z // 2
            
            for i in range(max(0, source_z-3), min(dim_z, source_z+4)):
                for j in range(5):
                    for k in range(max(0, source_x-3), min(dim_x, source_x+4)):
                        dz = i - source_z
                        dx = k - source_x
                        dist = np.sqrt(dx*dx + dz*dz)
                        
                        if dist < 3:
                            # Новый дым
                            self.density_field[i, j, k] = min(1.0, self.density_field[i, j, k] + 0.2)
                            
                            if hasattr(self, 'temperature_field'):
                                self.temperature_field[i, j, k] = 500.0
        
        elif self.texture_type == DynamicTextureType.FIRE:
            # Постоянный источник огня
            for _ in range(2):  # Несколько новых источников
                source_x = np.random.randint(dim_x//4, 3*dim_x//4)
                source_z = np.random.randint(dim_z//4, 3*dim_z//4)
                
                for i in range(max(0, source_z-2), min(dim_z, source_z+3)):
                    for j in range(5):
                        for k in range(max(0, source_x-2), min(dim_x, source_x+3)):
                            intensity = np.random.rand() * 0.5 + 0.3
                            self.density_field[i, j, k] = min(1.0, self.density_field[i, j, k] + intensity * 0.1)
                            
                            if hasattr(self, 'temperature_field'):
                                self.temperature_field[i, j, k] = self.params.fire_temperature
        
        # Общее затухание
        self.density_field *= 0.995
    
    def _update_temperature(self, velocity: np.ndarray):
        """Обновление температурного поля"""
        dim_z, dim_y, dim_x = self.dimensions
        
        # Адвекция температуры
        temp_new = np.zeros_like(self.temperature_field)
        
        for i in range(dim_z):
            for j in range(dim_y):
                for k in range(dim_x):
                    vx, vy, vz = velocity[i, j, k]
                    
                    # Полулагранжевая адвекция
                    prev_i = i - vz * self.params.time_step * dim_z
                    prev_j = j - vy * self.params.time_step * dim_y
                    prev_k = k - vx * self.params.time_step * dim_x
                    
                    prev_i = max(0, min(dim_z-1, prev_i))
                    prev_j = max(0, min(dim_y-1, prev_j))
                    prev_k = max(0, min(dim_x-1, prev_k))
                    
                    # Интерполяция
                    i0 = int(np.floor(prev_i))
                    j0 = int(np.floor(prev_j))
                    k0 = int(np.floor(prev_k))
                    
                    i1 = min(i0 + 1, dim_z - 1)
                    j1 = min(j0 + 1, dim_y - 1)
                    k1 = min(k0 + 1, dim_x - 1)
                    
                    di = prev_i - i0
                    dj = prev_j - j0
                    dk = prev_k - k0
                    
                    c000 = self.temperature_field[i0, j0, k0]
                    c001 = self.temperature_field[i0, j0, k1]
                    c010 = self.temperature_field[i0, j1, k0]
                    c011 = self.temperature_field[i0, j1, k1]
                    c100 = self.temperature_field[i1, j0, k0]
                    c101 = self.temperature_field[i1, j0, k1]
                    c110 = self.temperature_field[i1, j1, k0]
                    c111 = self.temperature_field[i1, j1, k1]
                    
                    c00 = c000 * (1 - dk) + c001 * dk
                    c01 = c010 * (1 - dk) + c011 * dk
                    c10 = c100 * (1 - dk) + c101 * dk
                    c11 = c110 * (1 - dk) + c111 * dk
                    
                    c0 = c00 * (1 - dj) + c01 * dj
                    c1 = c10 * (1 - dj) + c11 * dj
                    
                    temp_new[i, j, k] = c0 * (1 - di) + c1 * di
        
        # Теплопроводность
        if self.params.thermal_conductivity > 0:
            for i in range(1, dim_z-1):
                for j in range(1, dim_y-1):
                    for k in range(1, dim_x-1):
                        laplacian = (temp_new[i-1, j, k] + temp_new[i+1, j, k] +
                                    temp_new[i, j-1, k] + temp_new[i, j+1, k] +
                                    temp_new[i, j, k-1] + temp_new[i, j, k+1] -
                                    6 * temp_new[i, j, k])
                        
                        temp_new[i, j, k] += self.params.thermal_conductivity * laplacian
        
        # Охлаждение/затухание
        temp_new *= self.params.temperature_decay
        
        # Ограничение температуры
        self.temperature_field = np.clip(temp_new, 0, self.params.max_temperature)
    
    def _update_color_field(self):
        """Обновление цветового поля на основе текущего состояния"""
        if self.texture_type == DynamicTextureType.LAVA_FLOW:
            self._update_lava_color()
        elif self.texture_type == DynamicTextureType.WATER_FLOW:
            self._update_water_color()
        elif self.texture_type == DynamicTextureType.SMOKE_PLUME:
            self._update_smoke_color()
        elif self.texture_type == DynamicTextureType.FIRE:
            self._update_fire_color()
        elif self.texture_type == DynamicTextureType.CLOUD_DRIFT:
            self._update_cloud_color()
    
    def get_state(self, time: Optional[float] = None) -> DynamicTextureState:
        """
        Получение состояния текстуры в определенное время
        (интерполяция между сохраненными состояниями)
        """
        if time is None:
            time = self.time
        
        # Если запрашиваем текущее время
        if abs(time - self.time) < self.params.time_step * 0.5:
            return DynamicTextureState(
                time=self.time,
                data=self.color_field.copy(),
                velocity_field=self.navier_stokes.velocity.copy(),
                temperature_field=self.temperature_field.copy() if hasattr(self, 'temperature_field') else None
            )
        
        # Ищем два ближайших состояния для интерполяции
        states = list(self.state_history)
        if len(states) < 2:
            return states[-1] if states else self.update(0)
        
        # Сортируем по времени
        states.sort(key=lambda s: s.time)
        
        # Находим состояния до и после запрашиваемого времени
        prev_state = None
        next_state = None
        
        for state in states:
            if state.time <= time:
                prev_state = state
            else:
                next_state = state
                break
        
        # Если время вне диапазона
        if prev_state is None:
            return next_state
        if next_state is None:
            return prev_state
        
        # Линейная интерполяция
        t = (time - prev_state.time) / (next_state.time - prev_state.time)
        
        # Интерполяция данных
        data_interp = prev_state.data * (1 - t) + next_state.data * t
        
        # Создаем интерполированное состояние
        return DynamicTextureState(
            time=time,
            data=data_interp,
            velocity_field=prev_state.velocity_field if prev_state.velocity_field is not None else None,
            temperature_field=prev_state.temperature_field if prev_state.temperature_field is not None else None
        )
    
    def reset(self):
        """Сброс симуляции в начальное состояние"""
        self._init_fields()
        self.time = 0.0
        self.state_history.clear()
        self.cache.clear()

# ----------------------------------------------------------------------
# Система потоковых динамических текстур для больших миров
# ----------------------------------------------------------------------

class StreamingDynamicTextures:
    """Управление множеством динамических текстур в потоковом режиме"""
    
    def __init__(self, 
                 chunk_size: Tuple[int, int, int] = (32, 32, 32),
                 max_active_chunks: int = 8,
                 physics_params: Optional[PhysicsParameters] = None):
        
        self.chunk_size = chunk_size
        self.max_active_chunks = max_active_chunks
        self.physics_params = physics_params or PhysicsParameters()
        
        # Активные чанки
        self.active_chunks = {}  # (cx, cy, cz) -> DynamicTextureGenerator3D
        
        # Кэш состояний
        self.state_cache = {}
        
        # Приоритетная очередь для обновления
        self.update_queue = []
        
        # Статистика
        self.stats = {
            'updates': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'chunks_created': 0,
            'chunks_evicted': 0
        }
    
    def request_chunk(self, 
                     chunk_coords: Tuple[int, int, int],
                     texture_type: DynamicTextureType,
                     priority: float = 1.0) -> Optional[DynamicTextureState]:
        """
        Запрос состояния чанка динамической текстуры
        
        Args:
            chunk_coords: Координаты чанка
            texture_type: Тип текстуры
            priority: Приоритет (выше = важнее)
            
        Returns:
            Состояние чанка или None если не готово
        """
        chunk_key = (*chunk_coords, texture_type.value)
        
        # Проверяем кэш
        if chunk_key in self.state_cache:
            self.stats['cache_hits'] += 1
            return self.state_cache[chunk_key]
        
        self.stats['cache_misses'] += 1
        
        # Проверяем активные чанки
        if chunk_coords in self.active_chunks:
            generator = self.active_chunks[chunk_coords]
            
            # Обновляем приоритет
            self._update_priority(chunk_coords, priority)
            
            # Получаем текущее состояние
            state = generator.get_state()
            self.state_cache[chunk_key] = state
            
            return state
        
        # Если не активно, создаем новый чанк
        if len(self.active_chunks) < self.max_active_chunks:
            self._create_chunk(chunk_coords, texture_type, priority)
        
        return None
    
    def update_all(self, dt: float):
        """Обновление всех активных чанков"""
        # Сортируем по приоритету
        sorted_chunks = sorted(self.update_queue, key=lambda x: x[0], reverse=True)
        
        # Обновляем только верхние N чанков для производительности
        chunks_to_update = min(len(sorted_chunks), self.max_active_chunks // 2)
        
        for i in range(chunks_to_update):
            priority, chunk_coords = sorted_chunks[i]
            
            if chunk_coords in self.active_chunks:
                generator = self.active_chunks[chunk_coords]
                new_state = generator.update(dt)
                
                # Обновляем кэш
                chunk_key = (*chunk_coords, generator.texture_type.value)
                self.state_cache[chunk_key] = new_state
        
        self.stats['updates'] += 1
        
        # Очистка устаревших данных
        self._cleanup_old_data()
    
    def _create_chunk(self, 
                     chunk_coords: Tuple[int, int, int],
                     texture_type: DynamicTextureType,
                     priority: float):
        """Создание нового чанка динамической текстуры"""
        # Если достигнут лимит, вытесняем самый низкоприоритетный чанк
        if len(self.active_chunks) >= self.max_active_chunks:
            self._evict_lowest_priority_chunk()
        
        # Создаем генератор
        generator = DynamicTextureGenerator3D(
            dimensions=self.chunk_size,
            texture_type=texture_type,
            physics_params=self.physics_params,
            seed=self._chunk_seed(chunk_coords, texture_type)
        )
        
        # Инициализация в зависимости от соседних чанков
        self._initialize_from_neighbors(chunk_coords, generator)
        
        # Добавляем в активные
        self.active_chunks[chunk_coords] = generator
        self._update_priority(chunk_coords, priority)
        
        self.stats['chunks_created'] += 1
    
    def _evict_lowest_priority_chunk(self):
        """Вытеснение чанка с самым низким приоритетом"""
        if not self.update_queue:
            return
        
        # Находим чанк с минимальным приоритетом
        min_priority = float('inf')
        chunk_to_evict = None
        
        for priority, chunk_coords in self.update_queue:
            if priority < min_priority:
                min_priority = priority
                chunk_to_evict = chunk_coords
        
        if chunk_to_evict and chunk_to_evict in self.active_chunks:
            # Сохраняем финальное состояние в кэш
            generator = self.active_chunks[chunk_to_evict]
            final_state = generator.get_state()
            chunk_key = (*chunk_to_evict, generator.texture_type.value)
            self.state_cache[chunk_key] = final_state
            
            # Удаляем из активных
            del self.active_chunks[chunk_to_evict]
            
            # Удаляем из очереди обновлений
            self.update_queue = [(p, c) for p, c in self.update_queue if c != chunk_to_evict]
            
            self.stats['chunks_evicted'] += 1
    
    def _update_priority(self, chunk_coords: Tuple[int, int, int], new_priority: float):
        """Обновление приоритета чанка"""
        # Удаляем старый приоритет
        self.update_queue = [(p, c) for p, c in self.update_queue if c != chunk_coords]
        
        # Добавляем новый
        self.update_queue.append((new_priority, chunk_coords))
    
    def _chunk_seed(self, chunk_coords: Tuple[int, int, int], texture_type: DynamicTextureType) -> int:
        """Генерация seed для чанка"""
        seed_str = f"{chunk_coords[0]}_{chunk_coords[1]}_{chunk_coords[2]}_{texture_type.value}"
        return int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2**31)
    
    def _initialize_from_neighbors(self, 
                                  chunk_coords: Tuple[int, int, int],
                                  generator: DynamicTextureGenerator3D):
        """Инициализация чанка на основе соседних"""
        # Здесь могла бы быть логика передачи состояния между чанками
        # Например, течение воды из одного чанка в другой
        
        # Пока просто оставляем стандартную инициализацию
        pass
    
    def _cleanup_old_data(self):
        """Очистка устаревших данных из кэша"""
        max_cache_size = 100
        
        if len(self.state_cache) > max_cache_size:
            # Удаляем самые старые записи
            keys_to_remove = list(self.state_cache.keys())[:len(self.state_cache) - max_cache_size]
            for key in keys_to_remove:
                del self.state_cache[key]
    
    def get_stats(self) -> Dict:
        """Получение статистики системы"""
        return {
            **self.stats,
            'active_chunks': len(self.active_chunks),
            'cached_states': len(self.state_cache),
            'queue_size': len(self.update_queue)
        }

# ----------------------------------------------------------------------
# Визуализация динамических текстур
# ----------------------------------------------------------------------

class DynamicTextureVisualizer:
    """Визуализатор для динамических 3D текстур"""
    
    def __init__(self, render_method: str = "raycast"):
        self.render_method = render_method
        
    def render_state(self, 
                    state: DynamicTextureState,
                    camera_pos: Tuple[float, float, float] = (0.5, 0.5, 2.0),
                    camera_target: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                    image_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """
        Рендеринг состояния динамической текстуры
        
        Args:
            state: Состояние текстуры
            camera_pos: Позиция камеры
            camera_target: Цель камеры
            image_size: Размер изображения
            
        Returns:
            2D изображение (H, W, 4) RGBA
        """
        if self.render_method == "raycast":
            return self._raycast_state(state, camera_pos, camera_target, image_size)
        elif self.render_method == "mip":
            return self._mip_state(state, image_size)
        elif self.render_method == "slice":
            return self._slice_state(state, image_size)
        else:
            raise ValueError(f"Unknown render method: {self.render_method}")
    
    def _raycast_state(self, 
                      state: DynamicTextureState,
                      camera_pos: Tuple[float, float, float],
                      camera_target: Tuple[float, float, float],
                      image_size: Tuple[int, int]) -> np.ndarray:
        """Рейкастинг для динамической текстуры"""
        width, height = image_size
        image = np.zeros((height, width, 4), dtype=np.float32)
        
        # Базис камеры
        camera_dir = np.array(camera_target) - np.array(camera_pos)
        camera_dir = camera_dir / np.linalg.norm(camera_dir)
        
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(camera_dir, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, camera_dir)
        
        # FOV
        fov = 60.0
        aspect = width / height
        half_height = np.tan(np.radians(fov) / 2.0)
        half_width = aspect * half_height
        
        dim_z, dim_y, dim_x, channels = state.shape
        
        # Параметры рендеринга
        max_steps = 128
        step_size = 1.0 / max(dim_x, dim_y, dim_z)
        
        for y in range(height):
            for x in range(width):
                # Направление луча
                u = (2.0 * x / width - 1.0) * half_width
                v = (1.0 - 2.0 * y / height) * half_height
                
                ray_dir = camera_dir + u * right + v * up
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # Стартовая позиция
                ray_pos = np.array(camera_pos, dtype=np.float32)
                
                # Интегрирование вдоль луча
                color = np.zeros(4, dtype=np.float32)
                
                for step in range(max_steps):
                    # Проверяем границы
                    if (ray_pos[0] < 0 or ray_pos[0] >= 1 or
                        ray_pos[1] < 0 or ray_pos[1] >= 1 or
                        ray_pos[2] < 0 or ray_pos[2] >= 1):
                        break
                    
                    # Трилинейная интерполяция
                    fx = ray_pos[0] * (dim_x - 1)
                    fy = ray_pos[1] * (dim_y - 1)
                    fz = ray_pos[2] * (dim_z - 1)
                    
                    ix0 = int(np.floor(fx))
                    iy0 = int(np.floor(fy))
                    iz0 = int(np.floor(fz))
                    
                    ix1 = min(ix0 + 1, dim_x - 1)
                    iy1 = min(iy0 + 1, dim_y - 1)
                    iz1 = min(iz0 + 1, dim_z - 1)
                    
                    dx = fx - ix0
                    dy = fy - iy0
                    dz = fz - iz0
                    
                    # Интерполяция для каждого канала
                    sample = np.zeros(channels, dtype=np.float32)
                    
                    for c in range(channels):
                        c000 = state.data[iz0, iy0, ix0, c]
                        c001 = state.data[iz0, iy0, ix1, c]
                        c010 = state.data[iz0, iy1, ix0, c]
                        c011 = state.data[iz0, iy1, ix1, c]
                        c100 = state.data[iz1, iy0, ix0, c]
                        c101 = state.data[iz1, iy0, ix1, c]
                        c110 = state.data[iz1, iy1, ix0, c]
                        c111 = state.data[iz1, iy1, ix1, c]
                        
                        c00 = c000 * (1 - dx) + c001 * dx
                        c01 = c010 * (1 - dx) + c011 * dx
                        c10 = c100 * (1 - dx) + c101 * dx
                        c11 = c110 * (1 - dx) + c111 * dx
                        
                        c0 = c00 * (1 - dy) + c01 * dy
                        c1 = c10 * (1 - dy) + c11 * dy
                        
                        sample[c] = c0 * (1 - dz) + c1 * dz
                    
                    # Фронтально-заднее смешивание
                    alpha = sample[3]
                    color = color + (1.0 - color[3]) * alpha * sample
                    
                    # Если полностью непрозрачный
                    if color[3] >= 0.99:
                        break
                    
                    # Двигаем луч
                    ray_pos += ray_dir * step_size
                
                image[y, x] = color
        
        return np.clip(image, 0, 1)
    
    def _mip_state(self, state: DynamicTextureState, image_size: Tuple[int, int]) -> np.ndarray:
        """MIP (Maximum Intensity Projection) рендеринг"""
        width, height = image_size
        dim_z, dim_y, dim_x, channels = state.shape
        
        # Выбираем ось проекции (по умолчанию Z)
        axis = 'z'
        
        if axis == 'x':
            mip = np.max(state.data, axis=2)
        elif axis == 'y':
            mip = np.max(state.data, axis=1)
        else:  # 'z'
            mip = np.max(state.data, axis=0)
        
        # Масштабируем до нужного размера
        from scipy import ndimage
        
        if mip.shape[0] != height or mip.shape[1] != width:
            scale_y = height / mip.shape[0]
            scale_x = width / mip.shape[1]
            
            mip_resized = np.zeros((height, width, channels), dtype=np.float32)
            
            for c in range(channels):
                mip_resized[:, :, c] = ndimage.zoom(mip[:, :, c], (scale_y, scale_x), order=1)
            
            mip = mip_resized
        
        return mip
    
    def _slice_state(self, state: DynamicTextureState, image_size: Tuple[int, int]) -> np.ndarray:
        """Рендеринг 2D среза"""
        width, height = image_size
        dim_z, dim_y, dim_x, channels = state.shape
        
        # Серединный срез по оси Z
        slice_idx = dim_z // 2
        slice_data = state.data[slice_idx, :, :, :]
        
        # Масштабируем
        from scipy import ndimage
        
        if slice_data.shape[0] != height or slice_data.shape[1] != width:
            scale_y = height / slice_data.shape[0]
            scale_x = width / slice_data.shape[1]
            
            slice_resized = np.zeros((height, width, channels), dtype=np.float32)
            
            for c in range(channels):
                slice_resized[:, :, c] = ndimage.zoom(slice_data[:, :, c], (scale_y, scale_x), order=1)
            
            slice_data = slice_resized
        
        return slice_data

# ----------------------------------------------------------------------
# Примеры использования
# ----------------------------------------------------------------------

def example_lava_flow():
    """Пример лавового потока"""
    
    print("Lava flow example...")
    
    # Создаем генератор лавы
    generator = DynamicTextureGenerator3D(
        dimensions=(48, 48, 48),
        texture_type=DynamicTextureType.LAVA_FLOW,
        seed=42
    )
    
    states = []
    
    # Симуляция нескольких шагов
    print("Simulating lava flow...")
    for i in range(30):
        state = generator.update()
        states.append(state)
        
        if i % 10 == 0:
            print(f"  Step {i}, time: {state.time:.3f}")
    
    print(f"Generated {len(states)} states")
    
    # Визуализация
    visualizer = DynamicTextureVisualizer(render_method="slice")
    
    # Рендерим последнее состояние
    last_state = states[-1]
    image = visualizer.render_state(
        last_state,
        camera_pos=(0.5, 0.5, 1.5),
        camera_target=(0.5, 0.5, 0.5),
        image_size=(512, 512)
    )
    
    print(f"Rendered image shape: {image.shape}")
    
    return states, image

def example_water_flow():
    """Пример течения воды"""
    
    print("\nWater flow example...")
    
    generator = DynamicTextureGenerator3D(
        dimensions=(64, 32, 64),  # Шире, но ниже (как река)
        texture_type=DynamicTextureType.WATER_FLOW,
        seed=123
    )
    
    states = []
    
    # Симуляция
    print("Simulating water flow...")
    for i in range(50):
        state = generator.update()
        states.append(state)
    
    print(f"Generated {len(states)} states")
    
    # Анимация
    visualizer = DynamicTextureVisualizer(render_method="raycast")
    
    # Рендерим несколько кадров
    frames = []
    for i in range(0, len(states), 5):
        frame = visualizer.render_state(
            states[i],
            camera_pos=(0.5, 0.7, 1.8),
            camera_target=(0.5, 0.3, 0.2),
            image_size=(256, 256)
        )
        frames.append(frame)
    
    print(f"Rendered {len(frames)} frames")
    
    return states, frames

def example_fire_simulation():
    """Пример симуляции огня"""
    
    print("\nFire simulation example...")
    
    generator = DynamicTextureGenerator3D(
        dimensions=(32, 48, 32),
        texture_type=DynamicTextureType.FIRE,
        seed=456
    )
    
    states = []
    
    # Быстрая симуляция огня
    print("Simulating fire...")
    for i in range(40):
        state = generator.update(dt=0.02)  # Больший шаг для скорости
        states.append(state)
    
    print(f"Generated {len(states)} states")
    
    # Визуализация с подсветкой
    visualizer = DynamicTextureVisualizer(render_method="raycast")
    
    images = []
    for i in range(0, len(states), 2):
        img = visualizer.render_state(
            states[i],
            camera_pos=(0.5, 0.3, 1.0),
            camera_target=(0.5, 0.2, 0.0),
            image_size=(256, 256)
        )
        images.append(img)
    
    print(f"Rendered {len(images)} fire frames")
    
    return states, images

def example_streaming_system():
    """Пример потоковой системы динамических текстур"""
    
    print("\nStreaming dynamic textures system example...")
    
    # Создаем потоковую систему
    streamer = StreamingDynamicTextures(
        chunk_size=(32, 32, 32),
        max_active_chunks=4,
        physics_params=PhysicsParameters(
            viscosity=0.01,
            diffusion_rate=0.02,
            time_step=0.01
        )
    )
    
    # Запрашиваем несколько чанков с водой
    water_chunks = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    
    print("Requesting water chunks...")
    states = []
    
    for coords in water_chunks:
        state = streamer.request_chunk(
            coords,
            DynamicTextureType.WATER_FLOW,
            priority=1.0
        )
        
        if state is not None:
            states.append((coords, state))
            print(f"  Chunk {coords}: ready")
        else:
            print(f"  Chunk {coords}: generating...")
    
    # Обновляем систему несколько раз
    print("\nUpdating streaming system...")
    for i in range(10):
        streamer.update_all(dt=0.01)
        
        if i % 2 == 0:
            stats = streamer.get_stats()
            print(f"  Step {i}: {stats['active_chunks']} active chunks")
    
    # Проверяем чанки после обновления
    print("\nChecking chunks after updates...")
    for coords in water_chunks:
        state = streamer.request_chunk(coords, DynamicTextureType.WATER_FLOW)
        if state is not None:
            print(f"  Chunk {coords}: ready (time: {state.time:.3f})")
    
    stats = streamer.get_stats()
    print(f"\nFinal stats: {stats}")
    
    return streamer, states

def example_cloud_drift():
    """Пример дрейфа облаков"""
    
    print("\nCloud drift example...")
    
    generator = DynamicTextureGenerator3D(
        dimensions=(64, 32, 64),
        texture_type=DynamicTextureType.CLOUD_DRIFT,
        seed=789
    )
    
    states = []
    
    print("Simulating cloud drift...")
    for i in range(60):
        state = generator.update()
        states.append(state)
    
    print(f"Generated {len(states)} cloud states")
    
    # Визуализация с неба
    visualizer = DynamicTextureVisualizer(render_method="raycast")
    
    images = []
    for i in range(0, len(states), 3):
        img = visualizer.render_state(
            states[i],
            camera_pos=(0.5, 0.8, 1.5),  # Смотрим сверху вниз
            camera_target=(0.5, 0.2, 0.5),
            image_size=(512, 256)
        )
        images.append(img)
    
    print(f"Rendered {len(images)} cloud frames")
    
    return states, images

if __name__ == "__main__":
    print("Dynamic 3D Textures System")
    print("=" * 60)
    
    # Пример 1: Лавовые потоки
    lava_states, lava_image = example_lava_flow()
    
    # Пример 2: Течение воды
    water_states, water_frames = example_water_flow()
    
    # Пример 3: Огонь
    fire_states, fire_images = example_fire_simulation()
    
    # Пример 4: Потоковая система
    streamer, streamed_states = example_streaming_system()
    
    # Пример 5: Дрейф облаков
    cloud_states, cloud_images = example_cloud_drift()
    
    print("\n" + "=" * 60)
    print("Dynamic 3D Textures Features:")
    print("-" * 40)
    print("1. Physics-based simulation (Navier-Stokes)")
    print("2. Multiple dynamic texture types:")
    print("   - Lava flows with temperature")
    print("   - Water flow with obstacles")
    print("   - Fire with turbulence")
    print("   - Smoke plumes with buoyancy")
    print("   - Cloud drift with wind")
    print("3. Streaming system for large worlds")
    print("4. Real-time visualization methods")
    print("5. Optimized with Numba JIT compilation")
    
    print("\nPerformance considerations:")
    print("- Smaller volumes for real-time (32^3 - 64^3)")
    print("- Adjust time step for stability")
    print("- Use simplified physics when possible")
    print("- Implement level-of-detail for distant effects")
    print("- Consider GPU acceleration for production")
    
    print("\nIntegration with game engine:")
    print("""
# Пример интеграции
class GameDynamicTextures:
    def __init__(self):
        self.streamer = StreamingDynamicTextures(
            chunk_size=(32, 32, 32),
            max_active_chunks=16
        )
        
    def update(self, dt, player_position):
        # Обновляем чанки рядом с игроком
        player_chunk = self._world_to_chunk(player_position)
        
        for dx in range(-2, 3):
            for dy in range(-1, 2):
                for dz in range(-2, 3):
                    chunk_coords = (
                        player_chunk[0] + dx,
                        player_chunk[1] + dy,
                        player_chunk[2] + dz
                    )
                    
                    # Запрашиваем чанк
                    priority = 1.0 / (dx*dx + dy*dy + dz*dz + 1)
                    state = self.streamer.request_chunk(
                        chunk_coords,
                        DynamicTextureType.WATER_FLOW,
                        priority
                    )
                    
                    if state:
                        self._render_chunk(chunk_coords, state)
        
        # Обновляем симуляцию
        self.streamer.update_all(dt)
    """)
    
    print("\nDynamic 3D textures system ready for interactive worlds!")
