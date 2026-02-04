# fractex/volume_scattering.py
"""
Система объемного рассеяния света (Volume Light Scattering)
Поддержка атмосферного рассеяния, подводного рассеяния, свечения частиц
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from numba import jit, prange, vectorize, float32, float64, int32, int64
import warnings
import math
from enum import Enum

# ----------------------------------------------------------------------
# Константы и типы данных
# ----------------------------------------------------------------------

class ScatteringType(Enum):
    """Типы рассеяния света"""
    RAYLEIGH = 1      # Релеевское рассеяние (атмосфера, синее небо)
    MIE = 2           # Рассеяние Ми (облака, туман, подводная взвесь)
    HENYEY_GREENSTEIN = 3  # Анизотропное рассеяние (для объемных материалов)
    ISOTROPIC = 4     # Изотропное рассеяние (равномерное во все стороны)
    PHASE_FUNCTION = 5 # Кастомная фазовая функция

@dataclass
class MediumProperties:
    """Свойства среды для рассеяния света"""
    scattering_coefficient: float  # Коэффициент рассеяния (σ_s)
    absorption_coefficient: float  # Коэффициент поглощения (σ_a)
    extinction_coefficient: float  # Коэффициент экстинкции (σ_t = σ_s + σ_a)
    scattering_albedo: float       # Альбедо рассеяния (ω = σ_s / σ_t)
    phase_function_g: float        # Параметр асимметрии для HG (-1 до 1)
    density: float                 # Плотность среды (0-1)
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Цвет рассеяния
    
    def __post_init__(self):
        """Вычисление производных параметров"""
        self.extinction_coefficient = (self.scattering_coefficient + 
                                      self.absorption_coefficient)
        if self.extinction_coefficient > 0:
            self.scattering_albedo = (self.scattering_coefficient / 
                                     self.extinction_coefficient)
        else:
            self.scattering_albedo = 0.0

class LightSource:
    """Источник света для объемного рассеяния"""
    
    def __init__(self, 
                 position: Tuple[float, float, float] = (0, 0, 0),
                 direction: Tuple[float, float, float] = (0, -1, 0),
                 color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 intensity: float = 1.0,
                 light_type: str = "directional"):  # "directional", "point", "spot"
        
        self.position = np.array(position, dtype=np.float32)
        self.direction = np.array(direction, dtype=np.float32)
        self.direction = self.direction / np.linalg.norm(self.direction)
        self.color = np.array(color, dtype=np.float32)
        self.intensity = intensity
        self.light_type = light_type
        
        # Для точечного и прожекторного света
        self.radius = 0.0  # Радиус источника (для soft shadows)
        self.attenuation = (1.0, 0.0, 0.0)  # Постоянная, линейная, квадратичная

# ----------------------------------------------------------------------
# Фазовые функции рассеяния (оптимизированные с Numba)
# ----------------------------------------------------------------------

@jit(nopython=True, cache=True)
def phase_function_isotropic(cos_theta: float) -> float:
    """Изотропная фазовая функция (рассеяние равномерно во все стороны)"""
    return 1.0 / (4.0 * np.pi)

@jit(nopython=True, cache=True)
def phase_function_rayleigh(cos_theta: float) -> float:
    """Релеевская фазовая функция (рассеяние на малых частицах, атмосфера)"""
    return (3.0 / (16.0 * np.pi)) * (1.0 + cos_theta * cos_theta)

@jit(nopython=True, cache=True)
def phase_function_henyey_greenstein(cos_theta: float, g: float) -> float:
    """
    Фазовая функция Хеньи-Гринстейна для анизотропного рассеяния
    
    Args:
        cos_theta: Косинус угла между направлением луча и света
        g: Параметр асимметрии (-1: назад, 0: изотропно, 1: вперед)
    
    Returns:
        Значение фазовой функции
    """
    g2 = g * g
    denominator = 1.0 + g2 - 2.0 * g * cos_theta
    if denominator <= 0:
        return 0.0
    return (1.0 - g2) / (4.0 * np.pi * np.power(denominator, 1.5))

@jit(nopython=True, cache=True)
def phase_function_mie(cos_theta: float, g: float = 0.76) -> float:
    """Фазовая функция Ми (для крупных частиц, облаков)"""
    # Используем HG как аппроксимацию для Ми
    return phase_function_henyey_greenstein(cos_theta, g)

@jit(nopython=True, cache=True)
def schlick_phase_function(cos_theta: float, g: float) -> float:
    """
    Аппроксимация Шлика для фазовой функции HG
    Быстрее вычисляется, часто используется в real-time графике
    """
    k = 1.55 * g - 0.55 * g * g * g
    return (1.0 - k * k) / (4.0 * np.pi * (1.0 + k * cos_theta) * (1.0 + k * cos_theta))

# ----------------------------------------------------------------------
# Функции для расчета рассеяния в среде
# ----------------------------------------------------------------------

@jit(nopython=True, cache=True)
def compute_optical_depth(density: np.ndarray, 
                         extinction: float,
                         step_size: float) -> float:
    """
    Вычисление оптической глубины (затухание света в среде)
    
    τ = ∫ σ_t * ρ(x) dx
    
    Args:
        density: Плотность вдоль пути
        extinction: Коэффициент экстинкции
        step_size: Длина шага
        
    Returns:
        Оптическая глубина
    """
    optical_depth = 0.0
    for i in range(len(density)):
        optical_depth += density[i] * extinction * step_size
    return optical_depth

@jit(nopython=True, cache=True)
def transmittance(optical_depth: float) -> float:
    """
    Пропускание (трансмиттанс) света через среду
    
    T = exp(-τ)
    """
    return np.exp(-optical_depth)

@jit(nopython=True, cache=True)
def in_scattering(source_radiance: float,
                 phase_function: float,
                 scattering_coef: float,
                 density: float,
                 transmittance: float,
                 step_size: float) -> float:
    """
    Расчет ин-скеттеринга (вклада рассеянного света)
    
    L_in = σ_s * ρ * P(θ) * L_source * T * Δx
    """
    return scattering_coef * density * phase_function * source_radiance * transmittance * step_size

# ----------------------------------------------------------------------
# Класс объемного рассеяния для рендеринга
# ----------------------------------------------------------------------

class VolumeScatteringRenderer:
    """Рендерер с учетом объемного рассеяния света"""
    
    def __init__(self, 
                 volume: 'VolumeTexture3D',
                 medium: MediumProperties,
                 light_sources: List[LightSource],
                 use_multiple_scattering: bool = False,
                 num_scattering_events: int = 2):
        
        self.volume = volume
        self.medium = medium
        self.light_sources = light_sources
        self.use_multiple_scattering = use_multiple_scattering
        self.num_scattering_events = num_scattering_events
        
        # Кэш для ускорения расчетов
        self.density_cache = {}
        self.transmittance_cache = {}
        
        # Предварительные вычисления
        self._precompute_phase_functions()
    
    def _precompute_phase_functions(self):
        """Предварительное вычисление фазовых функций для ускорения"""
        self.phase_functions = {
            ScatteringType.ISOTROPIC: phase_function_isotropic,
            ScatteringType.RAYLEIGH: phase_function_rayleigh,
            ScatteringType.MIE: lambda cos: phase_function_mie(cos, self.medium.phase_function_g),
            ScatteringType.HENYEY_GREENSTEIN: lambda cos: phase_function_henyey_greenstein(cos, self.medium.phase_function_g),
        }
    
    def render_single_scattering(self,
                                camera_pos: np.ndarray,
                                ray_dir: np.ndarray,
                                max_steps: int = 256,
                                step_size: float = 0.005) -> np.ndarray:
        """
        Рендеринг с учетом однократного рассеяния (single scattering)
        
        Args:
            camera_pos: Позиция камеры
            ray_dir: Направление луча (нормализованное)
            max_steps: Максимальное количество шагов
            step_size: Размер шага
            
        Returns:
            Цвет с учетом рассеяния (RGB)
        """
        # Инициализация
        accumulated_color = np.zeros(3, dtype=np.float32)
        transmittance = 1.0
        ray_pos = camera_pos.copy()
        
        for step in range(max_steps):
            # Проверяем границы объема
            if not self._is_inside_volume(ray_pos):
                break
            
            # Получаем плотность в текущей точке
            density = self._sample_density(ray_pos)
            
            if density > 0:
                # Вычисляем вклад от каждого источника света
                for light in self.light_sources:
                    # Пропускание от точки до источника света
                    light_transmittance = self._compute_light_transmittance(ray_pos, light)
                    
                    # Косинус угла между лучом и направлением к свету
                    light_dir = self._get_light_direction(ray_pos, light)
                    cos_theta = np.dot(ray_dir, light_dir)
                    
                    # Фазовая функция
                    phase = self.phase_functions[ScatteringType.HENYEY_GREENSTEIN](cos_theta)
                    
                    # Вклад рассеяния
                    scattering = in_scattering(
                        source_radiance=light.intensity,
                        phase_function=phase,
                        scattering_coef=self.medium.scattering_coefficient,
                        density=density,
                        transmittance=transmittance * light_transmittance,
                        step_size=step_size
                    )
                    
                    # Учитываем цвет источника и среды
                    scattering_color = scattering * light.color * self.medium.color
                    accumulated_color += scattering_color
                
                # Обновляем пропускание
                transmittance *= np.exp(-self.medium.extinction_coefficient * density * step_size)
            
            # Продвигаем луч
            ray_pos += ray_dir * step_size
            
            # Если почти непрозрачно, останавливаемся
            if transmittance < 0.01:
                break
        
        return np.clip(accumulated_color, 0, 1)
    
    def render_multiple_scattering(self,
                                  camera_pos: np.ndarray,
                                  ray_dir: np.ndarray,
                                  max_steps: int = 128,
                                  step_size: float = 0.01) -> np.ndarray:
        """
        Рендеринг с учетом многократного рассеяния (multiple scattering)
        Использует упрощенный алгоритм для real-time
        
        Args:
            camera_pos: Позиция камеры
            ray_dir: Направление луча
            max_steps: Шаги для первичного луча
            step_size: Размер шага
            
        Returns:
            Цвет с учетом многократного рассеяния
        """
        # Шаг 1: Однократное рассеяние (прямое освещение)
        direct_scattering = self.render_single_scattering(
            camera_pos, ray_dir, max_steps, step_size
        )
        
        if not self.use_multiple_scattering:
            return direct_scattering
        
        # Шаг 2: Многократное рассеяние (упрощенное)
        # Используем аппроксимацию диффузного рассеяния
        indirect_scattering = self._compute_indirect_scattering(
            camera_pos, ray_dir, max_steps, step_size
        )
        
        # Комбинируем прямой и рассеянный свет
        return np.clip(direct_scattering + indirect_scattering * 0.5, 0, 1)
    
    def _compute_indirect_scattering(self,
                                    camera_pos: np.ndarray,
                                    ray_dir: np.ndarray,
                                    max_steps: int,
                                    step_size: float) -> np.ndarray:
        """
        Упрощенное вычисление многократного рассеяния
        (диффузная аппроксимация)
        """
        accumulated_color = np.zeros(3, dtype=np.float32)
        ray_pos = camera_pos.copy()
        
        # Собираем плотности вдоль луча
        densities = []
        positions = []
        
        for step in range(max_steps):
            if not self._is_inside_volume(ray_pos):
                break
            
            density = self._sample_density(ray_pos)
            if density > 0:
                densities.append(density)
                positions.append(ray_pos.copy())
            
            ray_pos += ray_dir * step_size
        
        if not densities:
            return accumulated_color
        
        # Для каждой точки с ненулевой плотностью
        for i, (pos, density) in enumerate(zip(positions, densities)):
            # Оцениваем рассеянный свет от соседних областей
            # Упрощенная модель: считаем, что свет равномерно рассеивается
            local_scattering = 0.0
            
            # Смотрим на соседние точки
            for j, (other_pos, other_density) in enumerate(zip(positions, densities)):
                if i == j:
                    continue
                
                # Расстояние между точками
                dist = np.linalg.norm(pos - other_pos)
                if dist < 0.1:  # Только близкие точки
                    # Упрощенный вклад рассеяния
                    phase = phase_function_isotropic(1.0)  # Изотропное
                    attenuation = np.exp(-self.medium.extinction_coefficient * dist)
                    local_scattering += (other_density * phase * attenuation)
            
            # Нормализуем
            if len(densities) > 1:
                local_scattering /= (len(densities) - 1)
            
            # Учитываем в общем цвете
            scattering_strength = density * self.medium.scattering_coefficient
            accumulated_color += scattering_strength * local_scattering * self.medium.color
        
        return accumulated_color / len(densities)
    
    def _sample_density(self, position: np.ndarray) -> float:
        """Выборка плотности из объемной текстуры"""
        # Нормализуем координаты к [0, 1]
        x = np.clip(position[0], 0, 1)
        y = np.clip(position[1], 0, 1)
        z = np.clip(position[2], 0, 1)
        
        # Используем кэш для ускорения
        cache_key = (int(x * 100), int(y * 100), int(z * 100))
        if cache_key in self.density_cache:
            return self.density_cache[cache_key]
        
        # Трилинейная интерполяция
        depth, height, width = self.volume.dimensions
        
        fx = x * (width - 1)
        fy = y * (height - 1)
        fz = z * (depth - 1)
        
        ix0 = int(np.floor(fx))
        iy0 = int(np.floor(fy))
        iz0 = int(np.floor(fz))
        
        ix1 = min(ix0 + 1, width - 1)
        iy1 = min(iy0 + 1, height - 1)
        iz1 = min(iz0 + 1, depth - 1)
        
        dx = fx - ix0
        dy = fy - iy0
        dz = fz - iz0
        
        # Берем первый канал как плотность
        c000 = self.volume.data[iz0, iy0, ix0, 0]
        c001 = self.volume.data[iz0, iy0, ix1, 0]
        c010 = self.volume.data[iz0, iy1, ix0, 0]
        c011 = self.volume.data[iz0, iy1, ix1, 0]
        c100 = self.volume.data[iz1, iy0, ix0, 0]
        c101 = self.volume.data[iz1, iy0, ix1, 0]
        c110 = self.volume.data[iz1, iy1, ix0, 0]
        c111 = self.volume.data[iz1, iy1, ix1, 0]
        
        # Трилинейная интерполяция
        c00 = c000 * (1 - dx) + c001 * dx
        c01 = c010 * (1 - dx) + c011 * dx
        c10 = c100 * (1 - dx) + c101 * dx
        c11 = c110 * (1 - dx) + c111 * dx
        
        c0 = c00 * (1 - dy) + c01 * dy
        c1 = c10 * (1 - dy) + c11 * dy
        
        density = c0 * (1 - dz) + c1 * dz
        
        # Кэшируем
        self.density_cache[cache_key] = density
        if len(self.density_cache) > 10000:
            self.density_cache.pop(next(iter(self.density_cache)))
        
        return density
    
    def _is_inside_volume(self, position: np.ndarray) -> bool:
        """Проверка, находится ли точка внутри объема"""
        return (0 <= position[0] <= 1 and 
                0 <= position[1] <= 1 and 
                0 <= position[2] <= 1)
    
    def _get_light_direction(self, position: np.ndarray, light: LightSource) -> np.ndarray:
        """Получение направления к источнику света"""
        if light.light_type == "directional":
            return -light.direction  # Источник бесконечно далеко
        
        # Для точечного источника
        light_dir = light.position - position
        return light_dir / np.linalg.norm(light_dir)
    
    def _compute_light_transmittance(self, position: np.ndarray, light: LightSource) -> float:
        """
        Вычисление пропускания от точки до источника света
        (shadow ray)
        """
        if light.light_type == "directional":
            # Для направленного света: луч в противоположном направлении
            light_dir = -light.direction
            ray_pos = position.copy()
            optical_depth = 0.0
            
            # Идем до границы объема
            for _ in range(64):  # Ограниченное количество шагов
                ray_pos += light_dir * 0.01
                if not self._is_inside_volume(ray_pos):
                    break
                
                density = self._sample_density(ray_pos)
                optical_depth += density * self.medium.extinction_coefficient * 0.01
        
        else:
            # Для точечного источника
            light_dir = self._get_light_direction(position, light)
            distance = np.linalg.norm(light.position - position)
            ray_pos = position.copy()
            step_size = distance / 64
            optical_depth = 0.0
            
            for i in range(64):
                ray_pos += light_dir * step_size
                if not self._is_inside_volume(ray_pos):
                    break
                
                density = self._sample_density(ray_pos)
                optical_depth += density * self.medium.extinction_coefficient * step_size
        
        return transmittance(optical_depth)
    
    def render_volumetric_light(self,
                               camera_pos: Tuple[float, float, float],
                               camera_target: Tuple[float, float, float],
                               image_size: Tuple[int, int],
                               max_steps: int = 128,
                               step_size: float = 0.01) -> np.ndarray:
        """
        Рендеринг объема с учетом рассеяния света (вей освещения)
        
        Args:
            camera_pos: Позиция камеры
            camera_target: Цель камеры
            image_size: Размер выходного изображения
            max_steps: Максимальное количество шагов луча
            step_size: Размер шага
            
        Returns:
            2D изображение (H, W, 3) RGB
        """
        width, height = image_size
        image = np.zeros((height, width, 3), dtype=np.float32)
        
        # Вычисляем базис камеры
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
        
        print(f"Rendering volumetric light {width}x{height}...")
        
        # Для каждого пикселя
        for y in prange(height):
            for x in range(width):
                # Вычисляем направление луча
                u = (2.0 * x / width - 1.0) * half_width
                v = (1.0 - 2.0 * y / height) * half_height
                
                ray_dir = camera_dir + u * right + v * up
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # Рендеринг с рассеянием
                if self.use_multiple_scattering:
                    color = self.render_multiple_scattering(
                        np.array(camera_pos), ray_dir, max_steps, step_size
                    )
                else:
                    color = self.render_single_scattering(
                        np.array(camera_pos), ray_dir, max_steps, step_size
                    )
                
                image[y, x] = color
        
        return np.clip(image, 0, 1)

# ----------------------------------------------------------------------
# Специализированные среды для разных эффектов
# ----------------------------------------------------------------------

class AtmosphereScattering:
    """Рассеяние в атмосфере (релеевское и ми)"""
    
    # Константы для атмосферы Земли
    RAYLEIGH_SCATTERING = np.array([5.8e-6, 1.35e-5, 3.31e-5])  # RGB коэффициенты
    MIE_SCATTERING = 2e-5
    RAYLEIGH_SCALE_HEIGHT = 8000.0  # метров
    MIE_SCALE_HEIGHT = 1200.0  # метров
    EARTH_RADIUS = 6371000.0  # метров
    ATMOSPHERE_HEIGHT = 100000.0  # метров
    
    def __init__(self):
        self.sun_direction = np.array([0.0, 1.0, 0.0])
        self.sun_intensity = 20.0
        self.ground_albedo = 0.3
        self.enable_ozone = True
        
    def compute_atmosphere_scattering(self,
                                     ray_origin: np.ndarray,
                                     ray_direction: np.ndarray,
                                     sun_direction: np.ndarray,
                                     samples: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисление рассеяния в атмосфере (упрощенная модель)
        
        Returns:
            (цвет рассеяния, цвет вторичного рассеяния)
        """
        # Преобразуем в локальные координаты (высота над Землей)
        ray_height = np.linalg.norm(ray_origin) - self.EARTH_RADIUS
        ray_height = max(ray_height, 0.0)
        
        # Инициализация
        total_rayleigh = np.zeros(3, dtype=np.float32)
        total_mie = np.zeros(3, dtype=np.float32)
        optical_depth_rayleigh = 0.0
        optical_depth_mie = 0.0
        
        # Интегрируем вдоль луча
        step_size = min(self.ATMOSPHERE_HEIGHT / samples, 1000.0)
        current_pos = ray_origin.copy()
        
        for i in range(samples):
            # Высота текущей точки
            height = np.linalg.norm(current_pos) - self.EARTH_RADIUS
            
            if height < 0 or height > self.ATMOSPHERE_HEIGHT:
                break
            
            # Плотность на этой высоте (экспоненциальное убывание)
            rayleigh_density = np.exp(-height / self.RAYLEIGH_SCALE_HEIGHT)
            mie_density = np.exp(-height / self.MIE_SCALE_HEIGHT)
            
            # Оптическая глубина
            optical_depth_rayleigh += rayleigh_density * step_size
            optical_depth_mie += mie_density * step_size
            
            # Рассеяние к солнцу от этой точки
            sun_transmittance_rayleigh = np.exp(-optical_depth_rayleigh * self.RAYLEIGH_SCATTERING)
            sun_transmittance_mie = np.exp(-optical_depth_mie * self.MIE_SCATTERING)
            
            # Фазовые функции
            cos_theta = np.dot(ray_direction, sun_direction)
            phase_rayleigh = phase_function_rayleigh(cos_theta)
            phase_mie = phase_function_mie(cos_theta, g=0.76)
            
            # Вклад рассеяния
            light_path = sun_transmittance_rayleigh * sun_transmittance_mie
            total_rayleigh += rayleigh_density * phase_rayleigh * light_path * step_size
            total_mie += mie_density * phase_mie * light_path * step_size
            
            # Двигаем луч
            current_pos += ray_direction * step_size
        
        # Учитываем интенсивность солнца
        rayleigh_color = total_rayleigh * self.RAYLEIGH_SCATTERING * self.sun_intensity
        mie_color = total_mie * self.MIE_SCATTERING * self.sun_intensity
        
        # Цвет неба (комбинация релеевского и ми)
        sky_color = rayleigh_color + mie_color
        
        # Вторичное рассеяние (упрощенное)
        secondary_scattering = sky_color * 0.5 * self.ground_albedo
        
        return sky_color, secondary_scattering
    
    def render_sky(self,
                  camera_pos: Tuple[float, float, float],
                  view_direction: Tuple[float, float, float],
                  sun_direction: Tuple[float, float, float],
                  image_size: Tuple[int, int]) -> np.ndarray:
        """
        Рендеринг неба с атмосферным рассеянием
        """
        width, height = image_size
        image = np.zeros((height, width, 3), dtype=np.float32)
        
        # Базис камеры
        camera_dir = np.array(view_direction, dtype=np.float32)
        camera_dir = camera_dir / np.linalg.norm(camera_dir)
        
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(camera_dir, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, camera_dir)
        
        # FOV
        fov = 90.0
        aspect = width / height
        half_height = np.tan(np.radians(fov) / 2.0)
        half_width = aspect * half_height
        
        # Позиция камеры в мировых координатах
        camera_world_pos = np.array(camera_pos) + np.array([0, self.EARTH_RADIUS + 100, 0])
        
        for y in range(height):
            for x in range(width):
                # Направление луча
                u = (2.0 * x / width - 1.0) * half_width
                v = (1.0 - 2.0 * y / height) * half_height
                
                ray_dir = camera_dir + u * right + v * up
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # Вычисляем рассеяние
                sky_color, secondary = self.compute_atmosphere_scattering(
                    camera_world_pos, ray_dir, sun_direction
                )
                
                # Комбинируем
                pixel_color = sky_color + secondary
                
                # Добавляем солнце
                sun_angle = np.dot(ray_dir, sun_direction)
                if sun_angle > 0.9995:
                    # Диск солнца
                    sun_size = 0.01
                    if sun_angle > 1.0 - sun_size:
                        sun_intensity = 10.0
                        pixel_color = np.array([1.0, 0.9, 0.8]) * sun_intensity
                
                image[y, x] = np.clip(pixel_color, 0, 1)
        
        return image

class UnderwaterScattering:
    """Рассеяние света под водой"""
    
    def __init__(self):
        # Свойства воды
        self.water_color = np.array([0.0, 0.4, 0.8])  # Синий цвет воды
        self.scattering_coefficient = 0.1
        self.absorption_coefficient = 0.05
        self.density = 0.5
        self.phase_function_g = 0.8  # Сильное рассеяние вперед
        
        # Источники света
        self.sun_direction = np.array([0.0, 1.0, 0.0])
        self.sun_color = np.array([1.0, 0.9, 0.7])
        self.ambient_light = np.array([0.1, 0.2, 0.3])
        
        # Каустика
        self.enable_caustics = True
        self.caustics_intensity = 0.5
        
    def render_underwater(self,
                         camera_pos: Tuple[float, float, float],
                         view_direction: Tuple[float, float, float],
                         water_surface_height: float = 0.0,
                         image_size: Tuple[int, int] = (256, 256),
                         max_depth: float = 100.0) -> np.ndarray:
        """
        Рендеринг подводной сцены с учетом рассеяния
        
        Args:
            camera_pos: Позиция камеры (под водой)
            view_direction: Направление взгляда
            water_surface_height: Высота поверхности воды (Y координата)
            image_size: Размер изображения
            max_depth: Максимальная глубина видимости
            
        Returns:
            Подводное изображение с рассеянием
        """
        width, height = image_size
        image = np.zeros((height, width, 3), dtype=np.float32)
        
        # Базис камеры
        camera_dir = np.array(view_direction, dtype=np.float32)
        camera_dir = camera_dir / np.linalg.norm(camera_dir)
        
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(camera_dir, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, camera_dir)
        
        # FOV
        fov = 70.0
        aspect = width / height
        half_height = np.tan(np.radians(fov) / 2.0)
        half_width = aspect * half_height
        
        camera_world_pos = np.array(camera_pos, dtype=np.float32)
        
        print(f"Rendering underwater scene {width}x{height}...")
        
        for y in range(height):
            for x in range(width):
                # Направление луча
                u = (2.0 * x / width - 1.0) * half_width
                v = (1.0 - 2.0 * y / height) * half_height
                
                ray_dir = camera_dir + u * right + v * up
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # Начинаем с позиции камеры
                ray_pos = camera_world_pos.copy()
                accumulated_color = np.zeros(3, dtype=np.float32)
                transmittance = 1.0
                
                # Интегрируем вдоль луча
                step_size = 0.5
                max_steps = int(max_depth / step_size)
                
                for step in range(max_steps):
                    # Проверяем, не вышли ли мы из воды
                    if ray_pos[1] > water_surface_height:
                        # Мы на поверхности, добавляем цвет неба
                        sky_contribution = self._get_sky_light(ray_pos, ray_dir)
                        accumulated_color += transmittance * sky_contribution
                        break
                    
                    # Расстояние от камеры
                    distance = step * step_size
                    
                    # Плотность воды (может меняться с глубиной)
                    depth = water_surface_height - ray_pos[1]
                    density = self.density * (1.0 - np.exp(-depth / 10.0))
                    
                    # Рассеяние от солнца
                    sun_direction_normalized = self.sun_direction / np.linalg.norm(self.sun_direction)
                    cos_theta = np.dot(ray_dir, sun_direction_normalized)
                    phase = phase_function_henyey_greenstein(cos_theta, self.phase_function_g)
                    
                    # Затухание света от солнца до этой точки
                    # (упрощенно: учитываем только глубину)
                    sun_attenuation = np.exp(-depth * (self.scattering_coefficient + self.absorption_coefficient))
                    
                    # Вклад рассеяния
                    scattering = (self.scattering_coefficient * density * phase *
                                 self.sun_color * sun_attenuation * step_size)
                    
                    # Учитываем цвет воды
                    scattering *= self.water_color
                    
                    # Добавляем к накопленному цвету
                    accumulated_color += transmittance * scattering
                    
                    # Обновляем пропускание
                    extinction = self.scattering_coefficient + self.absorption_coefficient
                    transmittance *= np.exp(-extinction * density * step_size)
                    
                    # Двигаем луч
                    ray_pos += ray_dir * step_size
                    
                    # Если почти непрозрачно, останавливаемся
                    if transmittance < 0.01:
                        break
                
                # Добавляем каустику если включена
                if self.enable_caustics and self.sun_direction[1] > 0:
                    caustics = self._compute_caustics(camera_world_pos, ray_dir, water_surface_height)
                    accumulated_color += caustics * self.caustics_intensity
                
                # Добавляем ambient light
                accumulated_color += self.ambient_light * (1.0 - transmittance)
                
                image[y, x] = np.clip(accumulated_color, 0, 1)
        
        return image
    
    def _get_sky_light(self, position: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Получение цвета неба для подводного рендеринга"""
        # Упрощенная модель: цвет неба зависит от направления
        horizon_factor = max(0.0, direction[1])  # Чем выше смотрим, тем светлее
        sky_color = np.array([0.5, 0.6, 0.8]) * (0.2 + 0.8 * horizon_factor)
        return sky_color
    
    def _compute_caustics(self,
                         camera_pos: np.ndarray,
                         view_dir: np.ndarray,
                         water_surface: float) -> np.ndarray:
        """
        Вычисление каустики (игр света на дне под водой)
        Упрощенная модель на основе шума
        """
        # Точка пересечения с дном (упрощенно: плоскость на глубине 10м)
        sea_floor_depth = 10.0
        if view_dir[1] < -0.01:  # Смотрим вниз
            t = (camera_pos[1] + sea_floor_depth) / -view_dir[1]
            floor_pos = camera_pos + view_dir * t
            
            # Шум для каустики
            noise = np.sin(floor_pos[0] * 10.0) * np.cos(floor_pos[2] * 10.0)
            caustics = np.array([1.0, 1.0, 0.9]) * (noise * 0.5 + 0.5) * 0.3
            
            # Затухание с глубиной
            depth_factor = np.exp(-sea_floor_depth * 0.1)
            return caustics * depth_factor
        
        return np.zeros(3, dtype=np.float32)

# ----------------------------------------------------------------------
# Оптимизированные функции для real-time рендеринга
# ----------------------------------------------------------------------

@jit(nopython=True, parallel=True, cache=True)
def fast_volume_scattering_kernel(
    width: int,
    height: int,
    camera_pos: np.ndarray,
    camera_dir: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    half_width: float,
    half_height: float,
    light_dir: np.ndarray,
    light_color: np.ndarray,
    volume_data: np.ndarray,
    scattering_coef: float,
    absorption_coef: float,
    phase_g: float,
    max_steps: int,
    step_size: float
) -> np.ndarray:
    """
    Оптимизированное ядро для объемного рассеяния (работает на CPU)
    
    Args:
        Все параметры для рендеринга
        
    Returns:
        Изображение (H, W, 3)
    """
    image = np.zeros((height, width, 3), dtype=np.float32)
    extinction_coef = scattering_coef + absorption_coef
    
    for y in prange(height):
        for x in range(width):
            # Направление луча
            u = (2.0 * x / width - 1.0) * half_width
            v = (1.0 - 2.0 * y / height) * half_height
            
            ray_dir = camera_dir + u * right + v * up
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            
            # Рейкастинг
            ray_pos = camera_pos.copy()
            accumulated_color = np.zeros(3, dtype=np.float32)
            transmittance = 1.0
            
            for step in range(max_steps):
                # Проверяем границы
                if (ray_pos[0] < 0 or ray_pos[0] >= 1 or
                    ray_pos[1] < 0 or ray_pos[1] >= 1 or
                    ray_pos[2] < 0 or ray_pos[2] >= 1):
                    break
                
                # Трилинейная интерполяция плотности
                depth, vol_height, vol_width = volume_data.shape[:3]
                
                fx = ray_pos[0] * (vol_width - 1)
                fy = ray_pos[1] * (vol_height - 1)
                fz = ray_pos[2] * (depth - 1)
                
                ix0 = int(np.floor(fx))
                iy0 = int(np.floor(fy))
                iz0 = int(np.floor(fz))
                
                ix1 = min(ix0 + 1, vol_width - 1)
                iy1 = min(iy0 + 1, vol_height - 1)
                iz1 = min(iz0 + 1, depth - 1)
                
                dx = fx - ix0
                dy = fy - iy0
                dz = fz - iz0
                
                # Берем плотность (первый канал)
                c000 = volume_data[iz0, iy0, ix0, 0]
                c001 = volume_data[iz0, iy0, ix1, 0]
                c010 = volume_data[iz0, iy1, ix0, 0]
                c011 = volume_data[iz0, iy1, ix1, 0]
                c100 = volume_data[iz1, iy0, ix0, 0]
                c101 = volume_data[iz1, iy0, ix1, 0]
                c110 = volume_data[iz1, iy1, ix0, 0]
                c111 = volume_data[iz1, iy1, ix1, 0]
                
                c00 = c000 * (1 - dx) + c001 * dx
                c01 = c010 * (1 - dx) + c011 * dx
                c10 = c100 * (1 - dx) + c101 * dx
                c11 = c110 * (1 - dx) + c111 * dx
                
                c0 = c00 * (1 - dy) + c01 * dy
                c1 = c10 * (1 - dy) + c11 * dy
                
                density = c0 * (1 - dz) + c1 * dz
                
                if density > 0:
                    # Фазовая функция
                    cos_theta = np.dot(ray_dir, light_dir)
                    phase = (1.0 - phase_g * phase_g) / \
                           (4.0 * np.pi * np.power(1.0 + phase_g * phase_g - 2.0 * phase_g * cos_theta, 1.5))
                    
                    # Вклад рассеяния
                    scattering = (scattering_coef * density * phase *
                                light_color * transmittance * step_size)
                    
                    accumulated_color += scattering
                    
                    # Обновляем пропускание
                    transmittance *= np.exp(-extinction_coef * density * step_size)
                
                # Продвигаем луч
                ray_pos += ray_dir * step_size
                
                # Ранний выход
                if transmittance < 0.01:
                    break
            
            image[y, x] = np.clip(accumulated_color, 0, 1)
    
    return image

# ----------------------------------------------------------------------
# Примеры использования
# ----------------------------------------------------------------------

def example_atmospheric_scattering():
    """Пример атмосферного рассеяния (небо и облака)"""
    
    print("Atmospheric scattering example...")
    
    # Создаем объем облаков
    from .volume_textures import VolumeTextureGenerator3D
    generator = VolumeTextureGenerator3D(seed=42)
    clouds = generator.generate_clouds_3d(
        width=128, height=64, depth=128,
        scale=0.02, density=0.3, detail=3
    )
    
    # Настройка среды (атмосфера с облаками)
    medium = MediumProperties(
        scattering_coefficient=0.1,  # Рассеяние в облаках
        absorption_coefficient=0.02,  # Поглощение
        phase_function_g=0.7,  # Рассеяние вперед (облака)
        density=1.0,
        color=(1.0, 1.0, 1.0)  # Белый свет
    )
    
    # Источник света (солнце)
    sun_light = LightSource(
        direction=(0.3, 1.0, 0.2),  # Солнце высоко
        color=(1.0, 0.9, 0.7),  # Теплый солнечный свет
        intensity=2.0,
        light_type="directional"
    )
    
    # Рендерер с рассеянием
    renderer = VolumeScatteringRenderer(
        volume=clouds,
        medium=medium,
        light_sources=[sun_light],
        use_multiple_scattering=True,
        num_scattering_events=2
    )
    
    # Рендеринг
    camera_pos = (0.5, 0.5, 2.0)
    camera_target = (0.5, 0.5, 0.0)
    
    image = renderer.render_volumetric_light(
        camera_pos=camera_pos,
        camera_target=camera_target,
        image_size=(512, 256),
        max_steps=128,
        step_size=0.01
    )
    
    print(f"Rendered image shape: {image.shape}")
    
    return image, clouds

def example_underwater_scene():
    """Пример подводного рассеяния"""
    
    print("\nUnderwater scattering example...")
    
    # Создаем подводную среду (плотность воды с частицами)
    from .volume_textures import VolumeTextureGenerator3D
    generator = VolumeTextureGenerator3D(seed=123)
    
    # Объем для плотности воды (силуэты водорослей, пузырей)
    water_volume = generator.generate_perlin_3d(
        width=96, height=96, depth=96,
        scale=0.05, octaves=3
    )
    
    # Настройка подводной среды
    medium = MediumProperties(
        scattering_coefficient=0.15,  # Сильное рассеяние в воде
        absorption_coefficient=0.1,  # Поглощение синим светом меньше
        phase_function_g=0.8,  # Очень сильное рассеяние вперед
        density=0.8,
        color=(0.1, 0.3, 0.6)  # Синий цвет воды
    )
    
    # Солнечный свет, проникающий через воду
    sun_light = LightSource(
        direction=(0.1, 1.0, 0.0),  # Солнце над водой
        color=(0.7, 0.8, 1.0),  # Голубоватый подводный свет
        intensity=1.5,
        light_type="directional"
    )
    
    # Ambient light от рассеянного подводного света
    ambient_light = LightSource(
        direction=(0, 1, 0),
        color=(0.1, 0.2, 0.4),
        intensity=0.3,
        light_type="directional"
    )
    
    # Рендерер
    renderer = VolumeScatteringRenderer(
        volume=water_volume,
        medium=medium,
        light_sources=[sun_light, ambient_light],
        use_multiple_scattering=False  # Для производительности
    )
    
    # Рендеринг под водой
    camera_pos = (0.5, 0.3, 0.5)  # Камера под водой
    camera_target = (0.5, 0.2, 0.0)  # Смотрим вниз
    
    image = renderer.render_volumetric_light(
        camera_pos=camera_pos,
        camera_target=camera_target,
        image_size=(512, 256),
        max_steps=64,  # Меньше шагов для производительности
        step_size=0.02
    )
    
    print(f"Underwater image shape: {image.shape}")
    
    return image, water_volume

def example_fast_volume_light():
    """Пример быстрого объемного освещения для real-time"""
    
    print("\nFast volume light example (real-time optimized)...")
    
    # Создаем маленький объем для производительности
    from .volume_textures import VolumeTextureGenerator3D
    generator = VolumeTextureGenerator3D(seed=42)
    volume = generator.generate_clouds_3d(
        width=64, height=64, depth=64,
        scale=0.05, density=0.4, detail=2
    )
    
    # Параметры камеры
    camera_pos = np.array([0.5, 0.5, 1.5], dtype=np.float32)
    camera_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    camera_dir = camera_dir / np.linalg.norm(camera_dir)
    
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(camera_dir, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, camera_dir)
    
    # Параметры рассеяния
    width, height = 256, 256
    fov = 60.0
    aspect = width / height
    half_height = np.tan(np.radians(fov) / 2.0)
    half_width = aspect * half_height
    
    light_dir = np.array([0.3, 1.0, 0.2], dtype=np.float32)
    light_dir = light_dir / np.linalg.norm(light_dir)
    light_color = np.array([1.0, 0.9, 0.7], dtype=np.float32)
    
    scattering_coef = 0.1
    absorption_coef = 0.05
    phase_g = 0.7
    max_steps = 64
    step_size = 0.02
    
    # Запускаем оптимизированное ядро
    image = fast_volume_scattering_kernel(
        width, height,
        camera_pos, camera_dir, right, up,
        half_width, half_height,
        light_dir, light_color,
        volume.data,
        scattering_coef, absorption_coef, phase_g,
        max_steps, step_size
    )
    
    print(f"Fast volume light image shape: {image.shape}")
    
    return image

def example_volumetric_fog():
    """Пример объемного тумана с рассеянием"""
    
    print("\nVolumetric fog example...")
    
    # Создаем простой объем тумана (однородный с небольшими вариациями)
    from .volume_textures import VolumeTextureGenerator3D
    generator = VolumeTextureGenerator3D(seed=42)
    
    # Генерируем однородный туман с небольшими вариациями
    fog_volume = generator.generate_perlin_3d(
        width=128, height=64, depth=128,
        scale=0.03, octaves=2
    )
    
    # Делаем туман более однородным
    fog_data = fog_volume.data
    fog_density = np.clip(fog_data[..., 0] * 0.5 + 0.3, 0, 1)  # Плотность 0.3-0.8
    fog_data[..., 0] = fog_density
    
    # Настройка среды (туман)
    medium = MediumProperties(
        scattering_coefficient=0.08,
        absorption_coefficient=0.02,
        phase_function_g=0.2,  # Слабое направленное рассеяние
        density=1.0,
        color=(0.9, 0.9, 0.9)  # Сероватый туман
    )
    
    # Несколько источников света (уличные фонари)
    lights = [
        LightSource(
            position=(0.3, 0.2, 0.3),
            color=(1.0, 0.9, 0.7),
            intensity=3.0,
            light_type="point"
        ),
        LightSource(
            position=(0.7, 0.2, 0.7),
            color=(0.7, 0.9, 1.0),
            intensity=2.5,
            light_type="point"
        ),
        LightSource(
            direction=(0.1, -1.0, 0.1),  # Лунный свет
            color=(0.6, 0.7, 1.0),
            intensity=0.5,
            light_type="directional"
        )
    ]
    
    # Рендерер
    renderer = VolumeScatteringRenderer(
        volume=fog_volume,
        medium=medium,
        light_sources=lights,
        use_multiple_scattering=True
    )
    
    # Рендеринг туманной сцены
    camera_pos = (0.5, 0.3, 1.0)
    camera_target = (0.5, 0.2, 0.0)
    
    image = renderer.render_volumetric_light(
        camera_pos=camera_pos,
        camera_target=camera_target,
        image_size=(512, 256),
        max_steps=96,
        step_size=0.015
    )
    
    print(f"Volumetric fog image shape: {image.shape}")
    
    return image, fog_volume

if __name__ == "__main__":
    print("Volume Light Scattering System")
    print("=" * 60)
    
    # Пример 1: Атмосферное рассеяние (облака)
    cloud_image, clouds = example_atmospheric_scattering()
    
    # Пример 2: Подводное рассеяние
    underwater_image, water_volume = example_underwater_scene()
    
    # Пример 3: Быстрое объемное освещение
    fast_image = example_fast_volume_light()
    
    # Пример 4: Объемный туман
    fog_image, fog_volume = example_volumetric_fog()
    
    print("\n" + "=" * 60)
    print("Volume Light Scattering Features:")
    print("-" * 40)
    print("1. Multiple scattering types: Rayleigh, Mie, Henyey-Greenstein")
    print("2. Atmospheric scattering for realistic skies")
    print("3. Underwater scattering with caustics")
    print("4. Volumetric fog and god rays")
    print("5. Single and multiple scattering support")
    print("6. Optimized kernels for real-time performance")
    print("7. Support for multiple light sources")
    
    print("\nPerformance optimization tips:")
    print("- Use lower resolution volumes for real-time")
    print("- Reduce number of scattering events")
    print("- Use simplified phase functions (Schlick approximation)")
    print("- Implement level-of-detail for distant volumes")
    print("- Consider GPU acceleration for production use")
    
    print("\nVolume light scattering system ready for realistic atmospheric effects!")
