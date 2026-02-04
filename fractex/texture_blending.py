# fractex/texture_blending.py
"""
Продвинутые алгоритмы смешивания фрактальных текстур
Поддержка многослойных материалов, плавных переходов и сложных эффектов
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from numba import jit, prange, vectorize, float32, float64
import warnings

# ----------------------------------------------------------------------
# Базовые функции смешивания (оптимизированные с Numba)
# ----------------------------------------------------------------------

@vectorize([float32(float32, float32, float32), 
            float64(float64, float64, float64)])
def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Линейная интерполяция (быстрая векторизованная версия)"""
    return a + t * (b - a)

@jit(nopython=True, cache=True)
def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Гладкая ступенчатая функция"""
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

@jit(nopython=True, cache=True)
def smootherstep(edge0: float, edge1: float, x: float) -> float:
    """Еще более гладкая ступенчатая функция"""
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)

@jit(nopython=True, cache=True)
def sigmoid_mix(x: float, sharpness: float = 10.0) -> float:
    """Сигмоидальное смешивание для плавных переходов"""
    return 1.0 / (1.0 + np.exp(-sharpness * (x - 0.5)))

# ----------------------------------------------------------------------
# Класс для работы с масками смешивания
# ----------------------------------------------------------------------

class BlendMask:
    """Маска для контролируемого смешивания текстур"""
    
    def __init__(self, 
                 mask_type: str = "linear",
                 seed: int = 42,
                 parameters: Optional[Dict] = None):
        
        self.mask_type = mask_type
        self.seed = seed
        self.params = parameters or {}
        self._cache = {}
        
        # Доступные типы масок
        self.mask_generators = {
            "linear": self._linear_mask,
            "gradient": self._gradient_mask,
            "noise": self._noise_mask,
            "radial": self._radial_mask,
            "voronoi": self._voronoi_mask,
            "cellular": self._cellular_mask,
            "fractal": self._fractal_mask,
            "height_based": self._height_based_mask,
            "slope_based": self._slope_based_mask,
        }
    
    def generate(self, 
                width: int, 
                height: int, 
                position: Tuple[float, float] = (0, 0),
                scale: float = 1.0) -> np.ndarray:
        """Генерация маски смешивания"""
        
        # Проверка кэша
        cache_key = f"{self.mask_type}_{width}_{height}_{position}_{scale}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # Генерация координатной сетки
        x = np.linspace(position[0], position[0] + width * scale, width)
        y = np.linspace(position[1], position[1] + height * scale, height)
        xx, yy = np.meshgrid(x, y)
        
        # Выбор генератора маски
        if self.mask_type in self.mask_generators:
            mask = self.mask_generators[self.mask_type](xx, yy, **self.params)
        else:
            mask = np.ones((height, width), dtype=np.float32)
        
        # Кэширование
        self._cache[cache_key] = mask.copy()
        if len(self._cache) > 50:  # Ограничение размера кэша
            self._cache.pop(next(iter(self._cache)))
        
        return mask
    
    def _linear_mask(self, xx, yy, direction='horizontal', **kwargs):
        """Линейный градиент"""
        if direction == 'horizontal':
            return (xx - xx.min()) / (xx.max() - xx.min())
        elif direction == 'vertical':
            return (yy - yy.min()) / (yy.max() - yy.min())
        elif direction == 'diagonal':
            return ((xx - xx.min()) + (yy - yy.min())) / \
                   ((xx.max() - xx.min()) + (yy.max() - yy.min()))
        else:
            # Произвольное направление
            angle = kwargs.get('angle', 45)
            angle_rad = np.radians(angle)
            return (xx * np.cos(angle_rad) + yy * np.sin(angle_rad))
    
    def _gradient_mask(self, xx, yy, center=(0.5, 0.5), radius=1.0, **kwargs):
        """Радиальный градиент"""
        center_x = center[0] * (xx.max() - xx.min()) + xx.min()
        center_y = center[1] * (yy.max() - yy.min()) + yy.min()
        
        dist = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        mask = 1.0 - np.clip(dist / radius, 0, 1)
        
        # Применяем функцию смягчения
        falloff = kwargs.get('falloff', 'smoothstep')
        if falloff == 'smoothstep':
            mask = smoothstep(0.0, 1.0, mask)
        elif falloff == 'smootherstep':
            mask = smootherstep(0.0, 1.0, mask)
        elif falloff == 'exponential':
            exponent = kwargs.get('exponent', 2.0)
            mask = np.power(mask, exponent)
        
        return mask
    
    def _noise_mask(self, xx, yy, noise_scale=0.01, octaves=3, **kwargs):
        """Маска на основе шума"""
        try:
            from .simplex_noise import SimplexNoise
            noise_gen = SimplexNoise(seed=self.seed)
            
            # Генерируем фрактальный шум
            noise = noise_gen.fractal_noise_2d(
                xx, yy, 
                octaves=octaves,
                persistence=kwargs.get('persistence', 0.5),
                lacunarity=kwargs.get('lacunarity', 2.0),
                base_scale=noise_scale
            )
            
            # Нормализуем к [0, 1]
            noise = (noise - noise.min()) / (noise.max() - noise.min())
            
            # Применяем пороговое значение если нужно
            threshold = kwargs.get('threshold', None)
            if threshold is not None:
                noise = np.where(noise > threshold, 1.0, 0.0)
            
            return noise
        except ImportError:
            # Запасной вариант - простой синусоидальный паттерн
            return (np.sin(xx * 0.1) * np.sin(yy * 0.1) + 1) / 2
    
    def _radial_mask(self, xx, yy, center=(0.5, 0.5), **kwargs):
        """Концентрические круги"""
        center_x = center[0] * (xx.max() - xx.min()) + xx.min()
        center_y = center[1] * (yy.max() - yy.min()) + yy.min()
        
        dist = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        frequency = kwargs.get('frequency', 10.0)
        
        mask = np.sin(dist * frequency) * 0.5 + 0.5
        return mask
    
    def _voronoi_mask(self, xx, yy, num_points=10, **kwargs):
        """Маска на основе диаграмм Вороного"""
        np.random.seed(self.seed)
        
        # Генерируем случайные точки
        points_x = np.random.rand(num_points) * (xx.max() - xx.min()) + xx.min()
        points_y = np.random.rand(num_points) * (yy.max() - yy.min()) + yy.min()
        
        # Для каждой точки вычисляем расстояние до ближайшей точки Вороного
        mask = np.zeros_like(xx, dtype=np.float32)
        
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                distances = np.sqrt((points_x - xx[i, j])**2 + 
                                  (points_y - yy[i, j])**2)
                mask[i, j] = np.min(distances)
        
        # Нормализация
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        
        # Инвертирование если нужно
        if kwargs.get('invert', False):
            mask = 1.0 - mask
        
        return mask
    
    def _cellular_mask(self, xx, yy, **kwargs):
        """Клеточный шум (cellular noise)"""
        np.random.seed(self.seed)
        
        # Упрощенная версия клеточного шума
        num_cells = kwargs.get('num_cells', 20)
        cell_size = kwargs.get('cell_size', 0.1)
        
        # Создаем случайные центры клеток
        cells_x = np.random.rand(num_cells)
        cells_y = np.random.rand(num_cells)
        
        # Нормализуем координаты к [0, 1] для простоты
        xx_norm = (xx - xx.min()) / (xx.max() - xx.min())
        yy_norm = (yy - yy.min()) / (yy.max() - yy.min())
        
        mask = np.zeros_like(xx_norm)
        
        for i in range(xx_norm.shape[0]):
            for j in range(xx_norm.shape[1]):
                # Находим ближайшую и вторую ближайшую клетки
                distances = []
                for c in range(num_cells):
                    dx = xx_norm[i, j] - cells_x[c]
                    dy = yy_norm[i, j] - cells_y[c]
                    dist = np.sqrt(dx*dx + dy*dy)
                    distances.append(dist)
                
                distances.sort()
                # Используем разницу между ближайшими клетками
                mask[i, j] = distances[1] - distances[0]
        
        # Нормализация
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        return mask
    
    def _fractal_mask(self, xx, yy, **kwargs):
        """Фрактальная маска (комбинация нескольких масок)"""
        # Создаем несколько масок с разными параметрами
        mask1 = self._noise_mask(xx, yy, noise_scale=0.01, octaves=2)
        mask2 = self._noise_mask(xx * 2, yy * 2, noise_scale=0.02, octaves=3)
        mask3 = self._radial_mask(xx, yy, frequency=5)
        
        # Комбинируем
        blend_mode = kwargs.get('blend_mode', 'multiply')
        
        if blend_mode == 'multiply':
            mask = mask1 * mask2 * mask3
        elif blend_mode == 'add':
            mask = (mask1 + mask2 + mask3) / 3
        elif blend_mode == 'max':
            mask = np.maximum(np.maximum(mask1, mask2), mask3)
        else:
            mask = mask1
        
        return mask
    
    def _height_based_mask(self, xx, yy, height_map=None, **kwargs):
        """Маска на основе карты высот"""
        if height_map is None:
            # Генерируем простую карту высот если не предоставлена
            height_map = self._noise_mask(xx, yy, noise_scale=0.02, octaves=4)
        
        min_height = kwargs.get('min_height', 0.3)
        max_height = kwargs.get('max_height', 0.7)
        softness = kwargs.get('softness', 0.1)
        
        # Создаем маску с плавным переходом
        mask = np.clip((height_map - min_height) / softness, 0, 1)
        mask *= np.clip(1 - (height_map - max_height) / softness, 0, 1)
        
        return mask
    
    def _slope_based_mask(self, xx, yy, height_map=None, **kwargs):
        """Маска на основе наклона (производной высоты)"""
        if height_map is None:
            height_map = self._noise_mask(xx, yy, noise_scale=0.02, octaves=4)
        
        # Вычисляем градиент (наклон)
        gradient_x = np.gradient(height_map, axis=1)
        gradient_y = np.gradient(height_map, axis=0)
        slope = np.sqrt(gradient_x**2 + gradient_y**2)
        
        max_slope = kwargs.get('max_slope', 0.5)
        mask = np.clip(1.0 - slope / max_slope, 0, 1)
        
        return mask

# ----------------------------------------------------------------------
# Основной класс смешивания текстур
# ----------------------------------------------------------------------

class TextureBlender:
    """Продвинутый блендер для смешивания фрактальных текстур"""
    
    def __init__(self, 
                 blend_space: str = "linear",  # linear, logarithmic, perceptual
                 gamma_correction: bool = True,
                 cache_size: int = 100):
        
        self.blend_space = blend_space
        self.gamma_correction = gamma_correction
        self.cache_size = cache_size
        self._result_cache = {}
        
        # Режимы смешивания
        self.blend_modes = {
            # Основные режимы
            "overlay": self._blend_overlay,
            "multiply": self._blend_multiply,
            "screen": self._blend_screen,
            "add": self._blend_add,
            "subtract": self._blend_subtract,
            "difference": self._blend_difference,
            "divide": self._blend_divide,
            
            # Режимы наложения
            "normal": self._blend_normal,
            "dissolve": self._blend_dissolve,
            
            # Режимы затемнения
            "darken": self._blend_darken,
            "color_burn": self._blend_color_burn,
            "linear_burn": self._blend_linear_burn,
            
            # Режимы осветления
            "lighten": self._blend_lighten,
            "color_dodge": self._blend_color_dodge,
            "linear_dodge": self._blend_linear_dodge,
            
            # Контрастные режимы
            "soft_light": self._blend_soft_light,
            "hard_light": self._blend_hard_light,
            "vivid_light": self._blend_vivid_light,
            "linear_light": self._blend_linear_light,
            "pin_light": self._blend_pin_light,
            "hard_mix": self._blend_hard_mix,
            
            # Компонентные режимы
            "hue": self._blend_hue,
            "saturation": self._blend_saturation,
            "color": self._blend_color,
            "luminosity": self._blend_luminosity,
            
            # Кастомные режимы для текстур
            "height_blend": self._blend_height_based,
            "slope_blend": self._blend_slope_based,
            "edge_blend": self._blend_edge_aware,
            "detail_preserving": self._blend_detail_preserving,
        }
    
    def blend(self, 
              texture_a: np.ndarray, 
              texture_b: np.ndarray,
              blend_mode: str = "overlay",
              opacity: float = 1.0,
              mask: Optional[np.ndarray] = None,
              **kwargs) -> np.ndarray:
        """
        Основная функция смешивания двух текстур
        
        Args:
            texture_a: Базовая текстура (H, W, C)
            texture_b: Накладываемая текстура (H, W, C)
            blend_mode: Режим смешивания
            opacity: Прозрачность наложения (0-1)
            mask: Маска смешивания (H, W) или (H, W, 1)
            **kwargs: Дополнительные параметры для режима смешивания
            
        Returns:
            Смешанная текстура (H, W, C)
        """
        # Проверка размеров
        if texture_a.shape != texture_b.shape:
            raise ValueError(f"Texture shapes must match: {texture_a.shape} != {texture_b.shape}")
        
        # Проверка маски
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[..., np.newaxis]
            if mask.shape[:2] != texture_a.shape[:2]:
                raise ValueError(f"Mask shape {mask.shape[:2]} doesn't match texture shape {texture_a.shape[:2]}")
        
        # Выбор режима смешивания
        if blend_mode not in self.blend_modes:
            warnings.warn(f"Blend mode '{blend_mode}' not found. Using 'overlay'.")
            blend_mode = "overlay"
        
        blend_func = self.blend_modes[blend_mode]
        
        # Применяем гамма-коррекцию если нужно
        if self.gamma_correction and blend_mode not in ["hue", "saturation", "color", "luminosity"]:
            texture_a = self._gamma_correct(texture_a, inverse=False)
            texture_b = self._gamma_correct(texture_b, inverse=False)
        
        # Смешивание
        result = blend_func(texture_a, texture_b, **kwargs)
        
        # Применяем маску если есть
        if mask is not None:
            result = lerp(texture_a, result, mask)
        
        # Применяем общую прозрачность
        if opacity < 1.0:
            result = lerp(texture_a, result, opacity)
        
        # Обратная гамма-коррекция
        if self.gamma_correction and blend_mode not in ["hue", "saturation", "color", "luminosity"]:
            result = self._gamma_correct(result, inverse=True)
        
        return np.clip(result, 0, 1)
    
    def blend_multiple(self,
                      textures: List[np.ndarray],
                      blend_modes: List[str],
                      opacities: List[float],
                      masks: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Смешивание нескольких текстур последовательно
        
        Args:
            textures: Список текстур для смешивания
            blend_modes: Список режимов смешивания
            opacities: Список прозрачностей
            masks: Список масок (может быть None)
            
        Returns:
            Итоговая смешанная текстура
        """
        if len(textures) < 2:
            return textures[0] if textures else np.array([])
        
        if masks is None:
            masks = [None] * len(textures)
        
        # Начинаем с первой текстуры
        result = textures[0].copy()
        
        # Последовательно применяем смешивание
        for i in range(1, len(textures)):
            result = self.blend(
                result, textures[i],
                blend_mode=blend_modes[i-1] if i-1 < len(blend_modes) else "overlay",
                opacity=opacities[i-1] if i-1 < len(opacities) else 1.0,
                mask=masks[i-1] if i-1 < len(masks) else None
            )
        
        return result
    
    def blend_layer_stack(self,
                         base_texture: np.ndarray,
                         layers: List[Dict]) -> np.ndarray:
        """
        Смешивание текстуры со стеком слоев
        
        Args:
            base_texture: Базовая текстура
            layers: Список слоев, каждый слой - словарь с параметрами:
                   {
                       'texture': np.ndarray,
                       'blend_mode': str,
                       'opacity': float,
                       'mask': Optional[np.ndarray],
                       'mask_params': Dict (опционально для генерации маски)
                   }
        
        Returns:
            Итоговая текстура
        """
        result = base_texture.copy()
        
        for layer in layers:
            # Генерация маски если есть параметры
            mask = layer.get('mask')
            if mask is None and 'mask_params' in layer:
                mask_params = layer['mask_params']
                height, width = base_texture.shape[:2]
                
                # Создаем маску
                mask_gen = BlendMask(**mask_params)
                position = mask_params.get('position', (0, 0))
                scale = mask_params.get('scale', 1.0)
                mask = mask_gen.generate(width, height, position, scale)
            
            # Смешивание
            result = self.blend(
                result, layer['texture'],
                blend_mode=layer.get('blend_mode', 'overlay'),
                opacity=layer.get('opacity', 1.0),
                mask=mask
            )
        
        return result
    
    # ------------------------------------------------------------------
    # Реализации режимов смешивания
    # ------------------------------------------------------------------
    
    def _blend_normal(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Обычное наложение (просто заменяет a на b)"""
        return b
    
    def _blend_dissolve(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Диссолв - случайная замена пикселей"""
        np.random.seed(kwargs.get('seed', 42))
        random_mask = np.random.rand(*a.shape[:2])
        threshold = kwargs.get('threshold', 0.5)
        
        mask = (random_mask > threshold).astype(np.float32)[..., np.newaxis]
        if a.shape[-1] == 4:  # С альфа-каналом
            mask = np.repeat(mask, 4, axis=2)
        
        return lerp(a, b, mask)
    
    def _blend_darken(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Берет темнейший из двух пикселей"""
        return np.minimum(a, b)
    
    def _blend_multiply(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Умножение - затемняет изображение"""
        return a * b
    
    def _blend_color_burn(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Color Burn - сильное затемнение"""
        eps = 1e-7
        return 1 - (1 - a) / np.maximum(b, eps)
    
    def _blend_linear_burn(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Linear Burn - линейное затемнение"""
        return np.maximum(a + b - 1, 0)
    
    def _blend_lighten(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Берет светлейший из двух пикселей"""
        return np.maximum(a, b)
    
    def _blend_screen(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Screen - осветляет изображение"""
        return 1 - (1 - a) * (1 - b)
    
    def _blend_color_dodge(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Color Dodge - сильное осветление"""
        eps = 1e-7
        return a / np.maximum(1 - b, eps)
    
    def _blend_linear_dodge(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Linear Dodge (Add) - сложение"""
        return np.minimum(a + b, 1)
    
    def _blend_add(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Простое сложение"""
        return self._blend_linear_dodge(a, b)
    
    def _blend_overlay(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Overlay - комбинация Multiply и Screen"""
        mask = a < 0.5
        result = np.zeros_like(a)
        result[mask] = 2 * a[mask] * b[mask]
        result[~mask] = 1 - 2 * (1 - a[~mask]) * (1 - b[~mask])
        return result
    
    def _blend_soft_light(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Soft Light - мягкий свет"""
        # Формула из Photoshop
        mask = b < 0.5
        result = np.zeros_like(a)
        result[mask] = 2 * a[mask] * b[mask] + a[mask] * a[mask] * (1 - 2 * b[mask])
        result[~mask] = 2 * a[~mask] * (1 - b[~mask]) + np.sqrt(a[~mask]) * (2 * b[~mask] - 1)
        return result
    
    def _blend_hard_light(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Hard Light - Overlay, но с swapped inputs"""
        return self._blend_overlay(b, a)  # Просто меняем местами a и b
    
    def _blend_vivid_light(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Vivid Light - комбинация Color Burn и Color Dodge"""
        eps = 1e-7
        mask = b < 0.5
        result = np.zeros_like(a)
        
        # Color Burn для темных областей
        result[mask] = 1 - (1 - a[mask]) / np.maximum(2 * b[mask], eps)
        
        # Color Dodge для светлых областей
        result[~mask] = a[~mask] / np.maximum(2 * (1 - b[~mask]), eps)
        
        return np.clip(result, 0, 1)
    
    def _blend_linear_light(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Linear Light - комбинация Linear Burn и Linear Dodge"""
        result = a + 2 * b - 1
        return np.clip(result, 0, 1)
    
    def _blend_pin_light(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Pin Light - комбинация Darken и Lighten"""
        mask1 = b < 0.5
        mask2 = b > 0.5
        
        result = a.copy()
        result[mask1] = np.minimum(a[mask1], 2 * b[mask1])
        result[mask2] = np.maximum(a[mask2], 2 * (b[mask2] - 0.5))
        
        return result
    
    def _blend_hard_mix(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Hard Mix - пороговое смешивание"""
        result = self._blend_vivid_light(a, b)
        return np.where(result < 0.5, 0, 1)
    
    def _blend_difference(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Difference - абсолютная разница"""
        return np.abs(a - b)
    
    def _blend_subtract(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Subtract - вычитание"""
        return np.maximum(a - b, 0)
    
    def _blend_divide(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Divide - деление"""
        eps = 1e-7
        return a / np.maximum(b, eps)
    
    def _blend_hue(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Сохраняет hue из b, saturation и luminosity из a"""
        a_hsv = self._rgb_to_hsv(a)
        b_hsv = self._rgb_to_hsv(b)
        
        result_hsv = np.stack([b_hsv[..., 0], a_hsv[..., 1], a_hsv[..., 2]], axis=-1)
        return self._hsv_to_rgb(result_hsv)
    
    def _blend_saturation(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Сохраняет saturation из b, hue и luminosity из a"""
        a_hsv = self._rgb_to_hsv(a)
        b_hsv = self._rgb_to_hsv(b)
        
        result_hsv = np.stack([a_hsv[..., 0], b_hsv[..., 1], a_hsv[..., 2]], axis=-1)
        return self._hsv_to_rgb(result_hsv)
    
    def _blend_color(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Сохраняет hue и saturation из b, luminosity из a"""
        a_hsv = self._rgb_to_hsv(a)
        b_hsv = self._rgb_to_hsv(b)
        
        result_hsv = np.stack([b_hsv[..., 0], b_hsv[..., 1], a_hsv[..., 2]], axis=-1)
        return self._hsv_to_rgb(result_hsv)
    
    def _blend_luminosity(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Сохраняет luminosity из b, hue и saturation из a"""
        a_hsv = self._rgb_to_hsv(a)
        b_hsv = self._rgb_to_hsv(b)
        
        result_hsv = np.stack([a_hsv[..., 0], a_hsv[..., 1], b_hsv[..., 2]], axis=-1)
        return self._hsv_to_rgb(result_hsv)
    
    def _blend_height_based(self, a: np.ndarray, b: np.ndarray, 
                           height_map: np.ndarray, **kwargs) -> np.ndarray:
        """Смешивание на основе карты высот"""
        # Нормализуем карту высот
        height_norm = (height_map - height_map.min()) / (height_map.max() - height_map.min())
        
        # Параметры перехода
        low_threshold = kwargs.get('low_threshold', 0.3)
        high_threshold = kwargs.get('high_threshold', 0.7)
        transition = kwargs.get('transition', 0.1)
        
        # Создаем маску смешивания
        blend_mask = np.zeros_like(height_norm)
        
        # Нижний переход
        mask_low = height_norm < low_threshold
        blend_mask[mask_low] = 0
        
        # Верхний переход
        mask_high = height_norm > high_threshold
        blend_mask[mask_high] = 1
        
        # Переходная зона
        mask_transition = ~mask_low & ~mask_high
        height_transition = height_norm[mask_transition]
        
        # Плавный переход в переходной зоне
        t = (height_transition - low_threshold) / (high_threshold - low_threshold)
        if transition > 0:
            t = smoothstep(0, 1, t)
        
        blend_mask[mask_transition] = t
        
        # Применяем смешивание
        if blend_mask.ndim == 2 and a.ndim == 3:
            blend_mask = blend_mask[..., np.newaxis]
        
        return lerp(a, b, blend_mask)
    
    def _blend_slope_based(self, a: np.ndarray, b: np.ndarray,
                          height_map: np.ndarray, **kwargs) -> np.ndarray:
        """Смешивание на основе наклона (slope)"""
        # Вычисляем градиент высоты
        grad_x = np.gradient(height_map, axis=1)
        grad_y = np.gradient(height_map, axis=0)
        slope = np.sqrt(grad_x**2 + grad_y**2)
        
        # Нормализуем наклон
        slope_norm = slope / np.max(slope)
        
        # Параметры
        max_slope = kwargs.get('max_slope', 0.5)
        sharpness = kwargs.get('sharpness', 10.0)
        
        # Маска: 1 на ровных поверхностях, 0 на крутых склонах
        blend_mask = np.clip(1.0 - slope_norm / max_slope, 0, 1)
        
        # Применяем резкость перехода
        if sharpness != 1.0:
            blend_mask = np.power(blend_mask, sharpness)
        
        # Расширяем маску для многоканальных текстур
        if blend_mask.ndim == 2 and a.ndim == 3:
            blend_mask = blend_mask[..., np.newaxis]
        
        return lerp(b, a, blend_mask)  # На склонах - текстура A, на равнинах - B
    
    def _blend_edge_aware(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Edge-aware blending - сохраняет границы"""
        # Вычисляем градиенты обеих текстур
        if a.ndim == 3:
            # Используем luminance для градиентов
            a_gray = self._rgb_to_luminance(a)
            b_gray = self._rgb_to_luminance(b)
        else:
            a_gray = a
            b_gray = b
        
        # Градиенты
        a_grad_x = np.gradient(a_gray, axis=1)
        a_grad_y = np.gradient(a_gray, axis=0)
        a_grad = np.sqrt(a_grad_x**2 + a_grad_y**2)
        
        b_grad_x = np.gradient(b_gray, axis=1)
        b_grad_y = np.gradient(b_gray, axis=0)
        b_grad = np.sqrt(b_grad_x**2 + b_grad_y**2)
        
        # Маска на основе сравнения градиентов
        # Сохраняем более выраженные границы
        edge_strength = np.maximum(a_grad, b_grad)
        mask = a_grad / np.maximum(edge_strength, 1e-7)
        
        # Расширяем маску
        if mask.ndim == 2 and a.ndim == 3:
            mask = mask[..., np.newaxis]
        
        return lerp(b, a, mask)
    
    def _blend_detail_preserving(self, a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
        """Смешивание с сохранением деталей"""
        # Выделяем высокочастотные детали из A
        from scipy import ndimage
        
        # Низкочастотная версия A
        a_low = ndimage.gaussian_filter(a, sigma=kwargs.get('sigma', 2.0))
        
        # Высокочастотные детали A
        a_high = a - a_low
        
        # Смешиваем низкие частоты
        b_low = ndimage.gaussian_filter(b, sigma=kwargs.get('sigma', 2.0))
        blended_low = self._blend_overlay(a_low, b_low)
        
        # Добавляем обратно высокочастотные детали из A
        result = blended_low + a_high * kwargs.get('detail_strength', 0.7)
        
        return np.clip(result, 0, 1)
    
    # ------------------------------------------------------------------
    # Вспомогательные функции
    # ------------------------------------------------------------------
    
    def _gamma_correct(self, img: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Гамма-коррекция для линейного/перцептуального пространства"""
        if self.blend_space == "linear" or not self.gamma_correction:
            return img
        
        gamma = 2.2
        if inverse:
            return np.power(img, gamma)
        else:
            return np.power(img, 1.0/gamma)
    
    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """Конвертация RGB в HSV"""
        # Упрощенная реализация
        max_val = np.max(rgb, axis=-1)
        min_val = np.min(rgb, axis=-1)
        delta = max_val - min_val
        
        hue = np.zeros_like(max_val)
        saturation = np.zeros_like(max_val)
        value = max_val
        
        # Hue
        mask = delta > 0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        
        # Красный - максимум
        mask_r = mask & (max_val == r)
        hue[mask_r] = (g[mask_r] - b[mask_r]) / delta[mask_r]
        
        # Зеленый - максимум
        mask_g = mask & (max_val == g)
        hue[mask_g] = 2.0 + (b[mask_g] - r[mask_g]) / delta[mask_g]
        
        # Синий - максимум
        mask_b = mask & (max_val == b)
        hue[mask_b] = 4.0 + (r[mask_b] - g[mask_b]) / delta[mask_b]
        
        hue = (hue / 6.0) % 1.0
        
        # Saturation
        saturation[mask] = delta[mask] / max_val[mask]
        
        return np.stack([hue, saturation, value], axis=-1)
    
    def _hsv_to_rgb(self, hsv: np.ndarray) -> np.ndarray:
        """Конвертация HSV в RGB"""
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        
        h = (h * 6.0) % 6.0
        i = np.floor(h).astype(int)
        f = h - i
        
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        rgb = np.zeros_like(hsv)
        
        # 6 случаев
        mask0 = i == 0
        rgb[mask0, 0] = v[mask0]
        rgb[mask0, 1] = t[mask0]
        rgb[mask0, 2] = p[mask0]
        
        mask1 = i == 1
        rgb[mask1, 0] = q[mask1]
        rgb[mask1, 1] = v[mask1]
        rgb[mask1, 2] = p[mask1]
        
        mask2 = i == 2
        rgb[mask2, 0] = p[mask2]
        rgb[mask2, 1] = v[mask2]
        rgb[mask2, 2] = t[mask2]
        
        mask3 = i == 3
        rgb[mask3, 0] = p[mask3]
        rgb[mask3, 1] = q[mask3]
        rgb[mask3, 2] = v[mask3]
        
        mask4 = i == 4
        rgb[mask4, 0] = t[mask4]
        rgb[mask4, 1] = p[mask4]
        rgb[mask4, 2] = v[mask4]
        
        mask5 = i == 5
        rgb[mask5, 0] = v[mask5]
        rgb[mask5, 1] = p[mask5]
        rgb[mask5, 2] = q[mask5]
        
        return np.clip(rgb, 0, 1)
    
    def _rgb_to_luminance(self, rgb: np.ndarray) -> np.ndarray:
        """Конвертация RGB в luminance (яркость)"""
        # Стандартные коэффициенты для восприятия
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

# ----------------------------------------------------------------------
# Специализированные блендеры для terrain и природных материалов
# ----------------------------------------------------------------------

class TerrainTextureBlender:
    """Специализированный блендер для создания terrain текстур"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.blender = TextureBlender()
        self.mask_gen = BlendMask(seed=seed)
        
        # Предопределенные наборы текстур для разных биомов
        self.biome_presets = {
            "temperate": {
                "textures": ["grass", "dirt", "rock", "forest"],
                "height_ranges": [(0.0, 0.3), (0.2, 0.5), (0.4, 0.8), (0.7, 1.0)],
                "slope_thresholds": [0.3, 0.5, 0.7],
            },
            "desert": {
                "textures": ["sand", "rock", "cliff"],
                "height_ranges": [(0.0, 0.4), (0.3, 0.7), (0.6, 1.0)],
                "slope_thresholds": [0.2, 0.4],
            },
            "arctic": {
                "textures": ["snow", "rock", "ice"],
                "height_ranges": [(0.0, 0.5), (0.4, 0.8), (0.7, 1.0)],
                "slope_thresholds": [0.4, 0.6],
            },
            "mountain": {
                "textures": ["forest", "rock", "snow"],
                "height_ranges": [(0.0, 0.4), (0.3, 0.7), (0.6, 1.0)],
                "slope_thresholds": [0.5, 0.8],
            }
        }
    
    def create_terrain_material(self,
                               height_map: np.ndarray,
                               texture_layers: Dict[str, np.ndarray],
                               biome: str = "temperate",
                               custom_params: Optional[Dict] = None) -> np.ndarray:
        """
        Создание сложного terrain материала на основе карты высот
        
        Args:
            height_map: Карта высот (H, W)
            texture_layers: Словарь текстур для слоев
            biome: Тип биома (preset)
            custom_params: Кастомные параметры смешивания
            
        Returns:
            Итоговая terrain текстура (H, W, 4)
        """
        # Получаем preset для биома
        if biome in self.biome_presets:
            preset = self.biome_presets[biome]
        else:
            preset = self.biome_presets["temperate"]
        
        # Объединяем с кастомными параметрами
        params = preset.copy()
        if custom_params:
            params.update(custom_params)
        
        # Нормализуем карту высот
        height_norm = (height_map - height_map.min()) / (height_map.max() - height_map.min())
        
        # Вычисляем карту наклона
        grad_x = np.gradient(height_norm, axis=1)
        grad_y = np.gradient(height_norm, axis=0)
        slope = np.sqrt(grad_x**2 + grad_y**2)
        
        # Начинаем с базовой текстуры
        base_texture_name = params["textures"][0]
        if base_texture_name in texture_layers:
            result = texture_layers[base_texture_name].copy()
        else:
            # Создаем простую текстуру если отсутствует
            result = np.ones((*height_map.shape, 4), dtype=np.float32) * 0.5
        
        # Последовательно добавляем слои
        for i in range(1, len(params["textures"])):
            texture_name = params["textures"][i]
            if texture_name not in texture_layers:
                continue
            
            # Получаем диапазон высот для этого слоя
            height_range = params["height_ranges"][i]
            
            # Создаем маску на основе высоты
            height_mask = np.zeros_like(height_norm)
            
            # Плавный переход между слоями
            transition = 0.1  # Ширина переходной зоны
            
            # Нижняя граница
            low_start = height_range[0] - transition/2
            low_end = height_range[0] + transition/2
            
            # Верхняя граница  
            high_start = height_range[1] - transition/2
            high_end = height_range[1] + transition/2
            
            # Нижний переход
            mask_low = height_norm <= low_start
            height_mask[mask_low] = 0
            
            # Верхний переход
            mask_high = height_norm >= high_end
            height_mask[mask_high] = 0
            
            # Нижняя переходная зона
            mask_low_trans = (height_norm > low_start) & (height_norm <= low_end)
            if np.any(mask_low_trans):
                t_low = (height_norm[mask_low_trans] - low_start) / transition
                height_mask[mask_low_trans] = smootherstep(0, 1, t_low)
            
            # Верхняя переходная зона
            mask_high_trans = (height_norm >= high_start) & (height_norm < high_end)
            if np.any(mask_high_trans):
                t_high = 1 - (height_norm[mask_high_trans] - high_start) / transition
                height_mask[mask_high_trans] = smootherstep(0, 1, t_high)
            
            # Основная зона
            mask_mid = (height_norm > low_end) & (height_norm < high_start)
            height_mask[mask_mid] = 1.0
            
            # Корректируем маску на основе наклона для некоторых слоев
            if i > 0:  # Для не-базовых слоев
                slope_threshold = params["slope_thresholds"][i-1] if i-1 < len(params["slope_thresholds"]) else 0.5
                slope_mask = np.where(slope > slope_threshold, 0, 1)
                height_mask *= slope_mask
            
            # Смешиваем слой
            result = self.blender.blend(
                result, texture_layers[texture_name],
                blend_mode="overlay",
                opacity=1.0,
                mask=height_mask[..., np.newaxis]
            )
        
        return result
    
    def add_detail_layers(self,
                         base_texture: np.ndarray,
                         detail_textures: List[np.ndarray],
                         scale_factors: List[float] = None,
                         blend_modes: List[str] = None) -> np.ndarray:
        """
        Добавление детализированных слоев (трава, камни, etc.)
        
        Args:
            base_texture: Базовая текстура
            detail_textures: Список текстур деталей
            scale_factors: Масштаб для каждой текстуры
            blend_modes: Режимы смешивания для каждой текстуры
            
        Returns:
            Детализированная текстура
        """
        result = base_texture.copy()
        
        if scale_factors is None:
            scale_factors = [1.0] * len(detail_textures)
        
        if blend_modes is None:
            blend_modes = ["overlay"] * len(detail_textures)
        
        for i, (detail_tex, scale, blend_mode) in enumerate(zip(detail_textures, scale_factors, blend_modes)):
            # Создаем маску для деталей на основе шума
            height, width = base_texture.shape[:2]
            mask_gen = BlendMask(
                mask_type="noise",
                parameters={
                    "noise_scale": 0.05 * scale,
                    "octaves": 3,
                    "threshold": 0.6,
                },
            )
            mask = mask_gen.generate(width, height)
            
            # Применяем детали
            result = self.blender.blend(
                result, detail_tex,
                blend_mode=blend_mode,
                opacity=0.7,
                mask=mask[..., np.newaxis]
            )
        
        return result
    
    def create_biome_transition(self,
                               terrain_a: np.ndarray,
                               terrain_b: np.ndarray,
                               transition_map: np.ndarray,
                               transition_width: float = 0.3) -> np.ndarray:
        """
        Создание плавного перехода между двумя биомами
        
        Args:
            terrain_a: Текстура первого биома
            terrain_b: Текстура второго биома  
            transition_map: Карта перехода (0 = биом A, 1 = биом B)
            transition_width: Ширина переходной зоны
            
        Returns:
            Текстура с плавным переходом
        """
        # Создаем маску перехода с плавными краями
        transition_smooth = np.zeros_like(transition_map)
        
        # Плавные переходы
        mask_low = transition_map <= 0.5 - transition_width/2
        transition_smooth[mask_low] = 0
        
        mask_high = transition_map >= 0.5 + transition_width/2
        transition_smooth[mask_high] = 1
        
        # Переходная зона
        mask_trans = ~mask_low & ~mask_high
        if np.any(mask_trans):
            t = (transition_map[mask_trans] - (0.5 - transition_width/2)) / transition_width
            transition_smooth[mask_trans] = smootherstep(0, 1, t)
        
        # Расширяем маску для RGB(A)
        if transition_smooth.ndim == 2 and terrain_a.ndim == 3:
            transition_smooth = transition_smooth[..., np.newaxis]
        
        # Смешиваем
        return lerp(terrain_a, terrain_b, transition_smooth)

# ----------------------------------------------------------------------
# Продвинутые алгоритмы смешивания для специальных эффектов
# ----------------------------------------------------------------------

class AdvancedTextureBlending:
    """Продвинутые алгоритмы для сложных эффектов смешивания"""
    
    @staticmethod
    def triplanar_mapping(texture: np.ndarray,
                         positions: np.ndarray,
                         normals: np.ndarray,
                         scale: float = 1.0) -> np.ndarray:
        """
        Triplanar mapping - проекция текстуры на 3D поверхность
        без искажений на крутых склонах
        
        Args:
            texture: 2D текстура для проекции
            positions: Позиции вершин (N, 3)
            normals: Нормали вершин (N, 3)
            scale: Масштаб текстуры
            
        Returns:
            Текстура, спроецированная на 3D поверхность
        """
        # Для каждой оси (X, Y, Z) проецируем текстуру
        # и смешиваем на основе нормалей
        
        # Упрощенная реализация для демонстрации
        # В реальности нужна более сложная логика для 3D текстур
        
        # Веса для каждой плоскости
        weights = np.abs(normals)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        
        # Здесь должна быть логика выборки из текстуры для каждой плоскости
        # ...
        
        # Возвращаем placeholder
        return np.zeros((*positions.shape[:-1], texture.shape[-1]), dtype=np.float32)
    
    @staticmethod
    def parallax_occlusion_mapping(height_map: np.ndarray,
                                  base_texture: np.ndarray,
                                  view_direction: np.ndarray,
                                  strength: float = 0.1,
                                  num_layers: int = 10) -> np.ndarray:
        """
        Parallax Occlusion Mapping - псевдо-3D эффект
        
        Args:
            height_map: Карта высот для смещения
            base_texture: Базовая текстура
            view_direction: Направление взгляда (H, W, 3)
            strength: Сила эффекта
            num_layers: Количество слоев для ray marching
            
        Returns:
            Текстура с эффектом глубины
        """
        height, width = height_map.shape[:2]
        result = np.zeros_like(base_texture)
        
        # Упрощенная реализация
        for i in range(num_layers):
            # Вычисляем текущий слой
            layer_depth = i / num_layers
            
            # Смещение на основе карты высот и направления взгляда
            offset = view_direction[..., :2] * strength * layer_depth * height_map[..., np.newaxis]
            
            # Вычисляем координаты для выборки
            # (нужна реальная логика билинейной интерполяции)
            pass
        
        return result
    
    @staticmethod
    def distance_field_blending(texture_a: np.ndarray,
                               texture_b: np.ndarray,
                               distance_field: np.ndarray,
                               threshold: float = 0.5,
                               smoothness: float = 0.1) -> np.ndarray:
        """
        Смешивание на основе distance field (полей расстояний)
        
        Args:
            texture_a: Текстура A
            texture_b: Текстура B
            distance_field: Поле расстояний (чем ближе к 0, тем ближе к границе)
            threshold: Пороговое значение
            smoothness: Плавность перехода
            
        Returns:
            Смешанная текстура
        """
        # Нормализуем distance field
        df_norm = (distance_field - distance_field.min()) / (distance_field.max() - distance_field.min())
        
        # Создаем маску на основе расстояния до границы
        mask = np.zeros_like(df_norm)
        
        # Область A
        mask_a = df_norm < threshold - smoothness
        mask[mask_a] = 0
        
        # Область B
        mask_b = df_norm > threshold + smoothness
        mask[mask_b] = 1
        
        # Переходная область
        mask_trans = ~mask_a & ~mask_b
        if np.any(mask_trans):
            t = (df_norm[mask_trans] - (threshold - smoothness)) / (2 * smoothness)
            mask[mask_trans] = smootherstep(0, 1, t)
        
        # Расширяем маску если нужно
        if mask.ndim == 2 and texture_a.ndim == 3:
            mask = mask[..., np.newaxis]
        
        return lerp(texture_a, texture_b, mask)

# ----------------------------------------------------------------------
# Примеры использования
# ----------------------------------------------------------------------

def example_terrain_creation():
    """Пример создания сложного terrain материала"""
    
    print("Creating complex terrain material example...")
    
    # Инициализация
    terrain_blender = TerrainTextureBlender(seed=42)
    
    # Создаем тестовые текстуры (в реальности они были бы сгенерированы)
    height, width = 512, 512
    
    # Карта высот (симуляция горного ландшафта)
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    xx, yy = np.meshgrid(x, y)
    
    height_map = np.sin(xx) * np.cos(yy) * 0.5 + 0.5
    
    # Тестовые текстуры слоев
    texture_layers = {}
    
    # Grass texture (зеленый)
    grass = np.zeros((height, width, 4), dtype=np.float32)
    grass[..., 0] = 0.2 + np.sin(xx * 5) * 0.1  # R
    grass[..., 1] = 0.6 + np.cos(yy * 3) * 0.2  # G
    grass[..., 2] = 0.1 + np.sin(xx + yy) * 0.05 # B
    grass[..., 3] = 1.0  # A
    texture_layers["grass"] = grass
    
    # Dirt texture (коричневый)
    dirt = np.zeros((height, width, 4), dtype=np.float32)
    dirt[..., 0] = 0.4 + np.sin(xx * 3) * 0.1  # R
    dirt[..., 1] = 0.3 + np.cos(yy * 2) * 0.1  # G
    dirt[..., 2] = 0.2 + np.sin(xx * yy) * 0.05 # B
    dirt[..., 3] = 1.0  # A
    texture_layers["dirt"] = dirt
    
    # Rock texture (серый)
    rock = np.zeros((height, width, 4), dtype=np.float32)
    rock[..., 0] = 0.5 + np.sin(xx * 10) * 0.2  # R
    rock[..., 1] = 0.5 + np.cos(yy * 8) * 0.2   # G
    rock[..., 2] = 0.5 + np.sin(xx * yy * 2) * 0.2 # B
    rock[..., 3] = 1.0  # A
    texture_layers["rock"] = rock
    
    # Forest texture (темно-зеленый)
    forest = np.zeros((height, width, 4), dtype=np.float32)
    forest[..., 0] = 0.1 + np.sin(xx * 7) * 0.05  # R
    forest[..., 1] = 0.4 + np.cos(yy * 5) * 0.1   # G
    forest[..., 2] = 0.1 + np.sin(xx + yy * 2) * 0.05 # B
    forest[..., 3] = 1.0  # A
    texture_layers["forest"] = forest
    
    # Создаем terrain материал
    terrain = terrain_blender.create_terrain_material(
        height_map=height_map,
        texture_layers=texture_layers,
        biome="mountain",
        custom_params={
            "height_ranges": [(0.0, 0.3), (0.2, 0.5), (0.4, 0.8), (0.7, 1.0)],
            "slope_thresholds": [0.4, 0.6, 0.8]
        }
    )
    
    print(f"Terrain texture shape: {terrain.shape}")
    print(f"Min/Max: {terrain.min():.3f}/{terrain.max():.3f}")
    
    return terrain

def example_advanced_blending():
    """Пример продвинутого смешивания текстур"""
    
    print("\nAdvanced texture blending example...")
    
    # Инициализация
    blender = TextureBlender()
    
    # Создаем тестовые текстуры
    size = 256
    texture1 = np.zeros((size, size, 3), dtype=np.float32)
    texture2 = np.zeros((size, size, 3), dtype=np.float32)
    
    # Простые градиентные текстуры
    for i in range(size):
        for j in range(size):
            # Текстура 1: Вертикальный градиент
            texture1[i, j, 0] = i / size  # Красный
            texture1[i, j, 1] = j / size  # Зеленый
            texture1[i, j, 2] = 0.5       # Синий
            
            # Текстура 2: Горизонтальный градиент с шумом
            texture2[i, j, 0] = j / size + np.sin(i * 0.1) * 0.2
            texture2[i, j, 1] = i / size * 0.5
            texture2[i, j, 2] = j / size
    
    # Создаем маску смешивания (радиальный градиент)
    mask_gen = BlendMask(mask_type="gradient", 
                        parameters={"center": (0.5, 0.5), "radius": 0.8})
    mask = mask_gen.generate(size, size, scale=0.01)
    
    # Тестируем разные режимы смешивания
    blend_modes = ["overlay", "multiply", "screen", "soft_light", "hard_light"]
    
    results = {}
    for mode in blend_modes:
        result = blender.blend(texture1, texture2, 
                              blend_mode=mode,
                              opacity=1.0,
                              mask=mask)
        results[mode] = result
    
    print(f"Generated {len(results)} blended textures")
    
    return results

if __name__ == "__main__":
    print("Texture Blending Algorithms")
    print("=" * 60)
    
    # Пример создания terrain
    terrain = example_terrain_creation()
    
    # Пример продвинутого смешивания
    blended_textures = example_advanced_blending()
    
    print("\n" + "=" * 60)
    print("Available blend modes in TextureBlender:")
    print("-" * 40)
    
    blender = TextureBlender()
    modes = list(blender.blend_modes.keys())
    
    # Группируем по категориям
    categories = {
        "Basic": ["normal", "dissolve"],
        "Darkening": ["darken", "multiply", "color_burn", "linear_burn"],
        "Lightening": ["lighten", "screen", "color_dodge", "linear_dodge", "add"],
        "Contrast": ["overlay", "soft_light", "hard_light", "vivid_light", 
                    "linear_light", "pin_light", "hard_mix"],
        "Comparative": ["difference", "subtract", "divide"],
        "Component": ["hue", "saturation", "color", "luminosity"],
        "Specialized": ["height_blend", "slope_blend", "edge_blend", "detail_preserving"]
    }
    
    for category, mode_list in categories.items():
        print(f"\n{category}:")
        print(f"  {', '.join(mode_list)}")
    
    print("\n" + "=" * 60)
    print("Texture blending system ready for use!")
