# fractex/simplex_noise.py
"""
Полная реализация 2D, 3D и 4D симплекс-шума
На основе алгоритма Стифана Густавсона (Stefan Gustavson)
"""

import numpy as np
from typing import Tuple, Optional, Union
from numba import jit, prange
import math

# ----------------------------------------------------------------------
# Константы для симплекс-шума
# ----------------------------------------------------------------------

# Градиенты для 2D, 3D и 4D шума
_GRAD3 = np.array([
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]
], dtype=np.float32)

_GRAD4 = np.array([
    [0, 1, 1, 1], [0, 1, 1, -1], [0, 1, -1, 1], [0, 1, -1, -1],
    [0, -1, 1, 1], [0, -1, 1, -1], [0, -1, -1, 1], [0, -1, -1, -1],
    [1, 0, 1, 1], [1, 0, 1, -1], [1, 0, -1, 1], [1, 0, -1, -1],
    [-1, 0, 1, 1], [-1, 0, 1, -1], [-1, 0, -1, 1], [-1, 0, -1, -1],
    [1, 1, 0, 1], [1, 1, 0, -1], [1, -1, 0, 1], [1, -1, 0, -1],
    [-1, 1, 0, 1], [-1, 1, 0, -1], [-1, -1, 0, 1], [-1, -1, 0, -1],
    [1, 1, 1, 0], [1, 1, -1, 0], [1, -1, 1, 0], [1, -1, -1, 0],
    [-1, 1, 1, 0], [-1, 1, -1, 0], [-1, -1, 1, 0], [-1, -1, -1, 0]
], dtype=np.float32)

# Таблица перестановок (классическая из шума Перлина)
_PERM = np.array([
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140,
    36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120,
    234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
    88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71,
    134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133,
    230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161,
    1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130,
    116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250,
    124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227,
    47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44,
    154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98,
    108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34,
    242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14,
    239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121,
    50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243,
    141, 128, 195, 78, 66, 215, 61, 156, 180
], dtype=np.uint8)

# Дублируем для быстрого доступа
_PERM_EXTENDED = np.concatenate([_PERM, _PERM])

# Коэффициенты для симплекс-шума
_F2 = 0.5 * (np.sqrt(3.0) - 1.0)
_G2 = (3.0 - np.sqrt(3.0)) / 6.0
_F3 = 1.0 / 3.0
_G3 = 1.0 / 6.0
_F4 = (np.sqrt(5.0) - 1.0) / 4.0
_G4 = (5.0 - np.sqrt(5.0)) / 20.0

# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------

@jit(nopython=True, cache=True)
def _dot2(g: np.ndarray, x: float, y: float) -> float:
    """Скалярное произведение 2D градиента с вектором (x, y)"""
    return g[0] * x + g[1] * y

@jit(nopython=True, cache=True)
def _dot3(g: np.ndarray, x: float, y: float, z: float) -> float:
    """Скалярное произведение 3D градиента с вектором (x, y, z)"""
    return g[0] * x + g[1] * y + g[2] * z

@jit(nopython=True, cache=True)
def _dot4(g: np.ndarray, x: float, y: float, z: float, w: float) -> float:
    """Скалярное произведение 4D градиента с вектором (x, y, z, w)"""
    return g[0] * x + g[1] * y + g[2] * z + g[3] * w

@jit(nopython=True, cache=True)
def _fast_floor(x: float) -> int:
    """Быстрое вычисление floor для положительных и отрицательных чисел"""
    xi = int(x)
    return xi if x >= xi else xi - 1

# ----------------------------------------------------------------------
# 2D симплекс-шум
# ----------------------------------------------------------------------

@jit(nopython=True, cache=True)
def simplex_noise_2d(x: float, y: float, perm: np.ndarray = _PERM_EXTENDED) -> float:
    """
    2D симплекс-шум
    
    Args:
        x, y: Координаты
        perm: Таблица перестановок (длиной 512)
    
    Returns:
        Значение шума в диапазоне примерно [-1, 1]
    """
    # Константы для 2D симплекса
    F2 = _F2
    G2 = _G2
    
    # Шаг 1: Скалярная сумма для определения симплекса
    s = (x + y) * F2
    i = _fast_floor(x + s)
    j = _fast_floor(y + s)
    
    t = (i + j) * G2
    X0 = i - t
    Y0 = j - t
    x0 = x - X0
    y0 = y - Y0
    
    # Шаг 2: Определяем, в каком треугольнике находимся (верхний или нижний)
    i1, j1 = (1, 0) if x0 > y0 else (0, 1)
    
    # Координаты внутри симплекса
    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2
    y2 = y0 - 1.0 + 2.0 * G2
    
    # Шаг 3: Хешируем углы симплекса
    ii = i & 255
    jj = j & 255
    
    # Градиенты в трех углах
    gi0 = perm[ii + perm[jj]] % 12
    gi1 = perm[ii + i1 + perm[jj + j1]] % 12
    gi2 = perm[ii + 1 + perm[jj + 1]] % 12
    
    # Шаг 4: Вычисляем вклад от каждого угла
    t0 = 0.5 - x0 * x0 - y0 * y0
    n0 = 0.0
    if t0 > 0:
        t0 *= t0
        n0 = t0 * t0 * _dot2(_GRAD3[gi0], x0, y0)
    
    t1 = 0.5 - x1 * x1 - y1 * y1
    n1 = 0.0
    if t1 > 0:
        t1 *= t1
        n1 = t1 * t1 * _dot2(_GRAD3[gi1], x1, y1)
    
    t2 = 0.5 - x2 * x2 - y2 * y2
    n2 = 0.0
    if t2 > 0:
        t2 *= t2
        n2 = t2 * t2 * _dot2(_GRAD3[gi2], x2, y2)
    
    # Шаг 5: Возвращаем результат
    return 70.0 * (n0 + n1 + n2)

# ----------------------------------------------------------------------
# 3D симплекс-шум
# ----------------------------------------------------------------------

@jit(nopython=True, cache=True)
def simplex_noise_3d(x: float, y: float, z: float, perm: np.ndarray = _PERM_EXTENDED) -> float:
    """
    3D симплекс-шум
    
    Args:
        x, y, z: Координаты
        perm: Таблица перестановок (длиной 512)
    
    Returns:
        Значение шума в диапазоне примерно [-1, 1]
    """
    # Константы для 3D симплекса
    F3 = _F3
    G3 = _G3
    
    # Шаг 1: Скалярная сумма для определения симплекса
    s = (x + y + z) * F3
    i = _fast_floor(x + s)
    j = _fast_floor(y + s)
    k = _fast_floor(z + s)
    
    t = (i + j + k) * G3
    X0 = i - t
    Y0 = j - t
    Z0 = k - t
    x0 = x - X0
    y0 = y - Y0
    z0 = z - Z0
    
    # Шаг 2: Определяем, в каком тетраэдре находимся
    if x0 >= y0:
        if y0 >= z0:        # XYZ order
            i1, j1, k1 = 1, 0, 0
            i2, j2, k2 = 1, 1, 0
        elif x0 >= z0:      # XZY order
            i1, j1, k1 = 1, 0, 0
            i2, j2, k2 = 1, 0, 1
        else:               # ZXY order
            i1, j1, k1 = 0, 0, 1
            i2, j2, k2 = 1, 0, 1
    else:
        if y0 < z0:         # ZYX order
            i1, j1, k1 = 0, 0, 1
            i2, j2, k2 = 0, 1, 1
        elif x0 < z0:       # YZX order
            i1, j1, k1 = 0, 1, 0
            i2, j2, k2 = 0, 1, 1
        else:               # YXZ order
            i1, j1, k1 = 0, 1, 0
            i2, j2, k2 = 1, 1, 0
    
    # Координаты внутри симплекса
    x1 = x0 - i1 + G3
    y1 = y0 - j1 + G3
    z1 = z0 - k1 + G3
    x2 = x0 - i2 + 2.0 * G3
    y2 = y0 - j2 + 2.0 * G3
    z2 = z0 - k2 + 2.0 * G3
    x3 = x0 - 1.0 + 3.0 * G3
    y3 = y0 - 1.0 + 3.0 * G3
    z3 = z0 - 1.0 + 3.0 * G3
    
    # Шаг 3: Хешируем углы тетраэдра
    ii = i & 255
    jj = j & 255
    kk = k & 255
    
    # Градиенты в четырех углах
    gi0 = perm[ii + perm[jj + perm[kk]]] % 12
    gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12
    gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12
    gi3 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12
    
    # Шаг 4: Вычисляем вклад от каждого угла
    t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
    n0 = 0.0
    if t0 > 0:
        t0 *= t0
        n0 = t0 * t0 * _dot3(_GRAD3[gi0], x0, y0, z0)
    
    t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
    n1 = 0.0
    if t1 > 0:
        t1 *= t1
        n1 = t1 * t1 * _dot3(_GRAD3[gi1], x1, y1, z1)
    
    t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
    n2 = 0.0
    if t2 > 0:
        t2 *= t2
        n2 = t2 * t2 * _dot3(_GRAD3[gi2], x2, y2, z2)
    
    t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
    n3 = 0.0
    if t3 > 0:
        t3 *= t3
        n3 = t3 * t3 * _dot3(_GRAD3[gi3], x3, y3, z3)
    
    # Шаг 5: Возвращаем результат
    return 32.0 * (n0 + n1 + n2 + n3)

# ----------------------------------------------------------------------
# 4D симплекс-шум (для анимированных текстур)
# ----------------------------------------------------------------------

@jit(nopython=True, cache=True)
def simplex_noise_4d(x: float, y: float, z: float, w: float, 
                     perm: np.ndarray = _PERM_EXTENDED) -> float:
    """
    4D симплекс-шум
    
    Args:
        x, y, z, w: Координаты
        perm: Таблица перестановок (длиной 512)
    
    Returns:
        Значение шума в диапазоне примерно [-1, 1]
    """
    # Константы для 4D симплекса
    F4 = _F4
    G4 = _G4
    
    # Шаг 1: Скалярная сумма для определения симплекса
    s = (x + y + z + w) * F4
    i = _fast_floor(x + s)
    j = _fast_floor(y + s)
    k = _fast_floor(z + s)
    l = _fast_floor(w + s)
    
    t = (i + j + k + l) * G4
    X0 = i - t
    Y0 = j - t
    Z0 = k - t
    W0 = l - t
    x0 = x - X0
    y0 = y - Y0
    z0 = z - Z0
    w0 = w - W0
    
    # Шаг 2: Сортируем координаты для определения симплекса
    # Это сложная часть - нужно определить, в каком из 24 4D симплексов мы находимся
    
    # Создаем массив для сортировки
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    
    if x0 > y0:
        c1 ^= 1
        c2 ^= 1
    if x0 > z0:
        c1 ^= 1
        c3 ^= 1
    if x0 > w0:
        c1 ^= 1
        c4 ^= 1
    if y0 > z0:
        c2 ^= 1
        c3 ^= 1
    if y0 > w0:
        c2 ^= 1
        c4 ^= 1
    if z0 > w0:
        c3 ^= 1
        c4 ^= 1
    
    # Определяем индексы симплекса
    i1 = 1 if (c1 != 0) else 0
    j1 = 1 if (c2 != 0) else 0
    k1 = 1 if (c3 != 0) else 0
    l1 = 1 if (c4 != 0) else 0
    
    i2 = 1 if (c1 ^ c2) != 0 else 0
    j2 = 1 if (c2 ^ c3) != 0 else 0
    k2 = 1 if (c3 ^ c4) != 0 else 0
    l2 = 1 if (c4 ^ c1) != 0 else 0
    
    i3 = 1 if (c1 ^ c2 ^ c3) != 0 else 0
    j3 = 1 if (c2 ^ c3 ^ c4) != 0 else 0
    k3 = 1 if (c3 ^ c4 ^ c1) != 0 else 0
    l3 = 1 if (c4 ^ c1 ^ c2) != 0 else 0
    
    i4 = 1
    j4 = 1
    k4 = 1
    l4 = 1
    
    # Координаты внутри симплекса
    x1 = x0 - i1 + G4
    y1 = y0 - j1 + G4
    z1 = z0 - k1 + G4
    w1 = w0 - l1 + G4
    x2 = x0 - i2 + 2.0 * G4
    y2 = y0 - j2 + 2.0 * G4
    z2 = z0 - k2 + 2.0 * G4
    w2 = w0 - l2 + 2.0 * G4
    x3 = x0 - i3 + 3.0 * G4
    y3 = y0 - j3 + 3.0 * G4
    z3 = z0 - k3 + 3.0 * G4
    w3 = w0 - l3 + 3.0 * G4
    x4 = x0 - i4 + 4.0 * G4
    y4 = y0 - j4 + 4.0 * G4
    z4 = z0 - k4 + 4.0 * G4
    w4 = w0 - l4 + 4.0 * G4
    
    # Шаг 3: Хешируем углы симплекса
    ii = i & 255
    jj = j & 255
    kk = k & 255
    ll = l & 255
    
    # Градиенты в пяти углах
    gi0 = perm[ii + perm[jj + perm[kk + perm[ll]]]] % 32
    gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1 + perm[ll + l1]]]] % 32
    gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2 + perm[ll + l2]]]] % 32
    gi3 = perm[ii + i3 + perm[jj + j3 + perm[kk + k3 + perm[ll + l3]]]] % 32
    gi4 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1 + perm[ll + 1]]]] % 32
    
    # Шаг 4: Вычисляем вклад от каждого угла
    t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0 - w0 * w0
    n0 = 0.0
    if t0 > 0:
        t0 *= t0
        n0 = t0 * t0 * _dot4(_GRAD4[gi0], x0, y0, z0, w0)
    
    t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1 - w1 * w1
    n1 = 0.0
    if t1 > 0:
        t1 *= t1
        n1 = t1 * t1 * _dot4(_GRAD4[gi1], x1, y1, z1, w1)
    
    t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2 - w2 * w2
    n2 = 0.0
    if t2 > 0:
        t2 *= t2
        n2 = t2 * t2 * _dot4(_GRAD4[gi2], x2, y2, z2, w2)
    
    t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3 - w3 * w3
    n3 = 0.0
    if t3 > 0:
        t3 *= t3
        n3 = t3 * t3 * _dot4(_GRAD4[gi3], x3, y3, z3, w3)
    
    t4 = 0.6 - x4 * x4 - y4 * y4 - z4 * z4 - w4 * w4
    n4 = 0.0
    if t4 > 0:
        t4 *= t4
        n4 = t4 * t4 * _dot4(_GRAD4[gi4], x4, y4, z4, w4)
    
    # Шаг 5: Возвращаем результат
    return 27.0 * (n0 + n1 + n2 + n3 + n4)

# ----------------------------------------------------------------------
# Векторизованные версии для работы с массивами
# ----------------------------------------------------------------------

@jit(nopython=True, parallel=True, cache=True)
def simplex_noise_2d_array(x: np.ndarray, y: np.ndarray, 
                          perm: np.ndarray = _PERM_EXTENDED) -> np.ndarray:
    """Векторизованная версия 2D симплекс-шума"""
    shape = x.shape
    result = np.zeros(shape, dtype=np.float32)
    
    for i in prange(shape[0]):
        for j in range(shape[1]):
            result[i, j] = simplex_noise_2d(x[i, j], y[i, j], perm)
    
    return result

@jit(nopython=True, parallel=True, cache=True)
def simplex_noise_3d_array(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          perm: np.ndarray = _PERM_EXTENDED) -> np.ndarray:
    """Векторизованная версия 3D симплекс-шума"""
    shape = x.shape
    result = np.zeros(shape, dtype=np.float32)
    
    for i in prange(shape[0]):
        for j in range(shape[1]):
            result[i, j] = simplex_noise_3d(x[i, j], y[i, j], z[i, j], perm)
    
    return result

# ----------------------------------------------------------------------
# Класс для удобной работы с симплекс-шумом
# ----------------------------------------------------------------------

class SimplexNoise:
    """Удобный интерфейс для работы с симплекс-шумом"""
    
    def __init__(self, seed: int = 42):
        """
        Инициализация генератора шума
        
        Args:
            seed: Семя для генерации таблицы перестановок
        """
        self.seed = seed
        self.perm = self._generate_permutation(seed)
        self.perm_extended = np.concatenate([self.perm, self.perm])
        
    def _generate_permutation(self, seed: int) -> np.ndarray:
        """Генерация таблицы перестановок на основе seed"""
        np.random.seed(seed)
        perm = np.random.permutation(256).astype(np.uint8)
        return perm
    
    def noise_2d(self, x: Union[float, np.ndarray], 
                 y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """2D симплекс-шум"""
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return simplex_noise_2d_array(x, y, self.perm_extended)
        else:
            return simplex_noise_2d(x, y, self.perm_extended)
    
    def noise_3d(self, x: Union[float, np.ndarray],
                 y: Union[float, np.ndarray],
                 z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """3D симплекс-шум"""
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(z, np.ndarray):
            if x.ndim <= 2:
                return simplex_noise_3d_array(x, y, z, self.perm_extended)
            result = np.zeros_like(x, dtype=np.float32)
            it = np.nditer(x, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                result[idx] = simplex_noise_3d(
                    float(x[idx]), float(y[idx]), float(z[idx]), self.perm_extended
                )
                it.iternext()
            return result
        return simplex_noise_3d(x, y, z, self.perm_extended)
    
    def noise_4d(self, x: float, y: float, z: float, w: float) -> float:
        """4D симплекс-шум"""
        return simplex_noise_4d(x, y, z, w, self.perm_extended)
    
    def fractal_noise_2d(self, x: np.ndarray, y: np.ndarray,
                         octaves: int = 8, persistence: float = 0.5,
                         lacunarity: float = 2.0, 
                         base_scale: float = 1.0) -> np.ndarray:
        """
        Фрактальный шум (fBm) на основе симплекс-шума
        
        Args:
            x, y: Координатные сетки
            octaves: Количество октав
            persistence: Сохранение амплитуды между октавами
            lacunarity: Умножение частоты между октавами
            base_scale: Базовый масштаб
            
        Returns:
            Массив значений фрактального шума
        """
        result = np.zeros_like(x, dtype=np.float32)
        amplitude = 1.0
        frequency = base_scale
        
        for i in range(octaves):
            # Генерируем шум для текущей октавы
            nx = x * frequency
            ny = y * frequency
            
            noise = self.noise_2d(nx, ny)
            result += amplitude * noise
            
            # Подготовка для следующей октавы
            amplitude *= persistence
            frequency *= lacunarity
        
        # Нормализация (примерная)
        max_val = (1 - persistence ** octaves) / (1 - persistence)
        return result / max_val if max_val > 0 else result

    def fractal_noise_3d(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         octaves: int = 8, persistence: float = 0.5,
                         lacunarity: float = 2.0,
                         base_scale: float = 1.0) -> np.ndarray:
        """
        Фрактальный 3D шум (fBm) на основе симплекс-шума
        """
        result = np.zeros_like(x, dtype=np.float32)
        amplitude = 1.0
        frequency = base_scale
        
        for _ in range(octaves):
            nx = x * frequency
            ny = y * frequency
            nz = z * frequency
            noise = self.noise_3d(nx, ny, nz)
            result += amplitude * noise
            amplitude *= persistence
            frequency *= lacunarity
        
        max_val = (1 - persistence ** octaves) / (1 - persistence)
        return result / max_val if max_val > 0 else result
    
    def ridged_noise_2d(self, x: np.ndarray, y: np.ndarray,
                        octaves: int = 8, persistence: float = 0.5,
                        lacunarity: float = 2.0, 
                        base_scale: float = 1.0,
                        offset: float = 1.0,
                        gain: float = 2.0) -> np.ndarray:
        """
        Риджид-шум (ridged multifractal) - для острых горных хребтов
        
        Args:
            x, y: Координатные сетки
            octaves: Количество октав
            persistence: Сохранение амплитуды
            lacunarity: Умножение частоты
            base_scale: Базовый масштаб
            offset: Смещение для создания риджей
            gain: Усиление сигнала
            
        Returns:
            Массив значений риджид-шума
        """
        result = np.zeros_like(x, dtype=np.float32)
        amplitude = 1.0
        frequency = base_scale
        weight = 1.0
        
        for i in range(octaves):
            # Генерируем шум для текущей октавы
            nx = x * frequency
            ny = y * frequency
            
            noise = self.noise_2d(nx, ny)
            
            # Делаем абсолютное значение и инвертируем для создания риджей
            signal = offset - np.abs(noise)
            signal *= signal
            signal *= weight
            
            weight = np.clip(signal * gain, 0.0, 1.0)
            
            result += signal * amplitude
            
            # Подготовка для следующей октавы
            amplitude *= persistence
            frequency *= lacunarity
        
        return result / octaves
    
    def billow_noise_2d(self, x: np.ndarray, y: np.ndarray,
                        octaves: int = 8, persistence: float = 0.5,
                        lacunarity: float = 2.0, 
                        base_scale: float = 1.0) -> np.ndarray:
        """
        Билоу-шум (billow noise) - пушистые облака
        
        Args:
            x, y: Координатные сетки
            octaves: Количество октав
            persistence: Сохранение амплитуды
            lacunarity: Умножение частоты
            base_scale: Базовый масштаб
            
        Returns:
            Массив значений билоу-шума
        """
        result = np.zeros_like(x, dtype=np.float32)
        amplitude = 1.0
        frequency = base_scale
        
        for i in range(octaves):
            # Генерируем шум для текущей октавы
            nx = x * frequency
            ny = y * frequency
            
            noise = self.noise_2d(nx, ny)
            # Используем абсолютное значение для создания "пушистости"
            noise = np.abs(noise)
            
            result += amplitude * noise
            
            # Подготовка для следующей октавы
            amplitude *= persistence
            frequency *= lacunarity
        
        # Нормализация
        max_val = (1 - persistence ** octaves) / (1 - persistence)
        return result / max_val if max_val > 0 else result

# ----------------------------------------------------------------------
# Специализированные генераторы на основе симплекс-шума
# ----------------------------------------------------------------------

class SimplexTextureGenerator:
    """Генератор текстур на основе симплекс-шума"""
    
    def __init__(self, seed: int = 42):
        self.simplex = SimplexNoise(seed)
        self.cache = {}
        
    def generate_clouds(self, width: int, height: int, 
                       scale: float = 0.01, time: float = 0.0) -> np.ndarray:
        """
        Генерация облачной текстуры
        
        Args:
            width, height: Размеры текстуры
            scale: Масштаб текстуры
            time: Время для анимации
            
        Returns:
            Текстура в формате (height, width, 4) RGBA
        """
        # Создаем координатную сетку
        x = np.linspace(0, width * scale, width)
        y = np.linspace(0, height * scale, height)
        xx, yy = np.meshgrid(x, y)
        
        if time > 0:
            # Анимированные облака с использованием 3D шума
            zz = np.ones_like(xx) * time * 0.1
            base_noise = self.simplex.noise_3d(xx, yy, zz)
        else:
            # Статичные облака
            base_noise = self.simplex.noise_2d(xx, yy)
        
        # Фрактальный шум для детализации
        fractal = self.simplex.fractal_noise_2d(
            xx, yy, octaves=6, persistence=0.6, 
            lacunarity=2.0, base_scale=scale * 2
        )
        
        # Комбинируем шумы
        clouds = base_noise * 0.7 + fractal * 0.3
        
        # Нормализация к [0, 1]
        clouds = (clouds + 1) * 0.5
        
        # Создаем RGBA текстуру
        texture = np.zeros((height, width, 4), dtype=np.float32)
        
        # Белые облака с легкой голубизной
        texture[..., 0] = clouds  # R
        texture[..., 1] = clouds  # G
        texture[..., 2] = clouds * 0.9 + 0.1  # B (легкая голубизна)
        
        # Альфа-канал: более плотные облаки в центре
        alpha = np.clip(clouds * 1.5 - 0.25, 0, 1)
        texture[..., 3] = alpha
        
        return texture
    
    def generate_marble(self, width: int, height: int,
                       scale: float = 0.005, veins: int = 3) -> np.ndarray:
        """
        Генерация мраморной текстуры
        
        Args:
            width, height: Размеры текстуры
            scale: Масштаб текстуры
            veins: Количество "вен" в мраморе
            
        Returns:
            Текстура в формате (height, width, 4) RGBA
        """
        # Создаем координатную сетку
        x = np.linspace(0, width * scale, width)
        y = np.linspace(0, height * scale, height)
        xx, yy = np.meshgrid(x, y)
        
        # Базовый шум для структуры
        base_noise = self.simplex.noise_2d(xx, yy)
        
        # Создаем синусоидальные вены
        marble_pattern = np.zeros_like(base_noise)
        for i in range(veins):
            angle = i * np.pi / veins
            # Поворачиваем координаты
            x_rot = xx * np.cos(angle) - yy * np.sin(angle)
            # Синусоида для создания вен
            vein = np.sin(x_rot * 10 + base_noise * 2) * 0.5 + 0.5
            marble_pattern += vein
        
        marble_pattern /= veins
        
        # Добавляем фрактальную детализацию
        detail = self.simplex.fractal_noise_2d(
            xx, yy, octaves=4, persistence=0.5,
            lacunarity=2.5, base_scale=scale * 10
        )
        marble_pattern = marble_pattern * 0.8 + detail * 0.2
        
        # Создаем RGBA текстуру
        texture = np.zeros((height, width, 4), dtype=np.float32)
        
        # Цвета мрамора
        base_color = np.array([0.92, 0.87, 0.82])  # Светлый мрамор
        vein_color = np.array([0.75, 0.65, 0.55])  # Темные вены
        
        # Интерполяция между цветами на основе паттерна
        for i in range(3):
            texture[..., i] = base_color[i] * (1 - marble_pattern) + vein_color[i] * marble_pattern
        
        # Легкие вариации цвета
        color_variation = self.simplex.noise_2d(xx * 0.1, yy * 0.1) * 0.1
        texture[..., :3] += color_variation[..., np.newaxis] * 0.1
        
        texture[..., 3] = 1.0  # Непрозрачный
        
        return np.clip(texture, 0, 1)
    
    def generate_wood(self, width: int, height: int,
                     scale: float = 0.01, rings: float = 20.0) -> np.ndarray:
        """
        Генерация деревянной текстуры
        
        Args:
            width, height: Размеры текстуры
            scale: Масштаб текстуры
            rings: Количество годичных колец
            
        Returns:
            Текстура в формате (height, width, 4) RGBA
        """
        # Создаем координатную сетку
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(x, y)
        
        # Создаем кольцевую структуру (годичные кольца)
        radius = np.sqrt(xx*xx + yy*yy) * rings
        
        # Добавляем шум для реалистичности
        noise = self.simplex.fractal_noise_2d(
            xx, yy, octaves=5, persistence=0.6,
            lacunarity=2.0, base_scale=scale * 5
        )
        
        # Комбинируем кольца с шумом
        wood_pattern = np.sin(radius * 2 * np.pi + noise * 2) * 0.5 + 0.5
        
        # Добавляем детали (поры дерева)
        detail = self.simplex.noise_2d(xx * 50, yy * 50) * 0.1
        wood_pattern += detail
        
        # Создаем RGBA текстуру
        texture = np.zeros((height, width, 4), dtype=np.float32)
        
        # Цвета дерева
        light_wood = np.array([0.7, 0.5, 0.3])  # Светлая древесина
        dark_wood = np.array([0.4, 0.25, 0.1])  # Темная древесина
        
        # Интерполяция между цветами
        for i in range(3):
            texture[..., i] = light_wood[i] * wood_pattern + dark_wood[i] * (1 - wood_pattern)
        
        # Добавляем текстуру волокон
        fiber = np.sin(xx * 100 + self.simplex.noise_2d(xx * 10, yy * 10)) * 0.05
        texture[..., :3] += fiber[..., np.newaxis]
        
        texture[..., 3] = 1.0  # Непрозрачный
        
        return np.clip(texture, 0, 1)
    
    def generate_lava(self, width: int, height: int,
                     scale: float = 0.01, time: float = 0.0) -> np.ndarray:
        """
        Генерация текстуры лавы
        
        Args:
            width, height: Размеры текстуры
            scale: Масштаб текстуры
            time: Время для анимации
            
        Returns:
            Текстура в формате (height, width, 4) RGBA
        """
        # Создаем координатную сетку
        x = np.linspace(0, width * scale, width)
        y = np.linspace(0, height * scale, height)
        xx, yy = np.meshgrid(x, y)
        
        # Анимированная текстура с использованием 3D шума
        zz = np.ones_like(xx) * time * 0.2
        
        # Базовый шум для лавы
        if time > 0:
            base_noise = self.simplex.noise_3d(xx, yy, zz)
        else:
            base_noise = self.simplex.noise_2d(xx, yy)
        
        # Детализированный шум для текстуры
        detail = self.simplex.fractal_noise_2d(
            xx, yy, octaves=6, persistence=0.7,
            lacunarity=1.8, base_scale=scale * 3
        )
        
        # Комбинируем шумы
        lava = base_noise * 0.6 + detail * 0.4
        
        # Нормализация
        lava = (lava + 1) * 0.5
        
        # Создаем RGBA текстуру
        texture = np.zeros((height, width, 4), dtype=np.float32)
        
        # Цвета лавы (от темно-красного до ярко-желтого)
        # Используем нелинейное смешивание цветов
        red = np.clip(lava * 1.5, 0, 1)
        green = np.clip(lava * 0.8 - 0.2, 0, 1)
        blue = np.clip(lava * 0.2 - 0.4, 0, 0.1)
        
        texture[..., 0] = red    # R
        texture[..., 1] = green  # G
        texture[..., 2] = blue   # B
        
        # Яркость для свечения
        brightness = np.power(lava, 2)  # Квадрат для более ярких областей
        
        # Альфа-канал с небольшими вариациями
        texture[..., 3] = 0.9 + brightness * 0.1
        
        return texture

    def generate_terrain(self, width: int, height: int,
                         scale: float = 0.005, octaves: int = 6,
                         persistence: float = 0.5, lacunarity: float = 2.0
                         ) -> np.ndarray:
        """
        Генерация 2D текстуры рельефа
        
        Args:
            width, height: Размеры текстуры
            scale: Масштаб текстуры
            octaves: Количество октав
            persistence: Сохранение амплитуды
            lacunarity: Умножение частоты
            
        Returns:
            Текстура в формате (height, width, 4) RGBA
        """
        x = np.linspace(0, width * scale, width)
        y = np.linspace(0, height * scale, height)
        xx, yy = np.meshgrid(x, y)
        
        base = self.simplex.fractal_noise_2d(
            xx, yy, octaves=octaves, persistence=persistence,
            lacunarity=lacunarity, base_scale=1.0
        )
        ridged = self.simplex.ridged_noise_2d(
            xx, yy, octaves=max(3, octaves // 2), persistence=0.6,
            lacunarity=2.2, base_scale=1.5
        )
        
        heightmap = base * 0.7 + ridged * 0.3
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
        
        texture = np.zeros((height, width, 4), dtype=np.float32)
        
        water = heightmap < 0.35
        sand = (heightmap >= 0.35) & (heightmap < 0.45)
        grass = (heightmap >= 0.45) & (heightmap < 0.7)
        rock = (heightmap >= 0.7) & (heightmap < 0.85)
        snow = heightmap >= 0.85
        
        texture[water, :3] = np.array([0.10, 0.20, 0.50])
        texture[sand, :3] = np.array([0.76, 0.70, 0.50])
        texture[grass, :3] = np.array([0.20, 0.55, 0.25])
        texture[rock, :3] = np.array([0.50, 0.50, 0.50])
        texture[snow, :3] = np.array([0.92, 0.92, 0.96])
        
        variation = self.simplex.noise_2d(xx * 0.2, yy * 0.2) * 0.05
        texture[..., :3] = np.clip(texture[..., :3] + variation[..., np.newaxis], 0, 1)
        texture[..., 3] = 1.0
        
        return texture

    def generate_grass(self, width: int, height: int,
                       scale: float = 0.02) -> np.ndarray:
        """
        Генерация текстуры травы
        
        Args:
            width, height: Размеры текстуры
            scale: Масштаб текстуры
            
        Returns:
            Текстура в формате (height, width, 4) RGBA
        """
        x = np.linspace(0, width * scale, width)
        y = np.linspace(0, height * scale, height)
        xx, yy = np.meshgrid(x, y)
        
        base = self.simplex.noise_2d(xx, yy)
        detail = self.simplex.fractal_noise_2d(
            xx, yy, octaves=4, persistence=0.6,
            lacunarity=2.0, base_scale=scale * 5
        )
        grass = (base * 0.6 + detail * 0.4)
        grass = (grass + 1) * 0.5
        
        texture = np.zeros((height, width, 4), dtype=np.float32)
        texture[..., 0] = np.clip(grass * 0.4, 0, 1)
        texture[..., 1] = np.clip(grass * 0.8 + 0.2, 0, 1)
        texture[..., 2] = np.clip(grass * 0.4, 0, 1)
        texture[..., 3] = 1.0
        
        return texture

# ----------------------------------------------------------------------
# Демонстрация и тестирование
# ----------------------------------------------------------------------

def benchmark_simplex_noise():
    """Бенчмарк производительности симплекс-шума"""
    import time
    
    print("Benchmarking Simplex Noise...")
    print("=" * 50)
    
    # Инициализация
    simplex = SimplexNoise(seed=42)
    
    # Тест 1: 2D шум на небольшом массиве
    size_small = 256
    x_small = np.random.randn(size_small, size_small)
    y_small = np.random.randn(size_small, size_small)
    
    start = time.time()
    result = simplex.noise_2d(x_small, y_small)
    elapsed = time.time() - start
    
    print(f"2D Noise ({size_small}x{size_small}): {elapsed:.4f}s")
    print(f"  Mean: {result.mean():.4f}, Std: {result.std():.4f}")
    
    # Тест 2: Фрактальный шум
    start = time.time()
    fractal = simplex.fractal_noise_2d(
        x_small, y_small, octaves=8, persistence=0.5,
        lacunarity=2.0, base_scale=1.0
    )
    elapsed = time.time() - start
    
    print(f"Fractal Noise (8 octaves): {elapsed:.4f}s")
    print(f"  Mean: {fractal.mean():.4f}, Std: {fractal.std():.4f}")
    
    # Тест 3: Генерация текстур
    tex_gen = SimplexTextureGenerator(seed=42)
    
    sizes = [(128, 128), (512, 512), (1024, 1024)]
    textures = ["clouds", "marble", "wood", "lava"]
    
    print("\nTexture Generation Benchmark:")
    print("-" * 30)
    
    for w, h in sizes:
        print(f"\nSize: {w}x{h}")
        for tex_type in textures:
            start = time.time()
            
            if tex_type == "clouds":
                texture = tex_gen.generate_clouds(w, h)
            elif tex_type == "marble":
                texture = tex_gen.generate_marble(w, h)
            elif tex_type == "wood":
                texture = tex_gen.generate_wood(w, h)
            elif tex_type == "lava":
                texture = tex_gen.generate_lava(w, h)
            
            elapsed = time.time() - start
            print(f"  {tex_type:10s}: {elapsed:.4f}s")

def visualize_noise_patterns():
    """Визуализация различных паттернов шума"""
    import matplotlib.pyplot as plt
    
    print("Generating noise patterns visualization...")
    
    # Создаем координатную сетку
    size = 512
    x = np.linspace(0, 10, size)
    y = np.linspace(0, 10, size)
    xx, yy = np.meshgrid(x, y)
    
    # Инициализация генератора
    simplex = SimplexNoise(seed=42)
    
    # Генерация различных типов шума
    patterns = {
        "Simplex 2D": simplex.noise_2d(xx, yy),
        "Fractal (fBm)": simplex.fractal_noise_2d(xx, yy, octaves=8),
        "Ridged": simplex.ridged_noise_2d(xx, yy, octaves=8),
        "Billow": simplex.billow_noise_2d(xx, yy, octaves=8)
    }
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for ax, (title, pattern) in zip(axes, patterns.items()):
        im = ax.imshow(pattern, cmap='viridis', origin='lower')
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('simplex_noise_patterns.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'simplex_noise_patterns.png'")

if __name__ == "__main__":
    print("Simplex Noise Implementation")
    print("=" * 60)
    
    # Запускаем бенчмарк
    benchmark_simplex_noise()
    
    print("\n" + "=" * 60)
    print("Example usage:")
    
    # Пример использования
    simplex = SimplexNoise(seed=123)
    
    # Одиночная точка
    value = simplex.noise_2d(1.5, 2.3)
    print(f"\nNoise at (1.5, 2.3): {value:.6f}")
    
    # Массив точек
    x = np.array([[0.0, 1.0], [2.0, 3.0]])
    y = np.array([[0.0, 0.5], [1.0, 1.5]])
    values = simplex.noise_2d(x, y)
    print(f"\nNoise array:\n{values}")
    
    # Генератор текстур
    tex_gen = SimplexTextureGenerator(seed=456)
    
    # Генерация небольшой текстуры облаков
    clouds = tex_gen.generate_clouds(64, 64)
    print(f"\nCloud texture shape: {clouds.shape}")
    print(f"Cloud texture range: [{clouds.min():.3f}, {clouds.max():.3f}]")
    
    # 3D шум
    value_3d = simplex.noise_3d(1.0, 2.0, 3.0)
    print(f"\n3D noise at (1.0, 2.0, 3.0): {value_3d:.6f}")
    
    # 4D шум
    value_4d = simplex.noise_4d(1.0, 2.0, 3.0, 4.0)
    print(f"4D noise at (1.0, 2.0, 3.0, 4.0): {value_4d:.6f}")
    
    # Для визуализации (раскомментируйте при необходимости)
    # visualize_noise_patterns()
