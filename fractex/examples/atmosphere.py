# Динамическая атмосфера с изменяющимся временем дня
import sys
from pathlib import Path
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DynamicAtmosphere:
    def __init__(self):
        self.sun_direction = np.array([0.0, 1.0, 0.0])
        self.sun_intensity = 1.0
        self.scattering_coefficient = 0.1
        self.phase_function_g = 0.7
        self.sun_color = (1.0, 0.95, 0.9)
    
    def update_time_of_day(self, time_of_day: float):
        # time_of_day: 0.0 (полночь) до 1.0 (полночь следующего дня)
        sun_angle = time_of_day * 2 * math.pi
        self.sun_direction = np.array([
            math.sin(sun_angle),
            math.cos(sun_angle),
            0.0
        ])
        
        # Цвет солнца
        if 0.2 < time_of_day < 0.3:
            self.sun_color = (1.0, 0.5, 0.3)
        elif 0.7 < time_of_day < 0.8:
            self.sun_color = (1.0, 0.4, 0.2)
        else:
            self.sun_color = (1.0, 0.95, 0.9)
        
        # Интенсивность
        sun_height = self.sun_direction[1]
        self.sun_intensity = max(0.1, sun_height * 2)
        
        # Параметры рассеяния
        if time_of_day < 0.25 or time_of_day > 0.75:
            self.scattering_coefficient = 0.03
            self.phase_function_g = 0.3
        else:
            self.scattering_coefficient = 0.1
            self.phase_function_g = 0.7


if __name__ == "__main__":
    atmosphere = DynamicAtmosphere()
    for t in [0.0, 0.25, 0.5, 0.75]:
        atmosphere.update_time_of_day(t)
        print(f"time={t:.2f} sun_dir={atmosphere.sun_direction} intensity={atmosphere.sun_intensity:.2f}")
