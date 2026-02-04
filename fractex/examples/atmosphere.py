# Динамическая атмосфера с изменяющимся временем дня
class DynamicAtmosphere:
    def update_time_of_day(self, time_of_day):
        # time_of_day: 0.0 (полночь) до 1.0 (полночь следующего дня)
        
        # Солнечное направление
        sun_angle = time_of_day * 2 * math.pi
        self.sun_direction = np.array([
            math.sin(sun_angle),
            math.cos(sun_angle),
            0.0
        ])
        
        # Цвет солнца
        if 0.2 < time_of_day < 0.3:  # Восход
            sun_color = (1.0, 0.5, 0.3)
        elif 0.7 < time_of_day < 0.8:  # Закат
            sun_color = (1.0, 0.4, 0.2)
        else:  # День
            sun_color = (1.0, 0.95, 0.9)
        
        # Интенсивность
        sun_height = self.sun_direction[1]
        self.sun_intensity = max(0.1, sun_height * 2)
        
        # Параметры рассеяния
        if time_of_day < 0.25 or time_of_day > 0.75:  # Ночь
            self.scattering_coefficient *= 0.3
            self.phase_function_g = 0.3
        else:  # День
            self.scattering_coefficient = 0.1
            self.phase_function_g = 0.7
