# Создание собственных паттернов
class CustomPatternGenerator:
    def generate_custom_pattern(self, dimensions, params):
        depth, height, width = dimensions
        texture = np.zeros((depth, height, width, 4))
        
        # Пользовательская логика
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    # Кастомная математическая функция
                    value = self.custom_math_function(i, j, k, params)
                    
                    if self.is_on_surface(value, params):
                        color = self.calculate_color(i, j, k, value)
                        texture[i, j, k] = color
        
        return texture
    
    def custom_math_function(self, x, y, z, params):
        # Пример: гиперболический параболоид
        return (x/params.scale)**2 - (y/params.scale)**2 - z
