def create_underwater_volcano():
    """Создание подводного вулкана с лавой и пузырями"""
    
    # Основной чанк с лавой
    lava_generator = DynamicTextureGenerator3D(
        dimensions=(64, 64, 64),
        texture_type=DynamicTextureType.LAVA_FLOW,
        seed=42
    )
    
    # Чанк с водой и пузырями
    water_generator = DynamicTextureGenerator3D(
        dimensions=(64, 64, 64),
        texture_type=DynamicTextureType.WATER_FLOW,
        seed=123
    )
    
    # Добавляем пузыри в воду
    def add_bubbles_to_water(state):
        # Где есть горячая лава под водой - создаем пузыри
        for i in range(64):
            for j in range(64):
                for k in range(64):
                    # Если под этой точкой есть горячая лава
                    if lava_state.temperature_field[i, j, k] > 800:
                        # Создаем пузырь
                        bubble_intensity = min(1.0, lava_state.temperature_field[i, j, k] / 1200.0)
                        water_state.density_field[i, j, k] += bubble_intensity * 0.1
        
        return water_state
    
    # Симуляция
    states = []
    for frame in range(100):
        # Обновляем лаву
        lava_state = lava_generator.update()
        
        # Обновляем воду с учетом лавы
        water_state = water_generator.update()
        water_state = add_bubbles_to_water(water_state)
        
        # Смешиваем состояния
        blended_state = blend_lava_and_water(lava_state, water_state)
        states.append(blended_state)
    
    return states
