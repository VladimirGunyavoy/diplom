import sys
import os
from ursina import *
import numpy as np

# --- Настройка путей для импорта ---
script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.logic.cost_function import CostFunction
from src.visual.cost_visualizer import CostVisualizer
from src.logic.spawn_area import SpawnArea

# --- Мок-объекты (заглушки) для зависимостей ---

# --- Основной код теста ---
if __name__ == '__main__':
    app = Ursina()

    # 1. Создаем логику
    goal_pos = np.array([2.0, 3.0])
    cost_logic = CostFunction(goal_position_2d=goal_pos)

    # 2. Создаем реальный объект SpawnArea вместо мока
    spawn_area_logic = SpawnArea(
        focus1=np.array([-3, 0]), 
        focus2=np.array([3, 0]), 
        eccentricity=0.8
    )
    
    mock_config = {
        'cost_surface': {
            'unlit': True,
            'alpha': 200,
            'mesh_generation': {
                'boundary_points': 64,
                'interior_min_radius': 0.5
            }
        }
    }

    # 3. Создаем визуализатор с реальным объектом
    cost_visualizer = CostVisualizer(
        cost_function=cost_logic,
        spawn_area=spawn_area_logic,
        config=mock_config
    )
    
    # Настройка камеры
    EditorCamera()
    
    # Запускаем приложение
    print("Визуальный тест для CostVisualizer.")
    print("Вы должны увидеть 3D-поверхность параболоидной формы.")
    print("Закройте окно для завершения теста.")
    app.run() 