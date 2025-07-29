import sys
import os
import numpy as np
from ursina import *

# Добавляем корневую директорию проекта в sys.path ('diplom/spores/10')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.logic.spawn_area import SpawnArea
from src.visual.spawn_area_visualizer import SpawnAreaVisualizer

def main():
    app = Ursina()

    # 1. Создаем логический объект
    focus1 = np.array([-4, -2])
    focus2 = np.array([4, 2])
    eccentricity = 0.8
    
    spawn_area_logic = SpawnArea(
        focus1=focus1,
        focus2=focus2,
        eccentricity=eccentricity
    )

    # 2. Создаем визуальный объект, передавая ему логику
    spawn_area_visual = SpawnAreaVisualizer(
        spawn_area=spawn_area_logic,
        color=color.cyan,
        thickness=3
    )
    
    # Настройки для удобного просмотра
    EditorCamera()
    Grid(10, 10)
    
    # Запускаем приложение
    app.run()

if __name__ == '__main__':
    main() 