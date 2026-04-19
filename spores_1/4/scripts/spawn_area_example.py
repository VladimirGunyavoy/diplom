from ursina import *
import os
import sys
import numpy as np

# Добавляем путь к проекту
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.color_manager import ColorManager
from src.spawn_area import SpawnArea

# Создаем приложение
app = Ursina()

# Создаем менеджер цветов
color_manager = ColorManager()

# Настройка сцены с цветовой схемой
window.color = color_manager.get_color('scene', 'window_background')
camera.position = (0, 5, -10)
camera.rotation_x = 20

# Создаем пол
floor = Entity(
    model='cube',
    scale=(20, 0.1, 20),
    position=(0, -1, 0),
    color=color_manager.get_color('scene', 'floor')
)

# Создаем освещение
DirectionalLight(
    direction=(1, -1, 1),
    color=color_manager.get_color('scene', 'directional_light')
)
AmbientLight(color=color_manager.get_color('scene', 'ambient_light'))

# Создаем фокусы для эллипса
focus1 = Entity(
    model='sphere',
    scale=0.2,
    position=(-2, 0, 0),
    color=color_manager.get_color('spore', 'goal')
)

focus2 = Entity(
    model='sphere',
    scale=0.2,
    position=(2, 0, 0),
    color=color_manager.get_color('spore', 'goal')
)

# Создаем область спавна (эллипс)
spawn_area = SpawnArea(
    focus1=focus1.position,
    focus2=focus2.position,
    eccentricity=0.5,
    dimensions=2,
    resolution=64,
    mode='line',
    color=color_manager.get_color('spawn_area', 'boundary'),
    color_manager=color_manager
)

# Создаем инструкции
instructions = Text(
    text=dedent('''
    <white>УПРАВЛЕНИЕ:
    WASD - перемещение
    Мышь - поворот камеры
    3/4 - изменение эксцентриситета
    ESC - выход
    ''').strip(),
    position=(-0.75, 0.35),
    scale=0.75,
    color=color_manager.get_color('ui', 'text_primary'),
    background=True,
    background_color=color_manager.get_color('ui', 'background_transparent'),
)

# Обработчик ввода
def input(key):
    if key == 'escape':
        quit()
    
    # Передаем ввод в SpawnArea
    spawn_area.input_handler(key)

# Обновление
def update():
    spawn_area.update()

# Настройка контролов камеры
EditorCamera()

if __name__ == '__main__':
    app.run() 