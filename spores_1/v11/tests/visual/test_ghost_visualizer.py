import sys
import os
# Добавляем корневую директорию проекта в sys.path ('diplom/spores/10')
# Это нужно сделать до всех остальных импортов из src.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Этот импорт должен быть первым, чтобы настроить пути для Ursina
import src.config.ursina_setup 

# Патч для Ursina теперь будет применен автоматически при импорте
# любого модуля из src.utils, например, ColorManager, который импортируется ниже.

import numpy as np
from ursina import *

from src.logic.pendulum import PendulumSystem
from src.logic.ghost_processor import GhostProcessor
from src.visual.ghost_visualizer import GhostVisualizer
from src.visual.scene_setup import SceneSetup
from src.managers.color_manager import ColorManager
from src.visual.frame import Frame
from src.utils.scalable import Scalable

app = Ursina()

color_manager = ColorManager()

# --- 1. Настройка логики ---
pendulum = PendulumSystem()
dt = 0.1
ghost_processor = GhostProcessor(pendulum, dt)

# Генерируем набор управляющих воздействий для призраков
controls = np.linspace(-1, 1, 5)

# --- 2. Настройка визуализации ---
# "Настоящая" спора, за которой будут следовать призраки
spore_entity = Scalable(model='sphere', color=color.azure, scale=0.2, position=(2, 0, 1))

# Визуализатор призраков
ghost_visualizer = GhostVisualizer(max_ghosts=len(controls), scale=0.1)

# --- 3. Главный цикл (update) ---
def update():
    # Заставляем "настоящую" спору двигаться по кругу, чтобы тест был динамичным.
    spore_entity.x = np.sin(time.time() * 0.5) * 3
    spore_entity.z = np.cos(time.time() * 0.5) * 3
    
    # Получаем текущую 2D позицию "споры"
    current_pos_2d = np.array([spore_entity.x, spore_entity.z])
    
    # Вычисляем позиции призраков
    ghost_positions = ghost_processor.process(current_pos_2d, controls)
    
    # Обновляем их визуализацию
    ghost_visualizer.update_ghosts(ghost_positions)
    scene_setup.update(time.dt)

# Настройки для удобного просмотра
# Создаем scene_setup и передаем ему color_manager
scene_setup = SceneSetup(color_manager=color_manager)
# Создаем Frame и передаем ему тот же color_manager
scene_setup.frame = Frame(color_manager=color_manager, scale=1) 

def input(key):
    scene_setup.input_handler(key)

app.run() 