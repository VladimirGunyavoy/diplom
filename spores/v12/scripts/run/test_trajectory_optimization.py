"""
ТЕСТ ОПТИМИЗАЦИИ ТРАЕКТОРИЙ
===========================

Тестовый скрипт для проверки функциональности объединения траекторий.
Создает две споры в позициях (0.79, 0) и (0, 2), где вторая должна 
прийти к первой и создать связь вместо новой споры.

Использование:
- F: создать следующую спору (с проверкой пересечений)
- Ожидаемое поведение: вторая спора должна объединиться с первой
"""

import sys
import os
from ursina import *
import numpy as np
import time
from colorsys import hsv_to_rgb
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import json
import collections.abc

# --- Помощник для глубокого слияния конфигов ---
def deep_merge(d, u):
    """Рекурсивно объединяет словари."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_merge(d.get(k, {}), v)
        else:
            d[k] = v
    return d

# --- Настройка путей для импорта ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Загрузка конфигурации ---
with open(os.path.join(project_root, 'config', 'json', 'config.json'), 'r') as f:
    config = json.load(f)

with open(os.path.join(project_root, 'config', 'json', 'sizes.json'), 'r') as f:
    sizes_config = json.load(f)
    config = deep_merge(config, sizes_config)

# --- Основные импорты ---
from src.visual.spawn_area_visualizer import SpawnAreaVisualizer
from src.logic.spawn_area import SpawnArea as SpawnAreaLogic
from src.logic.cost_function import CostFunction
from src.visual.cost_visualizer import CostVisualizer
from src.managers.color_manager import ColorManager
from src.visual.scene_setup import SceneSetup
from src.visual.frame import Frame
from src.managers.zoom_manager import ZoomManager
from src.utils.scalable import Scalable, ScalableFrame, ScalableFloor
from src.logic.pendulum import PendulumSystem
from src.managers.window_manager import WindowManager
from src.core.spore import Spore
from src.managers.spore_manager import SporeManager
from src.managers.param_manager import ParamManager
from src.visual.ui_setup import UI_setup
from src.managers.angel_manager import AngelManager
from src.managers.input_manager import InputManager
from src.managers.update_manager import UpdateManager
from src.managers.spawn_area_manager import SpawnAreaManager

print("=== ТЕСТ ОПТИМИЗАЦИИ ТРАЕКТОРИЙ ===")
print("🎯 Проверка объединения схожих траекторий")
print("📍 Начальные позиции: (0.79, 0) и (0, 2)")
print("🔗 Ожидается: вторая спора объединится с первой")
print("=" * 50)

# ===== ИНИЦИАЛИЗАЦИЯ =====
app = Ursina()
color_manager = ColorManager()

# ===== СОЗДАНИЕ СЦЕНЫ =====
scene_config = config['scene_setup']
scene_setup = SceneSetup(
    init_position=scene_config['init_position'], 
    init_rotation_x=scene_config['init_rotation_x'], 
    init_rotation_y=scene_config['init_rotation_y'], 
    color_manager=color_manager
)
frame = Frame(
    color_manager=color_manager,
    origin_scale=config.get('frame', {}).get('origin_scale', 0.05)
)
scene_setup.frame = frame
window_manager = WindowManager()

# ===== СОЗДАНИЕ ZOOM MANAGER =====
zoom_manager = ZoomManager(scene_setup, color_manager=color_manager)
angel_manager = AngelManager(color_manager=color_manager, zoom_manager=zoom_manager, config=config)

# Регистрируем дочерние элементы Frame в ZoomManager
for i, entity in enumerate(frame.entities):
    zoom_manager.register_object(entity, name=f'frame_child_{i}')

# --- СОЗДАНИЕ МАСШТАБИРУЕМОГО ПОЛА ---
floor = ScalableFloor(
    model='quad',
    scale=config['scene']['floor_scale'],
    rotation_x=90,
    color=color_manager.get_color('scene', 'floor'),
    texture='white_cube',
    texture_scale=(40, 40)
)
zoom_manager.register_object(floor, name='floor')

# ===== СОЗДАНИЕ PARAM MANAGER =====
settings_param = ParamManager(0, show=False, color_manager=color_manager)

# ===== СОЗДАНИЕ СИСТЕМЫ МАЯТНИКА =====
pendulum_config = config['pendulum']
pendulum = PendulumSystem(damping=pendulum_config['damping'])

# ===== СОЗДАНИЕ ТЕСТОВЫХ СПОР =====
print("\n🌟 Создание тестовых спор...")

goal_position = config['spore']['goal_position']
goal = Spore(
    pendulum=pendulum,
    dt=pendulum_config['dt'],
    scale=config['spore']['scale'],
    position=(goal_position[0], 0, goal_position[1]),
    goal_position=goal_position,
    is_goal=True,
    color_manager=color_manager
)

# ТЕСТОВАЯ СПОРА 1: позиция (0.68, 0.12)
print("📍 Создание споры 1 в позиции (0.68, 0.12)")
spore1 = Spore(
    pendulum=pendulum,
    dt=pendulum_config['dt'],
    goal_position=goal_position,
    scale=config['spore']['scale'],
    position=(0.68, 0, 0.12),  # (x, y, z) где z соответствует второй координате в 2D
    color_manager=color_manager,
    config=config.get('spore', {})
)

# ТЕСТОВАЯ СПОРА 2: позиция (0, 2) 
print("📍 Создание споры 2 в позиции (0, 2)")
spore2 = Spore(
    pendulum=pendulum,
    dt=pendulum_config['dt'],
    goal_position=goal_position,
    scale=config['spore']['scale'],
    position=(0, 0, 2),  # (x, y, z) где z соответствует второй координате в 2D
    color_manager=color_manager,
    config=config.get('spore', {})
)

# ===== СОЗДАНИЕ ОКРУЖЕНИЯ =====
spawn_area_config = config['spawn_area']
spawn_area_logic = SpawnAreaLogic(
    focus1=spore1.logic.position_2d,
    focus2=goal.logic.position_2d,
    eccentricity=spawn_area_config['eccentricity']
)

spawn_area_visualizer = SpawnAreaVisualizer(
    spawn_area=spawn_area_logic,
    resolution=spawn_area_config['resolution'],
    color=color_manager.get_color('spawn_area', 'default')
)

# ===== СОЗДАНИЕ МЕНЕДЖЕРА СПОР =====
spore_manager = SporeManager(
    pendulum=pendulum, 
    zoom_manager=zoom_manager, 
    settings_param=settings_param, 
    color_manager=color_manager,
    angel_manager=angel_manager,
    config=config,
    spawn_area=spawn_area_logic
)

# Добавляем споры в менеджер
spore_manager.add_spore(goal)
spore_manager.add_spore(spore1)
spore_manager.add_spore(spore2)

# Регистрируем споры в zoom manager
zoom_manager.register_object(goal, "goal_spore")
zoom_manager.register_object(spore1, "test_spore_1")
zoom_manager.register_object(spore2, "test_spore_2")

# --- Поверхность стоимости ---
cost_surface_parent = Scalable()
cost_logic = CostFunction(
    goal_position_2d=np.array([goal.position[0], goal.position[2]]),
)
cost_surface = CostVisualizer(
    cost_function=cost_logic,
    spawn_area=spawn_area_logic,
    parent_entity=cost_surface_parent,
    color_manager=color_manager,
    config=config['cost_surface']
)
angel_manager.cost_function = cost_logic

# ===== НАСТРОЙКА UI =====
print("\n📊 Настройка UI...")
ui_setup = UI_setup(color_manager=color_manager)

# ===== СОЗДАНИЕ МЕНЕДЖЕРОВ =====
spawn_area_manager = SpawnAreaManager(
    spawn_area_logic=spawn_area_logic,
    spawn_area_visualizer=spawn_area_visualizer,
    cost_visualizer=cost_surface
)

input_manager = InputManager(
    scene_setup=scene_setup,
    zoom_manager=zoom_manager,
    spore_manager=spore_manager,
    spawn_area_manager=spawn_area_manager,
    param_manager=settings_param,
    ui_setup=ui_setup,
    angel_manager=angel_manager
)

update_manager = UpdateManager(
    ui_setup=ui_setup,
    settings_param=settings_param,
    spore_manager=spore_manager,
    zoom_manager=zoom_manager,
    spawn_area_manager=spawn_area_manager,
    angel_manager=angel_manager
)

# Настройка UI с провайдерами данных
data_providers = {
    'get_spore_count': lambda: len(spore_manager.objects),
    'get_param_value': lambda: settings_param.value,
    'get_zoom_info': lambda: (zoom_manager.a_transformation, zoom_manager.spores_scale),
    'get_candidate_info': lambda: (spore_manager.min_radius, spore_manager.candidate_count),
}

ui_elements = ui_setup.setup_demo_ui(
    data_providers=data_providers,
    spawn_area=spawn_area_logic
)

print("\n🎮 Управление:")
print("F - создать следующую спору (с проверкой пересечений)")
print("E/T - масштаб, R - сброс")
print("Q - выход")
print("\n✨ Тест начат! Нажмите F для создания спор...")

# Текущая активная спора (для F)
current_active = spore2  # Начинаем со второй споры

def update():
    update_manager.update()

def input(key):
    if key == 'q' or key == 'escape':
        print("\n📊 Результаты теста:")
        print(f"   Всего спор: {len(spore_manager.objects)}")
        print(f"   Всего связей: {len(spore_manager.links)}")
        print("👋 Тест завершен!")
        application.quit()
        return
    
    # Специальная обработка для тестового режима
    if key == 'f':
        print(f"\n⚡ Генерация споры от активной споры {spore_manager.objects[-1].id}")
        result = spore_manager.generate_new_spore()
        if result:
            print(f"✅ Результат: спора {result.id}")
        return
    
    input_manager.handle_input(key)

# Регистрируем поверхность стоимости в zoom manager
zoom_manager.register_object(cost_surface_parent, name='cost_surface')

app.run() 