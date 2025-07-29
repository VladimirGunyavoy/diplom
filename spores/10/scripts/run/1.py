"""
ДЕМОНСТРАЦИЯ UI_SETUP
====================

Этот файл демонстрирует использование класса UI_setup для удобной настройки UI:
- Вся логика UI инкапсулирована в одном классе
- Автоматическое создание всех необходимых элементов
- Готовые функции обновления и обработки команд
- Простой импорт и настройка одной командой
- Интеграция с UI Manager и Color Manager

Преимущества:
✅ Минимум кода в демо скриптах
✅ Готовые настройки для типичных сценариев
✅ Автоматическое управление всеми UI элементами
✅ Легкая интеграция с существующими системами
✅ Консистентный интерфейс во всех демо
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
# Добавляем корневую папку проекта 'spores/9', чтобы можно было делать импорты от 'src'
script_dir = os.path.dirname(os.path.abspath(__file__))
# Поднимаемся от spores/9/scripts/run до spores/9
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Загрузка конфигурации ---
# Загружаем основной конфиг
with open(os.path.join(project_root, 'config', 'json', 'config.json'), 'r') as f:
    config = json.load(f)

# Загружаем конфиг с размерами и объединяем
with open(os.path.join(project_root, 'config', 'json', 'sizes.json'), 'r') as f:
    sizes_config = json.load(f)
    config = deep_merge(config, sizes_config)

# ColorManager загружает свой конфиг 'colors.json' самостоятельно

# --- Основные импорты из нашего проекта ---
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

print("=== ДЕМОНСТРАЦИЯ UI_SETUP ===")
print("🎨 Готовые настройки UI для демо скриптов")
print("📦 Инкапсуляция всей логики в одном классе")
print("🔄 Автоматическое управление всеми элементами")
print("=" * 40)

# ===== ИНИЦИАЛИЗАЦИЯ =====
app = Ursina()

# ===== СОЗДАНИЕ МЕНЕДЖЕРОВ =====
color_manager = ColorManager()

# ===== СОЗДАНИЕ СЦЕНЫ И ДРУГИХ СИСТЕМ =====
scene_config = config['scene_setup']
scene_setup = SceneSetup(
    init_position=scene_config['init_position'], 
    init_rotation_x=scene_config['init_rotation_x'], 
    init_rotation_y=scene_config['init_rotation_y'], 
    color_manager=color_manager
)
frame = Frame(color_manager=color_manager)
scene_setup.frame = frame
window_manager = WindowManager(color_manager=color_manager)

print("\n🌍 2. Сцена создана")

# ===== СОЗДАНИЕ ZOOM MANAGER =====
zoom_manager = ZoomManager(scene_setup, color_manager=color_manager)
angel_manager = AngelManager(color_manager=color_manager, zoom_manager=zoom_manager, config=config)

print("   ✓ Zoom Manager создан")

# РЕГИСТРИРУЕМ FRAME В ZOOM MANAGER
zoom_manager.register_object(frame, name='frame')

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
# ------------------------------------

# ===== СОЗДАНИЕ PARAM MANAGER =====
settings_param = ParamManager(0, show=False, color_manager=color_manager)

print("   ✓ Param Manager создан")

# ===== СОЗДАНИЕ СИСТЕМЫ МАЯТНИКА И МЕНЕДЖЕРА СПОР =====
pendulum_config = config['pendulum']
pendulum = PendulumSystem(
    damping=pendulum_config['damping']
)
spore_manager = SporeManager(
    pendulum=pendulum, 
    zoom_manager=zoom_manager, 
    settings_param=settings_param, 
    color_manager=color_manager,
    angel_manager=angel_manager,
    config=config
)

# ===== СОЗДАНИЕ ТЕСТОВЫХ СПОР =====
print("\n🌟 4. Создание тестовых спор...")

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

spore_config = config['spore']
spore = Spore(
    pendulum=pendulum,
    dt=pendulum_config['dt'],
    goal_position=goal_position,
    scale=config['spore']['scale'],
    link_thickness=spore_config.get('link_thickness', 1),
    position=spore_config['initial_position'],
    color_manager=color_manager,
    config=spore_config
)

# ===== СОЗДАНИЕ ОКРУЖЕНИЯ (SPAWN AREA И COST SURFACE) =====
spawn_area_config = config['spawn_area']

# Создаем логику области спавна
spawn_area_logic = SpawnAreaLogic(
    focus1=spore.logic.position_2d,
    focus2=goal.logic.position_2d,
    eccentricity=spawn_area_config['eccentricity']
)

# Создаем визуализацию области спавна
spawn_area_visualizer = SpawnAreaVisualizer(
    spawn_area=spawn_area_logic,
    resolution=spawn_area_config['resolution'],
    color=color_manager.get_color('spawn_area', 'default')
)

# --- Поверхность стоимости (Cost) ---
# Создаем родительский Entity для всей поверхности стоимости
cost_surface_parent = Scalable()

# Создаем логику
cost_logic = CostFunction(
    goal_position_2d=np.array([goal.position[0], goal.position[2]]),
)
# Создаем визуализацию
cost_surface = CostVisualizer(
    cost_function=cost_logic,
    spawn_area=spawn_area_logic, # <-- Передаем логический объект
    parent_entity=cost_surface_parent, # <--- Указываем нового родителя
    color_manager=color_manager,
    config=config['cost_surface']
)
angel_manager.cost_function = cost_logic # <-- Передаем логический объект

# ===== НАСТРОЙКА UI ЧЕРЕЗ UI_SETUP С КОЛБЭКАМИ =====
print("\n📊 3. Настройка полного UI через UI_setup с колбэками...")

# --- ИНИЦИАЛИЗАЦИЯ UI_SETUP ---
ui_setup = UI_setup(color_manager=color_manager)

# ===== СОЗДАНИЕ SPAWN AREA MANAGER =====
spawn_area_manager = SpawnAreaManager(
    spawn_area_logic=spawn_area_logic,
    spawn_area_visualizer=spawn_area_visualizer,
    cost_visualizer=cost_surface
)

# ===== СОЗДАНИЕ INPUT MANAGER =====
input_manager = InputManager(
    scene_setup=scene_setup,
    zoom_manager=zoom_manager,
    spore_manager=spore_manager,
    spawn_area_manager=spawn_area_manager,
    param_manager=settings_param,
    ui_setup=ui_setup
)

# ===== СОЗДАНИЕ UPDATE MANAGER =====
update_manager = UpdateManager(
    scene_setup=scene_setup,
    zoom_manager=zoom_manager,
    param_manager=settings_param,
    ui_setup=ui_setup
)

# Создаем словарь с поставщиками данных
data_providers = {
    'get_spore_count': lambda: len(spore_manager.objects),
    'get_camera_info': lambda: (
        scene_setup.player.position.x,
        scene_setup.player.position.y,
        scene_setup.player.position.z,
        scene_setup.player.camera_pivot.rotation_x,
        scene_setup.player.rotation_y,
        0  # rot_z
    ),
    'get_cursor_status': lambda: scene_setup.cursor_locked,
    'get_look_point_info': lambda: (
        *zoom_manager.identify_invariant_point(),
        getattr(zoom_manager.scene_setup.frame.x_axis, 'scale', [1])[0],
        getattr(zoom_manager, 'spores_scale', 1)
    ),
    'get_param_info': lambda: (settings_param.param, settings_param.show)
}

# Настраиваем весь UI одной командой, передавая колбэки
ui_elements = ui_setup.setup_demo_ui(data_providers, spawn_area=spawn_area_logic)

# ===== РЕГИСТРАЦИЯ ОБЪЕКТОВ В МЕНЕДЖЕРАХ =====
spore_manager.add_spore(goal)
spore_manager.add_spore(spore)

zoom_manager.register_object(goal)
zoom_manager.register_object(spore)
zoom_manager.register_object(spawn_area_visualizer)
# cost_surface больше не Scalable, регистрируем его общий родительский Entity
zoom_manager.register_object(cost_surface_parent)

zoom_manager.update_transform()

print(f"   ✓ Создано спор: {len(spore_manager.objects)}")
print(f"   📍 Стартовая спора в позиции: {spore.position}")
print(f"   🎯 Целевая спора в позиции: {goal.position}")
print(f"   📷 Камера в позиции: {scene_setup.player.position}")

# ===== ФУНКЦИИ ОБНОВЛЕНИЯ ДЕЛЕГИРОВАНЫ UI_SETUP =====
# Вся логика обновления UI теперь внутри UI_setup!

# ===== КОМАНДЫ ДЕМОНСТРАЦИИ =====
print("\n🎮 5. Команды демонстрации UI:")
ui_setup.show_demo_commands_help()

# ===== ОБРАБОТКА КОМАНД ДЕЛЕГИРОВАНА UI_SETUP =====
# Вся логика обработки команд UI теперь внутри UI_setup!

# ===== ОСНОВНЫЕ ФУНКЦИИ =====
def update():
    """Общий апдейт, вызываемый каждый кадр"""
    update_manager.update_all()

def input(key):
    """
    Главный обработчик ввода.
    Перехватывает все нажатия клавиш и делегирует их менеджерам.
    """
    # Если ввод заморожен, полностью игнорируем любые нажатия.
    # Логика разморозки происходит в SceneSetup.update()
    if scene_setup.input_frozen:
        return

    # Передаем управление в InputManager
    input_manager.handle_input(key)

# ===== ФИНАЛЬНАЯ ДЕМОНСТРАЦИЯ =====
print("\n🎉 6. UI_setup готов к демонстрации!")
print("\n📋 ДОСТУПНЫЕ КОМАНДЫ:")
print("   ОБЫЧНЫЕ: WASD, Space/Shift, мышь, Alt, Q")
print("   ZOOM: E/T (зум), R (сброс), 1/2 (споры)")
print("   PARAM: Z/X (параметр)")
print("   UI: H (скрыть/показать весь UI)")
print("\n" + "="*40)
print("�� UI_SETUP DEMO ЗАПУЩЕНА!")
print("="*40)

# ===== ЗАПУСК =====
if __name__ == '__main__':
    app.run() 