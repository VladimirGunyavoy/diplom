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
from src.scene.spawn_area import SpawnArea
from src.scene.cost import Cost
from src.managers.color_manager import ColorManager
from src.scene.scene_setup import SceneSetup
from src.scene.frame import Frame
from src.managers.zoom_manager import ZoomManager
from src.utils.scalable import Scalable, ScalableFrame, ScalableFloor
from src.core.pendulum import PendulumSystem
from src.managers.window_manager import WindowManager
from src.core.spore import Spore
from src.managers.spore_manager import SporeManager
from src.managers.param_manager import ParamManager
from src.ui.ui_setup import UI_setup
from src.managers.angel_manager import AngelManager

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
    dt=pendulum_config['dt'], 
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
    position=spore_config['initial_position'],
    color_manager=color_manager
)

# ===== СОЗДАНИЕ ОКРУЖЕНИЯ (SPAWN AREA И COST SURFACE) =====
spawn_area_config = config['spawn_area']
spawn_area = SpawnArea(
    focus1=spore,
    focus2=goal,
    eccentricity=spawn_area_config['eccentricity'],
    dimensions=2,
    resolution=spawn_area_config['resolution'],
    mode='line',
    color_manager=color_manager
)

# --- Поверхность стоимости (Cost) ---
cost_surface = Cost(
    goal_position=goal,
    spawn_area=spawn_area,
    color_manager=color_manager,
    config=config
)
cost_surface.generate_surface()
angel_manager.cost_surface = cost_surface

# ===== НАСТРОЙКА UI ЧЕРЕЗ UI_SETUP С КОЛБЭКАМИ =====
print("\n📊 3. Настройка полного UI через UI_setup с колбэками...")

# --- ИНИЦИАЛИЗАЦИЯ UI_SETUP ---
ui_setup = UI_setup(color_manager=color_manager)

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
ui_elements = ui_setup.setup_demo_ui(data_providers, spawn_area=spawn_area)

# ===== РЕГИСТРАЦИЯ ОБЪЕКТОВ В МЕНЕДЖЕРАХ =====
spore_manager.add_spore(goal)
spore_manager.add_spore(spore)

zoom_manager.register_object(goal)
zoom_manager.register_object(spore)
zoom_manager.register_object(spawn_area)
zoom_manager.register_object(cost_surface)

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
    """Обновление системы"""
    # Обновляем сцену
    scene_setup.update(time.dt)
    
    # Обновляем zoom manager (который обновляет look point)
    zoom_manager.identify_invariant_point()
    
    # Обновляем param manager
    settings_param.update()
    
    # Обновляем все UI элементы через UI_setup (который вызывает UIManager)
    ui_setup.update()

def input(key):
    """Обрабатывает ввод пользователя для всех компонентов."""
    scene_setup.input_handler(key)
    zoom_manager.input_handler(key)
    spore_manager.input_handler(key)
    if ui_setup:
        ui_setup.handle_demo_commands(key)
    
    # -> ВОТ ИСПРАВЛЕНИЕ: Добавляем обработку ввода для SpawnArea
    spawn_area.input_handler(key)

# ===== ФИНАЛЬНАЯ ДЕМОНСТРАЦИЯ =====
print("\n🎉 6. UI_setup готов к демонстрации!")
print("\n📋 ДОСТУПНЫЕ КОМАНДЫ:")
print("   ОБЫЧНЫЕ: WASD, Space/Shift, мышь, Alt, Q")
print("   ZOOM: E/T (зум), R (сброс), 1/2 (споры)")
print("   PARAM: Z/X (параметр)")
print("   UI: H (скрыть/показать весь UI)")
print("\n" + "="*40)
print("🚀 UI_SETUP DEMO ЗАПУЩЕНА!")
print("="*40)

# ===== ЗАПУСК =====
if __name__ == '__main__':
    app.run() 