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
from src.managers.manual_spore_manager import ManualSporeManager  # v13_manual: для ручного создания спор
from src.utils.debug_output import init_debug_output
from src.managers.dt_manager import DTManager

print("=== ДЕМОНСТРАЦИЯ UI_SETUP ===")
print("🎨 Готовые настройки UI для демо скриптов")
print("📦 Инкапсуляция всей логики в одном классе")
print("🔄 Автоматическое управление всеми элементами")
print("=" * 40)

# ===== ИНИЦИАЛИЗАЦИЯ ОТЛАДОЧНОГО ВЫВОДА =====
init_debug_output(config)
print("✅ Система отладочного вывода инициализирована")

# ===== НАСТРОЙКИ v13_manual =====
USE_SPAWN_AREA = False  # v13_manual: отключаем автоматический spawn area для ручного создания спор

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
frame = Frame(
    color_manager=color_manager,
    origin_scale=config.get('frame', {}).get('origin_scale', 0.05)
)
scene_setup.frame = frame
window_manager = WindowManager(monitor='left')

# ===== НАСТРОЙКА UI ПОЗИЦИЙ ДЛЯ ТЕКУЩЕГО МОНИТОРА =====
from src.visual.ui_constants import UI_POSITIONS
UI_POSITIONS.set_monitor(window_manager.get_current_monitor())
print(f"   ✓ UI позиции настроены для монитора: {window_manager.get_current_monitor()}")

print("\n🌍 2. Сцена создана")

# ===== СОЗДАНИЕ ZOOM MANAGER =====
zoom_manager = ZoomManager(scene_setup, color_manager=color_manager)

# ===== УСЛОВНОЕ СОЗДАНИЕ ANGEL MANAGER =====
# Создаем только если cost_surface или angels включены в конфиге
cost_enabled = config.get('cost_surface', {}).get('enabled', False)
angels_enabled = config.get('angel', {}).get('show_angels', False)

if cost_enabled or angels_enabled:
    angel_manager = AngelManager(color_manager=color_manager, zoom_manager=zoom_manager, config=config)
    print("   ✓ Angel Manager создан")
else:
    angel_manager = None
    print("   ⏭️ Angel Manager пропущен (cost_surface и angels отключены)")

print("   ✓ Zoom Manager создан")

# РЕГИСТРИРУЕМ FRAME В ZOOM MANAGER - УБИРАЕМ ЭТО
# zoom_manager.register_object(frame, name='frame')

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
# ------------------------------------

# ===== СОЗДАНИЕ PARAM MANAGER =====
settings_param = ParamManager(0, show=False, color_manager=color_manager)

print("   ✓ Param Manager создан")

# ===== СОЗДАНИЕ СИСТЕМЫ МАЯТНИКА =====
pendulum_config = config['pendulum']
pendulum = PendulumSystem(
    damping=pendulum_config['damping'],
    max_control=pendulum_config['max_control']
)

dt_manager = DTManager(config, pendulum)
print("   ✓ DT Manager создан")

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
initial_position_3d = (
    spore_config['initial_position'][0], 
    0, 
    spore_config['initial_position'][2] if len(spore_config['initial_position']) > 2 else spore_config['initial_position'][1]
)
# spore = Spore(
#     pendulum=pendulum,
#     dt=pendulum_config['dt'],
#     goal_position=goal_position,
#     scale=config['spore']['scale'],
#     link_thickness=spore_config.get('link_thickness', 1),
#     position=initial_position_3d,
#     color_manager=color_manager,
#     config=spore_config
# )

# ===== СОЗДАНИЕ ОКРУЖЕНИЯ (SPAWN AREA И COST SURFACE) =====
spawn_area_config = config['spawn_area']

# v13_manual: условное создание spawn area
if USE_SPAWN_AREA:
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
    print("   ✓ Spawn Area создана")
else:
    spawn_area_logic = None
    spawn_area_visualizer = None  
    print("   ⏭️ Spawn Area отключена (v13_manual: ручное создание спор)")

# ===== СОЗДАНИЕ МЕНЕДЖЕРА СПОР =====
spore_manager = SporeManager(
    pendulum=pendulum, 
    zoom_manager=zoom_manager, 
    settings_param=settings_param, 
    color_manager=color_manager,
    angel_manager=angel_manager,
    config=config,
    spawn_area=spawn_area_logic  # Может быть None
)

# --- УСЛОВНОЕ СОЗДАНИЕ ПОВЕРХНОСТИ СТОИМОСТИ (Cost) ---
if cost_enabled:
    # Создаем родительский Entity для всей поверхности стоимости
    cost_surface_parent = Scalable()
    
    # Создаем логику - оптимизация создания массива
    goal_pos_2d = np.empty(2, dtype=float)
    goal_pos_2d[0] = goal.position[0]
    goal_pos_2d[1] = goal.position[2]
    cost_logic = CostFunction(
        goal_position_2d=goal_pos_2d,
    )
    # Создаем визуализацию
    cost_surface = CostVisualizer(
        cost_function=cost_logic,
        spawn_area=spawn_area_logic, # <-- Передаем логический объект
        parent_entity=cost_surface_parent, # <--- Указываем нового родителя
        color_manager=color_manager,
        config=config['cost_surface']
    )
    
    # Передаем логический объект в angel_manager если он существует
    if angel_manager:
        angel_manager.cost_function = cost_logic
    
    print("   ✓ Cost Surface создана")
else:
    cost_surface = None
    cost_logic = None
    cost_surface_parent = None
    print("   ⏭️ Cost Surface пропущена (cost_surface.enabled = false)")

# ===== НАСТРОЙКА UI ЧЕРЕЗ UI_SETUP С КОЛБЭКАМИ =====
print("\n📊 3. Настройка полного UI через UI_setup с колбэками...")

# --- ИНИЦИАЛИЗАЦИЯ UI_SETUP ---
ui_setup = UI_setup(color_manager=color_manager)

# ===== СОЗДАНИЕ SPAWN AREA MANAGER =====
# v13_manual: условное создание spawn area manager
if USE_SPAWN_AREA:
    spawn_area_manager = SpawnAreaManager(
        spawn_area_logic=spawn_area_logic,
        spawn_area_visualizer=spawn_area_visualizer,
        cost_visualizer=cost_surface  # Может быть None
    )
    print("   ✓ Spawn Area Manager создан")
else:
    spawn_area_manager = None
    print("   ⏭️ Spawn Area Manager отключен (v13_manual: ручное создание спор)")

# ===== СОЗДАНИЕ MANUAL SPORE MANAGER =====
# v13_manual: менеджер для ручного создания спор с превью
manual_spore_manager = ManualSporeManager(
    spore_manager=spore_manager,
    zoom_manager=zoom_manager, 
    pendulum=pendulum,
    color_manager=color_manager,
    config=config
)

spore_manager._manual_spore_manager_ref = manual_spore_manager

dt_manager = DTManager(config, pendulum)
dt_manager.spore_manager = spore_manager  # 🆕 Связываем с SporeManager

# Устанавливаем dt_manager в manual_spore_manager
manual_spore_manager.set_dt_manager(dt_manager)

# ===== СОЗДАНИЕ INPUT MANAGER =====
input_manager = InputManager(
    scene_setup=scene_setup,
    zoom_manager=zoom_manager,
    spore_manager=spore_manager,
    spawn_area_manager=spawn_area_manager,  # Может быть None
    param_manager=settings_param,
    ui_setup=ui_setup,
    angel_manager=angel_manager,
    cost_visualizer=cost_surface,
    manual_spore_manager=manual_spore_manager,  # v13_manual: для обработки ЛКМ
    dt_manager=dt_manager
)

# ===== СОЗДАНИЕ UPDATE MANAGER =====
update_manager = UpdateManager(
    scene_setup=scene_setup,
    zoom_manager=zoom_manager,
    param_manager=settings_param,
    ui_setup=ui_setup,
    input_manager=input_manager,
    manual_spore_manager=manual_spore_manager  # v13_manual: для обновления курсора превью
)

# Создаем словарь с поставщиками данных

# ===== ФУНКЦИЯ ДЛЯ ИСПРАВЛЕННОГО LOOK POINT =====
def get_corrected_look_point():
    """Возвращает исправленный look point с учетом зума по формуле (look_point - frame_origin) / scale"""
    # Получаем сырую точку взгляда
    raw_x, raw_z = zoom_manager.identify_invariant_point()
    
    # Получаем позицию origin_cube с проверкой существования
    try:
        origin_x = zoom_manager.scene_setup.frame.origin_cube.position.x
        origin_z = zoom_manager.scene_setup.frame.origin_cube.position.z
    except:
        origin_x, origin_z = 0.0, 0.0
    
    # Применяем формулу коррекции
    scale = zoom_manager.a_transformation
    corrected_x = (raw_x - origin_x) / scale
    corrected_z = (raw_z - origin_z) / scale
    
    return corrected_x, corrected_z

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
        *get_corrected_look_point(),  # Используем исправленные координаты
        float(getattr(zoom_manager.scene_setup.frame.x_axis, 'scale_x', getattr(zoom_manager.scene_setup.frame.x_axis, 'scale', 1.0))),
        getattr(zoom_manager, 'spores_scale', 1)
    ),
    'get_param_info': lambda: (settings_param.param, settings_param.show),
    'get_candidate_info': lambda: (spore_manager.min_radius, spore_manager.candidate_count),
    'get_dt_info': lambda: dt_manager.get_stats()
}

# Настраиваем весь UI одной командой, передавая колбэки
ui_elements = ui_setup.setup_demo_ui(data_providers, spawn_area=spawn_area_logic)  # Может быть None

# ===== РЕГИСТРАЦИЯ ОБЪЕКТОВ В МЕНЕДЖЕРАХ =====
spore_manager.add_spore(goal)
# spore_manager.add_spore(spore)

zoom_manager.register_object(goal)
# zoom_manager.register_object(spore)
# v13_manual: условная регистрация spawn area
if spawn_area_visualizer:
    zoom_manager.register_object(spawn_area_visualizer)
# cost_surface больше не Scalable, регистрируем его общий родительский Entity (если существует)
if cost_surface_parent:
    zoom_manager.register_object(cost_surface_parent)

zoom_manager.update_transform()

# ===== ГЕНЕРАЦИЯ КАНДИДАТСКИХ СПОР =====
print("\n🎯 Генерация кандидатов...")
spore_manager.generate_candidate_spores()

print(f"   ✓ Создано спор: {len(spore_manager.objects)}")
print(f"   👻 Создано кандидатов: {spore_manager.candidate_count}")
# print(f"   📍 Стартовая спора в позиции: {spore.position}")
print(f"   🎯 Целевая спора в позиции: {goal.position}")
print(f"   📷 Камера в позиции: {scene_setup.player.position}")

# ===== ФУНКЦИИ ОБНОВЛЕНИЯ ДЕЛЕГИРОВАНЫ UI_SETUP =====
# Вся логика обновления UI теперь внутри UI_setup!

# ===== КОМАНДЫ ДЕМОНСТРАЦИИ =====
print("\n🎮 5. Команды демонстрации UI:")
# ui_setup.show_demo_commands_help()
# ui_setup.show_game_commands_help()

# ===== ОБРАБОТКА КОМАНД ДЕЛЕГИРОВАНА UI_SETUP =====
# Вся логика обработки команд UI теперь внутри UI_setup!

# ===== ОСНОВНЫЕ ФУНКЦИИ =====
def update():
    """Глобальный обработчик обновлений."""
    update_manager.update_all()

def input(key):
    """Глобальный обработчик ввода."""
    # Обработка выхода из приложения на самом верхнем уровне,
    # чтобы она работала независимо от состояния "заморозки" ввода.
    if key == 'q' or key == 'escape':
        application.quit()
        return

    # Передаем управление в централизованный InputManager
    input_manager.handle_input(key)

# ===== ФИНАЛЬНАЯ ДЕМОНСТРАЦИЯ =====
print("\n🎉 6. UI_setup готов к демонстрации!")
print("\n📋 ДОСТУПНЫЕ КОМАНДЫ:")
print("   ОБЫЧНЫЕ: WASD, Space/Shift, мышь, Alt, Q")
print("   СПОРЫ: F (новая от последней), G (активировать кандидата), V (развить всех кандидатов)")
print("   ZOOM: E/T (зум), R (сброс), 1/2 (споры)")
print("   PARAM: Z/X (параметр)")
print("   КАНДИДАТЫ: 5/6 (радиус генерации)")
print("   ВИЗУАЛИЗАЦИЯ: Y (ангелы вкл/выкл), C (поверхность стоимости вкл/выкл)")
print("   UI: H (скрыть/показать весь UI)")
print("\n" + "="*40)
print("🚀 СИМУЛЯЦИЯ ЗАПУЩЕНА 🚀")
print("="*40)

# ===== ЗАПУСК =====
if __name__ == '__main__':
    app.run() 