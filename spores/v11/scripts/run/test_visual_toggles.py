#!/usr/bin/env python3
"""
ТЕСТ ПЕРЕКЛЮЧЕНИЯ ВИЗУАЛИЗАЦИИ
=============================

Проверяет новую функциональность переключения визуализации:
- Y: Переключение ангелов (angels/pillars)  
- C: Переключение поверхности стоимости (cost surface)

Что тестируется:
1. По умолчанию и ангелы, и cost surface выключены
2. Кнопка Y корректно переключает ангелов
3. Кнопка C корректно переключает cost surface
4. Состояние сохраняется правильно
"""

import os
import sys
import json
import collections.abc
from ursina import *
import numpy as np

# Настройка путей
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def deep_merge(d, u):
    """Рекурсивно объединяет словари."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_merge(d.get(k, {}), v)
        else:
            d[k] = v
    return d

print("🧪 ТЕСТ ПЕРЕКЛЮЧЕНИЯ ВИЗУАЛИЗАЦИИ")
print("=" * 50)

# Загружаем конфигурацию
config_path = os.path.join(project_root, 'config', 'json', 'config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

print("📋 ПРОВЕРКА КОНФИГУРАЦИИ ПО УМОЛЧАНИЮ:")
print(f"   cost_surface.enabled: {config.get('cost_surface', {}).get('enabled', 'НЕ НАЙДЕН')}")
print(f"   angel.show_angels: {config.get('angel', {}).get('show_angels', 'НЕ НАЙДЕН')}")
print(f"   angel.show_pillars: {config.get('angel', {}).get('show_pillars', 'НЕ НАЙДЕН')}")

# Проверяем что по умолчанию всё выключено
cost_enabled = config.get('cost_surface', {}).get('enabled', True)
angels_enabled = config.get('angel', {}).get('show_angels', True)
pillars_enabled = config.get('angel', {}).get('show_pillars', True)

if not cost_enabled and not angels_enabled and not pillars_enabled:
    print("✅ Конфигурация корректна - всё выключено по умолчанию")
else:
    print("⚠️  ВНИМАНИЕ: Не все элементы выключены по умолчанию:")
    print(f"   cost_surface.enabled: {cost_enabled}")
    print(f"   angel.show_angels: {angels_enabled}")
    print(f"   angel.show_pillars: {pillars_enabled}")

print("\n🎮 УПРАВЛЕНИЕ В ТЕСТЕ:")
print("   Y - Переключить ангелов")
print("   C - Переключить поверхность стоимости")
print("   Q - Выход")
print("   WASD - Движение, мышь - поворот")

# Импорты для создания минимального тестового сценария
try:
    from src.visual.spawn_area_visualizer import SpawnAreaVisualizer
    from src.logic.spawn_area import SpawnArea as SpawnAreaLogic
    from src.logic.cost_function import CostFunction
    from src.visual.cost_visualizer import CostVisualizer
    from src.managers.color_manager import ColorManager
    from src.visual.scene_setup import SceneSetup
    from src.managers.zoom_manager import ZoomManager
    from src.utils.scalable import Scalable, ScalableFloor
    from src.logic.pendulum import PendulumSystem
    from src.core.spore import Spore
    from src.managers.spore_manager import SporeManager
    from src.managers.param_manager import ParamManager
    from src.managers.angel_manager import AngelManager
    from src.managers.input_manager import InputManager

    print("\n✅ Все модули успешно загружены")

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

    # ===== СОЗДАНИЕ ZOOM MANAGER =====
    zoom_manager = ZoomManager(scene_setup, color_manager=color_manager)
    angel_manager = AngelManager(color_manager=color_manager, zoom_manager=zoom_manager, config=config)

    # --- СОЗДАНИЕ ПОЛА ---
    floor = ScalableFloor(
        model='quad',
        scale=config['scene']['floor_scale'],
        rotation_x=90,
        color=color_manager.get_color('scene', 'floor'),
        texture='white_cube',
        texture_scale=(40, 40)
    )
    zoom_manager.register_object(floor, name='floor')

    # ===== СОЗДАНИЕ СИСТЕМЫ МАЯТНИКА И СПОР =====
    pendulum_config = config['pendulum']
    pendulum = PendulumSystem(damping=pendulum_config['damping'])

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

    spore = Spore(
        pendulum=pendulum,
        dt=pendulum_config['dt'],
        goal_position=goal_position,
        scale=config['spore']['scale'],
        position=(0, 0, 2),
        color_manager=color_manager,
        config=config.get('spore', {})
    )

    # ===== СОЗДАНИЕ ОБЛАСТИ СПАВНА =====
    spawn_area_logic = SpawnAreaLogic(
        focus1=spore.logic.position_2d,
        focus2=goal.logic.position_2d,
        eccentricity=config['spawn_area']['eccentricity']
    )

    # ===== СОЗДАНИЕ COST SURFACE =====
    cost_surface_parent = Scalable()
    cost_logic = CostFunction(goal_position_2d=np.array([goal.position[0], goal.position[2]]))
    cost_surface = CostVisualizer(
        cost_function=cost_logic,
        spawn_area=spawn_area_logic,
        parent_entity=cost_surface_parent,
        color_manager=color_manager,
        config=config['cost_surface']
    )
    angel_manager.cost_function = cost_logic

    # ===== СОЗДАНИЕ МЕНЕДЖЕРОВ =====
    settings_param = ParamManager(0, show=False, color_manager=color_manager)
    spore_manager = SporeManager(
        pendulum=pendulum, 
        zoom_manager=zoom_manager, 
        settings_param=settings_param, 
        color_manager=color_manager,
        angel_manager=angel_manager,
        config=config,
        spawn_area=spawn_area_logic
    )

    # ===== СОЗДАНИЕ INPUT MANAGER =====
    input_manager = InputManager(
        scene_setup=scene_setup,
        zoom_manager=zoom_manager,
        spore_manager=spore_manager,
        spawn_area_manager=None,
        param_manager=settings_param,
        ui_setup=None,
        angel_manager=angel_manager,
        cost_visualizer=cost_surface
    )

    # ===== РЕГИСТРАЦИЯ ОБЪЕКТОВ =====
    spore_manager.add_spore(goal)
    spore_manager.add_spore(spore)
    zoom_manager.register_object(goal)
    zoom_manager.register_object(spore)
    zoom_manager.register_object(cost_surface_parent)
    zoom_manager.update_transform()

    print("✅ Тестовая сцена создана")
    print(f"📊 НАЧАЛЬНОЕ СОСТОЯНИЕ:")
    print(f"   Cost surface видима: {cost_surface.visible}")
    print(f"   Ангелы видимы: {angel_manager.angels_visible}")
    print(f"   Столбы видимы: {angel_manager.pillars_visible}")

    # ===== ОБРАБОТЧИКИ =====
    def input(key):
        """Обработчик ввода."""
        if key == 'q' or key == 'escape':
            application.quit()
            return
        
        input_manager.handle_input(key)

    def update():
        """Обработчик обновления.""" 
        pass

    print("\n🚀 СИМУЛЯЦИЯ ЗАПУЩЕНА")
    print("Используйте Y и C для переключения визуализации")
    
    # ===== ЗАПУСК =====
    app.run()

except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("ℹ️  Возможно отсутствует Ursina или другие зависимости")
except Exception as e:
    print(f"❌ Ошибка во время теста: {e}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    pass  # Код уже выполнен выше 