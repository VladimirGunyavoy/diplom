"""
УПРОЩЕННЫЙ ТЕСТ
===============

Простая проверка основной функциональности без сложной логики.
"""

import sys
import os
from ursina import *
import numpy as np
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

# --- Настройка путей ---
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

# --- Импорты ---
from src.managers.color_manager import ColorManager
from src.visual.scene_setup import SceneSetup
from src.visual.frame import Frame
from src.managers.zoom_manager import ZoomManager
from src.utils.scalable import ScalableFloor
from src.logic.pendulum import PendulumSystem
from src.core.spore import Spore
from src.managers.spore_manager import SporeManager

print("=== УПРОЩЕННЫЙ ТЕСТ ===")
print("🎯 Проверка базовой функциональности")
print("=" * 30)

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

# --- СОЗДАНИЕ ПОЛА ---
floor = ScalableFloor(
    model='quad',
    scale=10,
    rotation_x=90,
    color=color_manager.get_color('scene', 'floor'),
    texture='white_cube',
    texture_scale=(40, 40)
)
zoom_manager.register_object(floor, name='floor')

# ===== СОЗДАНИЕ СИСТЕМЫ МАЯТНИКА =====
pendulum_config = config['pendulum']
pendulum = PendulumSystem(damping=pendulum_config['damping'])

# ===== СОЗДАНИЕ ТЕСТОВЫХ СПОР =====
print("\n🌟 Создание тестовых спор...")

goal_position = config['spore']['goal_position']

# ЦЕЛЬ
print("📍 Создание цели в позиции", goal_position)
goal = Spore(
    pendulum=pendulum,
    dt=pendulum_config['dt'],
    scale=0.1,  # Увеличиваем размер для видимости
    position=(goal_position[0], 0, goal_position[1]),
    goal_position=goal_position,
    is_goal=True,
    color_manager=color_manager
)

# СПОРА 1: позиция (0.68, 0.12) - в 3D это (0.68, 0, 0.12)
print("📍 Создание споры 1 в позиции (0.68, 0.12)")
spore1 = Spore(
    pendulum=pendulum,
    dt=pendulum_config['dt'],
    goal_position=goal_position,
    scale=0.1,  # Увеличиваем размер
    position=(0.68, 0, 0.12),
    color_manager=color_manager,
    config=config.get('spore', {})
)

# СПОРА 2: позиция (0, 2) - в 3D это (0, 0, 2)
print("📍 Создание споры 2 в позиции (0, 2)")
spore2 = Spore(
    pendulum=pendulum,
    dt=pendulum_config['dt'],
    goal_position=goal_position,
    scale=0.1,  # Увеличиваем размер
    position=(0, 0, 2),
    color_manager=color_manager,
    config=config.get('spore', {})
)

# ===== СОЗДАНИЕ МЕНЕДЖЕРА СПОР =====
spore_manager = SporeManager(
    pendulum=pendulum, 
    zoom_manager=zoom_manager, 
    settings_param=None,  # Упрощаем
    color_manager=color_manager,
    angel_manager=None,  # Упрощаем
    config=config,
    spawn_area=None  # Упрощаем
)

# Добавляем споры
print("\n➕ Добавление спор в менеджер...")
spore_manager.add_spore(goal)
spore_manager.add_spore(spore1)
spore_manager.add_spore(spore2)

# Регистрируем в zoom manager
zoom_manager.register_object(goal, "goal_spore")
zoom_manager.register_object(spore1, "test_spore_1")
zoom_manager.register_object(spore2, "test_spore_2")

print(f"\n✅ Создано спор: {len(spore_manager.objects)}")
print("📊 Позиции спор:")
for i, spore in enumerate(spore_manager.objects):
    print(f"   Спора {i}: 3D={spore.real_position}, 2D={spore.calc_2d_pos()}")

print("\n🎮 Управление:")
print("F - создать следующую спору")
print("E/T - масштаб, R - сброс")
print("Q - выход")

def update():
    # Базовое обновление
    pass

def input(key):
    print(f"🎹 КЛАВИША НАЖАТА: '{key}'")  # Отладочная информация
    
    if key == 'q' or key == 'escape':
        print("\n📊 Завершение теста")
        print(f"   Спор: {len(spore_manager.objects)}")
        print(f"   Связей: {len(spore_manager.links)}")
        application.quit()
        return
    
    if key == 'f':
        print(f"\n⚡ F НАЖАТА! Обработка...")
        print(f"   Всего спор: {len(spore_manager.objects)}")
        if spore_manager.objects:
            last_spore = spore_manager.objects[-1]
            print(f"   Последняя спора: ID={last_spore.id}")
            print(f"   Позиция: 3D={last_spore.real_position}, 2D={last_spore.calc_2d_pos()}")
            print(f"   Жива: {last_spore.is_alive()}")
            print(f"   Оптимальное управление: {last_spore.logic.optimal_control}")
            print(f"   Оптимальное dt: {last_spore.logic.optimal_dt}")
            
            print("   🚀 Вызываем generate_new_spore()...")
            try:
                result = spore_manager.generate_new_spore()
                if result:
                    print(f"✅ Создана новая спора: ID={result.id}, позиция: {result.real_position}")
                    print(f"   Всего спор теперь: {len(spore_manager.objects)}")
                else:
                    print("❌ generate_new_spore() вернул None")
            except Exception as e:
                print(f"❌ Ошибка в generate_new_spore(): {e}")
                import traceback
                traceback.print_exc()
        else:
            print("   ⚠️ Нет спор в менеджере")
        return
    
    # Базовые команды масштабирования
    if key == 'e':
        print("🔍 Увеличение масштаба")
        zoom_manager.zoom_in()
    elif key == 't':
        print("🔍 Уменьшение масштаба")
        zoom_manager.zoom_out()
    elif key == 'r':
        print("🔍 Сброс масштаба")
        zoom_manager.reset_zoom()
    else:
        print(f"   ❓ Неизвестная клавиша: '{key}'")

print("\n✨ Простой тест запущен!")
app.run() 