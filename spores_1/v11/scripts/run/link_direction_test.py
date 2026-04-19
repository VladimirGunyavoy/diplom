"""
ТЕСТ НАПРАВЛЕНИЯ СВЯЗЕЙ
=======================

Простой тест для проверки правильности направления связей после исправления.
"""

import sys
import os
import json
import collections.abc

# Настройка путей
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from ursina import *
import numpy as np

# Импорты
from src.logic.pendulum import PendulumSystem
from src.managers.color_manager import ColorManager
from src.managers.zoom_manager import ZoomManager
from src.managers.spore_manager import SporeManager
from src.visual.scene_setup import SceneSetup
from src.core.spore import Spore
from src.managers.window_manager import WindowManager

def deep_merge(d, u):
    """Рекурсивно объединяет словари."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_merge(d.get(k, {}), v)
        else:
            d[k] = v
    return d

# Загружаем конфиги
with open(os.path.join(project_root, 'config', 'json', 'config.json'), 'r') as f:
    config = json.load(f)

with open(os.path.join(project_root, 'config', 'json', 'sizes.json'), 'r') as f:
    sizes_config = json.load(f)
    config = deep_merge(config, sizes_config)

print("="*60)
print("🔗 ТЕСТ НАПРАВЛЕНИЯ СВЯЗЕЙ")
print("="*60)
print(f"📊 trajectory_merge_tolerance: {config.get('trajectory_optimization', {}).get('trajectory_merge_tolerance')}")
print("="*60)

# Инициализация
app = Ursina()
color_manager = ColorManager()
scene_setup = SceneSetup(
    init_position=config['scene_setup']['init_position'], 
    init_rotation_x=config['scene_setup']['init_rotation_x'], 
    init_rotation_y=config['scene_setup']['init_rotation_y'], 
    color_manager=color_manager
)
zoom_manager = ZoomManager(scene_setup, color_manager=color_manager)
window_manager = WindowManager()

# Создание системы
pendulum = PendulumSystem(damping=config['pendulum']['damping'])

# Создание менеджера спор
spore_manager = SporeManager(
    pendulum=pendulum,
    zoom_manager=zoom_manager,
    settings_param=None,
    color_manager=color_manager,
    config=config
)

# Создание целевой споры (РОЗОВАЯ)
goal_position = config['spore']['goal_position']
goal = Spore(
    pendulum=pendulum,
    dt=config['pendulum']['dt'],
    scale=config['spore']['scale'],
    position=(goal_position[0], 0, goal_position[1]),
    goal_position=goal_position,
    is_goal=True,
    color_manager=color_manager
)

# Создание обычной споры (ГОЛУБАЯ) близко к цели
close_to_goal = Spore(
    pendulum=pendulum,
    dt=config['pendulum']['dt'],
    goal_position=goal_position,
    scale=config['spore']['scale'],
    position=(goal_position[0] - 0.15, 0, goal_position[1] + 0.05),  # Близко к цели
    color_manager=color_manager,
    config=config.get('spore', {})
)

# Добавляем споры в менеджер
spore_manager.add_spore(goal)
spore_manager.add_spore(close_to_goal)

# Регистрируем в zoom_manager
zoom_manager.register_object(goal)
zoom_manager.register_object(close_to_goal)

print(f"\n📍 Создание тестовых спор:")
print(f"   🌸 Цель (РОЗОВАЯ): {goal.calc_2d_pos()}")
print(f"   🔵 Обычная (ГОЛУБАЯ): {close_to_goal.calc_2d_pos()}")

print(f"\n🎮 УПРАВЛЕНИЕ:")
print(f"   F - создать спору от голубой (должно объединиться с розовой)")
print(f"   Q - выход")

def input(key):
    if key == 'q' or key == 'escape':
        application.quit()
        return
    
    if key == 'f':
        print(f"\n{'='*60}")
        print(f"🔗 ТЕСТ СОЗДАНИЯ СВЯЗИ")
        print(f"{'='*60}")
        
        # Делаем голубую спору последней (источник объединения)
        spore_manager.objects = [goal, close_to_goal]
        
        print(f"📊 До объединения:")
        print(f"   Голубая спора: {close_to_goal.calc_2d_pos()}")
        print(f"   Розовая цель: {goal.calc_2d_pos()}")
        
        result = spore_manager.generate_new_spore()
        
        if result == goal:
            print(f"✅ ОБЪЕДИНЕНИЕ ПРОИЗОШЛО!")
            print(f"   Стрелка должна идти: Розовая цель → Голубая спора")
            print(f"   (показывает что голубая спора 'достигла' розовую цель)")
        elif result and result != close_to_goal:
            print(f"❌ Создана новая спора {result.id} вместо объединения")
            zoom_manager.register_object(result)
        else:
            print(f"❌ Спора не создалась")
        
        print(f"{'='*60}")

def update():
    pass

if __name__ == '__main__':
    app.run() 