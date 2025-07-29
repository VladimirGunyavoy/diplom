"""
ДИАГНОСТИКА ЭВОЛЮЦИИ СПОР
=========================

Скрипт для анализа состояния спор и выявления причин остановки эволюции.
Показывает подробную информацию о каждой споре.
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

print("="*70)
print("🔍 ДИАГНОСТИКА ЭВОЛЮЦИИ СПОР")
print("="*70)
print(f"📊 Конфигурация trajectory_optimization:")
traj_config = config.get('trajectory_optimization', {})
for key, value in traj_config.items():
    print(f"   {key}: {value}")
print("="*70)

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

# Создание целевой споры
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

# Создание стартовой споры
spore_start = Spore(
    pendulum=pendulum,
    dt=config['pendulum']['dt'],
    goal_position=goal_position,
    scale=config['spore']['scale'],
    position=(0, 0, 2),
    color_manager=color_manager,
    config=config.get('spore', {})
)

# Добавляем споры в менеджер
spore_manager.add_spore(goal)
spore_manager.add_spore(spore_start)

# Регистрируем в zoom_manager
zoom_manager.register_object(goal)
zoom_manager.register_object(spore_start)

def analyze_spore(spore):
    """Анализирует состояние споры и возвращает диагностическую информацию."""
    info = {
        'id': spore.id,
        'position_2d': spore.calc_2d_pos(),
        'cost': spore.logic.cost,
        'is_alive': spore.is_alive(),
        'can_evolve': spore.can_evolve() if hasattr(spore, 'can_evolve') else 'N/A',
        'evolution_completed': getattr(spore, 'evolution_completed', False),
        'optimal_control': spore.logic.optimal_control,
        'optimal_dt': spore.logic.optimal_dt,
        'is_goal': getattr(spore, 'is_goal', False)
    }
    
    # Анализ причин невозможности эволюции
    if not info['can_evolve']:
        if not info['is_alive']:
            info['evolution_status'] = "МЕРТВА (optimal_dt=0)"
        elif info['evolution_completed']:
            info['evolution_status'] = "ЗАВЕРШЕНА (объединение)"
        else:
            info['evolution_status'] = "НЕИЗВЕСТНАЯ ПРИЧИНА"
    else:
        info['evolution_status'] = "МОЖЕТ ЭВОЛЮЦИОНИРОВАТЬ"
    
    return info

def print_spore_info(info):
    """Выводит информацию о споре в читаемом формате."""
    status_emoji = {
        "МОЖЕТ ЭВОЛЮЦИОНИРОВАТЬ": "✅",
        "МЕРТВА (optimal_dt=0)": "💀", 
        "ЗАВЕРШЕНА (объединение)": "🏁",
        "НЕИЗВЕСТНАЯ ПРИЧИНА": "❓"
    }
    
    emoji = status_emoji.get(info['evolution_status'], "❓")
    print(f"   {emoji} Спора {info['id']}:")
    print(f"      Позиция: {info['position_2d']}")
    print(f"      Стоимость: {info['cost']:.4f}")
    print(f"      Статус: {info['evolution_status']}")
    if info['optimal_control'] is not None:
        print(f"      Управление: {info['optimal_control']}")
    if info['optimal_dt'] is not None:
        print(f"      dt: {info['optimal_dt']}")
    print()

def diagnose_evolution():
    """Проводит полный анализ состояния эволюции."""
    print(f"\n📊 АНАЛИЗ СОСТОЯНИЯ СПОР:")
    print(f"   Общее количество: {len(spore_manager.objects)}")
    
    can_evolve_count = 0
    dead_count = 0 
    completed_count = 0
    
    for spore in spore_manager.objects:
        info = analyze_spore(spore)
        print_spore_info(info)
        
        if info['evolution_status'] == "МОЖЕТ ЭВОЛЮЦИОНИРОВАТЬ":
            can_evolve_count += 1
        elif info['evolution_status'] == "МЕРТВА (optimal_dt=0)":
            dead_count += 1
        elif info['evolution_status'] == "ЗАВЕРШЕНА (объединение)":
            completed_count += 1
    
    print(f"📈 СВОДКА:")
    print(f"   ✅ Могут эволюционировать: {can_evolve_count}")
    print(f"   💀 Мертвые споры: {dead_count}")
    print(f"   🏁 Завершенные споры: {completed_count}")
    print(f"   ❓ Другие: {len(spore_manager.objects) - can_evolve_count - dead_count - completed_count}")
    
    if can_evolve_count == 0:
        print(f"\n⚠️  ПРОБЛЕМА: Нет спор, способных к эволюции!")
        print(f"   💡 Возможные решения:")
        print(f"      - Увеличить trajectory_merge_tolerance")
        print(f"      - Проверить логику завершения эволюции")
        print(f"      - Создать новую спору в другой позиции")
    else:
        print(f"\n✅ Эволюция может продолжаться")

print(f"\n📍 Начальное состояние:")
diagnose_evolution()

print(f"\n🎮 УПРАВЛЕНИЕ:")
print(f"   F - создать новую спору")
print(f"   D - диагностика состояния")
print(f"   Q - выход")

def input(key):
    if key == 'q' or key == 'escape':
        application.quit()
        return
    
    if key == 'f':
        print(f"\n{'='*50}")
        print(f"🚀 СОЗДАНИЕ НОВОЙ СПОРЫ")
        print(f"{'='*50}")
        
        result = spore_manager.generate_new_spore()
        if result:
            zoom_manager.register_object(result)
            print(f"✅ Создана спора {result.id}")
        else:
            print(f"❌ Спора не создалась")
        
        diagnose_evolution()
        print(f"{'='*50}")
    
    elif key == 'd':
        print(f"\n{'='*50}")
        print(f"🔍 ДИАГНОСТИКА")
        print(f"{'='*50}")
        diagnose_evolution()
        print(f"{'='*50}")

def update():
    pass

if __name__ == '__main__':
    app.run() 