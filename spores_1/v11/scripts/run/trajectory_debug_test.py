"""
ТЕСТ ОТЛАДКИ ТРАЕКТОРИЙ
=======================

Скрипт для диагностики логики проверки пересечения траекторий.
Создает исходный сценарий и показывает подробную отладочную информацию.
"""

import sys
import os
import json

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

# Загрузка конфигов
import collections.abc

def deep_merge(d, u):
    """Рекурсивно объединяет словари."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_merge(d.get(k, {}), v)
        else:
            d[k] = v
    return d

# Загружаем основной конфиг
with open(os.path.join(project_root, 'config', 'json', 'config.json'), 'r') as f:
    config = json.load(f)

# Загружаем конфиг с размерами и объединяем
with open(os.path.join(project_root, 'config', 'json', 'sizes.json'), 'r') as f:
    sizes_config = json.load(f)
    config = deep_merge(config, sizes_config)

print("="*50)
print("🔍 ТЕСТ ОТЛАДКИ ТРАЕКТОРИЙ")
print("="*50)
print(f"📊 trajectory_merge_tolerance: {config.get('trajectory_optimization', {}).get('trajectory_merge_tolerance', 'НЕ НАЙДЕН')}")
print(f"📊 merge_tolerance: {config.get('trajectory_optimization', {}).get('merge_tolerance', 'НЕ НАЙДЕН')}")
print("="*50)

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

# Создание спор в тестовых позициях (как на скриншоте)
spore_a = Spore(
    pendulum=pendulum,
    dt=config['pendulum']['dt'],
    goal_position=goal_position,
    scale=config['spore']['scale'],
    position=(0.79, 0, 0),  # Первая спора
    color_manager=color_manager,
    config=config.get('spore', {})
)

spore_b = Spore(
    pendulum=pendulum,
    dt=config['pendulum']['dt'],
    goal_position=goal_position,
    scale=config['spore']['scale'],
    position=(0, 0, 2),  # Вторая спора
    color_manager=color_manager,
    config=config.get('spore', {})
)

# Добавляем споры в менеджер
spore_manager.add_spore(goal)
spore_manager.add_spore(spore_a)  
spore_manager.add_spore(spore_b)

# Регистрируем в zoom_manager
zoom_manager.register_object(goal)
zoom_manager.register_object(spore_a)
zoom_manager.register_object(spore_b)

print("\n📍 Созданы споры:")
print(f"   🎯 Цель: {goal.calc_2d_pos()}")
print(f"   🟣 Спора A: {spore_a.calc_2d_pos()}")
print(f"   🟣 Спора B: {spore_b.calc_2d_pos()}")

print("\n🎮 УПРАВЛЕНИЕ:")
print("   F - создать новую спору от последней")
print("   Q - выход")
print("   1/2 - переключение между спорами для тестирования")

current_test_spore = spore_b  # Начинаем с споры B

def input(key):
    global current_test_spore
    
    if key == 'q' or key == 'escape':
        application.quit()
        return
    
    if key == 'f':
        print(f"\n{'='*60}")
        print(f"🚀 СОЗДАНИЕ СПОРЫ ОТ {current_test_spore.id}")
        print(f"{'='*60}")
        
        # Временно делаем текущую спору последней в списке
        spore_manager.objects = [s for s in spore_manager.objects if s != current_test_spore]
        spore_manager.objects.append(current_test_spore)
        
        result = spore_manager.generate_new_spore()
        if result:
            zoom_manager.register_object(result)
            if result != current_test_spore:  # Если создалась новая спора
                print(f"✅ Создана новая спора {result.id}")
            else:  # Если произошло объединение
                print(f"🔗 Произошло объединение")
        
        print(f"{'='*60}")
    
    elif key == '1':
        current_test_spore = spore_a
        print(f"🔄 Переключились на спору A: {spore_a.calc_2d_pos()}")
    
    elif key == '2':
        current_test_spore = spore_b  
        print(f"🔄 Переключились на спору B: {spore_b.calc_2d_pos()}")

def update():
    pass

if __name__ == '__main__':
    app.run() 