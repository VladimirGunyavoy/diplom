"""
ТЕСТ СЛИЯНИЯ ТРАЕКТОРИЙ  
=======================

Специальный тест для проверки объединения траекторий.
Создаем споры, которые должны встретиться при оптимальном управлении.
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
print("🔗 ТЕСТ СЛИЯНИЯ ТРАЕКТОРИЙ")
print("="*60)
print(f"📊 trajectory_merge_tolerance: {config.get('trajectory_optimization', {}).get('trajectory_merge_tolerance', 'НЕ НАЙДЕН')}")
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

# СПЕЦИАЛЬНЫЙ СЦЕНАРИЙ: Создаем споры, которые должны встретиться
# Спора A - существующая "целевая" точка для слияния
spore_target = Spore(
    pendulum=pendulum,
    dt=config['pendulum']['dt'],
    goal_position=goal_position,
    scale=config['spore']['scale'],
    position=(1.0, 0, 0.5),  # Целевая позиция для слияния
    color_manager=color_manager,
    config=config.get('spore', {})
)

# Спора B - должна прийти к sporet_target после одного шага оптимизации
# Найдем позицию, из которой следующий шаг приведет к spore_target
target_pos_2d = spore_target.calc_2d_pos()
print(f"🎯 Целевая позиция для слияния: {target_pos_2d}")

# Создаем тестовую спору для расчета обратного шага
# Попробуем несколько позиций и найдем ту, которая ведет к target
test_positions = [
    (0.8, 0, 0.6),   # Близко к цели
    (0.9, 0, 0.4),   # Другой вариант
    (1.1, 0, 0.6),   # Еще один вариант
    (0.95, 0, 0.45), # Очень близко
]

best_distance = float('inf')
best_position = test_positions[0]

for test_pos in test_positions:
    test_spore = Spore(
        pendulum=pendulum,
        dt=config['pendulum']['dt'],
        goal_position=goal_position,
        scale=config['spore']['scale'],
        position=test_pos,
        color_manager=color_manager,
        config=config.get('spore', {})
    )
    
    # Принудительно запускаем оптимизацию через optimizer
    from src.logic.optimizer import SporeOptimizer
    optimizer = SporeOptimizer(pendulum, config)
    result = optimizer.find_optimal_step(test_spore)
    optimal_control, optimal_dt = result['x']
    test_spore.logic.optimal_control = np.array([optimal_control])
    test_spore.logic.optimal_dt = optimal_dt
    
    # Рассчитываем следующую позицию
    if test_spore.logic.optimal_control is not None and test_spore.logic.optimal_dt is not None:
        next_pos_3d = test_spore.evolve_3d(
            control=test_spore.logic.optimal_control,
            dt=test_spore.logic.optimal_dt
        )
    else:
        # Если оптимизация не сработала, используем управление по умолчанию
        next_pos_3d = test_spore.evolve_3d(
            control=np.array([0.0]),
            dt=config['pendulum']['dt']
        )
    next_pos_2d = np.array([next_pos_3d[0], next_pos_3d[2]])
    
    distance_to_target = np.linalg.norm(next_pos_2d - target_pos_2d)
    print(f"📍 Тест позиции {test_pos} -> {next_pos_2d}, расстояние до цели: {distance_to_target:.4f}")
    
    if distance_to_target < best_distance:
        best_distance = distance_to_target
        best_position = test_pos
    
    # Удаляем тестовую спору
    destroy(test_spore)

print(f"✅ Лучшая стартовая позиция: {best_position}, расстояние: {best_distance:.4f}")

# Создаем спору, которая должна придти к spore_target
spore_source = Spore(
    pendulum=pendulum,
    dt=config['pendulum']['dt'],
    goal_position=goal_position,
    scale=config['spore']['scale'],
    position=best_position,
    color_manager=color_manager,
    config=config.get('spore', {})
)

# Запускаем оптимизацию для источника через optimizer
from src.logic.optimizer import SporeOptimizer
optimizer = SporeOptimizer(pendulum, config)
result = optimizer.find_optimal_step(spore_source)
optimal_control, optimal_dt = result['x']
spore_source.logic.optimal_control = np.array([optimal_control])
spore_source.logic.optimal_dt = optimal_dt

# Добавляем споры в менеджер
spore_manager.add_spore(goal)
spore_manager.add_spore(spore_target)
spore_manager.add_spore(spore_source)

# Регистрируем в zoom_manager
zoom_manager.register_object(goal)
zoom_manager.register_object(spore_target)
zoom_manager.register_object(spore_source)

print(f"\n📍 Созданы споры:")
print(f"   🎯 Цель: {goal.calc_2d_pos()}")
print(f"   🔵 Целевая спора (для слияния): {spore_target.calc_2d_pos()}")
print(f"   🟢 Источник (должна слиться): {spore_source.calc_2d_pos()}")

print(f"\n🎮 УПРАВЛЕНИЕ:")
print(f"   F - создать новую спору от источника (должно произойти слияние!)")
print(f"   Q - выход")

def input(key):
    if key == 'q' or key == 'escape':
        application.quit()
        return
    
    if key == 'f':
        print(f"\n{'='*80}")
        print(f"🚀 ПОПЫТКА СЛИЯНИЯ")
        print(f"{'='*80}")
        
        # Делаем источник последней спорой
        spore_manager.objects = [goal, spore_target, spore_source]
        
        print(f"📊 Состояние перед слиянием:")
        print(f"   Источник {spore_source.id}: {spore_source.calc_2d_pos()}")
        print(f"   Цель слияния {spore_target.id}: {spore_target.calc_2d_pos()}")
        print(f"   Управление источника: {spore_source.logic.optimal_control}")
        print(f"   dt источника: {spore_source.logic.optimal_dt}")
        
        result = spore_manager.generate_new_spore()
        
        if result == spore_target:
            print(f"🎉 УСПЕШНОЕ СЛИЯНИЕ! Спора {spore_source.id} объединилась с {spore_target.id}")
        elif result and result != spore_source:
            print(f"❌ Создана новая спора {result.id} вместо слияния")
            zoom_manager.register_object(result)
        else:
            print(f"❌ Спора не создалась")
        
        print(f"{'='*80}")

def update():
    pass

if __name__ == '__main__':
    app.run() 