#!/usr/bin/env python3
"""
ТЕСТ ИСПРАВЛЕНИЯ ПРЕЖДЕВРЕМЕННОЙ СМЕРТИ СПОР
============================================

Проверяет, что споры теперь действительно доходят до красной линии (минимума коста)
вместо того чтобы останавливаться за шаг до неё.
"""

import os
import sys
import json
import collections.abc
import numpy as np
from ursina import *

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

# Загружаем конфиги
with open(os.path.join(project_root, 'config', 'json', 'config.json'), 'r') as f:
    config = json.load(f)

with open(os.path.join(project_root, 'config', 'json', 'sizes.json'), 'r') as f:
    sizes_config = json.load(f)
    config = deep_merge(config, sizes_config)

# Включаем отладочный вывод для детального анализа
config['trajectory_optimization']['debug_output'] = True

# Импорты
from src.logic.pendulum import PendulumSystem
from src.managers.color_manager import ColorManager
from src.managers.zoom_manager import ZoomManager
from src.managers.spore_manager import SporeManager
from src.visual.scene_setup import SceneSetup
from src.core.spore import Spore
from src.managers.window_manager import WindowManager

print("=" * 80)
print("🔧 ТЕСТ ИСПРАВЛЕНИЯ ПРЕЖДЕВРЕМЕННОЙ СМЕРТИ СПОР")
print("=" * 80)
print("📋 Цель теста:")
print("   • Проверить, что споры доходят до красной линии")
print("   • Убедиться, что оптимизация траекторий работает корректно")
print("   • Показать пошаговое движение к минимуму коста")
print("=" * 80)

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

# Создание стартовой споры чуть дальше от цели
start_position = (1.0, 0, 0.5)  # Позиция, откуда хорошо видно движение к цели
spore_start = Spore(
    pendulum=pendulum,
    dt=config['pendulum']['dt'],
    goal_position=goal_position,
    scale=config['spore']['scale'],
    position=start_position,
    color_manager=color_manager,
    config=config.get('spore', {})
)

# Добавляем споры в менеджер
spore_manager.add_spore(goal)
spore_manager.add_spore(spore_start)

# Регистрируем в zoom_manager
zoom_manager.register_object(goal)
zoom_manager.register_object(spore_start)

step_count = 0

def show_detailed_status():
    """Показывает подробный статус всех спор."""
    print(f"\n📊 ДЕТАЛЬНЫЙ СТАТУС (шаг {step_count}):")
    print("=" * 60)
    
    for i, spore in enumerate(spore_manager.objects):
        if hasattr(spore, 'is_goal') and spore.is_goal:
            print(f"🎯 ЦЕЛЬ {spore.id}: позиция {spore.calc_2d_pos()}, cost {spore.logic.cost:.4f}")
            continue
            
        status_parts = []
        if spore.is_alive():
            status_parts.append("💚 ЖИВА")
        else:
            status_parts.append("💀 МЕРТВА")
            
        if hasattr(spore, 'evolution_completed') and spore.evolution_completed:
            status_parts.append("🔵 ЭВОЛЮЦИЯ ЗАВЕРШЕНА")
            
        if hasattr(spore, 'can_evolve') and spore.can_evolve():
            status_parts.append("🚀 МОЖЕТ ЭВОЛЮЦИОНИРОВАТЬ")
        else:
            status_parts.append("🚫 НЕ МОЖЕТ ЭВОЛЮЦИОНИРОВАТЬ")
        
        print(f"🔸 Спора {spore.id}:")
        print(f"   📍 Позиция: {spore.calc_2d_pos()}")
        print(f"   💰 Cost: {spore.logic.cost:.6f}")
        print(f"   📊 Статус: {' | '.join(status_parts)}")
        if spore.logic.optimal_control is not None:
            print(f"   🎮 Оптимальное управление: {spore.logic.optimal_control}")
            print(f"   ⏱️  Оптимальное dt: {spore.logic.optimal_dt}")
        print()
    
    print(f"📈 СВОДКА:")
    print(f"   Всего спор: {len(spore_manager.objects)}")
    print(f"   Всего связей: {len(spore_manager.links)}")
    print("=" * 60)

def update():
    pass

def input(key):
    global step_count
    
    if key == 'q' or key == 'escape':
        print(f"\n📋 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ТЕСТА:")
        print("=" * 60)
        print(f"   Сделано шагов: {step_count}")
        print(f"   Итого спор: {len(spore_manager.objects)}")
        print(f"   Итого связей: {len(spore_manager.links)}")
        
        # Анализ финального состояния
        for spore in spore_manager.objects:
            if not (hasattr(spore, 'is_goal') and spore.is_goal):
                final_cost = spore.logic.cost
                if final_cost < 0.1:
                    print(f"   ✅ Спора {spore.id} близко к цели (cost: {final_cost:.6f})")
                elif not spore.is_alive():
                    print(f"   🪦 Спора {spore.id} достигла минимума и умерла (cost: {final_cost:.6f})")
                else:
                    print(f"   ⏳ Спора {spore.id} еще в пути (cost: {final_cost:.6f})")
        
        print("=" * 60)
        print("👋 Тест завершен!")
        application.quit()
        return
    
    if key == 'f':
        print(f"\n⚡ ШАГ {step_count + 1}: СОЗДАНИЕ НОВОЙ СПОРЫ")
        print("-" * 50)
        
        last_spore = spore_manager.objects[-1]
        if hasattr(last_spore, 'is_goal') and last_spore.is_goal:
            print("⚠️  Активна целевая спора, переключаемся на предыдущую")
            if len(spore_manager.objects) > 1:
                # Перемещаем не-цель в конец
                for spore in spore_manager.objects:
                    if not (hasattr(spore, 'is_goal') and spore.is_goal):
                        spore_manager.objects.remove(spore)
                        spore_manager.objects.append(spore)
                        break
                last_spore = spore_manager.objects[-1]
        
        print(f"🎯 Активная спора: {last_spore.id}")
        print(f"   📍 Текущая позиция: {last_spore.calc_2d_pos()}")
        print(f"   💰 Текущая стоимость: {last_spore.logic.cost:.6f}")
        print(f"   💚 Жива: {last_spore.is_alive()}")
        print(f"   🚀 Может эволюционировать: {last_spore.can_evolve()}")
        
        if last_spore.logic.optimal_control is not None:
            print(f"   🎮 Оптимальное управление: {last_spore.logic.optimal_control}")
            print(f"   ⏱️  Оптимальное dt: {last_spore.logic.optimal_dt}")
            
            # Предсказываем, куда пойдет спора
            predicted_pos_3d = last_spore.evolve_3d(
                control=last_spore.logic.optimal_control,
                dt=last_spore.logic.optimal_dt
            )
            predicted_pos_2d = np.array([predicted_pos_3d[0], predicted_pos_3d[2]])
            predicted_cost = np.linalg.norm(predicted_pos_2d - np.array(goal_position))
            print(f"   🔮 Предсказанная позиция: {predicted_pos_2d}")
            print(f"   🔮 Предсказанная стоимость: {predicted_cost:.6f}")
        
        print("\n🚀 Вызываем generate_new_spore()...")
        result = spore_manager.generate_new_spore()
        
        if result:
            step_count += 1
            if result != last_spore:  # Новая спора создана
                print(f"✅ Создана новая спора: {result.id}")
                print(f"   📍 Позиция: {result.calc_2d_pos()}")
                print(f"   💰 Стоимость: {result.logic.cost:.6f}")
            else:  # Произошло объединение
                print(f"🔄 Произошло объединение с существующей спорой")
        else:
            print(f"❌ Спора не создана (возможно, эволюция завершена)")
            
        show_detailed_status()
        return
    
    if key == 'i':
        show_detailed_status()
        return
        
    if key == 'auto':
        # Автоматический прогон до завершения эволюции
        print(f"\n🤖 АВТОМАТИЧЕСКИЙ РЕЖИМ")
        max_steps = 20
        while step_count < max_steps:
            result = spore_manager.generate_new_spore() 
            if result:
                step_count += 1
                print(f"Шаг {step_count}: спора {result.id}, cost: {result.logic.cost:.6f}")
            else:
                print(f"Эволюция завершена на шаге {step_count}")
                break
        show_detailed_status()

print(f"\n🎮 УПРАВЛЕНИЕ:")
print("=" * 40)
print("   F - создать следующую спору")
print("   I - показать детальный статус") 
print("   Q - выход с отчетом")
print("   AUTO - автоматический прогон")
print("=" * 40)
print()
print("🔍 ЧТО НАБЛЮДАТЬ:")
print("   • Споры должны реально доходить до красной линии")
print("   • Cost должен уменьшаться при приближении к цели")
print("   • При достижении минимума optimal_dt → 0 и спора умирает")
print("   • Никаких преждевременных остановок!")
print("=" * 40)

show_detailed_status()

app.run() 