#!/usr/bin/env python3
"""
Тест логики завершения эволюции спор после объединения
Проверяет, что споры прекращают создавать детей после объединения с существующими спорами
"""

import os
import sys
import json
import collections.abc
import numpy as np
from ursina import *

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
from src.managers.window_manager import WindowManager
from src.visual.scene_setup import SceneSetup
from src.managers.zoom_manager import ZoomManager
from src.utils.scalable import ScalableFloor
from src.logic.pendulum import PendulumSystem
from src.core.spore import Spore
from src.managers.spore_manager import SporeManager
from src.visual.link import Link

print("=== ТЕСТ ЗАВЕРШЕНИЯ ЭВОЛЮЦИИ ===")
print("🎯 Проверка логики прекращения эволюции после объединения")
print("=" * 50)

# ===== ИНИЦИАЛИЗАЦИЯ =====
app = Ursina()

# ===== НАСТРОЙКА ОКНА =====
window_manager = WindowManager(
    title="Тест завершения эволюции - Споры v11",
    size=(900, 700)
)
window_manager.set_background_color(color.dark_gray)

color_manager = ColorManager()

print("\n🪟 НАСТРОЙКА ОКОН:")
print("• Окно Ursina: 900x700 пикселей")
print("• Консоль покажет пошаговую логику завершения эволюции")

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
    scale=12,
    rotation_x=90,
    color=color_manager.get_color('scene', 'floor'),
    texture='white_cube',
    texture_scale=(50, 50)
)
zoom_manager.register_object(floor, name='floor')

# ===== СОЗДАНИЕ СИСТЕМЫ МАЯТНИКА =====
pendulum_config = config['pendulum']
pendulum = PendulumSystem(damping=pendulum_config['damping'])

# ===== СОЗДАНИЕ ТЕСТОВОГО СЦЕНАРИЯ =====
print("\n🌟 Создание тестового сценария...")

goal_position = config['spore']['goal_position']

# ЦЕЛЬ
print("📍 Создание цели в позиции", goal_position)
goal = Spore(
    pendulum=pendulum,
    dt=pendulum_config['dt'],
    scale=0.1,
    position=(goal_position[0], 0, goal_position[1]),
    goal_position=goal_position,
    is_goal=True,
    color_manager=color_manager
)

# СПОРА A: Будет объединяться со спорой B
print("📍 Создание споры A в позиции (0.5, 0.1)")
spore_a = Spore(
    pendulum=pendulum,
    dt=pendulum_config['dt'],
    goal_position=goal_position,
    scale=0.09,
    position=(0.5, 0, 0.1),
    color_manager=color_manager,
    config=config.get('spore', {})
)

# СПОРА B: Целевая для объединения
print("📍 Создание споры B в позиции (0.52, 0.08)")
spore_b = Spore(
    pendulum=pendulum,
    dt=pendulum_config['dt'],
    goal_position=goal_position,
    scale=0.09,
    position=(0.52, 0, 0.08),
    color_manager=color_manager,
    config=config.get('spore', {})
)

# СПОРА C: Независимая спора
print("📍 Создание споры C в позиции (0.3, 0.3)")
spore_c = Spore(
    pendulum=pendulum,
    dt=pendulum_config['dt'],
    goal_position=goal_position,
    scale=0.09,
    position=(0.3, 0, 0.3),
    color_manager=color_manager,
    config=config.get('spore', {})
)

# ===== СОЗДАНИЕ МЕНЕДЖЕРА СПОР =====
spore_manager = SporeManager(
    pendulum=pendulum, 
    zoom_manager=zoom_manager, 
    settings_param=None,
    color_manager=color_manager,
    angel_manager=None,
    config=config,
    spawn_area=None
)

# Добавляем споры в менеджер
print("\n➕ Добавление спор в менеджер...")
spore_manager.add_spore(goal)
spore_manager.add_spore(spore_a)
spore_manager.add_spore(spore_b)
spore_manager.add_spore(spore_c)

# Регистрируем в zoom manager
zoom_manager.register_object(goal, "goal_spore")
zoom_manager.register_object(spore_a, "spore_a")
zoom_manager.register_object(spore_b, "spore_b") 
zoom_manager.register_object(spore_c, "spore_c")

print(f"\n✅ Создана тестовая среда:")
print(f"   Спор: {len(spore_manager.objects)} (включая цель)")
print(f"   Связей: {len(spore_manager.links)}")
print("📊 Цветовая схема:")
print("   🟣 Фиолетовый - обычные споры")
print("   🔵 Синий - споры с завершенной эволюцией")
print("   🟢 Зеленый - цель")
print("   🔴 Серый - мертвые споры")

current_active_index = len(spore_manager.objects) - 1

def show_spore_status():
    """Показывает подробный статус всех спор"""
    print(f"\n📊 СТАТУС ВСЕХ СПОР:")
    print("=" * 50)
    for i, spore in enumerate(spore_manager.objects):
        status_markers = []
        
        if hasattr(spore, 'is_goal') and spore.is_goal:
            status_markers.append("🏁 ЦЕЛЬ")
        else:
            if spore.is_alive():
                status_markers.append("💚 ЖИВА")
            else:
                status_markers.append("💀 МЕРТВА")
                
            if hasattr(spore, 'evolution_completed') and spore.evolution_completed:
                status_markers.append("🔵 ЭВОЛЮЦИЯ ЗАВЕРШЕНА")
                
            if hasattr(spore, 'can_evolve') and spore.can_evolve():
                status_markers.append("🚀 МОЖЕТ ЭВОЛЮЦИОНИРОВАТЬ")
            else:
                status_markers.append("🚫 НЕ МОЖЕТ ЭВОЛЮЦИОНИРОВАТЬ")
        
        active_marker = "🎯" if i == current_active_index else "  "
        
        print(f"{active_marker} Спора {spore.id}: {spore.calc_2d_pos()} | {' | '.join(status_markers)}")
    
    print("=" * 50)

def switch_active_spore(direction):
    """Переключает активную спору"""
    global current_active_index
    
    non_goal_spores = [i for i, spore in enumerate(spore_manager.objects) 
                       if not (hasattr(spore, 'is_goal') and spore.is_goal)]
    
    if direction == 'next':
        current_idx = non_goal_spores.index(current_active_index)
        next_idx = (current_idx + 1) % len(non_goal_spores)
        current_active_index = non_goal_spores[next_idx]
    elif direction == 'prev':
        current_idx = non_goal_spores.index(current_active_index)
        prev_idx = (current_idx - 1) % len(non_goal_spores)
        current_active_index = non_goal_spores[prev_idx]
    
    # Перемещаем активную спору в конец списка
    active_spore = spore_manager.objects[current_active_index]
    spore_manager.objects.remove(active_spore)
    spore_manager.objects.append(active_spore)
    
    current_active_index = len(spore_manager.objects) - 1
    
    print(f"\n🎯 АКТИВНАЯ СПОРА: {active_spore.id}")
    print(f"   Позиция: {active_spore.calc_2d_pos()}")
    print(f"   Может эволюционировать: {active_spore.can_evolve()}")

print("\n🎮 ПОДРОБНОЕ УПРАВЛЕНИЕ:")
print("=" * 50)
print("📍 ОСНОВНЫЕ КОМАНДЫ:")
print("   F - создать спору от активной (тест завершения эволюции)")
print("   Q - выход с отчетом")
print()
print("🔄 СМЕНА АКТИВНОЙ СПОРЫ:")
print("   N - следующая активная спора")
print("   P - предыдущая активная спора") 
print("   1,2,3 - выбрать спору A,B,C напрямую")
print()
print("📊 ИНФОРМАЦИЯ:")
print("   I - показать детальный статус всех спор")
print()
print("🔍 ВИЗУАЛИЗАЦИЯ:")
print("   E/T - изменить масштаб")
print("   R - сброс масштаба")
print()
print("🎯 ЦЕЛЬ ТЕСТА:")
print("   1. Объединить спору A со спорой B")
print("   2. Убедиться, что спора A больше не может эволюционировать")
print("   3. Проверить, что спора C может продолжать эволюцию")
print("=" * 50)

def update():
    pass

def input(key):
    print(f"\n🎹 КЛАВИША НАЖАТА: '{key}'")
    print("-" * 30)
    
    if key == 'q' or key == 'escape':
        print("📊 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ТЕСТА:")
        print("=" * 50)
        
        completed_spores = 0
        evolving_spores = 0
        dead_spores = 0
        
        for spore in spore_manager.objects:
            if hasattr(spore, 'is_goal') and spore.is_goal:
                continue
            elif hasattr(spore, 'evolution_completed') and spore.evolution_completed:
                completed_spores += 1
            elif not spore.is_alive():
                dead_spores += 1
            elif hasattr(spore, 'can_evolve') and spore.can_evolve():
                evolving_spores += 1
        
        print(f"   Всего спор: {len(spore_manager.objects)}")
        print(f"   Всего связей: {len(spore_manager.links)}")
        print(f"   🔵 Завершенные эволюции: {completed_spores}")
        print(f"   🚀 Способные эволюционировать: {evolving_spores}")
        print(f"   💀 Мертвые споры: {dead_spores}")
        print("=" * 50)
        print("👋 Тест завершен!")
        application.quit()
        return
    
    if key == 'f':
        active_spore = spore_manager.objects[-1]
        print(f"⚡ ПОПЫТКА СОЗДАНИЯ СПОРЫ ОТ {active_spore.id}")
        print(f"   📍 Текущая позиция: {active_spore.calc_2d_pos()}")
        print(f"   💚 Жива: {active_spore.is_alive()}")
        print(f"   🔵 Эволюция завершена: {getattr(active_spore, 'evolution_completed', False)}")
        print(f"   🚀 Может эволюционировать: {active_spore.can_evolve()}")
        
        result = spore_manager.generate_new_spore()
        
        if result:
            if result in spore_manager.objects[:-1]:
                print(f"   ✅ ОБЪЕДИНЕНИЕ! Создана связь к споре {result.id}")
                print(f"   🔵 Родительская спора {active_spore.id} завершила эволюцию")
            else:
                print(f"   ✅ Создана новая спора: {result.id}")
        else:
            print(f"   ❌ Спора не создана")
            
        print(f"   📈 Состояние: {len(spore_manager.objects)} спор, {len(spore_manager.links)} связей")
        return
    
    if key == 'i':
        show_spore_status()
        return
    
    if key == 'n':
        switch_active_spore('next')
        return
        
    if key == 'p':
        switch_active_spore('prev')
        return
    
    # Прямой выбор спор
    if key == '1':
        for i, spore in enumerate(spore_manager.objects):
            if hasattr(spore, 'id') and spore.id == spore_a.id:
                spore_manager.objects.append(spore_manager.objects.pop(i))
                current_active_index = len(spore_manager.objects) - 1
                print(f"🎯 Выбрана спора A (ID: {spore_a.id})")
                break
        return
    elif key == '2':
        for i, spore in enumerate(spore_manager.objects):
            if hasattr(spore, 'id') and spore.id == spore_b.id:
                spore_manager.objects.append(spore_manager.objects.pop(i))
                current_active_index = len(spore_manager.objects) - 1
                print(f"🎯 Выбрана спора B (ID: {spore_b.id})")
                break
        return
    elif key == '3':
        for i, spore in enumerate(spore_manager.objects):
            if hasattr(spore, 'id') and spore.id == spore_c.id:
                spore_manager.objects.append(spore_manager.objects.pop(i))
                current_active_index = len(spore_manager.objects) - 1
                print(f"🎯 Выбрана спора C (ID: {spore_c.id})")
                break
        return
    
    # Базовые команды масштабирования
    if key == 'e':
        zoom_manager.zoom_in()
    elif key == 't':
        zoom_manager.zoom_out()
    elif key == 'r':
        zoom_manager.reset_zoom()
    else:
        print(f"   ❓ Неизвестная клавиша")

print("\n✨ Тест завершения эволюции запущен!")
print("🔍 Используйте клавиши для тестирования логики")

# Показываем начальный статус
show_spore_status()

app.run() 