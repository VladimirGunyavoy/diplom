#!/usr/bin/env python3
"""
Тест граничных случаев оптимизации траекторий
- Споры с существующими связями
- Разные направления движения
- Ручное управление активной спорой
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

print("=== ТЕСТ ГРАНИЧНЫХ СЛУЧАЕВ ===")
print("🎯 Проверка поведения при существующих связях")
print("=" * 40)

# ===== ИНИЦИАЛИЗАЦИЯ =====
app = Ursina()

# ===== НАСТРОЙКА ОКНА ДЛЯ ПАРАЛЛЕЛЬНОГО ПРОСМОТРА =====
window_manager = WindowManager(
    title="Тест граничных случаев - Споры v11",
    size=(800, 600)  # Компактный размер для размещения рядом с консолью
)
window_manager.set_background_color(color.dark_gray)

color_manager = ColorManager()

print("\n🪟 НАСТРОЙКА ОКОН:")
print("• Разместите окно Ursina справа от консоли")
print("• Размер окна: 800x600 пикселей")
print("• Консоль слева покажет детальную информацию")

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
print("\n🌟 Создание сложного сценария...")

goal_position = config['spore']['goal_position']

# ЦЕЛЬ
print("📍 Создание цели в позиции", goal_position)
goal = Spore(
    pendulum=pendulum,
    dt=pendulum_config['dt'],
    scale=0.08,
    position=(goal_position[0], 0, goal_position[1]),
    goal_position=goal_position,
    is_goal=True,
    color_manager=color_manager
)

# СПОРА A: стартовая позиция (0.5, 0.1)
print("📍 Создание споры A в позиции (0.5, 0.1)")
spore_a = Spore(
    pendulum=pendulum,
    dt=pendulum_config['dt'],
    goal_position=goal_position,
    scale=0.08,
    position=(0.5, 0, 0.1),
    color_manager=color_manager,
    config=config.get('spore', {})
)

# СПОРА B: промежуточная позиция (0.6, 0.15)
print("📍 Создание споры B в позиции (0.6, 0.15)")
spore_b = Spore(
    pendulum=pendulum,
    dt=pendulum_config['dt'],
    goal_position=goal_position,
    scale=0.08,
    position=(0.6, 0, 0.15),
    color_manager=color_manager,
    config=config.get('spore', {})
)

# СПОРА C: целевая позиция для объединения (0.68, 0.12)
print("📍 Создание споры C в позиции (0.68, 0.12)")
spore_c = Spore(
    pendulum=pendulum,
    dt=pendulum_config['dt'],
    goal_position=goal_position,
    scale=0.08,
    position=(0.68, 0, 0.12),
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

# ===== СОЗДАНИЕ НАЧАЛЬНЫХ СВЯЗЕЙ =====
print("\n🔗 Создание цепочки связей A → B → C...")

# Связь A → B
link_ab = Link(spore_a, spore_b, 
               color_manager=color_manager,
               zoom_manager=zoom_manager,
               config=config)
spore_manager.links.append(link_ab)
zoom_manager.register_object(link_ab, "link_ab")

# Связь B → C  
link_bc = Link(spore_b, spore_c,
               color_manager=color_manager, 
               zoom_manager=zoom_manager,
               config=config)
spore_manager.links.append(link_bc)
zoom_manager.register_object(link_bc, "link_bc")

# Обновляем геометрию связей
link_ab.update_geometry()
link_bc.update_geometry()
zoom_manager.update_transform()

print(f"\n✅ Создана тестовая среда:")
print(f"   Спор: {len(spore_manager.objects)} (включая цель)")
print(f"   Связей: {len(spore_manager.links)}")
print("📊 Структура:")
print("   Цель ← A → B → C")
print("   Последняя спора (активная): C")

# ===== ФУНКЦИЯ ДЛЯ СМЕНЫ АКТИВНОЙ СПОРЫ =====
current_active_index = len(spore_manager.objects) - 1  # Индекс активной споры

def switch_active_spore(direction):
    """Переключает активную спору"""
    global current_active_index
    
    # Исключаем цель из выбора (она всегда первая)
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
    
    # Перемещаем активную спору в конец списка (чтобы стала "последней")
    active_spore = spore_manager.objects[current_active_index]
    spore_manager.objects.remove(active_spore)
    spore_manager.objects.append(active_spore)
    
    # Обновляем индекс
    current_active_index = len(spore_manager.objects) - 1
    
    print(f"🎯 Активная спора: {active_spore.id} в позиции {active_spore.calc_2d_pos()}")

print("\n🎮 ПОДРОБНОЕ УПРАВЛЕНИЕ:")
print("=" * 40)
print("📍 ОСНОВНЫЕ КОМАНДЫ:")
print("   F - создать спору от активной (проверка объединения)")
print("   Q - выход с отчетом")
print()
print("🔄 СМЕНА АКТИВНОЙ СПОРЫ:")
print("   N - следующая активная спора")
print("   P - предыдущая активная спора") 
print("   1 - выбрать спору A (позиция 0.5, 0.1)")
print("   2 - выбрать спору B (позиция 0.6, 0.15)")
print("   3 - выбрать спору C (позиция 0.68, 0.12)")
print()
print("📊 ИНФОРМАЦИЯ:")
print("   I - показать детальную информацию о всех спорах")
print()
print("🔍 ВИЗУАЛИЗАЦИЯ:")
print("   E/T - изменить масштаб")
print("   R - сброс масштаба")
print()
print("🎯 ЦЕЛЬ ТЕСТА:")
print("   Проверить объединение траекторий при существующих связях")
print("   Желтые связи = объединение, оранжевые = обычные")
print("=" * 40)

def update():
    pass

def input(key):
    print(f"\n🎹 КЛАВИША НАЖАТА: '{key}'")
    print("-" * 30)
    
    if key == 'q' or key == 'escape':
        print("📊 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ТЕСТА:")
        print("=" * 40)
        print(f"   Всего спор: {len(spore_manager.objects)}")
        print(f"   Всего связей: {len(spore_manager.links)}")
        print("   Типы связей:")
        
        # Анализируем типы связей
        normal_links = 0
        merge_links = 0
        for link in spore_manager.links:
            if hasattr(link, 'color'):
                if 'active' in str(link.color).lower() or 'yellow' in str(link.color).lower():
                    merge_links += 1
                else:
                    normal_links += 1
            else:
                normal_links += 1
        
        print(f"     🔗 Обычные связи: {normal_links}")
        print(f"     💛 Объединенные связи: {merge_links}")
        print("=" * 40)
        print("👋 Тест завершен!")
        application.quit()
        return
    
    if key == 'f':
        active_spore = spore_manager.objects[-1]
        print(f"⚡ ГЕНЕРАЦИЯ СПОРЫ ОТ АКТИВНОЙ")
        print(f"   🎯 Активная спора: {active_spore.id}")
        print(f"   📍 Текущая позиция: {active_spore.calc_2d_pos()}")
        print(f"   🎮 Оптимальное управление: {active_spore.logic.optimal_control}")
        print(f"   ⏱️  Оптимальное dt: {active_spore.logic.optimal_dt}")
        print(f"   💚 Жива: {active_spore.is_alive()}")
        
        # Показываем, какие споры есть рядом
        if active_spore.logic.optimal_control is not None and active_spore.logic.optimal_dt is not None:
            next_pos_3d = active_spore.evolve_3d(
                control=active_spore.logic.optimal_control[0], 
                dt=active_spore.logic.optimal_dt
            )
            next_pos_2d = np.array([next_pos_3d[0], next_pos_3d[2]])
            print(f"   🎪 Следующая позиция: {next_pos_2d}")
            
            # Проверяем расстояния до всех спор
            tolerance = config.get('trajectory_optimization', {}).get('merge_tolerance', 0.1)
            print(f"   📏 Tolerance для объединения: {tolerance}")
            
            print("   🔍 Расстояния до других спор:")
            for i, other_spore in enumerate(spore_manager.objects):
                if other_spore.id != active_spore.id and not (hasattr(other_spore, 'is_goal') and other_spore.is_goal):
                    distance = np.linalg.norm(next_pos_2d - other_spore.calc_2d_pos())
                    status = "🎯 МОЖЕТ ОБЪЕДИНИТЬСЯ" if distance <= tolerance else "🚫 Слишком далеко"
                    print(f"     Спора {other_spore.id}: {distance:.6f} ({status})")
        
        print("   🚀 Вызываем generate_new_spore()...")
        
        result = spore_manager.generate_new_spore()
        
        print(f"   📊 Результат:")
        if result:
            if result in spore_manager.objects[:-1]:  # Если вернулась существующая спора
                print(f"   ✅ ОБЪЕДИНЕНИЕ! Создана связь к споре {result.id}")
                print(f"   💛 Это должна быть желтая связь!")
            else:
                print(f"   ✅ Создана новая спора: {result.id}")
                print(f"   🔗 Это должна быть оранжевая связь")
        else:
            print("   ❌ Спора не создана (мертвая или ошибка)")
        
        print(f"   📈 Текущее состояние: {len(spore_manager.objects)} спор, {len(spore_manager.links)} связей")
        return
    
    if key == 'n':
        switch_active_spore('next')
        return
        
    if key == 'p':
        switch_active_spore('prev')
        return
    
    # Прямой выбор спор
    if key == '1':
        # Спора A (индекс 1, так как 0 - цель)
        spore_manager.objects.append(spore_manager.objects.pop(1))
        current_active_index = len(spore_manager.objects) - 1
        print(f"🎯 Выбрана спора A")
        return
    elif key == '2':
        # Спора B (была индекс 2, но после перемещения A может измениться)
        for i, spore in enumerate(spore_manager.objects):
            if spore.calc_2d_pos()[0] == 0.6:  # Найдем спору B по координатам
                spore_manager.objects.append(spore_manager.objects.pop(i))
                break
        current_active_index = len(spore_manager.objects) - 1
        print(f"🎯 Выбрана спора B")
        return
    elif key == '3':
        # Спора C
        for i, spore in enumerate(spore_manager.objects):
            if abs(spore.calc_2d_pos()[0] - 0.68) < 0.01:  # Найдем спору C по координатам
                spore_manager.objects.append(spore_manager.objects.pop(i))
                break
        current_active_index = len(spore_manager.objects) - 1
        print(f"🎯 Выбрана спора C")
        return
    
    if key == 'i':
        print(f"\n📊 Информация о спорах:")
        print(f"   Всего спор: {len(spore_manager.objects)}")
        print(f"   Всего связей: {len(spore_manager.links)}")
        print("   Детали спор:")
        for i, spore in enumerate(spore_manager.objects):
            status = "🎯 АКТИВНАЯ" if i == len(spore_manager.objects) - 1 else ""
            goal_marker = "🏁" if hasattr(spore, 'is_goal') and spore.is_goal else ""
            print(f"     {i}: ID={spore.id}, pos={spore.calc_2d_pos()}, alive={spore.is_alive()} {goal_marker} {status}")
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

print("\n✨ Тест граничных случаев запущен!")
print("🔍 Проверьте поведение при объединении спор с существующими связями")

app.run() 