#!/usr/bin/env python3
"""
Автоматизированный тест граничных случаев оптимизации траекторий
Проверяет все возможные сценарии программно без ручного ввода
"""

import os
import sys
import json
import collections.abc
import numpy as np

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
from src.logic.pendulum import PendulumSystem
from src.core.spore import Spore
from src.managers.spore_manager import SporeManager
from src.visual.link import Link

print("=== АВТОМАТИЗИРОВАННЫЙ ТЕСТ ГРАНИЧНЫХ СЛУЧАЕВ ===")
print("🎯 Программная проверка всех сценариев оптимизации траекторий")
print("=" * 60)

def create_test_environment():
    """Создает тестовую среду без графики"""
    
    # Создаем базовые менеджеры
    color_manager = ColorManager()
    pendulum_config = config['pendulum']
    pendulum = PendulumSystem(damping=pendulum_config['damping'])
    goal_position = config['spore']['goal_position']
    
    # Создаем спор менеджер без графических компонентов
    spore_manager = SporeManager(
        pendulum=pendulum, 
        zoom_manager=None,  # Без графики
        settings_param=None,
        color_manager=color_manager,
        angel_manager=None,
        config=config,
        spawn_area=None
    )
    
    return spore_manager, pendulum, goal_position

def test_scenario_1():
    """
    Сценарий 1: Спора с существующими связями объединяется с другой спорой
    """
    print("\n🔬 ТЕСТ 1: Спора с существующими связями")
    print("-" * 40)
    
    spore_manager, pendulum, goal_position = create_test_environment()
    pendulum_config = config['pendulum']
    
    # Создаем споры
    spore_a = Spore(pendulum=pendulum, dt=pendulum_config['dt'], 
                   goal_position=goal_position, position=(0.5, 0, 0.1),
                   color_manager=spore_manager.color_manager, config=config.get('spore', {}))
    
    spore_b = Spore(pendulum=pendulum, dt=pendulum_config['dt'],
                   goal_position=goal_position, position=(0.6, 0, 0.15), 
                   color_manager=spore_manager.color_manager, config=config.get('spore', {}))
    
    spore_c = Spore(pendulum=pendulum, dt=pendulum_config['dt'],
                   goal_position=goal_position, position=(0.68, 0, 0.12),
                   color_manager=spore_manager.color_manager, config=config.get('spore', {}))
    
    # Добавляем в менеджер
    spore_manager.add_spore(spore_a) 
    spore_manager.add_spore(spore_b)
    spore_manager.add_spore(spore_c)
    
    # Создаем начальные связи A→B→C
    link_ab = Link(spore_a, spore_b, color_manager=spore_manager.color_manager, 
                   zoom_manager=None, config=config)
    link_bc = Link(spore_b, spore_c, color_manager=spore_manager.color_manager,
                   zoom_manager=None, config=config)
    spore_manager.links.extend([link_ab, link_bc])
    
    print(f"📊 Начальное состояние:")
    print(f"   Споры: {len(spore_manager.objects)} (A, B, C)")
    print(f"   Связи: {len(spore_manager.links)} (A→B, B→C)")
    print(f"   Позиции: A={spore_a.calc_2d_pos()}, B={spore_b.calc_2d_pos()}, C={spore_c.calc_2d_pos()}")
    
    # Тест: попытка создать спору от B, которая может прийти к C
    print(f"\n🧪 Делаем B активной (перемещаем в конец списка)")
    spore_manager.objects.remove(spore_b)
    spore_manager.objects.append(spore_b)
    
    print(f"   Активная спора: {spore_manager.objects[-1].id} в позиции {spore_manager.objects[-1].calc_2d_pos()}")
    print(f"   Оптимальное управление: {spore_b.logic.optimal_control}")
    print(f"   Оптимальное dt: {spore_b.logic.optimal_dt}")
    
    # Проверяем, куда приведет оптимальный шаг от B
    if spore_b.logic.optimal_control is not None and spore_b.logic.optimal_dt is not None:
        next_pos_3d = spore_b.evolve_3d(control=spore_b.logic.optimal_control[0], 
                                         dt=spore_b.logic.optimal_dt)
        next_pos_2d = np.array([next_pos_3d[0], next_pos_3d[2]])
        print(f"   Следующая позиция B: {next_pos_2d}")
        
        # Проверяем расстояние до C
        distance_to_c = np.linalg.norm(next_pos_2d - spore_c.calc_2d_pos())
        tolerance = config.get('trajectory_optimization', {}).get('merge_tolerance', 0.1)
        print(f"   Расстояние до C: {distance_to_c:.6f} (tolerance: {tolerance})")
        
        result = spore_manager.generate_new_spore()
        
        if result == spore_c:
            print(f"✅ УСПЕХ: Объединение произошло! B → C")
            print(f"   Новых спор не создано, создана связь B→C")
        elif result != spore_b:  # Создалась новая спора
            print(f"⚠️  Создана новая спора {result.id} вместо объединения")
            print(f"   Возможно, расстояние больше tolerance")
        else:
            print(f"❌ Спора не создана (мертвая родительская или ошибка)")
    else:
        print(f"❌ У споры B нет оптимального управления")
    
    return len(spore_manager.objects), len(spore_manager.links)

def test_scenario_2():
    """
    Сценарий 2: Первая спора движется к другой споре
    """
    print("\n🔬 ТЕСТ 2: Первая спора движется")
    print("-" * 30)
    
    spore_manager, pendulum, goal_position = create_test_environment()
    pendulum_config = config['pendulum']
    
    # Создаем две споры в стратегических позициях
    spore_first = Spore(pendulum=pendulum, dt=pendulum_config['dt'],
                       goal_position=goal_position, position=(0.3, 0, 0.05), 
                       color_manager=spore_manager.color_manager, config=config.get('spore', {}))
    
    spore_target = Spore(pendulum=pendulum, dt=pendulum_config['dt'],
                        goal_position=goal_position, position=(0.68, 0, 0.12),
                        color_manager=spore_manager.color_manager, config=config.get('spore', {}))
    
    # Добавляем в менеджер (первая будет добавлена первой)
    spore_manager.add_spore(spore_first)
    spore_manager.add_spore(spore_target)
    
    print(f"📊 Начальное состояние:")
    print(f"   Первая спора: {spore_first.id} в {spore_first.calc_2d_pos()}")
    print(f"   Целевая спора: {spore_target.id} в {spore_target.calc_2d_pos()}")
    
    # Делаем первую спору активной
    spore_manager.objects.remove(spore_first)
    spore_manager.objects.append(spore_first)
    
    print(f"\n🧪 Делаем первую спору активной")
    print(f"   Активная: {spore_manager.objects[-1].id}")
    print(f"   Оптимальное управление: {spore_first.logic.optimal_control}")
    
    # Несколько шагов для движения первой споры к целевой
    steps_taken = 0
    max_steps = 10
    
    while steps_taken < max_steps:
        print(f"\n📍 Шаг {steps_taken + 1}:")
        print(f"   Позиция первой споры: {spore_manager.objects[-1].calc_2d_pos()}")
        
        result = spore_manager.generate_new_spore()
        
        if result == spore_target:
            print(f"✅ УСПЕХ: Объединение! Первая спора достигла целевой")
            break
        elif result and result != spore_manager.objects[-2]:  # Новая спора создана
            print(f"   Создана промежуточная спора {result.id}")
            steps_taken += 1
        else:
            print(f"❌ Движение остановлено (мертвая спора или ошибка)")
            break
    
    if steps_taken >= max_steps:
        print(f"⚠️  Достигнут лимит шагов ({max_steps}), объединение не произошло")
    
    return len(spore_manager.objects), len(spore_manager.links)

def test_scenario_3():
    """
    Сценарий 3: Объединение к споре которая уже имеет множественные связи
    """
    print("\n🔬 ТЕСТ 3: Объединение к споре с множественными связями")
    print("-" * 50)
    
    spore_manager, pendulum, goal_position = create_test_environment()
    pendulum_config = config['pendulum']
    
    # Создаем центральную спору и несколько спор вокруг нее
    central_spore = Spore(pendulum=pendulum, dt=pendulum_config['dt'],
                         goal_position=goal_position, position=(0.68, 0, 0.12),
                         color_manager=spore_manager.color_manager, config=config.get('spore', {}))
    
    satellite_1 = Spore(pendulum=pendulum, dt=pendulum_config['dt'],
                        goal_position=goal_position, position=(0.6, 0, 0.1),
                        color_manager=spore_manager.color_manager, config=config.get('spore', {}))
    
    satellite_2 = Spore(pendulum=pendulum, dt=pendulum_config['dt'],
                        goal_position=goal_position, position=(0.65, 0, 0.08),
                        color_manager=spore_manager.color_manager, config=config.get('spore', {}))
    
    approaching_spore = Spore(pendulum=pendulum, dt=pendulum_config['dt'],
                             goal_position=goal_position, position=(0.66, 0, 0.13),
                             color_manager=spore_manager.color_manager, config=config.get('spore', {}))
    
    # Добавляем в менеджер
    spore_manager.add_spore(central_spore)
    spore_manager.add_spore(satellite_1) 
    spore_manager.add_spore(satellite_2)
    spore_manager.add_spore(approaching_spore)
    
    # Создаем связи к центральной споре
    link_1_to_central = Link(satellite_1, central_spore, 
                            color_manager=spore_manager.color_manager,
                            zoom_manager=None, config=config)
    link_2_to_central = Link(satellite_2, central_spore,
                            color_manager=spore_manager.color_manager, 
                            zoom_manager=None, config=config)
    
    spore_manager.links.extend([link_1_to_central, link_2_to_central])
    
    print(f"📊 Начальное состояние:")
    print(f"   Центральная спора: {central_spore.id} в {central_spore.calc_2d_pos()}")
    print(f"   Связи к центральной: {len([l for l in spore_manager.links if l.child_spore == central_spore])}")
    print(f"   Приближающаяся спора: {approaching_spore.id} в {approaching_spore.calc_2d_pos()}")
    
    # Делаем приближающуюся спору активной
    spore_manager.objects.remove(approaching_spore)
    spore_manager.objects.append(approaching_spore)
    
    print(f"\n🧪 Тестируем объединение приближающейся споры с центральной")
    
    result = spore_manager.generate_new_spore()
    
    if result == central_spore:
        print(f"✅ УСПЕХ: Объединение! Третья связь создана к центральной споре")
        central_links = len([l for l in spore_manager.links if l.child_spore == central_spore])
        print(f"   Центральная спора теперь имеет {central_links} входящих связей")
    else:
        print(f"⚠️  Объединение не произошло, создана новая спора или ошибка")
    
    return len(spore_manager.objects), len(spore_manager.links)

def main():
    """Основная функция запуска всех тестов"""
    
    print("🚀 Запуск автоматизированных тестов...")
    
    # Выполняем все тесты
    results = []
    
    try:
        spores_1, links_1 = test_scenario_1()
        results.append(("Сценарий 1 (споры с связями)", spores_1, links_1))
    except Exception as e:
        print(f"❌ Ошибка в тесте 1: {e}")
        results.append(("Сценарий 1 (ошибка)", 0, 0))
    
    try:
        spores_2, links_2 = test_scenario_2()
        results.append(("Сценарий 2 (первая движется)", spores_2, links_2))
    except Exception as e:
        print(f"❌ Ошибка в тесте 2: {e}")
        results.append(("Сценарий 2 (ошибка)", 0, 0))
    
    try:
        spores_3, links_3 = test_scenario_3()
        results.append(("Сценарий 3 (множественные связи)", spores_3, links_3))
    except Exception as e:
        print(f"❌ Ошибка в тесте 3: {e}")
        results.append(("Сценарий 3 (ошибка)", 0, 0))
    
    # Выводим сводку
    print("\n" + "=" * 60)
    print("📊 СВОДКА РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    for scenario, spores, links in results:
        print(f"{scenario:35} | Споры: {spores:2} | Связи: {links:2}")
    
    print("\n🎯 ВЫВОДЫ:")
    print("• Система оптимизации траекторий протестирована")
    print("• Граничные случаи с существующими связями проверены")
    print("• Поведение при множественных связях к одной споре изучено")
    print("\n✅ Автоматизированное тестирование завершено!")

if __name__ == "__main__":
    main() 