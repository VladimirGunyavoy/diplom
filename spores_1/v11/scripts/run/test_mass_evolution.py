#!/usr/bin/env python3
"""
ТЕСТ МАССОВОГО РАЗВИТИЯ КАНДИДАТОВ
=================================

Проверяет новую функциональность кнопки V - развитие всех кандидатов до смерти или остановки эволюции.

Что тестируется:
1. Создание кандидатских спор
2. Массовая активация всех кандидатов
3. Развитие каждого до завершения эволюции
4. Детальное логирование процесса
"""

import os
import sys
import json
import collections.abc
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

def simulate_mass_evolution():
    """Симулирует массовое развитие кандидатов без GUI."""
    
    print("🧪 СИМУЛЯЦИЯ МАССОВОГО РАЗВИТИЯ КАНДИДАТОВ")
    print("=" * 50)
    
    # Загружаем конфигурацию
    config_path = os.path.join(project_root, 'config', 'json', 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Включаем отладочный вывод
    test_config = {
        'trajectory_optimization': {
            'debug_output': True,
            'tolerance': 0.1
        },
        'spore': {
            'goal_position': [3.14159, 0],
            'scale': 0.05
        },
        'pendulum': {
            'dt': 0.1
        }
    }
    config = deep_merge(config, test_config)
    
    try:
        # Импортируем необходимые классы
        from src.logic.pendulum import PendulumSystem
        from src.logic.spawn_area import SpawnArea
        from src.managers.spore_manager import SporeManager
        from src.managers.color_manager import ColorManager
        from src.managers.zoom_manager import ZoomManager
        from src.visual.scene_setup import SceneSetup
        from src.core.spore import Spore
        
        print("✅ Все модули успешно загружены")
        
        # Создаем моки для объектов, которые нужны только для UI
        class MockZoomManager:
            def register_object(self, obj, name): 
                print(f"   🔧 [MOCK] Регистрация объекта: {name}")
            def unregister_object(self, name): 
                print(f"   🔧 [MOCK] Удаление объекта: {name}")
            def update_transform(self): 
                print(f"   🔧 [MOCK] Обновление трансформации")
        
        # Создаем основные компоненты
        pendulum = PendulumSystem(config.get('pendulum', {}))
        color_manager = ColorManager()
        zoom_manager = MockZoomManager()
        
        # Создаем область спавна
        spawn_area = SpawnArea(
            center=(0, 0),
            semi_axes=(2.0, 1.5),
            config=config.get('spawn_area', {})
        )
        
        # Создаем менеджер спор
        spore_manager = SporeManager(
            pendulum=pendulum,
            zoom_manager=zoom_manager,
            settings_param=None,
            color_manager=color_manager,
            angel_manager=None,
            config=config,
            spawn_area=spawn_area
        )
        
        print("✅ Компоненты инициализированы")
        
        # Создаем начальную спору
        initial_spore = Spore(
            pendulum=pendulum,
            dt=config.get('pendulum', {}).get('dt', 0.1),
            goal_position=config.get('spore', {}).get('goal_position', [3.14159, 0]),
            scale=config.get('spore', {}).get('scale', 0.05),
            position=(0, 0, 0),
            color_manager=color_manager,
            config=config.get('spore', {})
        )
        
        spore_manager.add_spore(initial_spore)
        print(f"✅ Создана начальная спора {initial_spore.id}")
        
        # Создаем кандидатских спор
        print("\n📊 СОЗДАНИЕ КАНДИДАТОВ:")
        spore_manager.generate_candidate_spores()
        
        candidates_count = len(spore_manager.candidate_spores)
        print(f"✅ Создано кандидатов: {candidates_count}")
        
        if candidates_count == 0:
            print("❌ Нет кандидатов для тестирования")
            return
        
        # Ограничиваем количество кандидатов для тестирования
        max_test_candidates = 3
        if candidates_count > max_test_candidates:
            print(f"📊 Для тестирования оставляем только первых {max_test_candidates} кандидатов")
            # Удаляем лишних кандидатов
            for candidate in spore_manager.candidate_spores[max_test_candidates:]:
                spore_manager.zoom_manager.unregister_object(candidate.id)
            spore_manager.candidate_spores = spore_manager.candidate_spores[:max_test_candidates]
        
        print(f"\n🚀 НАЧИНАЕМ ТЕСТ МАССОВОГО РАЗВИТИЯ:")
        print(f"   📊 Кандидатов для обработки: {len(spore_manager.candidate_spores)}")
        print(f"   📊 Активных спор до теста: {len(spore_manager.objects)}")
        print(f"   📊 Связей до теста: {len(spore_manager.links)}")
        
        # ОСНОВНОЙ ТЕСТ: массовое развитие
        spore_manager.evolve_all_candidates_to_completion()
        
        print(f"\n📈 РЕЗУЛЬТАТЫ ТЕСТА:")
        print(f"   📊 Активных спор после теста: {len(spore_manager.objects)}")
        print(f"   📊 Связей после теста: {len(spore_manager.links)}")
        print(f"   📊 Оставшихся кандидатов: {len(spore_manager.candidate_spores)}")
        
        # Анализ результатов
        alive_spores = [s for s in spore_manager.objects if s.is_alive()]
        dead_spores = [s for s in spore_manager.objects if not s.is_alive()]
        completed_spores = [s for s in spore_manager.objects if s.evolution_completed]
        
        print(f"\n🔍 ДЕТАЛЬНАЯ СТАТИСТИКА:")
        print(f"   💚 Живых спор: {len(alive_spores)}")
        print(f"   💀 Мертвых спор: {len(dead_spores)}")
        print(f"   🏁 Завершивших эволюцию: {len(completed_spores)}")
        
        # Показываем финальные позиции
        print(f"\n📍 ФИНАЛЬНЫЕ ПОЗИЦИИ СПОР:")
        for spore in spore_manager.objects:
            status = ""
            if not spore.is_alive():
                status = "💀 мертва"
            elif spore.evolution_completed:
                status = "🏁 завершена"
            else:
                status = "💚 жива"
                
            print(f"   Спора {spore.id}: {spore.calc_2d_pos()} - cost: {spore.logic.cost:.6f} - {status}")
        
        print(f"\n🎉 ТЕСТ МАССОВОГО РАЗВИТИЯ ЗАВЕРШЕН УСПЕШНО!")
        
    except ImportError as e:
        print(f"❌ Ошибка импорта (возможно отсутствует Ursina): {e}")
        print("ℹ️  Это нормально для headless тестирования")
    except Exception as e:
        print(f"❌ Ошибка во время теста: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simulate_mass_evolution() 