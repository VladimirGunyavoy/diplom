#!/usr/bin/env python3
"""
Тест интеграции оптимизации по площади в пайплайн создания дерева из пар.
"""

import json
import numpy as np
from src.logic.tree.pairs.create_tree_from_pairs import create_tree_from_pairs
from src.logic.tree.spore_tree import SporeTree
from src.logic.tree.spore_tree_config import SporeTreeConfig
from src.logic.pendulum import PendulumSystem

def test_area_optimization_integration():
    """Тестирует интеграцию оптимизации по площади в пайплайн."""
    print("🧪 ТЕСТ ИНТЕГРАЦИИ ОПТИМИЗАЦИИ ПО ПЛОЩАДИ")
    print("=" * 60)
    
    # Загружаем конфигурацию
    with open('config/json/config.json', 'r') as f:
        config = json.load(f)
    
    print("✅ Конфигурация загружена")
    print(f"   area_optimization.enabled: {config.get('tree', {}).get('area_optimization', {}).get('enabled', False)}")
    
    # Создаем маятник
    pendulum = PendulumSystem(config)
    print("✅ PendulumSystem создан")
    
    # Создаем исходное дерево
    tree_config = SporeTreeConfig(
        dt_base=config.get('pendulum', {}).get('dt', 0.05),
        dt_grandchildren_factor=config.get('tree', {}).get('dt_grandchildren_factor', 0.1),
        show_debug=False
    )
    
    original_tree = SporeTree(
        pendulum=pendulum,
        config=tree_config,
        auto_create=True,
        show=False
    )
    print("✅ Исходное дерево создано")
    print(f"   Детей: {len(original_tree.children)}")
    print(f"   Внуков: {len(original_tree.grandchildren)}")
    
    # Запускаем полный пайплайн с оптимизацией по площади
    print("\n🚀 Запуск пайплайна 'поиск пар → сбор дерева → оптимизация площади'...")
    
    result = create_tree_from_pairs(
        tree=original_tree,
        pendulum=pendulum,
        config=config,
        show=True
    )
    
    if result and result['success']:
        print("\n✅ Пайплайн выполнен успешно!")
        
        # Проверяем результаты
        stats = result['stats']
        area_opt = result.get('area_optimization')
        
        print(f"\n📊 СТАТИСТИКА ПАЙПЛАЙНА:")
        print(f"   Найдено пар: {stats['pairs_found']}")
        print(f"   Спарено внуков: {stats['paired_grandchildren']}/{stats['total_grandchildren']}")
        print(f"   Изменено времен: {stats['times_changed']}")
        
        if area_opt:
            if area_opt.get('success'):
                print(f"\n🔷 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ПО ПЛОЩАДИ:")
                print(f"   Исходная площадь: {area_opt['original_area']:.6f}")
                print(f"   Оптимизированная площадь: {area_opt['optimized_area']:.6f}")
                print(f"   Улучшение: {area_opt['area_improvement']:.6f}")
                print(f"   Улучшение (%): {area_opt['area_improvement_percent']:.2f}%")
                print(f"   Констрейнты удовлетворены: {area_opt['constraints_satisfied']}")
                print(f"   Количество констрейнтов: {area_opt['constraints_count']}")
                print(f"   Количество пар: {area_opt['pairs_count']}")
            else:
                print(f"\n⚠️ Оптимизация по площади не удалась")
        else:
            print(f"\nℹ️ Оптимизация по площади не выполнялась (enabled=false)")
        
        # Проверяем, что дерево создано
        optimized_tree = result['optimized_tree']
        if optimized_tree:
            print(f"\n🌳 ОПТИМИЗИРОВАННОЕ ДЕРЕВО:")
            print(f"   Детей: {len(optimized_tree.children)}")
            print(f"   Внуков: {len(optimized_tree.grandchildren)}")
            print(f"   Тип: {type(optimized_tree).__name__}")
        
    else:
        print(f"\n❌ Пайплайн не выполнен:")
        if result:
            print(f"   Ошибка: {result.get('error', 'неизвестная ошибка')}")
        else:
            print(f"   Критическая ошибка")
    
    print(f"\n🎉 ТЕСТ ЗАВЕРШЕН!")

if __name__ == "__main__":
    test_area_optimization_integration()
