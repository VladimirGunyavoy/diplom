#!/usr/bin/env python3
"""
Простой тест интеграции оптимизации по площади - проверяет импорты и конфигурацию.
"""

import json
import numpy as np

def test_area_optimization_imports():
    """Тестирует импорты и конфигурацию для оптимизации по площади."""
    print("🧪 ТЕСТ ИМПОРТОВ И КОНФИГУРАЦИИ ОПТИМИЗАЦИИ ПО ПЛОЩАДИ")
    print("=" * 60)
    
    # Проверяем импорты
    try:
        from src.logic.tree.pairs.create_tree_from_pairs import create_tree_from_pairs
        print("✅ create_tree_from_pairs импортирован")
    except Exception as e:
        print(f"❌ Ошибка импорта create_tree_from_pairs: {e}")
        return
    
    try:
        from src.logic.tree.area_opt.optimize_tree_area import optimize_tree_area
        print("✅ optimize_tree_area импортирован")
    except Exception as e:
        print(f"❌ Ошибка импорта optimize_tree_area: {e}")
        return
    
    try:
        from src.logic.tree.spore_tree_config import SporeTreeConfig
        print("✅ SporeTreeConfig импортирован")
    except Exception as e:
        print(f"❌ Ошибка импорта SporeTreeConfig: {e}")
        return
    
    # Загружаем конфигурацию
    try:
        with open('config/json/config.json', 'r') as f:
            config = json.load(f)
        print("✅ Конфигурация загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
        return
    
    # Проверяем настройки оптимизации по площади
    area_cfg = config.get('tree', {}).get('area_optimization', {})
    print(f"\n📋 НАСТРОЙКИ ОПТИМИЗАЦИИ ПО ПЛОЩАДИ:")
    print(f"   enabled: {area_cfg.get('enabled', False)}")
    print(f"   constraint_distance: {area_cfg.get('constraint_distance', 1e-4)}")
    print(f"   dt_bounds: {area_cfg.get('dt_bounds', [0.001, 0.2])}")
    print(f"   max_iterations: {area_cfg.get('max_iterations', 1500)}")
    print(f"   method: {area_cfg.get('method', 'SLSQP')}")
    
    # Проверяем создание SporeTreeConfig
    try:
        tree_config = SporeTreeConfig(
            dt_base=config.get('pendulum', {}).get('dt', 0.05),
            dt_grandchildren_factor=config.get('tree', {}).get('dt_grandchildren_factor', 0.1),
            show_debug=False
        )
        print("✅ SporeTreeConfig создан успешно")
        print(f"   dt_base: {tree_config.dt_base}")
        print(f"   dt_grandchildren_factor: {tree_config.dt_grandchildren_factor}")
        print(f"   initial_position: {tree_config.initial_position}")
    except Exception as e:
        print(f"❌ Ошибка создания SporeTreeConfig: {e}")
        return
    
    # Проверяем сигнатуру функции optimize_tree_area
    import inspect
    sig = inspect.signature(optimize_tree_area)
    print(f"\n🔍 СИГНАТУРА optimize_tree_area:")
    print(f"   {sig}")
    
    # Проверяем, что все необходимые параметры есть
    required_params = ['tree', 'pairs', 'pendulum']
    missing_params = [param for param in required_params if param not in sig.parameters]
    
    if missing_params:
        print(f"❌ Отсутствуют обязательные параметры: {missing_params}")
    else:
        print("✅ Все обязательные параметры присутствуют")
    
    print(f"\n🎉 ТЕСТ ИМПОРТОВ ЗАВЕРШЕН УСПЕШНО!")

if __name__ == "__main__":
    test_area_optimization_imports()
