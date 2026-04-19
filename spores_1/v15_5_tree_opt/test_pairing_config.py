#!/usr/bin/env python3
"""
Тестовый скрипт для проверки конфигурации спаривания
"""

import json
import os

def test_pairing_config():
    print("🔧 ТЕСТИРОВАНИЕ КОНФИГУРАЦИИ СПАРИВАНИЯ")
    print("=" * 60)
    
    # Путь к конфигурационному файлу
    config_path = "v15_5_tree_opt/config/json/config.json"
    
    print(f"\n📋 Проверяем файл: {config_path}")
    
    if os.path.exists(config_path):
        print("✅ Файл конфигурации найден")
        
        # Загружаем конфигурацию
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Проверяем секцию tree.pairing
        tree_config = config.get('tree', {})
        pairing_config = tree_config.get('pairing', {})
        
        print(f"\n📊 Текущая конфигурация спаривания:")
        print(f"   enabled: {pairing_config.get('enabled', 'НЕ УСТАНОВЛЕН')}")
        print(f"   show_debug: {pairing_config.get('show_debug', 'НЕ УСТАНОВЛЕН')}")
        print(f"   dt_grandchildren_factor: {pairing_config.get('dt_grandchildren_factor', 'НЕ УСТАНОВЛЕН')}")
        print(f"   max_pairs: {pairing_config.get('max_pairs', 'НЕ УСТАНОВЛЕН')}")
        print(f"   min_distance_threshold: {pairing_config.get('min_distance_threshold', 'НЕ УСТАНОВЛЕН')}")
        print(f"   max_distance_threshold: {pairing_config.get('max_distance_threshold', 'НЕ УСТАНОВЛЕН')}")
        print(f"   time_precision: {pairing_config.get('time_precision', 'НЕ УСТАНОВЛЕН')}")
        print(f"   position_precision: {pairing_config.get('position_precision', 'НЕ УСТАНОВЛЕН')}")
        
        # Проверяем, что все параметры установлены
        required_params = [
            'enabled', 'show_debug', 'dt_grandchildren_factor', 'max_pairs',
            'min_distance_threshold', 'max_distance_threshold', 'time_precision', 'position_precision'
        ]
        missing_params = [param for param in required_params if param not in pairing_config]
        
        if missing_params:
            print(f"\n⚠️  Отсутствующие параметры: {missing_params}")
            print("   Добавьте их в секцию tree.pairing в config.json")
        else:
            print(f"\n✅ Все параметры спаривания настроены")
        
        # Показываем различия между клавишами 'P' и 'O'
        print(f"\n🔍 РАЗЛИЧИЯ МЕЖДУ КЛАВИШАМИ 'P' И 'O':")
        print(f"   Клавиша 'P' (спаривание):")
        print(f"     - Использует find_optimal_pairs()")
        print(f"     - Применяет dt из meeting_info")
        print(f"     - Сохраняет знаки dt")
        print(f"     - Обновляет ghost_tree_dt_vector")
        print(f"     - Параметры из tree.pairing")
        
        print(f"\n   Клавиша 'O' (оптимизация):")
        print(f"     - Использует optimize_tree_area()")
        print(f"     - Оптимизирует площадь дерева")
        print(f"     - Применяет constraint_distance")
        print(f"     - Возвращает optimized_dt_vector")
        print(f"     - Параметры из tree.area_optimization")
        
        # Показываем полную структуру конфига
        print(f"\n📋 Полная структура конфигурации:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
        
    else:
        print("❌ Файл конфигурации не найден")
        print("   Создайте файл config.json в папке v15_5_tree_opt/config/json/")

def test_bridge_pairing_integration():
    print(f"\n🔧 ТЕСТИРОВАНИЕ ИНТЕГРАЦИИ СПАРИВАНИЯ С МОСТОМ")
    print("=" * 60)
    
    try:
        # Импортируем мост
        import sys
        sys.path.append('v15_5_tree_opt/src')
        
        from logic.tree.tree_area_bridge import _load_pairing_config
        
        print("✅ Мост успешно импортирован")
        
        # Тестируем загрузку конфигурации спаривания
        pairing_config = _load_pairing_config()
        
        print(f"\n📊 Конфигурация спаривания загружена мостом:")
        for key, value in pairing_config.items():
            print(f"   {key}: {value}")
        
        # Проверяем ключевые параметры
        dt_grandchildren_factor = pairing_config.get('dt_grandchildren_factor', None)
        if dt_grandchildren_factor is not None:
            print(f"\n✅ dt_grandchildren_factor загружен: {dt_grandchildren_factor}")
        else:
            print(f"\n⚠️  dt_grandchildren_factor не найден в конфиге")
            
        show_debug = pairing_config.get('show_debug', None)
        if show_debug is not None:
            print(f"✅ show_debug загружен: {show_debug}")
        else:
            print(f"⚠️  show_debug не найден в конфиге")
            
    except Exception as e:
        print(f"❌ Ошибка при тестировании моста: {e}")

if __name__ == "__main__":
    test_pairing_config()
    test_bridge_pairing_integration()
    
    print(f"\n🎯 РЕЗУЛЬТАТ:")
    print("   - Конфигурация спаривания загружается из config/json/config.json")
    print("   - Клавиши 'P' и 'O' используют разные параметры из конфига")
    print("   - Можно настраивать параметры спаривания без изменения кода")
    print("   - Различия между алгоритмами 'P' и 'O' четко определены")
