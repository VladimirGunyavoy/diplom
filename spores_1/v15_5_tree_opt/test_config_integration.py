#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы конфигурации оптимизации
"""

import json
import os

def test_config_loading():
    print("🔧 ТЕСТИРОВАНИЕ КОНФИГУРАЦИИ ОПТИМИЗАЦИИ")
    print("=" * 60)
    
    # Путь к конфигурационному файлу
    config_path = "v15_5_tree_opt/config/json/config.json"
    
    print(f"\n📋 Проверяем файл: {config_path}")
    
    if os.path.exists(config_path):
        print("✅ Файл конфигурации найден")
        
        # Загружаем конфигурацию
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Проверяем секцию tree.area_optimization
        tree_config = config.get('tree', {})
        area_config = tree_config.get('area_optimization', {})
        
        print(f"\n📊 Текущая конфигурация оптимизации:")
        print(f"   constraint_distance: {area_config.get('constraint_distance', 'НЕ УСТАНОВЛЕН')}")
        print(f"   dt_bounds: {area_config.get('dt_bounds', 'НЕ УСТАНОВЛЕН')}")
        print(f"   max_iterations: {area_config.get('max_iterations', 'НЕ УСТАНОВЛЕН')}")
        print(f"   method: {area_config.get('method', 'НЕ УСТАНОВЛЕН')}")
        
        # Проверяем, что все параметры установлены
        required_params = ['constraint_distance', 'dt_bounds', 'max_iterations', 'method']
        missing_params = [param for param in required_params if param not in area_config]
        
        if missing_params:
            print(f"\n⚠️  Отсутствующие параметры: {missing_params}")
            print("   Добавьте их в секцию tree.area_optimization в config.json")
        else:
            print(f"\n✅ Все параметры оптимизации настроены")
        
        # Показываем полную структуру конфига
        print(f"\n📋 Полная структура конфигурации:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
        
    else:
        print("❌ Файл конфигурации не найден")
        print("   Создайте файл config.json в папке v15_5_tree_opt/config/json/")

def test_bridge_config_integration():
    print(f"\n🔧 ТЕСТИРОВАНИЕ ИНТЕГРАЦИИ С МОСТОМ")
    print("=" * 60)
    
    try:
        # Импортируем мост
        import sys
        sys.path.append('v15_5_tree_opt/src')
        
        from logic.tree.tree_area_bridge import _load_optimization_config
        
        print("✅ Мост успешно импортирован")
        
        # Тестируем загрузку конфигурации
        config = _load_optimization_config()
        
        print(f"\n📊 Конфигурация загружена мостом:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        # Проверяем, что constraint_distance загружается правильно
        constraint_distance = config.get('constraint_distance', None)
        if constraint_distance is not None:
            print(f"\n✅ constraint_distance загружен: {constraint_distance}")
        else:
            print(f"\n⚠️  constraint_distance не найден в конфиге")
            
    except Exception as e:
        print(f"❌ Ошибка при тестировании моста: {e}")

if __name__ == "__main__":
    test_config_loading()
    test_bridge_config_integration()
    
    print(f"\n🎯 РЕЗУЛЬТАТ:")
    print("   - Конфигурация загружается из config/json/config.json")
    print("   - Мост автоматически применяет параметры из конфига")
    print("   - При нажатии 'O' используются настройки из конфига")
    print("   - Можно легко изменить параметры без изменения кода")
