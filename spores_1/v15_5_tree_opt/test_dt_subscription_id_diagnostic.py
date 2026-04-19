#!/usr/bin/env python3
"""
Тест ID-диагностики системы подписок на изменения dt.
"""

import json
import numpy as np

def test_dt_subscription_id_diagnostic():
    """Тестирует ID-диагностику системы подписок на изменения dt."""
    print("🧪 ТЕСТ ID-ДИАГНОСТИКИ ПОДПИСКИ НА ИЗМЕНЕНИЯ DT")
    print("=" * 60)
    
    # Проверяем импорты
    try:
        from src.managers.dt_manager import DTManager
        from src.managers.input_manager import InputManager
        from src.logic.pendulum import PendulumSystem
        print("✅ Все импорты успешны")
    except Exception as e:
        print(f"❌ Ошибка импорта: {e}")
        return
    
    # Загружаем конфигурацию
    try:
        with open('config/json/config.json', 'r') as f:
            config = json.load(f)
        print("✅ Конфигурация загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
        return
    
    # Создаем маятник
    pendulum = PendulumSystem(config)
    print("✅ PendulumSystem создан")
    
    # Создаем DTManager
    dt_manager = DTManager(config, pendulum)
    print("✅ DTManager создан")
    
    # Создаем InputManager
    input_manager = InputManager(dt_manager=dt_manager)
    print("✅ InputManager создан")
    
    # Проверяем ID экземпляров
    print(f"\n🔍 ID-ДИАГНОСТИКА:")
    print(f"   DTManager id: {id(dt_manager)}")
    print(f"   InputManager.dt_manager id: {id(input_manager.dt_manager) if input_manager.dt_manager else None}")
    print(f"   ID совпадают: {id(dt_manager) == id(input_manager.dt_manager) if input_manager.dt_manager else False}")
    
    # Проверяем подписчиков
    print(f"\n📧 ПОДПИСЧИКИ:")
    dt_manager.debug_subscribers()
    
    # Тестируем изменение dt с ID-логированием
    print(f"\n🔧 Тестируем изменение dt с ID-логированием...")
    initial_dt = dt_manager.get_dt()
    print(f"   Начальный dt: {initial_dt}")
    
    # Уменьшаем dt
    print(f"\n📝 Ожидаем логи при уменьшении dt:")
    dt_manager.decrease_dt()
    new_dt = dt_manager.get_dt()
    print(f"   Новый dt: {new_dt}")
    
    if new_dt < initial_dt:
        print("✅ dt успешно уменьшен")
    else:
        print("❌ dt не изменился")
    
    # Тестируем увеличение dt
    print(f"\n📝 Ожидаем логи при увеличении dt:")
    dt_manager.increase_dt()
    
    print(f"\n🎉 ТЕСТ ID-ДИАГНОСТИКИ ЗАВЕРШЕН!")
    print("📋 Проверьте логи выше на наличие:")
    print("   [DT] instance id=...")
    print("   [IM] constructed, dt_manager id=...")
    print("   [DT] notifying N subscriber(s) [DT id=...]")
    print("   [DT] -> subscriber[0] InputManager._on_dt_changed")
    print("   [IM] _on_dt_changed fired")

if __name__ == "__main__":
    test_dt_subscription_id_diagnostic()
