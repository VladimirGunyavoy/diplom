#!/usr/bin/env python3
"""
Тест логирования системы подписок на изменения dt.
"""

import json
import numpy as np

def test_dt_subscription_logging():
    """Тестирует логирование системы подписок на изменения dt."""
    print("🧪 ТЕСТ ЛОГИРОВАНИЯ ПОДПИСКИ НА ИЗМЕНЕНИЯ DT")
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
    
    # Проверяем подписку
    initial_subscribers = len(dt_manager._subscribers)
    print(f"📧 Начальное количество подписчиков: {initial_subscribers}")
    
    # Тестируем изменение dt с логированием
    print(f"\n🔧 Тестируем изменение dt с логированием...")
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
    
    # Проверяем количество подписчиков после изменения
    final_subscribers = len(dt_manager._subscribers)
    print(f"📧 Конечное количество подписчиков: {final_subscribers}")
    
    # Тестируем увеличение dt
    print(f"\n📝 Ожидаем логи при увеличении dt:")
    dt_manager.increase_dt()
    
    print(f"\n🎉 ТЕСТ ЛОГИРОВАНИЯ ЗАВЕРШЕН!")
    print("📋 Проверьте логи выше на наличие:")
    print("   [IM] subscribed to DTManager.on_change")
    print("   [DT] applying max_length to SporeManager.links")
    print("   [DT] notifying N subscriber(s)")
    print("   [DT] -> subscriber[0] InputManager._on_dt_changed")
    print("   [IM] _on_dt_changed fired")

if __name__ == "__main__":
    test_dt_subscription_logging()
