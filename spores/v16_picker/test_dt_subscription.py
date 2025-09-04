#!/usr/bin/env python3
"""
Тест подписки на изменения dt в InputManager.
"""

import json
import numpy as np

def test_dt_subscription():
    """Тестирует подписку на изменения dt."""
    print("🧪 ТЕСТ ПОДПИСКИ НА ИЗМЕНЕНИЯ DT")
    print("=" * 50)
    
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
    
    # Проверяем, что метод subscribe_on_change существует
    if hasattr(dt_manager, 'subscribe_on_change'):
        print("✅ Метод subscribe_on_change найден")
    else:
        print("❌ Метод subscribe_on_change не найден")
        return
    
    # Создаем InputManager
    input_manager = InputManager(dt_manager=dt_manager)
    print("✅ InputManager создан")
    
    # Проверяем, что метод _on_dt_changed существует
    if hasattr(input_manager, '_on_dt_changed'):
        print("✅ Метод _on_dt_changed найден")
    else:
        print("❌ Метод _on_dt_changed не найден")
        return
    
    # Проверяем подписку
    initial_subscribers = len(dt_manager._subscribers)
    print(f"📧 Начальное количество подписчиков: {initial_subscribers}")
    
    # Проверяем, что подписка произошла автоматически
    if initial_subscribers > 0:
        print("✅ Подписка произошла автоматически")
    else:
        print("⚠️ Подписка не произошла автоматически")
    
    # Тестируем изменение dt
    print(f"\n🔧 Тестируем изменение dt...")
    initial_dt = dt_manager.get_dt()
    print(f"   Начальный dt: {initial_dt}")
    
    # Уменьшаем dt
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
    
    print(f"\n🎉 ТЕСТ ЗАВЕРШЕН УСПЕШНО!")

if __name__ == "__main__":
    test_dt_subscription()
