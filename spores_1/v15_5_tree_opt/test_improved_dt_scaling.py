#!/usr/bin/env python3
"""
Тест улучшенного масштабирования dt-вектора с диагностикой.
"""

import json
import numpy as np

def test_improved_dt_scaling():
    """Тестирует улучшенное масштабирование dt-вектора с диагностикой."""
    print("🧪 ТЕСТ УЛУЧШЕННОГО МАСШТАБИРОВАНИЯ DT-ВЕКТОРА")
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
    
    # Создаем мок ManualSporeManager для тестирования
    class MockManualSporeManager:
        def __init__(self):
            self.ghost_tree_dt_vector = None
            self.ghost_dt_baseline = None
            self.prediction_manager = None
    
    mock_manual_spore_manager = MockManualSporeManager()
    input_manager.manual_spore_manager = mock_manual_spore_manager
    
    # Тестируем улучшенное масштабирование dt-вектора
    print(f"\n🔧 ТЕСТИРУЕМ УЛУЧШЕННОЕ МАСШТАБИРОВАНИЕ:")
    
    # Имитируем создание спаренного дерева с dt-вектором
    initial_dt = dt_manager.get_dt()
    print(f"   Начальный dt: {initial_dt}")
    
    # Создаем тестовый dt-вектор (4 детей + 8 внуков)
    test_dt_vector = np.array([
        0.1, -0.15, 0.2, -0.25,  # дети
        0.05, -0.08, 0.12, -0.18, 0.22, -0.28, 0.32, -0.35  # внуки
    ])
    
    # Имитируем сохранение baseline
    mock_manual_spore_manager.ghost_tree_dt_vector = test_dt_vector.copy()
    mock_manual_spore_manager.ghost_dt_baseline = initial_dt
    
    print(f"   Тестовый dt-вектор: {test_dt_vector}")
    print(f"   Baseline dt: {mock_manual_spore_manager.ghost_dt_baseline}")
    
    # Тестируем увеличение dt с улучшенной диагностикой
    print(f"\n📝 Тестируем увеличение dt с диагностикой:")
    dt_manager.increase_dt()
    new_dt = dt_manager.get_dt()
    print(f"   Новый dt: {new_dt}")
    
    # Проверяем масштабирование
    if mock_manual_spore_manager.ghost_tree_dt_vector is not None:
        scaled_vector = mock_manual_spore_manager.ghost_tree_dt_vector
        print(f"   Масштабированный вектор: {scaled_vector}")
        
        # Проверяем что знаки сохранились
        original_signs = np.sign(test_dt_vector)
        scaled_signs = np.sign(scaled_vector)
        signs_preserved = np.array_equal(original_signs, scaled_signs)
        print(f"   Знаки сохранены: {signs_preserved}")
        
        # Проверяем масштабирование
        expected_scale = new_dt / initial_dt
        actual_scale = np.abs(scaled_vector[0]) / np.abs(test_dt_vector[0]) if test_dt_vector[0] != 0 else 1
        scale_correct = abs(expected_scale - actual_scale) < 0.01
        print(f"   Масштабирование корректно: {scale_correct} (ожидалось: {expected_scale:.3f}, получилось: {actual_scale:.3f})")
        
        if signs_preserved and scale_correct:
            print("✅ Улучшенное масштабирование работает корректно!")
        else:
            print("❌ Ошибка в масштабировании")
    else:
        print("❌ Вектор не был масштабирован")
    
    # Тестируем уменьшение dt
    print(f"\n📝 Тестируем уменьшение dt:")
    dt_manager.decrease_dt()
    new_dt = dt_manager.get_dt()
    print(f"   Новый dt: {new_dt}")
    
    # Проверяем масштабирование
    if mock_manual_spore_manager.ghost_tree_dt_vector is not None:
        scaled_vector = mock_manual_spore_manager.ghost_tree_dt_vector
        print(f"   Масштабированный вектор: {scaled_vector}")
        
        # Проверяем что знаки сохранились
        original_signs = np.sign(test_dt_vector)
        scaled_signs = np.sign(scaled_vector)
        signs_preserved = np.array_equal(original_signs, scaled_signs)
        print(f"   Знаки сохранены: {signs_preserved}")
        
        # Проверяем масштабирование
        expected_scale = new_dt / initial_dt
        actual_scale = np.abs(scaled_vector[0]) / np.abs(test_dt_vector[0]) if test_dt_vector[0] != 0 else 1
        scale_correct = abs(expected_scale - actual_scale) < 0.01
        print(f"   Масштабирование корректно: {scale_correct} (ожидалось: {expected_scale:.3f}, получилось: {actual_scale:.3f})")
        
        if signs_preserved and scale_correct:
            print("✅ Улучшенное масштабирование работает корректно!")
        else:
            print("❌ Ошибка в масштабировании")
    else:
        print("❌ Вектор не был масштабирован")
    
    print(f"\n🎉 ТЕСТ УЛУЧШЕННОГО МАСШТАБИРОВАНИЯ ЗАВЕРШЕН!")
    print("📋 Проверьте логи выше на наличие:")
    print("   [InputManager._on_dt_changed] scale ghost dt-vector: baseline ...")
    print("   [InputManager._on_dt_changed] k=...")
    print("   [InputManager._on_dt_changed] dt_vec before: [...]")
    print("   [InputManager._on_dt_changed] dt_vec after : [...]")

if __name__ == "__main__":
    test_improved_dt_scaling()
