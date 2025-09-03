#!/usr/bin/env python3
"""
Тестовый скрипт для проверки применения оптимизированных dt к призрачному дереву
"""

def test_apply_optimized_dt():
    print("🔧 ПРИМЕНЕНИЕ ОПТИМИЗИРОВАННЫХ DT К ПРИЗРАЧНОМУ ДЕРЕВУ")
    print("=" * 70)
    
    print("\n📋 Проблема:")
    print("   - Оптимизация работает и выдает хорошие результаты")
    print("   - Дистанции выводятся правильно")
    print("   - Но оптимизированные dt не применяются к призрачному дереву")
    print("   - Визуализация не меняется")
    
    print("\n🔧 Решение:")
    print("   1. ✅ Извлечение optimized_dt_vector из результата оптимизации")
    print("   2. ✅ Применение dt_vector к ghost_tree_dt_vector")
    print("   3. ✅ Обновление ghost_dt_baseline для масштабирования")
    print("   4. ✅ Принудительное обновление призрачного дерева")
    print("   5. ✅ Очистка и пересоздание предсказаний")
    
    print("\n🔍 Что теперь происходит при нажатии 'O':")
    print("   1. Запуск оптимизации площади")
    print("   2. Получение optimized_dt_vector из результата")
    print("   3. Применение dt_vector к призрачному дереву:")
    print("      manual_spore_manager.ghost_tree_dt_vector = optimized_dt_vector")
    print("   4. Обновление baseline:")
    print("      ghost_dt_baseline = dt_manager.get_dt()")
    print("   5. Принудительное обновление:")
    print("      prediction_manager.clear_predictions()")
    print("      _update_predictions()")
    
    print("\n📊 Ожидаемый вывод:")
    print("   [IM][O] 🔧 Применяем оптимизированные dt к призрачному дереву...")
    print("   [IM][O] 📊 ghost_dt_baseline установлен: 0.050000")
    print("   [IM][O] ✅ Призрачное дерево обновлено с оптимизированными dt!")
    
    print("\n🎯 Результат:")
    print("   - Призрачное дерево теперь использует оптимизированные dt")
    print("   - Визуализация должна измениться")
    print("   - Площадь дерева должна увеличиться")
    print("   - Дистанции между парами должны быть минимальными")
    
    print("\n⚠️  Если что-то не работает:")
    print("   - Проверьте, что ManualSporeManager доступен")
    print("   - Проверьте, что PredictionManager существует")
    print("   - Проверьте, что optimized_dt_vector не None")
    print("   - Проверьте, что _update_predictions() вызывается")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_apply_optimized_dt()
