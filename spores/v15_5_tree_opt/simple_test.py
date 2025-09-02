#!/usr/bin/env python3
"""
Простой тест для проверки IDManager.
"""

import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_id_manager():
    """Тестирует IDManager."""
    print("🧪 Тестирование IDManager...")
    
    try:
        from managers.id_manager import IDManager
        
        # Создаем экземпляр
        id_manager = IDManager()
        print(f"   ✓ IDManager создан: {id_manager}")
        
        # Тестируем получение ID для спор
        spore_id_1 = id_manager.get_next_spore_id()
        spore_id_2 = id_manager.get_next_spore_id()
        
        print(f"   📊 ID спор: {spore_id_1}, {spore_id_2}")
        
        # Тестируем получение ID для линков
        link_id_1 = id_manager.get_next_link_id()
        link_id_2 = id_manager.get_next_link_id()
        
        print(f"   🔗 ID линков: {link_id_1}, {link_id_2}")
        
        # Тестируем получение статистики
        stats = id_manager.get_stats()
        print(f"   📈 Статистика: {stats}")
        
        # Тестируем сброс счетчиков
        print("   🔄 Сбрасываем счетчики...")
        id_manager.reset_counters()
        
        stats_after_reset = id_manager.get_stats()
        print(f"   📈 Статистика после сброса: {stats_after_reset}")
        
        print("✅ Тестирование IDManager завершено успешно!")
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_id_manager()
    if success:
        print("\n🎉 Тест прошел успешно!")
    else:
        print("\n💥 Тест завершился с ошибками!")
