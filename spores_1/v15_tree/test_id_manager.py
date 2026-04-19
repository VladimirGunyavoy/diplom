#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы IDManager.
"""

import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from managers.id_manager import IDManager

def test_id_manager():
    """Тестирует основные функции IDManager."""
    print("🧪 Тестирование IDManager...")
    
    # Создаем экземпляр
    id_manager = IDManager()
    print(f"   ✓ IDManager создан: {id_manager}")
    
    # Тестируем получение ID для спор
    spore_id_1 = id_manager.get_next_spore_id()
    spore_id_2 = id_manager.get_next_spore_id()
    spore_id_3 = id_manager.get_next_spore_id()
    
    print(f"   📊 ID спор: {spore_id_1}, {spore_id_2}, {spore_id_3}")
    
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
    
    # Тестируем получение новых ID после сброса
    new_spore_id = id_manager.get_next_spore_id()
    new_link_id = id_manager.get_next_link_id()
    
    print(f"   🆕 Новые ID после сброса: спора {new_spore_id}, линк {new_link_id}")
    
    # Тестируем специальные счетчики
    angel_id = id_manager.get_next_angel_id()
    pillar_id = id_manager.get_next_pillar_id()
    ghost_id = id_manager.get_next_ghost_id()
    
    print(f"   👼 ID специальных объектов: ангел {angel_id}, столб {pillar_id}, призрак {ghost_id}")
    
    # Финальная статистика
    final_stats = id_manager.get_stats()
    print(f"   🎯 Финальная статистика: {final_stats}")
    
    print("✅ Тестирование IDManager завершено успешно!")

if __name__ == "__main__":
    test_id_manager()
