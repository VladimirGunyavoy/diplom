#!/usr/bin/env python3
"""
Тестовый скрипт для проверки визуализации стрелок в BufferMergeManager
"""

import sys
import os

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from managers.buffer_merge_manager import BufferMergeManager

def test_visualization():
    """Тестирует визуализацию стрелок."""
    print("🧪 ТЕСТ ВИЗУАЛИЗАЦИИ СТРЕЛОК")
    print("="*50)
    
    # Создаем менеджер
    manager = BufferMergeManager()
    
    # Проверяем есть ли данные в буфере
    if not manager.has_buffer_data():
        print("❌ Буферный граф пуст - нет данных для тестирования")
        print("💡 Запустите сначала основной скрипт для создания буферного графа")
        return
    
    # Создаем схему направлений
    manager.create_debug_diagram()
    
    print("\n🎨 Тест завершен!")
    print("💡 Проверьте логи выше на соответствие JSON данным")

if __name__ == "__main__":
    test_visualization()
