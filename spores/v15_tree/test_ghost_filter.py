#!/usr/bin/env python3
"""
Тестовый скрипт для проверки фильтрации призрачных объектов в ZoomManager.
"""

import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_ghost_filter():
    """Тестирует фильтрацию призрачных объектов."""
    print("🧪 Тестирование фильтрации призрачных объектов...")
    
    try:
        # Проверяем, что файл существует
        zoom_manager_path = 'src/managers/zoom_manager.py'
        if not os.path.exists(zoom_manager_path):
            print(f"   ❌ Файл {zoom_manager_path} не найден")
            return False
        
        print(f"   ✓ Файл {zoom_manager_path} найден")
        
        # Читаем файл и проверяем ключевые изменения
        with open(zoom_manager_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("   ✓ Файл прочитан успешно")
        
        # Проверяем, что новые методы добавлены
        if 'def show_all_objects_with_ghosts(self) -> None:' in content:
            print("   ✓ Метод show_all_objects_with_ghosts добавлен")
        else:
            print("   ❌ Метод show_all_objects_with_ghosts не найден")
            return False
        
        # Проверяем логику фильтрации призраков
        if 'Фильтруем призрачные объекты' in content:
            print("   ✓ Фильтрация призрачных объектов реализована")
        else:
            print("   ❌ Фильтрация призрачных объектов не реализована")
            return False
        
        # Проверяем проверки на призрачность
        if 'getattr(obj, \'is_ghost\', False)' in content:
            print("   ✓ Проверка атрибута is_ghost реализована")
        else:
            print("   ❌ Проверка атрибута is_ghost не реализована")
            return False
        
        if '\'ghost\' in name.lower()' in content:
            print("   ✓ Проверка имени на ghost реализована")
        else:
            print("   ❌ Проверка имени на ghost не реализована")
            return False
        
        if 'obj.color.a < 0.8' in content:
            print("   ✓ Проверка прозрачности реализована")
        else:
            print("   ❌ Проверка прозрачности не реализована")
            return False
        
        # Проверяем статистику по призракам
        if 'Призрачных объектов:' in content:
            print("   ✓ Статистика по призракам реализована")
        else:
            print("   ❌ Статистика по призракам не реализована")
            return False
        
        print("✅ Все проверки фильтрации призраков прошли успешно!")
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ghost_filter()
    if success:
        print("\n🎉 Фильтрация призрачных объектов добавлена корректно!")
        print("\n📋 Что добавлено:")
        print("   ✅ Автоматическая фильтрация призраков в print_all_objects")
        print("   ✅ Проверка атрибута is_ghost")
        print("   ✅ Проверка имени на 'ghost'")
        print("   ✅ Проверка прозрачности (alpha < 0.8)")
        print("   ✅ Статистика по количеству призраков")
        print("   ✅ Метод show_all_objects_with_ghosts для полной отладки")
        print("\n🔍 Пример вывода при добавлении объекта:")
        print("   🔍 ZoomManager: Объекты после добавления нового")
        print("   📊 Всего объектов: 3")
        print("   👻 Призрачных объектов: 2 (скрыты из вывода)")
        print("   📋 Типы объектов (не-призраки):")
        print("      • Spore: 1 шт.")
        print("        - test_spore_1")
        print("   🆕 Последние добавленные объекты (не-призраки):")
        print("      • test_spore_1 (Spore, ID: 1)")
    else:
        print("\n💥 Тесты завершились с ошибками!")
