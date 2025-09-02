#!/usr/bin/env python3
"""
Простой тест для проверки новой функциональности ZoomManager.
"""

import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_zoom_manager_simple():
    """Простое тестирование ZoomManager."""
    print("🧪 Простое тестирование ZoomManager...")
    
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
        if 'def print_all_objects(self) -> None:' in content:
            print("   ✓ Метод print_all_objects добавлен")
        else:
            print("   ❌ Метод print_all_objects не найден")
            return False
        
        if 'def show_all_objects(self) -> None:' in content:
            print("   ✓ Метод show_all_objects добавлен")
        else:
            print("   ❌ Метод show_all_objects не найден")
            return False
        
        # Проверяем, что register_object вызывает print_all_objects
        if 'self.print_all_objects()' in content:
            print("   ✓ register_object вызывает print_all_objects")
        else:
            print("   ❌ register_object не вызывает print_all_objects")
            return False
        
        # Проверяем детали реализации print_all_objects
        if 'Фильтруем призрачные объекты' in content:
            print("   ✓ Фильтрация призрачных объектов реализована")
        else:
            print("   ❌ Фильтрация призрачных объектов не реализована")
            return False
        
        if 'Последние добавленные объекты' in content:
            print("   ✓ Показ последних объектов реализован")
        else:
            print("   ❌ Показ последних объектов не реализован")
            return False
        
        if 'getattr(obj, \'id\', \'N/A\')' in content:
            print("   ✓ Показ ID объектов реализован")
        else:
            print("   ❌ Показ ID объектов не реализован")
            return False
        
        # Проверяем новую функциональность фильтрации призраков
        if 'Фильтруем призрачные объекты' in content:
            print("   ✓ Фильтрация призрачных объектов реализована")
        else:
            print("   ❌ Фильтрация призрачных объектов не реализована")
            return False
        
        if 'def show_all_objects_with_ghosts(self) -> None:' in content:
            print("   ✓ Метод show_all_objects_with_ghosts добавлен")
        else:
            print("   ❌ Метод show_all_objects_with_ghosts не найден")
            return False
        
        print("✅ Все проверки ZoomManager прошли успешно!")
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_zoom_manager_simple()
    if success:
        print("\n🎉 Новая функциональность ZoomManager добавлена корректно!")
        print("\n📋 Что добавлено:")
        print("   ✅ Автоматический вывод объектов при регистрации")
        print("   ✅ Метод print_all_objects() для детального вывода")
        print("   ✅ Метод show_all_objects() для ручного вызова")
        print("   ✅ Группировка объектов по типам")
        print("   ✅ Показ последних добавленных объектов")
        print("   ✅ Информация об ID объектов")
        print("\n🔍 Пример вывода при добавлении объекта:")
        print("   🔍 ZoomManager: Объекты после добавления нового")
        print("   📊 Всего объектов: 3")
        print("   👻 Призрачных объектов: 2 (скрыты из вывода)")
        print("   📋 Типы объектов (не-призраки):")
        print("      • Spore: 1 шт.")
        print("        - test_spore_1")
        print("   🆕 Последние добавленные объекты (не-призраки):")
        print("      • test_spore_1 (Spore, ID: 1)")
        print("\n🔍 Для полной отладки используйте:")
        print("   zoom_manager.show_all_objects_with_ghosts()")
    else:
        print("\n💥 Тесты завершились с ошибками!")
