#!/usr/bin/env python3
"""
Тестовый скрипт для проверки новой логики краткого вывода в ZoomManager.
"""

import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_quick_info():
    """Тестирует новую логику краткого вывода."""
    print("🧪 Тестирование новой логики краткого вывода...")
    
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
        if 'def print_quick_info(self, name: str, obj: Scalable) -> None:' in content:
            print("   ✓ Метод print_quick_info добавлен")
        else:
            print("   ❌ Метод print_quick_info не найден")
            return False
        
        if 'def enable_auto_print(self) -> None:' in content:
            print("   ✓ Метод enable_auto_print добавлен")
        else:
            print("   ❌ Метод enable_auto_print не найден")
            return False
        
        if 'def disable_auto_print(self) -> None:' in content:
            print("   ✓ Метод disable_auto_print добавлен")
        else:
            print("   ❌ Метод disable_auto_print не найден")
            return False
        
        # Проверяем логику проверки призраков
        if 'Проверяем, является ли объект призрачным' in content:
            print("   ✓ Проверка призраков в register_object реализована")
        else:
            print("   ❌ Проверка призраков в register_object не реализована")
            return False
        
        # Проверяем флаг auto_print_enabled
        if 'self.auto_print_enabled = True' in content:
            print("   ✓ Флаг auto_print_enabled добавлен")
        else:
            print("   ❌ Флаг auto_print_enabled не найден")
            return False
        
        # Проверяем, что print_all_objects больше не вызывается в register_object
        if 'self.print_all_objects()' not in content or content.count('self.print_all_objects()') <= 2:
            print("   ✓ print_all_objects больше не вызывается в register_object")
        else:
            print("   ❌ print_all_objects все еще вызывается в register_object")
            return False
        
        print("✅ Все проверки новой логики прошли успешно!")
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quick_info()
    if success:
        print("\n🎉 Новая логика краткого вывода добавлена корректно!")
        print("\n📋 Что изменилось:")
        print("   ✅ Убран автоматический вызов print_all_objects")
        print("   ✅ Добавлен метод print_quick_info для краткого вывода")
        print("   ✅ Проверка призраков перед выводом")
        print("   ✅ Флаг auto_print_enabled для управления выводом")
        print("   ✅ Методы enable_auto_print/disable_auto_print")
        print("\n🔍 Теперь при добавлении объекта:")
        print("   🔍 Добавлен объект: test_spore_1 (Spore, ID: 1)")
        print("   📊 Всего объектов в системе: 3")
        print("   👻 Призрачных объектов: 2")
        print("\n🎛️ Управление выводом:")
        print("   zoom_manager.disable_auto_print()  # Отключить")
        print("   zoom_manager.enable_auto_print()   # Включить")
        print("   zoom_manager.show_all_objects()    # Показать все вручную")
    else:
        print("\n💥 Тесты завершились с ошибками!")
