#!/usr/bin/env python3
"""
Простой тест для проверки ManualSporeManager.
"""

import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_manual_spore_manager_simple():
    """Тестирует основные функции ManualSporeManager."""
    print("🧪 Простое тестирование ManualSporeManager...")
    
    try:
        # Проверяем, что файл существует и читается
        manual_spore_manager_path = os.path.join('src', 'managers', 'manual_spore_manager.py')
        if os.path.exists(manual_spore_manager_path):
            print(f"   ✓ Файл {manual_spore_manager_path} найден")
        else:
            print(f"   ❌ Файл {manual_spore_manager_path} не найден")
            return False
        
        # Читаем файл и проверяем ключевые изменения
        with open(manual_spore_manager_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("   ✓ Файл прочитан успешно")
        
        # Проверяем, что локальные счетчики удалены
        if 'self._link_counter = 0' not in content and 'self._spore_counter = 0' not in content:
            print("   ✓ Локальные счетчики удалены")
        else:
            print("   ❌ Локальные счетчики все еще присутствуют")
            return False
        
        # Проверяем, что добавлена ссылка на id_manager
        if 'self.id_manager = self.spore_manager.id_manager' in content:
            print("   ✓ Ссылка на id_manager добавлена")
        else:
            print("   ❌ Ссылка на id_manager не найдена")
            return False
        
        # Проверяем, что методы получения ID используют id_manager
        if 'return self.id_manager.get_next_link_id()' in content:
            print("   ✓ Метод _get_next_link_id использует id_manager")
        else:
            print("   ❌ Метод _get_next_link_id не использует id_manager")
            return False
        
        if 'return self.id_manager.get_next_spore_id()' in content:
            print("   ✓ Метод _get_next_spore_id использует id_manager")
        else:
            print("   ❌ Метод _get_next_spore_id не использует id_manager")
            return False
        
        # Проверяем, что создание объектов использует правильный порядок
        if 'center_spore.id = self._get_next_spore_id()' in content:
            print("   ✓ Центральная спора получает ID после создания")
        else:
            print("   ❌ Центральная спора не получает ID после создания")
            return False
        
        if 'child_spore.id = self._get_next_spore_id()' in content:
            print("   ✓ Дочерние споры получают ID после создания")
        else:
            print("   ❌ Дочерние споры не получают ID после создания")
            return False
        
        if 'spore_link.id = self._get_next_link_id()' in content:
            print("   ✓ Линки получают ID после создания")
        else:
            print("   ❌ Линки не получают ID после создания")
            return False
        
        # Проверяем, что регистрация использует правильные ID
        if 'manual_center_{center_spore.id}' in content:
            print("   ✓ Регистрация центральной споры использует правильный ID")
        else:
            print("   ❌ Регистрация центральной споры не использует правильный ID")
            return False
        
        if 'manual_link_{config["name"]}_{spore_link.id}' in content:
            print("   ✓ Регистрация линков использует правильный ID")
        else:
            print("   ❌ Регистрация линков не использует правильный ID")
            return False
        
        print("✅ Все проверки ManualSporeManager прошли успешно!")
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_manual_spore_manager_simple()
    if success:
        print("\n🎉 Все тесты прошли успешно!")
    else:
        print("\n💥 Тесты завершились с ошибками!")
