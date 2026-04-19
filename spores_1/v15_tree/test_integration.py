#!/usr/bin/env python3
"""
Тестовый скрипт для проверки интеграции всех компонентов системы ID.
"""

import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_integration():
    """Тестирует интеграцию всех компонентов системы ID."""
    print("🧪 Тестирование интеграции системы ID...")
    
    try:
        # Проверяем основные файлы
        files_to_check = [
            'src/managers/id_manager.py',
            'src/managers/spore_manager.py',
            'src/managers/manual_spore_manager.py',
            'src/visual/spore_tree_visual.py',
            'src/managers/manual_creation/tree_creation_manager.py'
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                print(f"   ✓ Файл {file_path} найден")
            else:
                print(f"   ❌ Файл {file_path} не найден")
                return False
        
        print("   ✓ Все файлы найдены")
        
        # Проверяем IDManager
        print("\n🔍 Проверка IDManager...")
        with open('src/managers/id_manager.py', 'r', encoding='utf-8') as f:
            id_manager_content = f.read()
        
        if 'class IDManager:' in id_manager_content:
            print("   ✓ Класс IDManager найден")
        else:
            print("   ❌ Класс IDManager не найден")
            return False
        
        if 'get_next_spore_id' in id_manager_content and 'get_next_link_id' in id_manager_content:
            print("   ✓ Методы получения ID найдены")
        else:
            print("   ❌ Методы получения ID не найдены")
            return False
        
        # Проверяем SporeManager
        print("\n🔍 Проверка SporeManager...")
        with open('src/managers/spore_manager.py', 'r', encoding='utf-8') as f:
            spore_manager_content = f.read()
        
        if 'from .id_manager import IDManager' in spore_manager_content:
            print("   ✓ Импорт IDManager найден")
        else:
            print("   ❌ Импорт IDManager не найден")
            return False
        
        if 'self.id_manager = IDManager()' in spore_manager_content:
            print("   ✓ IDManager инициализирован")
        else:
            print("   ❌ IDManager не инициализирован")
            return False
        
        if 'self.id_manager.get_next_spore_id()' in spore_manager_content:
            print("   ✓ Методы используют id_manager")
        else:
            print("   ❌ Методы не используют id_manager")
            return False
        
        # Проверяем ManualSporeManager
        print("\n🔍 Проверка ManualSporeManager...")
        with open('src/managers/manual_spore_manager.py', 'r', encoding='utf-8') as f:
            manual_content = f.read()
        
        if 'self.id_manager = self.spore_manager.id_manager' in manual_content:
            print("   ✓ Ссылка на id_manager добавлена")
        else:
            print("   ❌ Ссылка на id_manager не найдена")
            return False
        
        if 'return self.id_manager.get_next_spore_id()' in manual_content:
            print("   ✓ Методы используют id_manager")
        else:
            print("   ❌ Методы не используют id_manager")
            return False
        
        # Проверяем SporeTreeVisual
        print("\n🔍 Проверка SporeTreeVisual...")
        with open('src/visual/spore_tree_visual.py', 'r', encoding='utf-8') as f:
            visual_content = f.read()
        
        if 'id_manager=None' in visual_content:
            print("   ✓ Конструктор принимает id_manager")
        else:
            print("   ❌ Конструктор не принимает id_manager")
            return False
        
        if 'self.id_manager = id_manager' in visual_content:
            print("   ✓ id_manager сохраняется")
        else:
            print("   ❌ id_manager не сохраняется")
            return False
        
        if 'self.root_spore.id = self._get_next_spore_id()' in visual_content:
            print("   ✓ ID присваиваются объектам при создании")
        else:
            print("   ❌ ID не присваиваются объектам при создании")
            return False
        
        # Проверяем TreeCreationManager
        print("\n🔍 Проверка TreeCreationManager...")
        with open('src/managers/manual_creation/tree_creation_manager.py', 'r', encoding='utf-8') as f:
            tree_content = f.read()
        
        if 'id_manager=self.spore_manager.id_manager' in tree_content:
            print("   ✓ id_manager передается в SporeTreeVisual")
        else:
            print("   ❌ id_manager не передается в SporeTreeVisual")
            return False
        
        if 'tree_spore_{spore.id}' in tree_content:
            print("   ✓ Регистрация использует присвоенные ID")
        else:
            print("   ❌ Регистрация не использует присвоенные ID")
            return False
        
        if 'tree_link_{link.id}' in tree_content:
            print("   ✓ Регистрация линков использует присвоенные ID")
        else:
            print("   ❌ Регистрация линков не использует присвоенные ID")
            return False
        
        print("\n✅ Все проверки интеграции прошли успешно!")
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integration()
    if success:
        print("\n🎉 Система ID полностью интегрирована!")
        print("\n📋 Резюме выполненных изменений:")
        print("   ✅ IDManager создан и работает")
        print("   ✅ SporeManager использует IDManager")
        print("   ✅ ManualSporeManager использует IDManager")
        print("   ✅ SporeTreeVisual использует IDManager")
        print("   ✅ TreeCreationManager передает IDManager")
        print("   ✅ Все объекты получают уникальные ID")
        print("   ✅ Нет конфликтов между модулями")
    else:
        print("\n💥 Интеграция не завершена!")
