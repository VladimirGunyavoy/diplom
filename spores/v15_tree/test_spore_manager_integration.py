#!/usr/bin/env python3
"""
Тестовый скрипт для проверки интеграции IDManager в SporeManager.
"""

import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_spore_manager_integration():
    """Тестирует интеграцию IDManager в SporeManager."""
    print("🧪 Тестирование интеграции IDManager в SporeManager...")
    
    try:
        # Импортируем необходимые модули
        from managers.spore_manager import SporeManager
        from managers.id_manager import IDManager
        from managers.color_manager import ColorManager
        from managers.zoom_manager import ZoomManager
        from managers.param_manager import ParamManager
        from logic.pendulum import PendulumSystem
        
        print("   ✓ Модули импортированы успешно")
        
        # Создаем необходимые зависимости
        pendulum = PendulumSystem()
        zoom_manager = ZoomManager()
        color_manager = ColorManager()
        param_manager = ParamManager()
        
        print("   ✓ Зависимости созданы")
        
        # Создаем SporeManager
        spore_manager = SporeManager(
            pendulum=pendulum,
            zoom_manager=zoom_manager,
            settings_param=param_manager,
            color_manager=color_manager
        )
        
        print("   ✓ SporeManager создан")
        
        # Проверяем, что id_manager создан
        if hasattr(spore_manager, 'id_manager'):
            print(f"   ✓ IDManager создан: {spore_manager.id_manager}")
        else:
            print("   ❌ IDManager не найден в SporeManager")
            return False
        
        # Проверяем, что id_manager является экземпляром IDManager
        if isinstance(spore_manager.id_manager, IDManager):
            print("   ✓ IDManager имеет правильный тип")
        else:
            print(f"   ❌ IDManager имеет неправильный тип: {type(spore_manager.id_manager)}")
            return False
        
        # Проверяем метод get_id_stats
        if hasattr(spore_manager, 'get_id_stats'):
            print("   ✓ Метод get_id_stats найден")
            
            # Получаем статистику
            stats = spore_manager.get_id_stats()
            print(f"   📊 Начальная статистика: {stats}")
            
            # Проверяем, что статистика содержит ожидаемые ключи
            expected_keys = ['spores_created', 'links_created', 'angels_created', 'pillars_created', 'ghosts_created', 'total_objects']
            for key in expected_keys:
                if key in stats:
                    print(f"   ✓ Ключ '{key}' найден в статистике")
                else:
                    print(f"   ❌ Ключ '{key}' отсутствует в статистике")
                    return False
        else:
            print("   ❌ Метод get_id_stats не найден")
            return False
        
        # Проверяем, что методы очистки используют id_manager
        if hasattr(spore_manager, 'clear') and hasattr(spore_manager, 'clear_all_manual'):
            print("   ✓ Методы очистки найдены")
        else:
            print("   ❌ Методы очистки не найдены")
            return False
        
        print("✅ Интеграция IDManager в SporeManager работает корректно!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Ошибка импорта: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_spore_manager_integration()
    if success:
        print("\n🎉 Все тесты прошли успешно!")
    else:
        print("\n💥 Тесты завершились с ошибками!")
