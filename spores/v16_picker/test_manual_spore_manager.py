#!/usr/bin/env python3
"""
Тестовый скрипт для проверки обновленного ManualSporeManager.
"""

import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_manual_spore_manager():
    """Тестирует обновленный ManualSporeManager."""
    print("🧪 Тестирование обновленного ManualSporeManager...")
    
    try:
        from managers.manual_spore_manager import ManualSporeManager
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
        
        # Создаем SporeManager
        spore_manager = SporeManager(
            pendulum=pendulum,
            zoom_manager=zoom_manager,
            settings_param=param_manager,
            color_manager=color_manager
        )
        
        print("   ✓ SporeManager создан")
        
        # Создаем ManualSporeManager
        config = {
            'spore': {'scale': 0.1, 'goal_position': [0, 0]},
            'pendulum': {'dt': 0.1}
        }
        
        manual_manager = ManualSporeManager(
            spore_manager=spore_manager,
            zoom_manager=zoom_manager,
            pendulum=pendulum,
            color_manager=color_manager,
            config=config
        )
        
        print("   ✓ ManualSporeManager создан")
        
        # Проверяем, что id_manager создан и имеет правильный тип
        if hasattr(manual_manager, 'id_manager'):
            print(f"   ✓ IDManager найден: {manual_manager.id_manager}")
            
            if isinstance(manual_manager.id_manager, IDManager):
                print("   ✓ IDManager имеет правильный тип")
            else:
                print(f"   ❌ IDManager имеет неправильный тип: {type(manual_manager.id_manager)}")
                return False
        else:
            print("   ❌ IDManager не найден в ManualSporeManager")
            return False
        
        # Проверяем, что методы получения ID используют id_manager
        if hasattr(manual_manager, '_get_next_spore_id') and hasattr(manual_manager, '_get_next_link_id'):
            print("   ✓ Методы получения ID найдены")
            
            # Тестируем получение ID
            spore_id_1 = manual_manager._get_next_spore_id()
            spore_id_2 = manual_manager._get_next_spore_id()
            link_id_1 = manual_manager._get_next_link_id()
            
            print(f"   📊 ID спор: {spore_id_1}, {spore_id_2}")
            print(f"   🔗 ID линка: {link_id_1}")
            
            # Проверяем, что ID последовательные
            if spore_id_1 == 1 and spore_id_2 == 2 and link_id_1 == 1:
                print("   ✓ ID последовательные и корректные")
            else:
                print(f"   ❌ ID некорректные: ожидалось 1, 2, 1, получено {spore_id_1}, {spore_id_2}, {link_id_1}")
                return False
        else:
            print("   ❌ Методы получения ID не найдены")
            return False
        
        # Проверяем, что используется тот же IDManager что и в SporeManager
        if manual_manager.id_manager is spore_manager.id_manager:
            print("   ✓ Используется тот же IDManager что и в SporeManager")
        else:
            print("   ❌ Используется другой IDManager")
            return False
        
        print("✅ Обновленный ManualSporeManager работает корректно!")
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
    success = test_manual_spore_manager()
    if success:
        print("\n🎉 Все тесты прошли успешно!")
    else:
        print("\n💥 Тесты завершились с ошибками!")
