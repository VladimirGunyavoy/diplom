#!/usr/bin/env python3
"""
Тестовый скрипт для проверки новой функциональности ZoomManager.
"""

import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_zoom_manager():
    """Тестирует новую функциональность ZoomManager."""
    print("🧪 Тестирование новой функциональности ZoomManager...")
    
    try:
        from managers.zoom_manager import ZoomManager
        from managers.color_manager import ColorManager
        from visual.scene_setup import SceneSetup
        from core.spore import Spore
        from logic.pendulum import PendulumSystem
        
        print("   ✓ Модули импортированы успешно")
        
        # Создаем необходимые зависимости
        pendulum = PendulumSystem()
        color_manager = ColorManager()
        
        # Создаем SceneSetup (может потребоваться мок)
        class MockSceneSetup:
            def __init__(self):
                self.player = None
                self.base_speed = 1.0
                self.base_position = [0, 0, 0]
        
        scene_setup = MockSceneSetup()
        
        # Создаем ZoomManager
        zoom_manager = ZoomManager(
            scene_setup=scene_setup,
            color_manager=color_manager
        )
        
        print("   ✓ ZoomManager создан")
        
        # Проверяем, что методы существуют
        if hasattr(zoom_manager, 'print_all_objects'):
            print("   ✓ Метод print_all_objects найден")
        else:
            print("   ❌ Метод print_all_objects не найден")
            return False
        
        if hasattr(zoom_manager, 'show_all_objects'):
            print("   ✓ Метод show_all_objects найден")
        else:
            print("   ❌ Метод show_all_objects не найден")
            return False
        
        # Тестируем вывод пустого менеджера
        print("\n🔍 Тест 1: Пустой ZoomManager")
        zoom_manager.show_all_objects()
        
        # Создаем тестовые споры
        print("\n🔍 Тест 2: Добавление спор")
        spore1 = Spore(
            pendulum=pendulum,
            dt=0.1,
            goal_position=[0, 0],
            scale=0.1,
            position=[0, 0, 0],
            color_manager=color_manager,
            config={}
        )
        spore1.id = 1
        
        spore2 = Spore(
            pendulum=pendulum,
            dt=0.1,
            goal_position=[1, 1],
            scale=0.1,
            position=[1, 0, 1],
            color_manager=color_manager,
            config={}
        )
        spore2.id = 2
        
        # Регистрируем споры (должен автоматически вызвать print_all_objects)
        print("\n📝 Регистрируем первую спору...")
        zoom_manager.register_object(spore1, "test_spore_1")
        
        print("\n📝 Регистрируем вторую спору...")
        zoom_manager.register_object(spore2, "test_spore_2")
        
        # Тестируем ручной вызов
        print("\n🔍 Тест 3: Ручной вызов show_all_objects")
        zoom_manager.show_all_objects()
        
        # Проверяем количество объектов
        expected_count = 2
        actual_count = len(zoom_manager.objects)
        if actual_count == expected_count:
            print(f"   ✓ Количество объектов корректно: {actual_count}")
        else:
            print(f"   ❌ Неправильное количество объектов: ожидалось {expected_count}, получено {actual_count}")
            return False
        
        print("\n✅ Все тесты ZoomManager прошли успешно!")
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
    success = test_zoom_manager()
    if success:
        print("\n🎉 Новая функциональность ZoomManager работает корректно!")
        print("\n📋 Что добавлено:")
        print("   ✅ Автоматический вывод объектов при регистрации")
        print("   ✅ Метод print_all_objects() для детального вывода")
        print("   ✅ Метод show_all_objects() для ручного вызова")
        print("   ✅ Группировка объектов по типам")
        print("   ✅ Показ последних добавленных объектов")
        print("   ✅ Информация об ID объектов")
    else:
        print("\n💥 Тесты завершились с ошибками!")
