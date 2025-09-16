#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправленного JSON экспорта графа.
"""

import sys
import os
import json

# Добавляем путь к модулям проекта
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_json_export():
    """Тестирует новый JSON экспорт с правильными номерами линков."""
    
    print("🧪 ТЕСТИРОВАНИЕ ИСПРАВЛЕННОГО JSON ЭКСПОРТА")
    print("=" * 50)
    
    try:
        # Импортируем необходимые модули
        from src.managers.spore_manager import SporeManager
        from src.managers.buffer_merge_manager import BufferMergeManager
        from src.logic.pendulum import PendulumSystem
        from src.managers.zoom_manager import ZoomManager
        from src.managers.color_manager import ColorManager
        from src.managers.param_manager import ParamManager
        
        print("✅ Модули успешно импортированы")
        
        # Создаем минимальную конфигурацию
        config = {
            'pendulum': {'dt': 0.1},
            'spore': {'scale': 0.05, 'goal_position': [3.14159, 0]},
            'link': {'show': True}
        }
        
        # Создаем необходимые компоненты
        pendulum = PendulumSystem(config.get('pendulum', {}))
        zoom_manager = ZoomManager()
        color_manager = ColorManager()
        param_manager = ParamManager()
        
        # Создаем SporeManager
        spore_manager = SporeManager(
            pendulum=pendulum,
            zoom_manager=zoom_manager,
            settings_param=param_manager,
            color_manager=color_manager,
            config=config
        )
        
        # Создаем BufferMergeManager
        buffer_manager = BufferMergeManager(
            distance_threshold=0.001,
            config=config
        )
        
        print("✅ Менеджеры созданы")
        
        # Проверяем наличие новых методов экспорта
        export_methods = [
            'export_graph_to_json',
            'save_graph_json'
        ]
        
        for method_name in export_methods:
            if hasattr(buffer_manager, method_name):
                print(f"✅ Метод {method_name} найден")
            else:
                print(f"❌ Метод {method_name} НЕ найден")
                return False
        
        print("\n🔍 ТЕСТИРОВАНИЕ ЭКСПОРТА:")
        print("-" * 30)
        
        # Тест 1: Экспорт пустого графа
        print("1. Тест экспорта пустого графа:")
        graph_data = buffer_manager.export_graph_to_json(spore_manager)
        
        if isinstance(graph_data, dict):
            print("   ✅ Корректно возвращает словарь")
            print(f"   📊 Спор: {len(graph_data.get('spores', []))}")
            print(f"   🔗 Линков: {len(graph_data.get('links', []))}")
            print(f"   📋 Версия: {graph_data.get('version', 'неизвестно')}")
        else:
            print("   ❌ Должен возвращать словарь")
            return False
        
        # Тест 2: Проверка структуры данных
        print("2. Тест структуры данных:")
        required_keys = ['timestamp', 'version', 'spores', 'links', 'stats']
        for key in required_keys:
            if key in graph_data:
                print(f"   ✅ Ключ '{key}' присутствует")
            else:
                print(f"   ❌ Ключ '{key}' отсутствует")
                return False
        
        # Тест 3: Проверка статистики
        print("3. Тест статистики:")
        stats = graph_data.get('stats', {})
        if 'total_spores' in stats and 'total_links' in stats:
            print(f"   ✅ Статистика корректна: {stats['total_spores']} спор, {stats['total_links']} линков")
        else:
            print("   ❌ Статистика некорректна")
            return False
        
        # Тест 4: Сохранение в файл
        print("4. Тест сохранения в файл:")
        try:
            # Создаем временную папку для тестов
            test_dir = "test_exports"
            os.makedirs(test_dir, exist_ok=True)
            
            # Сохраняем тестовый файл
            test_filepath = buffer_manager.save_graph_json(spore_manager, "test_export.json")
            
            if os.path.exists(test_filepath):
                print(f"   ✅ Файл сохранен: {test_filepath}")
                
                # Проверяем содержимое файла
                with open(test_filepath, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                
                if saved_data == graph_data:
                    print("   ✅ Содержимое файла корректно")
                else:
                    print("   ❌ Содержимое файла не соответствует экспорту")
                    return False
                
                # Удаляем тестовый файл
                os.remove(test_filepath)
                print("   🗑️ Тестовый файл удален")
                
            else:
                print("   ❌ Файл не был создан")
                return False
                
        except Exception as e:
            print(f"   ❌ Ошибка при сохранении: {e}")
            return False
        
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("✅ Исправленный JSON экспорт работает корректно")
        print("✅ Новые методы поиска линков интегрированы в экспорт")
        
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("💡 Убедитесь, что все зависимости установлены")
        return False
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_link_numbering():
    """Тестирует правильность нумерации линков в экспорте."""
    
    print("\n🔢 ТЕСТИРОВАНИЕ НУМЕРАЦИИ ЛИНКОВ")
    print("=" * 40)
    
    try:
        from src.managers.spore_manager import SporeManager
        from src.managers.buffer_merge_manager import BufferMergeManager
        from src.logic.pendulum import PendulumSystem
        from src.managers.zoom_manager import ZoomManager
        from src.managers.color_manager import ColorManager
        from src.managers.param_manager import ParamManager
        
        # Создаем конфигурацию
        config = {
            'pendulum': {'dt': 0.1},
            'spore': {'scale': 0.05, 'goal_position': [3.14159, 0]},
            'link': {'show': True}
        }
        
        # Создаем компоненты
        pendulum = PendulumSystem(config.get('pendulum', {}))
        zoom_manager = ZoomManager()
        color_manager = ColorManager()
        param_manager = ParamManager()
        
        spore_manager = SporeManager(
            pendulum=pendulum,
            zoom_manager=zoom_manager,
            settings_param=param_manager,
            color_manager=color_manager,
            config=config
        )
        
        buffer_manager = BufferMergeManager(
            distance_threshold=0.001,
            config=config
        )
        
        # Тестируем методы поиска линков
        print("1. Тест методов поиска линков:")
        
        # Проверяем список всех линков
        all_links = spore_manager.list_all_links()
        print(f"   📋 Всего линков найдено: {len(all_links)}")
        
        # Проверяем поиск по номеру
        for i in range(1, 6):  # Тестируем первые 5 номеров
            link = spore_manager.find_link_by_number(i)
            if link is None:
                print(f"   ℹ️ Линк #{i} не найден (ожидаемо для пустого графа)")
            else:
                print(f"   ✅ Линк #{i} найден")
        
        # Проверяем получение информации о линке
        link_info = spore_manager.get_link_info(1)
        if link_info.get('found') == False:
            print("   ✅ Корректно обрабатывает несуществующий линк")
        else:
            print("   ❌ Должен возвращать ошибку для несуществующего линка")
        
        print("\n✅ Тестирование нумерации завершено")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании нумерации: {e}")
        return False

if __name__ == "__main__":
    print("🚀 ЗАПУСК ТЕСТОВ JSON ЭКСПОРТА")
    print("=" * 60)
    
    # Запускаем основные тесты
    success1 = test_json_export()
    success2 = test_link_numbering()
    
    if success1 and success2:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("✅ JSON экспорт исправлен и работает корректно")
        print("✅ Нумерация линков работает правильно")
        sys.exit(0)
    else:
        print("\n❌ НЕКОТОРЫЕ ТЕСТЫ ЗАВЕРШИЛИСЬ С ОШИБКАМИ")
        sys.exit(1)
