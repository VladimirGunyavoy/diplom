#!/usr/bin/env python3
"""
Диагностический скрипт для проверки состояния буферного графа.
Показывает состояние BufferMergeManager после нажатия M.
"""

import sys
import os

# Добавляем корневую папку проекта в путь
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def check_buffer_manager_debug():
    """Проверяет логи BufferMergeManager."""
    
    print("🔍 ДИАГНОСТИКА БУФЕРНОГО ГРАФА")
    print("=" * 50)
    
    print("Инструкция:")
    print("1. Запустите main_demo.py")
    print("2. Нажмите K (переключение в режим дерева)")
    print("3. Нажмите O (оптимизация)")
    print("4. Нажмите M (мердж в буферный граф)")
    print("5. Смотрите на вывод в консоли")
    print()
    print("Ищите в логах:")
    print("- '🔗 ОБРАБОТКА СВЯЗЕЙ' - создание связей в буферном графе")
    print("- '✅ Экспортировано X связей из графа' - экспорт в JSON")
    print("- 'DEBUG: График содержит X связей' - состояние spore_manager.graph")
    print()
    print("Если видите:")
    print("- '🔗 ОБРАБОТКА СВЯЗЕЙ' но потом '0 связей' → проблема в создании связей")
    print("- 'График содержит 0 связей' → связи не добавляются в граф")
    print("- 'Экспортировано 0 связей' → JSON экспорт не находит связи")
    
    return True

def check_recent_buffer_files():
    """Проверяет недавние файлы буферного графа."""
    
    buffer_dir = os.path.join(script_dir, 'scripts', 'run', 'buffer')
    
    if not os.path.exists(buffer_dir):
        print(f"❌ Папка buffer не найдена: {buffer_dir}")
        return False
    
    print(f"\n📁 ФАЙЛЫ В ПАПКЕ BUFFER:")
    
    files = []
    for filename in os.listdir(buffer_dir):
        filepath = os.path.join(buffer_dir, filename)
        if os.path.isfile(filepath):
            stat = os.stat(filepath)
            files.append({
                'name': filename,
                'size': stat.st_size,
                'modified': stat.st_mtime
            })
    
    # Сортируем по времени модификации
    files.sort(key=lambda x: x['modified'], reverse=True)
    
    for file_info in files:
        import datetime
        mod_time = datetime.datetime.fromtimestamp(file_info['modified'])
        print(f"   📄 {file_info['name']}: {file_info['size']} байт, {mod_time}")
    
    return True

if __name__ == '__main__':
    check_buffer_manager_debug()
    check_recent_buffer_files()

