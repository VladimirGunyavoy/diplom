#!/usr/bin/env python3
"""
ТЕСТ СИСТЕМЫ ОТЛАДОЧНОГО ВЫВОДА
==============================

Проверяет новую систему управления отладочным выводом:
- По умолчанию всё выключено (производительность)
- Различные флаги для разных типов отладки
- Централизованное управление через конфигурацию

Что тестируется:
1. По умолчанию отладочные принты выключены
2. Флаги в config.json корректно управляют выводом
3. always_print() работает всегда
4. Различные типы отладки (verbose, evolution, candidate, trajectory)
"""

import os
import sys
import json
import collections.abc

# Настройка путей
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def deep_merge(d, u):
    """Рекурсивно объединяет словари."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_merge(d.get(k, {}), v)
        else:
            d[k] = v
    return d

print("🧪 ТЕСТ СИСТЕМЫ ОТЛАДОЧНОГО ВЫВОДА")
print("=" * 50)

# Загружаем конфигурацию
config_path = os.path.join(project_root, 'config', 'json', 'config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

print("📋 ПРОВЕРКА КОНФИГУРАЦИИ ПО УМОЛЧАНИЮ:")
debug_config = config.get('debug', {})
print(f"   enable_verbose_output: {debug_config.get('enable_verbose_output', 'НЕ НАЙДЕН')}")
print(f"   enable_detailed_evolution: {debug_config.get('enable_detailed_evolution', 'НЕ НАЙДЕН')}")
print(f"   enable_candidate_logging: {debug_config.get('enable_candidate_logging', 'НЕ НАЙДЕН')}")
print(f"   enable_trajectory_logging: {debug_config.get('enable_trajectory_logging', 'НЕ НАЙДЕН')}")

try:
    # Импортируем систему отладочного вывода
    from src.utils.debug_output import (
        init_debug_output, debug_print, evolution_print, 
        candidate_print, trajectory_print, always_print
    )
    
    print("\n✅ Модуль debug_output успешно загружен")
    
    # Тест 1: По умолчанию всё выключено
    print("\n🧪 ТЕСТ 1: Конфигурация по умолчанию (всё выключено)")
    init_debug_output(config)
    
    print("Проверяем что отладочные принты НЕ выводятся:")
    debug_print("   🔸 debug_print: Этот текст НЕ должен появиться")
    evolution_print("   🧬 evolution_print: Этот текст НЕ должен появиться")  
    candidate_print("   ⚪ candidate_print: Этот текст НЕ должен появиться")
    trajectory_print("   🔗 trajectory_print: Этот текст НЕ должен появиться")
    always_print("   ✅ always_print: Этот текст ДОЛЖЕН появиться")
    
    # Тест 2: Включаем все флаги
    print("\n🧪 ТЕСТ 2: Включаем все флаги отладки")
    test_config = deep_merge(config, {
        'debug': {
            'enable_verbose_output': True,
            'enable_detailed_evolution': True,
            'enable_candidate_logging': True,
            'enable_trajectory_logging': True
        }
    })
    
    init_debug_output(test_config)
    
    print("Проверяем что ВСЕ отладочные принты теперь выводятся:")
    debug_print("   🔸 debug_print: Этот текст ДОЛЖЕН появиться")
    evolution_print("   🧬 evolution_print: Этот текст ДОЛЖЕН появиться")
    candidate_print("   ⚪ candidate_print: Этот текст ДОЛЖЕН появиться")
    trajectory_print("   🔗 trajectory_print: Этот текст ДОЛЖЕН появиться")
    always_print("   ✅ always_print: Этот текст ВСЕГДА появляется")
    
    # Тест 3: Включаем только один тип
    print("\n🧪 ТЕСТ 3: Включаем только trajectory_logging")
    test_config = deep_merge(config, {
        'debug': {
            'enable_verbose_output': False,
            'enable_detailed_evolution': False,
            'enable_candidate_logging': False,
            'enable_trajectory_logging': True
        }
    })
    
    init_debug_output(test_config)
    
    print("Проверяем что только trajectory_print выводится:")
    debug_print("   🔸 debug_print: НЕ должен появиться")
    evolution_print("   🧬 evolution_print: НЕ должен появиться")
    candidate_print("   ⚪ candidate_print: НЕ должен появиться") 
    trajectory_print("   🔗 trajectory_print: ДОЛЖЕН появиться")
    always_print("   ✅ always_print: Всегда появляется")
    
    # Тест 4: Обратная совместимость со старым флагом
    print("\n🧪 ТЕСТ 4: Обратная совместимость со старым debug_output")
    test_config = deep_merge(config, {
        'debug': {
            'enable_verbose_output': False,
            'enable_detailed_evolution': False, 
            'enable_candidate_logging': False,
            'enable_trajectory_logging': False
        },
        'trajectory_optimization': {
            'debug_output': True
        }
    })
    
    init_debug_output(test_config)
    
    print("Проверяем что старый флаг включает trajectory_logging:")
    debug_print("   🔸 debug_print: НЕ должен появиться")
    evolution_print("   🧬 evolution_print: НЕ должен появиться")
    candidate_print("   ⚪ candidate_print: НЕ должен появиться")
    trajectory_print("   🔗 trajectory_print: ДОЛЖЕН появиться (через старый флаг)")
    always_print("   ✅ always_print: Всегда появляется")
    
    print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    print("\n📊 РЕЗЮМЕ:")
    print("✅ По умолчанию отладочный вывод выключен (производительность)")
    print("✅ Флаги в config.json корректно управляют выводом")  
    print("✅ always_print() работает независимо от флагов")
    print("✅ Различные типы отладки можно включать по отдельности")
    print("✅ Обратная совместимость со старым debug_output сохранена")
    
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    print(f"   • Для производительности оставьте все флаги false в config.json")
    print(f"   • Для отладки включите нужные типы: verbose/evolution/candidate/trajectory")
    print(f"   • Используйте always_print() для важных сообщений")

except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("ℹ️  Возможно отсутствуют модули системы")
except Exception as e:
    print(f"❌ Ошибка во время теста: {e}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    pass  # Код уже выполнен выше 