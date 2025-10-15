#!/usr/bin/env python3
"""
Тестовый скрипт для проверки экспорта связей в JSON.
Проверяет что связи между спорами корректно добавляются в граф и экспортируются в JSON.
"""

import sys
import os
import json

# Добавляем корневую папку проекта в путь
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def test_links_export():
    """Тестирует экспорт связей между спорами."""
    
    print("🔍 ТЕСТ ЭКСПОРТА СВЯЗЕЙ В JSON")
    print("=" * 50)
    
    # Путь к последнему JSON файлу
    json_path = os.path.join(script_dir, 'scripts', 'run', 'buffer', 'real_graph_latest.json')
    
    if not os.path.exists(json_path):
        print(f"❌ JSON файл не найден: {json_path}")
        print("💡 Запустите main_demo.py и создайте дерево через K → O → M → ЛКМ")
        return False
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ JSON файл загружен: {json_path}")
        
        # Анализируем содержимое
        spores = data.get('spores', [])
        links = data.get('links', [])
        stats = data.get('statistics', {})
        
        print(f"\n📊 СТАТИСТИКА:")
        print(f"   Спор в JSON: {len(spores)}")
        print(f"   Связей в JSON: {len(links)}")
        print(f"   Общая статистика: {stats.get('total_spores', 0)} спор, {stats.get('total_links', 0)} связей")
        
        # Проверяем связи у каждой споры
        print(f"\n🔗 АНАЛИЗ СВЯЗЕЙ ПО СПОРАМ:")
        spores_with_links = 0
        total_in_links = 0
        total_out_links = 0
        
        for i, spore in enumerate(spores):
            in_links = spore.get('in_links', [])
            out_links = spore.get('out_links', [])
            
            if in_links or out_links:
                spores_with_links += 1
                total_in_links += len(in_links)
                total_out_links += len(out_links)
                
                print(f"   Спора {i} (ID: {spore.get('spore_id', 'unknown')}):")
                print(f"      Входящие: {in_links}")
                print(f"      Исходящие: {out_links}")
        
        print(f"\n📈 ИТОГИ:")
        print(f"   Спор со связями: {spores_with_links} из {len(spores)}")
        print(f"   Всего входящих: {total_in_links}")
        print(f"   Всего исходящих: {total_out_links}")
        print(f"   Отдельных связей: {len(links)}")
        
        # Детальная информация о связях
        if links:
            print(f"\n🔗 ДЕТАЛИ СВЯЗЕЙ:")
            for i, link in enumerate(links[:5]):  # Показываем первые 5
                print(f"   Связь {i}: {link.get('parent_spore_id')} → {link.get('child_spore_id')}")
                print(f"      Тип: {link.get('link_type', 'unknown')}")
                print(f"      Направление: {link.get('direction', 'unknown')}")
                print(f"      dt: {link.get('dt', 0.0)}")
            
            if len(links) > 5:
                print(f"   ... и еще {len(links) - 5} связей")
        
        # Проверяем успешность
        success = len(links) > 0 or total_in_links > 0 or total_out_links > 0
        
        if success:
            print(f"\n✅ ТЕСТ ПРОЙДЕН: Связи найдены в JSON!")
            print(f"💡 Исправление сработало - связи экспортируются из графа")
        else:
            print(f"\n❌ ТЕСТ НЕ ПРОЙДЕН: Связи не найдены в JSON")
            print(f"🔧 Проверьте что дерево было создано и материализовано")
        
        return success
        
    except Exception as e:
        print(f"❌ Ошибка чтения JSON: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_links_export()
