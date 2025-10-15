#!/usr/bin/env python3
import json

# Загружаем JSON
with open('spores/v16_picker/scripts/run/buffer/real_graph_latest.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("🔍 АНАЛИЗ СПОРЫ 2:")
print("="*50)

# Ищем спору с индексом 1 (это спора номер 2 в таблице)
spore_2_data = None
for spore in data['spores']:
    if spore['index'] == 1:  # index=1 это спора номер 2 в таблице
        spore_2_data = spore
        break

if spore_2_data:
    print(f"Спора ID: {spore_2_data['spore_id']}")
    print(f"Индекс: {spore_2_data['index']} (это спора #{spore_2_data['index']+1} в таблице)")
    print(f"Исходящие связи: {spore_2_data['out_links']}")
    print(f"Входящие связи: {spore_2_data['in_links']}")
    
    print(f"\n🔗 ДЕТАЛИ СВЯЗЕЙ ИЗ МАССИВА LINKS:")
    
    # Ищем все связи где участвует эта спора
    spore_id = spore_2_data['spore_id']
    
    print(f"\n📤 ИСХОДЯЩИЕ (parent_spore_id = {spore_id}):")
    for i, link in enumerate(data['links']):
        if link['parent_spore_id'] == spore_id:
            child_id = link['child_spore_id']
            control = link['control']
            dt = link['dt']
            dt_sign = link['dt_sign']
            dt_value = dt * dt_sign
            
            # Находим индекс целевой споры
            child_index = None
            for spore in data['spores']:
                if spore['spore_id'] == child_id:
                    child_index = spore['index']
                    break
            
            print(f"   → Спора {child_index+1} (ID {child_id}): control={control:+}, dt={dt_value:+.3f}")
    
    print(f"\n📥 ВХОДЯЩИЕ (child_spore_id = {spore_id}):")
    for i, link in enumerate(data['links']):
        if link['child_spore_id'] == spore_id:
            parent_id = link['parent_spore_id']
            control = link['control']
            dt = link['dt']
            dt_sign = link['dt_sign']
            dt_value = dt * dt_sign
            
            # Находим индекс родительской споры
            parent_index = None
            for spore in data['spores']:
                if spore['spore_id'] == parent_id:
                    parent_index = spore['index']
                    break
            
            print(f"   ← Спора {parent_index+1} (ID {parent_id}): control={control:+}, dt={dt_value:+.3f}")

else:
    print("❌ Спора с индексом 1 не найдена!")

