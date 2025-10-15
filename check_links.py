#!/usr/bin/env python3
import json

# Загружаем JSON
with open('spores/v16_picker/scripts/run/buffer/real_graph_latest.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Всего связей в JSON: {len(data['links'])}")
print("Связи:")
for i, link in enumerate(data['links']):
    parent_id = link['parent_spore_id']
    child_id = link['child_spore_id']
    control = link['control']
    dt = link['dt']
    dt_sign = link['dt_sign']
    dt_value = dt * dt_sign
    print(f"{i+1:2d}. {parent_id} -> {child_id}: control={control:+}, dt={dt_value:+}")

print(f"\nТакже проверим in_links/out_links у первых спор:")
for i, spore in enumerate(data['spores'][:3]):
    spore_id = spore['spore_id']
    in_links = spore.get('in_links', [])
    out_links = spore.get('out_links', [])
    print(f"Спора {spore_id}: in_links={in_links}, out_links={out_links}")

