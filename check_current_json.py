#!/usr/bin/env python3
import json

# Загружаем JSON
with open('spores/v16_picker/scripts/run/buffer/real_graph_latest.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Связей в JSON: {len(data.get('links', []))}")
print(f"Спор в JSON: {len(data.get('spores', []))}")

links = data.get('links', [])
if links:
    print("Первые 3 связи:")
    for i, link in enumerate(links[:3]):
        print(f"{i+1}. {link}")
else:
    print("❌ Связей нет!")
    
print(f"\nПроверяем in_links/out_links у спор:")
for i, spore in enumerate(data.get('spores', [])[:3]):
    spore_id = spore.get('spore_id', 'unknown')
    in_links = spore.get('in_links', [])
    out_links = spore.get('out_links', [])
    print(f"Спора {i+1} (ID: {spore_id}): in={len(in_links)}, out={len(out_links)}")

