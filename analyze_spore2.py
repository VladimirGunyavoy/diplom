import json

with open('spores/v16_picker/scripts/run/buffer/real_graph_latest.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Спора с index=1 (это спора 2 на картинке, spore_id="3")
spore = data['spores'][1]
print(f"Spore visual {spore['index'] + 1} (internal ID: {spore['spore_id']})")
print(f"Position: {spore['position']}")
print()

print('CHILDREN (OUT LINKS):')
for link in spore['out_links']:
    direction = 'forward' if link['dt_sign'] == 1 else 'backward'
    control_type = 'max' if link['control'] > 0 else 'min'
    print(f"  {direction}_{control_type}: to spore {link['to_spore_id']}, dt={link['dt']}")

print()
print('CHILDREN (IN LINKS - going backwards):')
for link in spore['in_links']:
    # For incoming links we go backwards, so invert everything
    direction = 'backward' if link['dt_sign'] == 1 else 'forward'
    control_type = 'min' if link['control'] > 0 else 'max'
    print(f"  {direction}_{control_type}: from spore {link['from_spore_id']}, dt={-link['dt']}")

print()
print('ALL 4 CHILDREN (in expected order):')
print('  forward_max')
print('  forward_min')
print('  backward_max')
print('  backward_min')
