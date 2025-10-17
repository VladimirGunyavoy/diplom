"""
Тест логики valence_manager
"""

# Проверяем что логика правильная
print("Testing valence_manager.py logic...")
print()

with open('spores/v16_picker/src/managers/valence_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Проверяем логику для исходящих связей (детей)
print("Outgoing links (children) logic:")
outgoing_section = content[content.find('# Исходящие связи'):content.find('# Входящие связи')]

checks = [
    ("time_direction = 'forward'", "Direction is always forward"),
    ("abs(raw_dt)", "dt is always positive"),
    ("control_value = self._convert_to_float(self._extract_control_from_edge(edge_info))", "Control is not inverted"),
]

for check, desc in checks:
    if check in outgoing_section:
        print(f"  OK: {desc}")
    else:
        print(f"  ERROR: {desc} - not found")

# Проверяем логику для входящих связей (родителей)
print()
print("Incoming links (parents) logic:")
incoming_section = content[content.find('# Входящие связи'):content.find('return neighbors', content.find('# Входящие связи'))]

checks = [
    ("time_direction = 'backward'", "Direction is always backward"),
    ("-abs(raw_dt)", "dt is always negative"),
    ("control_value = raw_control", "Control is NOT inverted (same as edge)"),
]

for check, desc in checks:
    if check in incoming_section:
        print(f"  OK: {desc}")
    else:
        print(f"  ERROR: {desc} - not found")

# Проверяем логику для внуков (альтернация управления)
print()
print("Grandchildren logic (control alternation):")
grandchildren_section = content[content.find('def _get_neighbors_at_distance_2'):content.find('def _extract_dt_from_edge')]

if 'if first_control_type == second_control_type:' in grandchildren_section and 'continue' in grandchildren_section:
    print("  OK: Routes without control alternation are filtered out")
else:
    print("  ERROR: Control alternation filter not found")

print()
print("=== LOGIC TEST COMPLETE ===")
