"""
Простой тест valence без импорта зависимостей
"""
import sys
import os

# Читаем файл valence.py напрямую и проверяем структуру
valence_file = 'spores/v16_picker/src/logic/valence.py'

print("Testing valence.py structure...")
print()

with open(valence_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Проверяем что есть правильный порядок слотов детей
children_slots_section = content[content.find('def _create_children_slots'):content.find('def _create_grandchildren_slots')]

print("Children slots defined in code:")
if "ValenceSlot('child', 'forward', 'max')" in children_slots_section:
    print("  1. forward_max - OK")
else:
    print("  1. forward_max - MISSING")

if "ValenceSlot('child', 'forward', 'min')" in children_slots_section:
    print("  2. forward_min - OK")
else:
    print("  2. forward_min - MISSING")

if "ValenceSlot('child', 'backward', 'max')" in children_slots_section:
    print("  3. backward_max - OK")
else:
    print("  3. backward_max - MISSING")

if "ValenceSlot('child', 'backward', 'min')" in children_slots_section:
    print("  4. backward_min - OK")
else:
    print("  4. backward_min - MISSING")

# Проверяем количество комбинаций внуков
grandchildren_section = content[content.find('def _create_grandchildren_slots'):content.find('return slots', content.find('def _create_grandchildren_slots'))]

combinations = grandchildren_section.count("('forward', 'max',") + \
               grandchildren_section.count("('forward', 'min',") + \
               grandchildren_section.count("('backward', 'max',") + \
               grandchildren_section.count("('backward', 'min',")

print()
print(f"Grandchildren combinations: {combinations}")
if combinations == 8:
    print("  OK: 8 combinations with alternation")
else:
    print(f"  ERROR: Expected 8, got {combinations}")

# Проверяем что есть все 8 комбинаций
expected = [
    "('forward', 'max', 'forward')",
    "('forward', 'max', 'backward')",
    "('forward', 'min', 'forward')",
    "('forward', 'min', 'backward')",
    "('backward', 'max', 'forward')",
    "('backward', 'max', 'backward')",
    "('backward', 'min', 'forward')",
    "('backward', 'min', 'backward')",
]

print()
print("Checking all expected grandchildren combinations:")
all_ok = True
for exp in expected:
    if exp in grandchildren_section:
        print(f"  OK: {exp}")
    else:
        print(f"  MISSING: {exp}")
        all_ok = False

if all_ok:
    print()
    print("=== ALL STRUCTURE TESTS PASSED ===")
else:
    print()
    print("=== ERRORS FOUND ===")
