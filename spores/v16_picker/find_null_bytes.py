#!/usr/bin/env python3
"""Поиск null bytes в файле spore.py"""

with open('src/core/spore.py', 'rb') as f:
    data = f.read()

null_positions = [i for i, b in enumerate(data) if b == 0]
print('Null byte positions:', null_positions)

print('\nContext around null bytes:')
for pos in null_positions:
    start = max(0, pos - 5)
    end = min(len(data), pos + 6)
    context = data[start:end]
    print(f'Position {pos}: {repr(context)}')

# Найдем строки с null bytes
lines = data.split(b'\n')
for i, line in enumerate(lines):
    if b'\x00' in line:
        print(f'\nLine {i+1} contains null bytes: {repr(line)}')
