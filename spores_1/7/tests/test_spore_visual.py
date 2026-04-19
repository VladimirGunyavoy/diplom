#!/usr/bin/env python3
"""
Тест для SporeVisual - проверяем визуализацию и 2D↔3D конвертацию.
Запустите: cd tests && python3 test_spore_visual.py
"""

import sys
import os
import numpy as np

# Добавляем путь к src (из tests в корень проекта)
sys.path.append('..')
sys.path.append('.')

# Мокаем ursina ДО импорта модулей, которые его используют
class MockColor:
    def rgba(self, r, g, b, a):
        return (r, g, b, a)

class MockEntity:
    def __init__(self, *args, **kwargs):
        self.position = kwargs.get('position', (0, 0, 0))
        self.scale = kwargs.get('scale', (1, 1, 1))
        self.color = kwargs.get('color', (1, 1, 1, 1))

# Мокаем модуль ursina перед импортом
mock_ursina = type('MockUrsina', (), {
    'Entity': MockEntity,
    'color': MockColor()
})()

sys.modules['ursina'] = mock_ursina

# Добавляем Entity в глобальное пространство для import *
import builtins
builtins.Entity = MockEntity

# Теперь можно импортировать модули
from src.pendulum import PendulumSystem
from src.spore_logic import SporeLogic
from src.color_manager import ColorManager
from src.spore_visual import SporeVisual

def test_spore_visual():
    print("=== Тестирование SporeVisual ===")
    
    # Создаем компоненты
    color_manager = ColorManager()
    print(f"✓ Создан ColorManager")
    
    # Тест 1: Создание SporeVisual
    print(f"\n--- Тест 1: Создание SporeVisual ---")
    spore_visual = SporeVisual(
        model='sphere',
        color_manager=color_manager,
        is_goal=False,
        y_coordinate=0.5,
        position=(1.0, 0.5, 2.0),
        scale=0.1
    )
    print(f"✓ Создан SporeVisual: {spore_visual}")
    print(f"✓ Y координата: {spore_visual.y_coordinate}")
    print(f"✓ real_position: {spore_visual.real_position}")
    
    # Тест 2: Конвертация 3D→2D
    print(f"\n--- Тест 2: Конвертация 3D→2D ---")
    pos_2d = spore_visual.get_logic_position_2d()
    print(f"✓ 3D позиция: {spore_visual.real_position}")
    print(f"✓ 2D позиция: {pos_2d}")
    expected_2d = np.array([1.0, 2.0])  # (x, z) из (1.0, 0.5, 2.0)
    assert np.allclose(pos_2d, expected_2d), f"Ожидали {expected_2d}, получили {pos_2d}"
    
    # Тест 3: Конвертация 2D→3D
    print(f"\n--- Тест 3: Конвертация 2D→3D ---")
    new_2d_pos = np.array([3.0, 4.0])
    spore_visual.set_logic_position_2d(new_2d_pos)
    print(f"✓ Новая 2D позиция: {new_2d_pos}")
    print(f"✓ Обновленная 3D позиция: {spore_visual.real_position}")
    expected_3d = np.array([3.0, 0.5, 4.0])  # сохраняем Y
    assert np.allclose(spore_visual.real_position, expected_3d), f"Ожидали {expected_3d}, получили {spore_visual.real_position}"
    
    # Тест 4: Синхронизация с SporeLogic
    print(f"\n--- Тест 4: Синхронизация с SporeLogic ---")
    pendulum = PendulumSystem(dt=0.1, damping=0.3)
    spore_logic = SporeLogic(
        pendulum=pendulum,
        dt=0.1,
        goal_position_2d=np.array([np.pi, 0.0]),
        initial_position_2d=np.array([0.0, 2.0])
    )
    print(f"✓ Создан SporeLogic: {spore_logic}")
    
    # Синхронизируем
    spore_visual.sync_with_logic(spore_logic)
    print(f"✓ После синхронизации:")
    print(f"  SporeLogic позиция: {spore_logic.get_position_2d()}")
    print(f"  SporeVisual 3D: {spore_visual.real_position}")
    print(f"  SporeVisual 2D: {spore_visual.get_logic_position_2d()}")
    
    # Проверяем корректность синхронизации
    assert np.allclose(spore_logic.get_position_2d(), spore_visual.get_logic_position_2d()), "Позиции должны совпадать после синхронизации"
    
    # Тест 5: Изменение Y координаты
    print(f"\n--- Тест 5: Изменение Y координаты ---")
    old_2d = spore_visual.get_logic_position_2d()
    old_3d = spore_visual.real_position.copy()
    print(f"✓ До изменения Y: 2D={old_2d}, 3D={old_3d}")
    
    spore_visual.set_y_coordinate(1.5)
    new_2d = spore_visual.get_logic_position_2d()
    new_3d = spore_visual.real_position
    print(f"✓ После изменения Y на 1.5: 2D={new_2d}, 3D={new_3d}")
    
    # 2D позиция должна остаться той же, Y должна измениться
    assert np.allclose(old_2d, new_2d), "2D позиция должна остаться неизменной"
    assert new_3d[1] == 1.5, f"Y координата должна быть 1.5, получили {new_3d[1]}"
    assert spore_visual.y_coordinate == 1.5, "y_coordinate должна обновиться"
    
    # Тест 6: Управление цветом
    print(f"\n--- Тест 6: Управление цветом ---")
    print(f"✓ Текущий цвет (default): {spore_visual.color}")
    
    spore_visual.set_color_type('ghost')
    print(f"✓ Цвет после смены на 'ghost': {spore_visual.color}")
    
    spore_visual.set_color_type('goal')  
    print(f"✓ Цвет после смены на 'goal': {spore_visual.color}")
    
    # Тест 7: Целевая спора (is_goal=True)
    print(f"\n--- Тест 7: Целевая спора ---")
    goal_visual = SporeVisual(
        color_manager=color_manager,
        is_goal=True,
        position=(np.pi, 0.0, 0.0)
    )
    print(f"✓ Создана целевая спора: {goal_visual}")
    print(f"✓ is_goal: {goal_visual.is_goal}")
    print(f"✓ Цвет целевой споры: {goal_visual.color}")
    
    # Тест 8: apply_transform (имитация ZoomManager)
    print(f"\n--- Тест 8: apply_transform ---")
    # Устанавливаем начальные значения
    spore_visual.real_position = np.array([1.0, 0.0, 2.0])
    spore_visual.real_scale = np.array([0.1, 0.1, 0.1])
    
    print(f"✓ До трансформации:")
    print(f"  real_position: {spore_visual.real_position}")
    print(f"  real_scale: {spore_visual.real_scale}")
    print(f"  position: {spore_visual.position}")
    print(f"  scale: {spore_visual.scale}")
    
    # Применяем трансформацию (a=2, b=[1,1,1], spores_scale=0.5)
    a = 2.0
    b = np.array([1.0, 1.0, 1.0])
    spores_scale = 0.5
    
    spore_visual.apply_transform(a, b, spores_scale=spores_scale)
    
    print(f"✓ После трансформации (a={a}, b={b}, spores_scale={spores_scale}):")
    print(f"  position: {spore_visual.position}")
    print(f"  scale: {spore_visual.scale}")
    
    # Проверяем формулы трансформации
    expected_position = spore_visual.real_position * a + b  # [1,0,2]*2 + [1,1,1] = [3,1,5]
    expected_scale = spore_visual.real_scale * a * spores_scale  # [0.1,0.1,0.1]*2*0.5 = [0.1,0.1,0.1]
    
    assert np.allclose(spore_visual.position, expected_position), f"Позиция: ожидали {expected_position}, получили {spore_visual.position}"
    assert np.allclose(spore_visual.scale, expected_scale), f"Масштаб: ожидали {expected_scale}, получили {spore_visual.scale}"
    
    # Тест 9: Проверка ошибок
    print(f"\n--- Тест 9: Проверка ошибок ---")
    try:
        spore_visual.set_logic_position_2d([1, 2, 3])  # 3D вместо 2D
        assert False, "Должна быть ошибка для 3D позиции"
    except ValueError as e:
        print(f"✓ Корректная ошибка для 3D позиции: {e}")
    
    print(f"\n🎉 Все тесты SporeVisual прошли успешно!")
    return True

if __name__ == "__main__":
    try:
        test_spore_visual()
        print(f"\n✅ SporeVisual готов к использованию!")
    except Exception as e:
        print(f"\n❌ Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 