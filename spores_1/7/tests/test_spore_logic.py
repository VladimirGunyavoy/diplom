#!/usr/bin/env python3
"""
Тест для SporeLogic - проверяем математическую логику без GUI.
Запустите: python test_spore_logic.py
"""

import sys
import os
import numpy as np

# Добавляем путь к src (из tests в корень проекта)
sys.path.append('../src')

from src.pendulum import PendulumSystem
from src.spore_logic import SporeLogic

def test_spore_logic():
    print("=== Тестирование SporeLogic ===")
    
    # Создаем систему маятника
    pendulum = PendulumSystem(dt=0.1, damping=0.3)
    print(f"✓ Создан PendulumSystem: dt={pendulum.dt}, damping={pendulum.damping}")
    
    # Тестовые параметры
    goal_pos_2d = np.array([np.pi, 0.0])  # (x, z)
    initial_pos_2d = np.array([0.0, 2.0])  # (x, z)
    dt = 0.1
    
    # Создаем SporeLogic
    print(f"\n--- Создание SporeLogic ---")
    spore_logic = SporeLogic(
        pendulum=pendulum,
        dt=dt,
        goal_position_2d=goal_pos_2d,
        initial_position_2d=initial_pos_2d
    )
    print(f"✓ Создан SporeLogic: {spore_logic}")
    
    # Тест 1: Проверяем начальное состояние
    print(f"\n--- Тест 1: Начальное состояние ---")
    pos = spore_logic.get_position_2d()
    print(f"✓ Позиция: {pos}")
    print(f"✓ Стоимость: {spore_logic.cost:.3f}")
    assert np.allclose(pos, initial_pos_2d), f"Ожидали {initial_pos_2d}, получили {pos}"
    
    # Тест 2: Эволюция без управления
    print(f"\n--- Тест 2: Эволюция без управления ---")
    next_state = spore_logic.evolve(control=0)
    print(f"✓ Эволюция {pos} -> {next_state}")
    assert len(next_state) == 2, f"Ожидали 2D состояние, получили {len(next_state)}D"
    
    # Тест 3: Эволюция с управлением
    print(f"\n--- Тест 3: Эволюция с управлением ---")
    control = 0.5
    next_state_with_control = spore_logic.evolve(control=control)
    print(f"✓ Эволюция с управлением {control}: {pos} -> {next_state_with_control}")
    
    # Тест 4: step (создание новой SporeLogic)
    print(f"\n--- Тест 4: Step (новая SporeLogic) ---")
    new_spore = spore_logic.step(control=0.2)
    print(f"✓ Исходная: {spore_logic}")
    print(f"✓ Новая: {new_spore}")
    assert not np.allclose(spore_logic.get_position_2d(), new_spore.get_position_2d()), "Позиции должны отличаться"
    
    # Тест 5: clone
    print(f"\n--- Тест 5: Клонирование ---")
    cloned_spore = spore_logic.clone()
    print(f"✓ Оригинал: {spore_logic}")
    print(f"✓ Клон: {cloned_spore}")
    assert np.allclose(spore_logic.get_position_2d(), cloned_spore.get_position_2d()), "Клон должен иметь ту же позицию"
    assert spore_logic is not cloned_spore, "Клон должен быть отдельным объектом"
    
    # Тест 6: Генерация управлений
    print(f"\n--- Тест 6: Генерация управлений ---")
    random_controls = spore_logic.sample_random_controls(5)
    mesh_controls = spore_logic.sample_mesh_controls(5)
    print(f"✓ Случайные управления: {random_controls}")
    print(f"✓ Сеточные управления: {mesh_controls}")
    
    a, b = pendulum.get_control_bounds()
    assert len(random_controls) == 5, "Должно быть 5 случайных управлений"
    assert len(mesh_controls) == 5, "Должно быть 5 сеточных управлений"
    assert np.all(random_controls >= a) and np.all(random_controls <= b), "Случайные управления должны быть в границах"
    
    # Тест 7: Симуляция управлений
    print(f"\n--- Тест 7: Симуляция управлений ---")
    controls = [0, 0.5, -0.5]
    simulated_states = spore_logic.simulate_controls(controls)
    print(f"✓ Управления: {controls}")
    print(f"✓ Результирующие состояния:")
    for i, state in enumerate(simulated_states):
        print(f"   {controls[i]} -> {state}")
    
    assert simulated_states.shape == (3, 2), f"Ожидали форму (3, 2), получили {simulated_states.shape}"
    
    # Тест 8: Проверка ошибок
    print(f"\n--- Тест 8: Проверка ошибок ---")
    try:
        # Неправильная размерность цели
        SporeLogic(pendulum, dt, [1, 2, 3], initial_pos_2d)
        assert False, "Должна быть ошибка для 3D цели"
    except ValueError as e:
        print(f"✓ Корректная ошибка для 3D цели: {e}")
    
    try:
        # Неправильная размерность начальной позиции
        SporeLogic(pendulum, dt, goal_pos_2d, [1, 2, 3])
        assert False, "Должна быть ошибка для 3D начальной позиции"
    except ValueError as e:
        print(f"✓ Корректная ошибка для 3D начальной позиции: {e}")
    
    try:
        # Неправильная размерность в evolve
        spore_logic.evolve(state=[1, 2, 3])
        assert False, "Должна быть ошибка для 3D состояния в evolve"
    except ValueError as e:
        print(f"✓ Корректная ошибка для 3D состояния в evolve: {e}")
    
    print(f"\n🎉 Все тесты SporeLogic прошли успешно!")
    return True

if __name__ == "__main__":
    try:
        test_spore_logic()
        print(f"\n✅ SporeLogic готов к использованию!")
    except Exception as e:
        print(f"\n❌ Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 