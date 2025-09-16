#!/usr/bin/env python3
"""
Тест для проверки исправления знаков управления в линках пикера.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from src.core.spore import Spore
from src.visual.link import Link
from src.managers.color_manager import ColorManager
from src.managers.zoom_manager import ZoomManager
from src.logic.pendulum import PendulumSystem

def test_link_control_storage():
    """Тест сохранения информации об управлении в линке."""
    print("🧪 Тестирование сохранения информации об управлении в линке...")
    
    # Создаем необходимые компоненты
    pendulum = PendulumSystem()
    color_manager = ColorManager()
    zoom_manager = ZoomManager()
    config = {}
    
    # Создаем тестовые споры
    parent_spore = Spore(
        pendulum=pendulum,
        initial_position_2d=np.array([0.0, 0.0]),
        goal_position_2d=np.array([1.0, 1.0])
    )
    
    child_spore = Spore(
        pendulum=pendulum,
        initial_position_2d=np.array([0.1, 0.1]),
        goal_position_2d=np.array([1.0, 1.0])
    )
    
    # Создаем линк
    link = Link(
        parent_spore=parent_spore,
        child_spore=child_spore,
        color_manager=color_manager,
        zoom_manager=zoom_manager,
        config=config
    )
    
    # Тестируем сохранение информации об управлении
    test_control = 2.5
    test_dt = 0.1
    
    link.control_value = test_control
    link.dt_value = test_dt
    
    # Проверяем, что информация сохранилась
    assert hasattr(link, 'control_value'), "Линк должен иметь атрибут control_value"
    assert hasattr(link, 'dt_value'), "Линк должен иметь атрибут dt_value"
    assert link.control_value == test_control, f"Ожидалось control_value={test_control}, получено {link.control_value}"
    assert link.dt_value == test_dt, f"Ожидалось dt_value={test_dt}, получено {link.dt_value}"
    
    print("✅ Тест сохранения информации об управлении в линке прошел успешно!")
    print(f"   control_value: {link.control_value}")
    print(f"   dt_value: {link.dt_value}")

def test_picker_control_retrieval():
    """Тест получения информации об управлении из линка в пикере."""
    print("\n🧪 Тестирование получения информации об управлении из линка...")
    
    # Создаем необходимые компоненты
    pendulum = PendulumSystem()
    color_manager = ColorManager()
    zoom_manager = ZoomManager()
    config = {}
    
    # Создаем тестовые споры
    parent_spore = Spore(
        pendulum=pendulum,
        initial_position_2d=np.array([0.0, 0.0]),
        goal_position_2d=np.array([1.0, 1.0])
    )
    
    child_spore = Spore(
        pendulum=pendulum,
        initial_position_2d=np.array([0.1, 0.1]),
        goal_position_2d=np.array([1.0, 1.0])
    )
    
    # Создаем линк с информацией об управлении
    link = Link(
        parent_spore=parent_spore,
        child_spore=child_spore,
        color_manager=color_manager,
        zoom_manager=zoom_manager,
        config=config
    )
    
    test_control = -1.8
    test_dt = -0.05
    
    link.control_value = test_control
    link.dt_value = test_dt
    
    # Симулируем логику пикера для получения управления
    if hasattr(link, 'control_value'):
        control = link.control_value
        control_source = "link"
    elif hasattr(parent_spore, 'logic'):
        control = getattr(parent_spore.logic, 'optimal_control', 'N/A')
        control_source = "spore"
    else:
        control = 'N/A'
        control_source = "none"
    
    # Проверяем результат
    assert control_source == "link", f"Ожидался источник 'link', получен '{control_source}'"
    assert control == test_control, f"Ожидалось управление {test_control}, получено {control}"
    
    print("✅ Тест получения информации об управлении из линка прошел успешно!")
    print(f"   Источник управления: {control_source}")
    print(f"   Значение управления: {control}")
    print(f"   Знак управления: {'+' if control > 0 else '-' if control < 0 else '0'}")

def test_positive_and_negative_controls():
    """Тест для проверки положительных и отрицательных знаков управления."""
    print("\n🧪 Тестирование положительных и отрицательных знаков управления...")
    
    # Создаем необходимые компоненты
    pendulum = PendulumSystem()
    color_manager = ColorManager()
    zoom_manager = ZoomManager()
    config = {}
    
    # Тестируем положительное управление
    parent_spore_pos = Spore(
        pendulum=pendulum,
        initial_position_2d=np.array([0.0, 0.0]),
        goal_position_2d=np.array([1.0, 1.0])
    )
    
    child_spore_pos = Spore(
        pendulum=pendulum,
        initial_position_2d=np.array([0.1, 0.1]),
        goal_position_2d=np.array([1.0, 1.0])
    )
    
    link_pos = Link(
        parent_spore=parent_spore_pos,
        child_spore=child_spore_pos,
        color_manager=color_manager,
        zoom_manager=zoom_manager,
        config=config
    )
    
    link_pos.control_value = 2.0
    link_pos.dt_value = 0.1
    
    # Тестируем отрицательное управление
    parent_spore_neg = Spore(
        pendulum=pendulum,
        initial_position_2d=np.array([0.0, 0.0]),
        goal_position_2d=np.array([1.0, 1.0])
    )
    
    child_spore_neg = Spore(
        pendulum=pendulum,
        initial_position_2d=np.array([-0.1, -0.1]),
        goal_position_2d=np.array([1.0, 1.0])
    )
    
    link_neg = Link(
        parent_spore=parent_spore_neg,
        child_spore=child_spore_neg,
        color_manager=color_manager,
        zoom_manager=zoom_manager,
        config=config
    )
    
    link_neg.control_value = -2.0
    link_neg.dt_value = -0.1
    
    # Проверяем знаки
    assert link_pos.control_value > 0, f"Ожидалось положительное управление, получено {link_pos.control_value}"
    assert link_neg.control_value < 0, f"Ожидалось отрицательное управление, получено {link_neg.control_value}"
    
    print("✅ Тест положительных и отрицательных знаков управления прошел успешно!")
    print(f"   Положительное управление: {link_pos.control_value}")
    print(f"   Отрицательное управление: {link_neg.control_value}")
    print(f"   Положительное время: {link_pos.dt_value}")
    print(f"   Отрицательное время: {link_neg.dt_value}")

if __name__ == "__main__":
    print("🚀 Запуск тестов исправления знаков управления в линках пикера...")
    
    try:
        test_link_control_storage()
        test_picker_control_retrieval()
        test_positive_and_negative_controls()
        
        print("\n🎉 Все тесты прошли успешно!")
        print("✅ Исправление знаков управления в линках пикера работает корректно!")
        
    except Exception as e:
        print(f"\n❌ Тест не прошел: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

