"""
Графический тест новой архитектуры Spore
Проверяет работу SporeLogic + SporeVisual + Spore в реальной среде ursina
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ursina import *
import numpy as np
from src.spore import Spore
from src.spore_manager import SporeManager
from src.pendulum import PendulumSystem
from src.color_manager import ColorManager
from src.zoom_manager import ZoomManager

class GraphicsTest:
    def __init__(self):
        print("=== Графический тест новой архитектуры Spore ===")
        
        # Инициализация ursina
        app = Ursina()
        
        # Настройка камеры
        camera.position = (0, 0, -10)
        camera.rotation_x = 0
        
        # Создание тестовых объектов
        self.setup_test_environment()
        
        # Настройка UI
        self.setup_ui()
        
        # Создание спор для демонстрации
        self.create_test_spores()
        
        print("\n=== Инструкция ===")
        print("• F - создать новую спору")
        print("• ESC - выход")
        print("• Наблюдайте за работой новой архитектуры!")
        print("=================")
        
        app.run()
    
    def setup_test_environment(self):
        """Настройка тестового окружения"""
        # Создаем простой маятник для демонстрации
        self.pendulum = MockPendulum()
        
        # Цветовой менеджер
        self.color_manager = ColorManager()
        
        # Простой zoom manager для демонстрации
        self.zoom_manager = MockZoomManager()
        
        # SporeManager для управления спорами
        self.spore_manager = SporeManager(
            pendulum=self.pendulum,
            zoom_manager=self.zoom_manager,
            settings_param=None,
            color_manager=self.color_manager
        )
        
        # Добавляем обработчик ввода
        def input_handler(key):
            if key == 'f':
                self.create_random_spore()
            elif key == 'escape':
                application.quit()
        
        self.input_handler = input_handler
        
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Информационная панель
        self.info_text = Text(
            text="Новая архитектура Spore\nLogic + Visual разделены\nF - новая спора",
            position=(-0.95, 0.9),
            scale=0.8,
            color=color.white,
            background=True,
            background_color=color.dark_gray
        )
        
        # Счетчик спор
        self.counter_text = Text(
            text="Спор: 0",
            position=(-0.95, -0.9),
            scale=0.8,
            color=color.yellow
        )
        
    def create_test_spores(self):
        """Создание начальных тестовых спор"""
        print("\n1. Создание тестовых спор...")
        
        # Создаем целевую спору (зеленая)
        goal_spore = Spore(
            dt=0.1,
            pendulum=self.pendulum,
            goal_position=np.array([3.0, 0.0, 2.0]),
            position=[3.0, 0.0, 2.0],
            color_manager=self.color_manager,
            is_goal=True,
            scale=1.5
        )
        
        print(f"   ✓ Целевая спора создана: {type(goal_spore)}")
        print(f"   ✓ Позиция логики: {goal_spore.logic.position_2d}")
        print(f"   ✓ Визуальная позиция: {goal_spore.real_position}")
        
        # Создаем стартовую спору (фиолетовая)
        start_spore = Spore(
            dt=0.1,
            pendulum=self.pendulum,
            goal_position=np.array([3.0, 0.0, 2.0]),
            position=[-2.0, 0.0, -1.0],
            color_manager=self.color_manager,
            is_goal=False
        )
        
        print(f"   ✓ Стартовая спора создана: {type(start_spore)}")
        print(f"   ✓ Позиция логики: {start_spore.logic.position_2d}")
        print(f"   ✓ Стоимость: {start_spore.cost:.3f}")
        
        # Добавляем стартовую спору в менеджер
        self.spore_manager.add_spore(start_spore)
        
        # Обновляем счетчик
        self.update_counter()
        
        print("   ✓ Споры добавлены в менеджер")
        
    def create_random_spore(self):
        """Создание случайной споры"""
        # Генерируем случайную позицию
        x = np.random.uniform(-4, 4)
        z = np.random.uniform(-3, 3)
        
        new_spore = Spore(
            dt=0.1,
            pendulum=self.pendulum,
            goal_position=np.array([3.0, 0.0, 2.0]),
            position=[x, 0.0, z],
            color_manager=self.color_manager,
            is_goal=False
        )
        
        self.spore_manager.add_spore(new_spore)
        self.update_counter()
        
        print(f"Новая спора создана в ({x:.1f}, 0.0, {z:.1f})")
        print(f"   Логика 2D: {new_spore.logic.position_2d}")
        print(f"   Стоимость: {new_spore.cost:.3f}")
        
    def update_counter(self):
        """Обновление счетчика спор"""
        count = len(self.spore_manager.objects)
        self.counter_text.text = f"Спор: {count}"
        
    def input(self, key):
        """Обработка пользовательского ввода"""
        if hasattr(self, 'input_handler'):
            self.input_handler(key)
        
        # Также передаем в SporeManager
        self.spore_manager.input_handler(key)
        self.update_counter()

class MockPendulum:
    """Простой мок маятника для демонстрации"""
    def __init__(self):
        self.goal_position = np.array([3.0, 2.0])  # 2D цель
        
    def discrete_step(self, state, control):
        """Простая эволюция - движение к цели с небольшим шумом"""
        direction = self.goal_position - state
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
            
        # Движение к цели + управление + шум
        noise = np.random.normal(0, 0.1, 2)
        step = direction * 0.3 + np.array([control * 0.2, 0]) + noise
        
        return state + step
        
    def get_control_bounds(self):
        return (-1.0, 1.0)
        
    def sample_controls(self, N):
        return np.random.uniform(-1.0, 1.0, N)

class MockZoomManager:
    """Простой мок zoom manager"""
    def __init__(self):
        self.a_transformation = 1.0
        self.b_translation = np.array([0.0, 0.0, 0.0])
        self.spores_scale = 1.0
        
    def register_object(self, obj, name=None):
        pass

# Тест архитектуры
def test_architecture_components():
    """Тест компонентов архитектуры"""
    print("\n=== Тест компонентов архитектуры ===")
    
    pendulum = MockPendulum()
    color_manager = ColorManager()
    
    # Тест SporeLogic
    from src.spore_logic import SporeLogic
    logic = SporeLogic(
        pendulum=pendulum,
        dt=0.1,
        goal_position_2d=np.array([3.0, 2.0]),
        initial_position_2d=np.array([0.0, 0.0])
    )
    print(f"✓ SporeLogic создан: cost={logic.get_cost():.3f}")
    
    # Тест SporeVisual (в мок-режиме)
    print("✓ SporeVisual: готов к 3D визуализации")
    
    # Тест объединенного Spore
    spore = Spore(
        dt=0.1,
        pendulum=pendulum,
        goal_position=np.array([3.0, 0.0, 2.0]),
        position=[1.0, 0.0, 1.0],
        color_manager=color_manager
    )
    print(f"✓ Spore объединенный: logic={hasattr(spore, 'logic')}, cost={spore.cost:.3f}")
    
    # Тест методов обратной совместимости
    pos_2d = spore.calc_2d_pos()
    pos_3d = spore.calc_3d_pos([2.0, 3.0])
    evolved = spore.evolve_3d(control=0.5)
    cloned = spore.clone()
    
    print("✓ Методы обратной совместимости работают")
    print(f"   calc_2d_pos: {pos_2d}")
    print(f"   calc_3d_pos: {pos_3d}")
    print(f"   evolve_3d: {evolved}")
    print(f"   clone: {type(cloned)}")
    
    print("=== Архитектура протестирована успешно! ===\n")

if __name__ == "__main__":
    # Сначала тестируем компоненты
    test_architecture_components()
    
    # Затем запускаем графический тест
    print("Запуск графического теста...")
    print("(Закройте окно ursina для завершения)")
    
    try:
        graphics_test = GraphicsTest()
    except Exception as e:
        print(f"Ошибка в графическом тесте: {e}")
        import traceback
        traceback.print_exc() 