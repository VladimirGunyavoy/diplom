"""
Полноценный 3D графический тест новой архитектуры Spore
Использует настоящий scene_setup и все 3D возможности
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
from src.scene_setup import SceneSetup

class GraphicsTest3D:
    def __init__(self):
        print("=== 3D Графический тест новой архитектуры Spore ===")
        
        # Инициализация ursina
        app = Ursina()
        
        # Настройка 3D сцены
        self.setup_3d_scene()
        
        # Создание реального окружения
        self.setup_real_environment()
        
        # Создание спор для демонстрации
        self.create_demo_spores()
        
        # Настройка UI для демонстрации
        self.setup_demo_ui()
        
        print("\n=== 3D DEMO ИНСТРУКЦИИ ===")
        print("• WASD - перемещение")
        print("• Пробел/Shift - вверх/вниз")
        print("• Мышь - осмотр")
        print("• F - создать новую спору")
        print("• G - эволюция всех спор")
        print("• R - сброс спор")
        print("• Alt - переключение курсора")
        print("• Q - выход")
        print("========================")
        
        app.run()
    
    def setup_3d_scene(self):
        """Настройка полноценной 3D сцены"""
        # Цветовой менеджер
        self.color_manager = ColorManager()
        
        # Настройка сцены с освещением, полом, камерой
        self.scene = SceneSetup(
            init_position=(0, 2, -8),  # Хорошая стартовая позиция для обзора
            init_rotation_x=15,
            init_rotation_y=0,
            color_manager=self.color_manager
        )
        
    def setup_real_environment(self):
        """Создание реального окружения с маятником и менеджерами"""
        # Создаем настоящую систему маятника
        self.pendulum = PendulumSystem()
        
        # Zoom manager для масштабирования (передаем scene_setup)
        self.zoom_manager = ZoomManager(self.scene)
        
        # SporeManager для управления спорами
        self.spore_manager = SporeManager(
            pendulum=self.pendulum,
            zoom_manager=self.zoom_manager,
            settings_param=None,
            color_manager=self.color_manager
        )
        
        # Регистрируем объекты в zoom_manager
        self.zoom_manager.register_object(self.scene.floor, "floor")
        
    def create_demo_spores(self):
        """Создание демонстрационных спор"""
        print("\n1. Создание демонстрационных 3D спор...")
        
        # Целевая спора (зеленая, большая)
        goal_position = np.array([4.0, 0.0, 3.0])
        self.goal_spore = Spore(
            dt=0.1,
            pendulum=self.pendulum,
            goal_position=goal_position,
            position=goal_position.tolist(),
            color_manager=self.color_manager,
            is_goal=True,
            scale=2.0
        )
        self.zoom_manager.register_object(self.goal_spore, "goal_spore")
        
        print(f"   ✓ Целевая спора: {self.goal_spore.real_position}")
        print(f"   ✓ Логика 2D: {self.goal_spore.logic.position_2d}")
        
        # Создаем несколько стартовых спор в разных позициях
        start_positions = [
            [-3.0, 0.0, -2.0],
            [0.0, 0.0, -4.0],
            [-1.0, 0.0, 2.0],
            [2.0, 0.0, -1.0]
        ]
        
        self.demo_spores = []
        for i, pos in enumerate(start_positions):
            spore = Spore(
                dt=0.1,
                pendulum=self.pendulum,
                goal_position=goal_position,
                position=pos,
                color_manager=self.color_manager,
                is_goal=False,
                scale=1.0
            )
            
            # Добавляем в менеджер
            self.spore_manager.add_spore(spore)
            self.zoom_manager.register_object(spore, f"demo_spore_{i}")
            self.demo_spores.append(spore)
            
            print(f"   ✓ Спора {i+1}: 3D={spore.real_position}, 2D={spore.logic.position_2d}, cost={spore.cost:.2f}")
        
        print(f"   ✓ Создано {len(self.demo_spores)} демо-спор")
        
    def setup_demo_ui(self):
        """Настройка UI для демонстрации"""
        # Информация о архитектуре
        self.architecture_info = Text(
            text="НОВАЯ АРХИТЕКТУРА SPORE\nLogic (2D) + Visual (3D)\nРазделение ответственности",
            position=(-0.95, 0.9),
            scale=0.8,
            color=self.color_manager.get_color('ui', 'text_primary'),
            background=True,
            background_color=self.color_manager.get_color('ui', 'background_transparent')
        )
        
        # Счетчик спор
        self.spore_counter = Text(
            text=f"Спор: {len(self.demo_spores)}",
            position=(-0.95, 0.6),
            scale=0.8,
            color=self.color_manager.get_color('ui', 'text_secondary')
        )
        
        # Информация о стоимости
        self.cost_info = Text(
            text="",
            position=(-0.95, 0.4),
            scale=0.7,
            color=self.color_manager.get_color('ui', 'text_secondary')
        )
        
    def create_random_spore(self):
        """Создание случайной споры"""
        # Генерируем случайную позицию в 3D
        x = np.random.uniform(-5, 5)
        y = 0.0  # Держим на уровне пола
        z = np.random.uniform(-5, 5)
        
        new_spore = Spore(
            dt=0.1,
            pendulum=self.pendulum,
            goal_position=np.array([4.0, 0.0, 3.0]),
            position=[x, y, z],
            color_manager=self.color_manager,
            is_goal=False,
            scale=1.0
        )
        
        # Добавляем в систему
        self.spore_manager.add_spore(new_spore)
        self.zoom_manager.register_object(new_spore, f"random_spore_{len(self.demo_spores)}")
        self.demo_spores.append(new_spore)
        
        print(f"Новая спора: 3D=({x:.1f}, {y:.1f}, {z:.1f}), cost={new_spore.cost:.2f}")
        
    def evolve_all_spores(self):
        """Эволюция всех спор"""
        print("\nЭволюция всех спор...")
        for i, spore in enumerate(self.demo_spores):
            old_cost = spore.cost
            old_pos = spore.logic.position_2d.copy()
            
            # Эволюция через логику
            control = np.random.uniform(-1.0, 1.0)
            new_pos = spore.logic.evolve(control)
            
            # Синхронизация с визуализацией
            spore.sync_with_logic()
            
            print(f"   Спора {i+1}: {old_pos} -> {new_pos}, cost: {old_cost:.2f} -> {spore.cost:.2f}")
        
    def reset_spores(self):
        """Сброс всех спор к начальным позициям"""
        print("Сброс спор...")
        
        # Удаляем все текущие споры
        for spore in self.demo_spores:
            spore.remove_self()
        
        self.demo_spores.clear()
        self.spore_manager.objects.clear()
        
        # Пересоздаем
        self.create_demo_spores()
        
    def update_info(self):
        """Обновление информации на экране"""
        # Обновляем счетчик
        self.spore_counter.text = f"Спор: {len(self.demo_spores)}"
        
        # Обновляем информацию о стоимости
        if self.demo_spores:
            total_cost = sum(spore.cost for spore in self.demo_spores)
            avg_cost = total_cost / len(self.demo_spores)
            min_cost = min(spore.cost for spore in self.demo_spores)
            
            self.cost_info.text = f"Общая стоимость: {total_cost:.1f}\nСредняя: {avg_cost:.2f}\nМинимальная: {min_cost:.2f}"
        
    def input(self, key):
        """Обработка пользовательского ввода"""
        # Передаем в scene_setup для базовых команд
        self.scene.input_handler(key)
        
        # Наши команды
        if key == 'f':
            self.create_random_spore()
        elif key == 'g':
            self.evolve_all_spores()
        elif key == 'r':
            self.reset_spores()
            
        # Передаем в spore_manager
        self.spore_manager.input_handler(key)
        
    def update(self):
        """Обновление каждый кадр"""
        # Обновляем сцену
        self.scene.update(time.dt)
        
        # Обновляем информацию
        self.update_info()

# Запуск теста
if __name__ == "__main__":
    print("Тестирование новой архитектуры Spore в 3D...")
    print("Проверка: SporeLogic (2D математика) + SporeVisual (3D отображение)")
    print("(Закройте окно для завершения)")
    
    try:
        # Создаем экземпляр для регистрации методов update и input
        test = GraphicsTest3D()
        
        # Регистрируем глобальные обработчики
        def global_update():
            test.update()
            
        def global_input(key):
            test.input(key)
            
        # Привязываем к ursina
        app.update = global_update
        app.input = global_input
        
    except Exception as e:
        print(f"Ошибка в 3D тесте: {e}")
        import traceback
        traceback.print_exc() 