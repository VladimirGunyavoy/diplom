from ursina import *
import numpy as np
import sys
import os

# Добавляем путь к корневой директории проекта (spores/1)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Импортируем необходимые классы
from src.pendulum import PendulumSystem
from src.scene_setup import SceneSetup
from src.fame import Frame

class ZoomManager:
    """Центральный менеджер системы зума"""
    
    def __init__(self):
        # Текущие значения зума по осям
        self.zoom_x = 1.0
        self.zoom_y = 1.0
        self.zoom_z = 1.0
        
        # Целевые значения для плавной анимации
        self.target_zoom_x = 1.0
        self.target_zoom_y = 1.0
        self.target_zoom_z = 1.0
        
        # Настройки
        self.zoom_speed = 1.0  # Скорость изменения зума (1.0 = 100% в секунду)
        self.zoom_limits = (0.001, 1000.0)  # Минимальный и максимальный зум
        
        # Список объектов для масштабирования
        self.scalable_objects = []
        
    def set_target_zoom(self, axis, zoom_factor):
        """Устанавливает целевой зум для указанной оси"""
        zoom_factor = max(self.zoom_limits[0], min(self.zoom_limits[1], zoom_factor))
        
        if axis == 'x':
            self.target_zoom_x = zoom_factor
        elif axis == 'y':
            self.target_zoom_y = zoom_factor
        elif axis == 'z':
            self.target_zoom_z = zoom_factor
            
    def update(self, dt):
        """Обновляет зум с плавной анимацией"""
        # Плавное приближение к целевым значениям
        self.zoom_x = lerp(self.zoom_x, self.target_zoom_x, self.zoom_speed * dt)
        self.zoom_y = lerp(self.zoom_y, self.target_zoom_y, self.zoom_speed * dt)
        self.zoom_z = lerp(self.zoom_z, self.target_zoom_z, self.zoom_speed * dt)
        
        # Обновляем все зарегистрированные объекты
        self._update_all_objects()
    
    def register_scalable(self, scalable_object):
        """Регистрирует объект для масштабирования"""
        self.scalable_objects.append(scalable_object)
        
    def unregister_scalable(self, scalable_object):
        """Удаляет объект из списка масштабируемых"""
        if scalable_object in self.scalable_objects:
            self.scalable_objects.remove(scalable_object)
    
    def _update_all_objects(self):
        """Обновляет трансформации всех зарегистрированных объектов"""
        for obj in self.scalable_objects:
            obj.update_visual_transform(self.zoom_x, self.zoom_y, self.zoom_z)
    
    def get_camera_speed_multiplier(self):
        """Возвращает множитель скорости камеры"""
        # Используем среднее значение зума для расчета скорости
        avg_zoom = (self.zoom_x + self.zoom_z) / 2.0
        return 1.0 / max(avg_zoom, 0.001)  # Избегаем деления на ноль
    
    def get_logical_position(self, camera_position):
        """Преобразует визуальную позицию камеры в логические координаты"""
        return Vec3(
            camera_position.x / self.zoom_x,  # θ
            camera_position.y / self.zoom_y,  # высота
            camera_position.z / self.zoom_z   # ω
        )
    
    def debug_print_state(self):
        """Выводит текущее состояние системы зума"""
        print(f"=== ZOOM DEBUG ===")
        print(f"Current zoom: X={self.zoom_x:.3f} Y={self.zoom_y:.3f} Z={self.zoom_z:.3f}")
        print(f"Target zoom:  X={self.target_zoom_x:.3f} Y={self.target_zoom_y:.3f} Z={self.target_zoom_z:.3f}")
        print(f"Registered objects: {len(self.scalable_objects)}")
        print(f"Speed multiplier: {self.get_camera_speed_multiplier():.3f}")

class ScalableObject:
    """Базовый класс для всех масштабируемых объектов"""
    
    def __init__(self, logical_position, entity):
        # Логические координаты (исходные, неизменные)
        self.logical_position = Vec3(logical_position)
        
        # Ursina Entity для визуального отображения
        self.entity = entity
        
        # Флаги масштабирования (по умолчанию все оси)
        self.scale_x = True
        self.scale_y = True
        self.scale_z = True
        
        # Исходный размер объекта
        self.original_scale = entity.scale if hasattr(entity, 'scale') else Vec3(1, 1, 1)
        
    def update_visual_transform(self, zoom_x, zoom_y, zoom_z):
        """Обновляет визуальную позицию и масштаб объекта"""
        # Рассчитываем новую визуальную позицию
        new_position = Vec3(
            self.logical_position.x * zoom_x if self.scale_x else self.logical_position.x,
            self.logical_position.y * zoom_y if self.scale_y else self.logical_position.y,
            self.logical_position.z * zoom_z if self.scale_z else self.logical_position.z
        )
        
        # Обновляем позицию entity
        self.entity.position = new_position
        
        # Опционально: масштабируем размер объекта (для точек, связей)
        if hasattr(self.entity, 'scale'):
            scale_factor = Vec3(
                zoom_x if self.scale_x else 1.0,
                zoom_y if self.scale_y else 1.0,
                zoom_z if self.scale_z else 1.0
            )
            # Применяем масштаб, но с ограничениями для читаемости
            final_scale = Vec3(
                self.original_scale.x * min(scale_factor.x, 10.0),
                self.original_scale.y * min(scale_factor.y, 10.0), 
                self.original_scale.z * min(scale_factor.z, 10.0)
            )
            self.entity.scale = final_scale
    
    def set_logical_position(self, new_logical_position):
        """Обновляет логическую позицию объекта"""
        self.logical_position = Vec3(new_logical_position)

class ZoomScene:
    """Сцена с поддержкой зума"""
    
    def __init__(self):
        # Создаем базовую сцену
        self.scene = SceneSetup()
        
        # Создаем систему координат
        self.frame = Frame(position=(0, 0, 0))
        
        # Создаем менеджер зума
        self.zoom_manager = ZoomManager()
        
        # Создаем тестовые объекты
        self.create_test_objects()
        
        # Создаем UI для отображения информации о зуме
        self.create_zoom_ui()
        
    def create_test_objects(self):
        """Создает тестовые объекты для проверки зума"""
        # Создаем тестовый куб
        cube_entity = Entity(
            model='cube',
            color=color.orange,
            scale=0.5,
            position=(1, 0, 1)
        )
        self.test_cube = ScalableObject((1, 0, 1), cube_entity)
        self.zoom_manager.register_scalable(self.test_cube)
        
        # Создаем тестовую сферу
        sphere_entity = Entity(
            model='sphere',
            color=color.cyan,
            scale=0.3,
            position=(-1, 0, 2)
        )
        self.test_sphere = ScalableObject((-1, 0, 2), sphere_entity)
        self.zoom_manager.register_scalable(self.test_sphere)
        
        # Создаем систему координат с увеличенным размером
        self.frame = Frame(position=(0, 0, 0), axis_scale=3.0)
        
        # Регистрируем каждую ось отдельно
        for entity in self.frame.entities:
            if entity != self.frame.origin:  # Не масштабируем точку начала координат
                scalable = ScalableObject((0, 0, 0), entity)
                # Отключаем масштабирование по осям, которые не соответствуют направлению стрелки
                if entity == self.frame.x_axis:
                    scalable.scale_y = False
                    scalable.scale_z = False
                elif entity == self.frame.y_axis:
                    scalable.scale_x = False
                    scalable.scale_z = False
                elif entity == self.frame.z_axis:
                    scalable.scale_x = False
                    scalable.scale_y = False
                self.zoom_manager.register_scalable(scalable)
    
    def create_zoom_ui(self):
        """Создает UI элементы для отображения информации о зуме"""
        # Создаем контейнер для UI элементов
        self.ui_container = Entity(parent=camera.ui)
        
        self.zoom_info = Text(
            text="Zoom: X=1.0 Y=1.0 Z=1.0",
            position=(-0.5, 0.45),
            scale=0.7,
            color=color.white,
            background=True,
            background_color=color.rgba(0, 0, 0, 0.7),
            parent=self.ui_container
        )
        
        self.coords_info = Text(
            text="Position: θ=0.0 h=0.0 ω=0.0",
            position=(-0.5, 0.4),
            scale=0.7,
            color=color.yellow,
            background=True,
            background_color=color.rgba(0, 0, 0, 0.7),
            parent=self.ui_container
        )
        
        self.controls_info = Text(
            text="Zoom: left/right (θ), up/down (ω), page up/down (h), home (reset), F1 (debug)",
            position=(-0.5, 0.35),
            scale=0.6,
            color=color.gray,
            background=True,
            background_color=color.rgba(0, 0, 0, 0.7),
            parent=self.ui_container
        )
    
    def update(self, dt):
        """Обновление сцены"""
        # Обновляем базовую сцену
        self.scene.update(dt)
        
        # Обработка управления зумом
        self.handle_zoom_input(dt)
        
        # Обновление зума
        self.zoom_manager.update(dt)
        
        # Обновление UI
        self.update_zoom_ui()
        
        # Адаптация скорости камеры
        base_speed = 5.0
        self.scene.player.speed = base_speed * self.zoom_manager.get_camera_speed_multiplier()
    
    def handle_zoom_input(self, dt):
        """Обработка ввода для управления зумом"""
        zoom_step = 0.8  # 10% изменение при нажатии
        
        # Зум по оси X (θ)
        if held_keys['left arrow']:
            current = self.zoom_manager.target_zoom_x
            self.zoom_manager.set_target_zoom('x', current * (1 - zoom_step * dt))
            
        if held_keys['right arrow']:
            current = self.zoom_manager.target_zoom_x
            self.zoom_manager.set_target_zoom('x', current * (1 + zoom_step * dt))
        
        # Зум по оси Z (ω)
        if held_keys['down arrow']:
            current = self.zoom_manager.target_zoom_z
            self.zoom_manager.set_target_zoom('z', current * (1 - zoom_step * dt))
            
        if held_keys['up arrow']:
            current = self.zoom_manager.target_zoom_z
            self.zoom_manager.set_target_zoom('z', current * (1 + zoom_step * dt))
        
        # Зум по оси Y (высота)
        if held_keys['page up']:
            current = self.zoom_manager.target_zoom_y
            self.zoom_manager.set_target_zoom('y', current * (1 + zoom_step * dt))
            
        if held_keys['page down']:
            current = self.zoom_manager.target_zoom_y
            self.zoom_manager.set_target_zoom('y', current * (1 - zoom_step * dt))
        
        # Сброс зума
        if held_keys['home']:
            self.zoom_manager.set_target_zoom('x', 1.0)
            self.zoom_manager.set_target_zoom('y', 1.0)
            self.zoom_manager.set_target_zoom('z', 1.0)
    
    def update_zoom_ui(self):
        """Обновляет информацию о зуме в UI"""
        # Информация о зуме
        self.zoom_info.text = f"Zoom: X={self.zoom_manager.zoom_x:.2f} Y={self.zoom_manager.zoom_y:.2f} Z={self.zoom_manager.zoom_z:.2f}"
        
        # Логические координаты камеры
        logical_pos = self.zoom_manager.get_logical_position(self.scene.player.position)
        self.coords_info.text = f"Position: θ={logical_pos.x:.2f} h={logical_pos.y:.2f} ω={logical_pos.z:.2f}"
    
    def input(self, key):
        """Обработка ввода"""
        # Передаем управление базовой сцене
        self.scene.input_handler(key)
        
        # Обработка отладочной информации
        if key == 'f1':
            self.zoom_manager.debug_print_state()

# Создаем приложение
app = Ursina()

# Создаем сцену
scene = ZoomScene()

def update():
    scene.update(time.dt)

def input(key):
    scene.input(key)

# Запускаем приложение
app.run() 