import numpy as np
from ursina import *
from src.utils.scalable import Scalable
from src.managers.color_manager import ColorManager
from src.core.spore_logic import SporeLogic

class SporeVisual(Scalable):
    """
    Класс для визуализации споры.
    Наследуется от Scalable и отвечает за:
    - 3D визуальное представление
    - Управление цветом через ColorManager
    - Конвертацию между 2D логикой и 3D визуализацией
    - Интеграцию с системой трансформаций (zoom, scale)
    """
    
    def __init__(self, model='sphere', color_manager=None, is_goal=False, 
                 y_coordinate=0.0, *args, **kwargs):
        """
        Инициализация визуальной части споры.
        
        Args:
            model: 3D модель для отображения
            color_manager: Менеджер цветов
            is_goal: Является ли спора целевой (влияет на цвет)
            y_coordinate: Y координата для 3D визуализации (обычно 0)
        """
        super().__init__(model=model, *args, **kwargs)
        
        # Управление цветом
        if color_manager is None:
            color_manager = ColorManager()
        self.color_manager = color_manager
        
        # Устанавливаем цвет в зависимости от типа споры
        color_type = 'goal' if is_goal else 'default'
        self.color = self.color_manager.get_color('spore', color_type)
        self.is_goal = is_goal
        
        # Y координата для 3D визуализации (математика работает в XZ плоскости)
        self.y_coordinate = y_coordinate
    
    def apply_transform(self, a, b, **kwargs):
        """
        Переопределяем для поддержки spores_scale.
        Применяет трансформации масштабирования и смещения от ZoomManager.
        
        Args:
            a: Коэффициент масштабирования
            b: Вектор смещения  
            **kwargs: Дополнительные параметры (spores_scale)
        """
        spores_scale = kwargs.get('spores_scale', 1.0)
        self.position = self.real_position * a + b
        self.scale = self.real_scale * a * spores_scale
    
    def sync_with_logic(self, spore_logic: SporeLogic):
        """
        Синхронизирует визуальное состояние с логическим.
        Конвертирует 2D позицию логики в 3D визуальную позицию.
        
        Args:
            spore_logic: Экземпляр SporeLogic для синхронизации
        """
        pos_2d = spore_logic.get_position_2d()
        # Преобразуем 2D логическую позицию в 3D визуальную: (x, y, z)
        # 2D: (x, z) -> 3D: (x, y_coordinate, z)
        self.real_position = np.array([pos_2d[0], self.y_coordinate, pos_2d[1]])
    
    def get_logic_position_2d(self) -> np.array:
        """
        Извлекает 2D позицию из 3D визуальной позиции для передачи в логику.
        
        Returns:
            2D позиция (x, z) для использования в SporeLogic
        """
        return np.array([self.real_position[0], self.real_position[2]])
    
    def set_logic_position_2d(self, position_2d: np.array):
        """
        Устанавливает 2D логическую позицию, обновляя 3D визуальную.
        
        Args:
            position_2d: Новая 2D позиция (x, z)
        """
        position_2d = np.array(position_2d, dtype=float)
        if len(position_2d) != 2:
            raise ValueError(f"position_2d должна быть 2D, получена {len(position_2d)}D")
        
        # Обновляем real_position, сохраняя Y координату
        self.real_position = np.array([position_2d[0], self.y_coordinate, position_2d[1]])
    
    def set_color_type(self, color_type: str):
        """
        Изменяет тип цвета споры.
        
        Args:
            color_type: Тип цвета ('default', 'ghost', 'goal')
        """
        self.color = self.color_manager.get_color('spore', color_type)
    
    def set_y_coordinate(self, y: float):
        """
        Изменяет Y координату для визуализации.
        
        Args:
            y: Новая Y координата
        """
        old_pos_2d = self.get_logic_position_2d()
        self.y_coordinate = y
        self.set_logic_position_2d(old_pos_2d)  # Пересчитываем 3D позицию
    
    def __str__(self):
        pos_2d = self.get_logic_position_2d()
        return f"SporeVisual(pos_2d={pos_2d}, y={self.y_coordinate}, is_goal={self.is_goal})"
    
    def __repr__(self):
        return self.__str__() 