from ursina import *
import numpy as np
from src.utils.scalable import Scalable
from src.core.spore import Spore
from src.managers.color_manager import ColorManager
from src.managers.zoom_manager import ZoomManager

class Link(Scalable):
    def __init__(self, 
                 parent_spore: Spore, 
                 child_spore: Spore, 
                 zoom_manager: ZoomManager,
                 color_manager=None, 
                 **kwargs):
        self.thickness = 0.5
        # Используем переданный ColorManager или создаем новый
        if color_manager is None:
            color_manager = ColorManager()
        self.color_manager = color_manager
        self.zoom_manager = zoom_manager

        self.parent_spore = parent_spore
        self.child_spore = child_spore

        super().__init__(
            model='models/arrow.obj',
            position=(0, 0, 0),  # Временная позиция, будет обновлена в update_geometry
            color=self.color_manager.get_color('link', 'default'),
            **kwargs
        )
        
        # Используем общий метод для настройки геометрии

        self.update_geometry()

    def update_geometry(self):
        """Обновляет позицию, ориентацию и масштаб линка между двумя точками"""
        ball_size = self.parent_spore.real_scale[0] * self.zoom_manager.spores_scale
        
        distance = np.linalg.norm(self.child_spore.real_position - self.parent_spore.real_position)  
        
        # Защита от деления на ноль и очень малых расстояний
        min_distance = ball_size * 0.1  # Минимальное расстояние
        if distance < min_distance:
            # Если споры слишком близко, делаем линк невидимым или очень маленьким
            self.visible = False
            return
        else:
            self.visible = True
        
        k = 1.1
        k1 = 2
        alpha = ball_size / distance / 2

        real_pos = (1-alpha) * self.parent_spore.real_position + (alpha) * self.child_spore.real_position

        arrow_width = self.thickness * self.zoom_manager.spores_scale
        real_scale = (arrow_width, (distance - ball_size) , arrow_width)
        
        # Проверяем на NaN перед установкой позиции
        if np.any(np.isnan(real_pos)):
            print(f"WARNING: NaN detected in Link position, hiding link")
            self.visible = False
            return
        
        # Сначала устанавливаем позицию в мировых координатах
        self.position = tuple(real_pos)
        
        # Затем ориентируем стрелку от parent к child
        # The arrow's "forward" is its Y-axis.
        self.look_at(Vec3(*self.child_spore.real_position), axis='up')
        
        # После этого устанавливаем real координаты и масштаб
        self.real_scale = np.array(real_scale)
        self.real_position = real_pos 