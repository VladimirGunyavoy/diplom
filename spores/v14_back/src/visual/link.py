from ursina import Vec3
import numpy as np
from typing import Optional, Dict, Any
from ..utils.scalable import Scalable
from ..core.spore import Spore
from ..managers.color_manager import ColorManager
from ..managers.zoom_manager import ZoomManager

class Link(Scalable):
    def __init__(self, 
                 parent_spore: Spore, 
                 child_spore: Spore, 
                 zoom_manager: ZoomManager,
                 color_manager: Optional[ColorManager] = None, 
                 config: Optional[Dict[str, Any]] = None,
                 link_type: str = 'default',
                 **kwargs):
        self.config: Dict[str, Any] = config if config is not None else {}
        
        thickness: float
        if link_type == 'angel':
            thickness = self.config.get('angel', {}).get('link_thickness', 1)
        else:
            thickness = self.config.get('link', {}).get('thickness', 1)
        self.thickness = thickness
            
        # Используем переданный ColorManager или создаем новый
        self.color_manager: ColorManager = color_manager if color_manager is not None else ColorManager()
        self.zoom_manager: ZoomManager = zoom_manager

        self.parent_spore: Spore = parent_spore
        self.child_spore: Spore = child_spore

        super().__init__(
            model='models/arrow.obj',
            position=(0, 0, 0),  # Временная позиция, будет обновлена в update_geometry
            color=self.color_manager.get_color('link', 'default'),
            render_queue=0,
            thickness=self.thickness,
            **kwargs
        )
        
        # Используем общий метод для настройки геометрии

        self.update_geometry()

    def update_geometry(self) -> None:
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
        # self.rotation_x = 0
        
        # После этого устанавливаем real координаты и масштаб
        self.real_scale = np.array(real_scale)
        self.real_position = real_pos
        
    def apply_transform(self, a: float, b: np.ndarray, **kwargs) -> None:
        spores_scale = kwargs.get('spores_scale', 1.0)
        self.position = self.real_position * a + b
        self.scale = self.real_scale * a 