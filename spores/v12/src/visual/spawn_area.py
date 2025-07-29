from ursina import Entity, color
import numpy as np
from typing import Optional

from ...utils.scalable import Scalable
from ...managers.color_manager import ColorManager
from ...visual.ellipse import Ellipse

class SpawnAreaVisual(Scalable):
    """
    Класс для визуализации области спавна в виде эллипса.
    """
    def __init__(self, 
                 center: tuple = (0, 0, 0), 
                 a: float = 1.0, 
                 b: float = 0.5, 
                 color_manager: Optional[ColorManager] = None, 
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.color_manager = color_manager if color_manager is not None else ColorManager()
        
        self.ellipse = Ellipse(
            a=a,
            b=b,
            thickness=3,
            color=self.color_manager.get_color('spawn_area', 'ellipse'),
            parent=self,
            position=center
        )

    def update_visuals(self, center: np.ndarray, a: float, b: float, rotation_matrix: np.ndarray) -> None:
        """
        Обновляет параметры эллипса.
        """
        self.ellipse.a = a
        self.ellipse.b = b
        self.position = tuple(center)
        
        # Устанавливаем вращение. Ursina использует градусы.
        angle_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        self.rotation_y = np.degrees(angle_rad)