from ursina import *
import os
import sys
from src.utils.scalable import Scalable
from src.managers.color_manager import ColorManager


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

class Frame(Scalable):
    """
    Класс для отображения локальной системы координат (frame) в виде трех
    цветных стрелок. Наследуется от Scalable для поддержки масштабирования.
    """
    def __init__(self, 
                 parent=scene, 
                 scale=1, 
                 thickness=1,
                 color_manager=None,
                 **kwargs):

        super().__init__(parent=parent, **kwargs)

        if color_manager is None:
            color_manager = ColorManager()
        self.color_manager = color_manager

        self.origin_cube = Scalable(
            parent=self,
            model='cube',
            color=self.color_manager.get_color('frame', 'origin'),
            scale=1/20
        )

        self.x_axis = Scalable(
            parent=self,
            model='arrow.obj',
            color=self.color_manager.get_color('frame', 'x_axis'),
            rotation=(0, 0, 90)
        )
        self.y_axis = Scalable(
            parent=self,
            model='arrow.obj',
            color=self.color_manager.get_color('frame', 'y_axis'),
            rotation=(0, 90, 0) # Эта ось уже направлена вверх
        )
        self.z_axis = Scalable(
            parent=self,
            model='arrow.obj',
            color=self.color_manager.get_color('frame', 'z_axis'),
            rotation=(90, 0, 0)
        )

        self.entities = [self.origin_cube, self.x_axis, self.y_axis, self.z_axis]

        self.scale = (scale, scale, scale)
        self.thickness = thickness

    def apply_transform(self, a, b, spores_scale=None):
        """Применяет трансформацию ко всем элементам Frame."""
        for entity in self.entities:
            # spores_scale не влияет на Frame, поэтому мы его не передаем
            entity.apply_transform(a, b)
