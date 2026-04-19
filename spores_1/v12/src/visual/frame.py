from ursina import Entity, scene
import os
import sys
from typing import Optional, List, Any
from ..utils.scalable import Scalable
from ..managers.color_manager import ColorManager
import numpy as np


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

class Frame(Entity):
    """
    Класс для отображения локальной системы координат (frame) в виде трех
    цветных стрелок.
    """
    def __init__(self, position=(0, 0, 0), color_manager=None, origin_scale: float = 0.05, **kwargs):
        if color_manager is None:
            color_manager = ColorManager()
        self.color_manager = color_manager
        
        super().__init__(
            position=position,
            **kwargs
        )
        
        self.parent = scene
        self.collider = None
        self.texture = None

        self.origin_cube: Scalable = Scalable(
            parent=self,
            model='cube',
            color=self.color_manager.get_color('frame', 'origin'),
            scale=origin_scale
        )

        self.x_axis: Scalable = Scalable(
            parent=self,
            model='arrow.obj',
            color=self.color_manager.get_color('frame', 'x_axis'),
            rotation=(0, 0, 90)
        )
        self.y_axis: Scalable = Scalable(
            parent=self,
            model='arrow.obj',
            color=self.color_manager.get_color('frame', 'y_axis'),
            rotation=(0, 90, 0)
        )
        self.z_axis: Scalable = Scalable(
            parent=self,
            model='arrow.obj',
            color=self.color_manager.get_color('frame', 'z_axis'),
            rotation=(90, 0, 0)
        )

        self.entities: List[Scalable] = [self.origin_cube, self.x_axis, self.y_axis, self.z_axis]
