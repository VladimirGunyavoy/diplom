from ursina import *
import os
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.utils.scalable import Scalable
from src.managers.color_manager import ColorManager

class Frame:
    def __init__(self, position=(0, 0, 0), color_manager=None):
        self.entities = []  # Store all created entities
        
        # Используем переданный ColorManager или создаем новый
        if color_manager is None:
            color_manager = ColorManager()
        self.color_manager = color_manager
        
        self.origin = Scalable(model='cube', 
                        color=self.color_manager.get_color('frame', 'origin'), 
                        scale=1/20,
                        position=position)
        self.entities.append(self.origin)
        print(os.getcwd())

        self.x_axis = Scalable(model='models/arrow.obj', 
                        color=self.color_manager.get_color('frame', 'x_axis'),
                        scale=1,
                        position=position,
                        rotation=(0, 0, 90))
        self.entities.append(self.x_axis)

        
        self.y_axis = Scalable(model='models/arrow.obj', 
                        color=self.color_manager.get_color('frame', 'y_axis'),
                        scale=1,
                        position=position,
                        rotation=(0, 90, 0))
        self.entities.append(self.y_axis)
        
        self.z_axis = Scalable(model='models/arrow.obj', 
                        color=self.color_manager.get_color('frame', 'z_axis'),
                        scale=1,
                        position=position,
                        rotation=(90, 0, 0))
        self.entities.append(self.z_axis)

    def apply_transform(self, a, b, spores_scale=None):
        """Применяет трансформацию ко всем элементам Frame."""
        for entity in self.entities:
            # spores_scale не влияет на Frame, поэтому мы его не передаем
            entity.apply_transform(a, b)
