from ursina import *
import os
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.scalable import Scalable

class Frame:
    def __init__(self, position=(0, 0, 0)):
        self.entities = []  # Store all created entities
        
        self.origin = Scalable(model='cube', 
                        color=color.white, 
                        scale=1/20,
                        position=position)
        self.entities.append(self.origin)
        print(os.getcwd())

        self.x_axis = Scalable(model='models/arrow.obj', 
                        color=color.rgb(0.863, 0.196, 0.184),  # Стильный красный
                        scale=1,
                        position=position,
                        rotation=(0, 0, 90))
        self.entities.append(self.x_axis)

        
        self.y_axis = Scalable(model='models/arrow.obj', 
                        color=color.rgb(0.2, 0.7, 0.25),  # Чистый зеленый
                        scale=1,
                        position=position,
                        rotation=(0, 90, 0))
        self.entities.append(self.y_axis)
        
        self.z_axis = Scalable(model='models/arrow.obj', 
                        color=color.rgb(0.149, 0.545, 0.824),  # Стильный синий
                        scale=1,
                        position=position,
                        rotation=(90, 0, 0))
        self.entities.append(self.z_axis)
