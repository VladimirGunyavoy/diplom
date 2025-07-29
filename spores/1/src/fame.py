from ursina import *

class Frame:
    def __init__(self, position=(0, 0, 0)):
        self.entities = []  # Store all created entities
        
        self.origin = Entity(model='cube', 
                        color=color.white, 
                        scale=1/20,
                        position=position)
        self.entities.append(self.origin)

        self.x_axis = Entity(model='src/models/arrow.obj', 
                        color=color.red, 
                        scale=1,
                        position=position,
                        rotation=(0, 0, 90))
        self.entities.append(self.x_axis)

        
        self.y_axis = Entity(model='src/models/arrow.obj', 
                        color=color.green, 
                        scale=1,
                        position=position,
                        rotation=(0, 90, 0))
        self.entities.append(self.y_axis)
        
        self.z_axis = Entity(model='src/models/arrow.obj', 
                        color=color.blue, 
                        scale=1,
                        position=position,
                        rotation=(90, 0, 0))
        self.entities.append(self.z_axis)
