from ursina import *
import numpy as np

class Scalable(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.real_position = np.array(self.position)
        self.real_scale = np.array(self.scale)

    def apply_transform(self, a, b, **kwargs):
        self.position = self.real_position * a + b
        self.scale = self.real_scale * a

    def __str__(self):
        return f'{self.position}'

    def __repr__(self):
        return f'{self.position}'
    

class ScalableFrame(Scalable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ScalableFloor(Scalable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Link(Scalable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    

