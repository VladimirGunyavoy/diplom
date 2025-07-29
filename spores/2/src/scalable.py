from ursina import *
import numpy as np

class Scalable(Entity):
    def __init__(self, is_frame=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zoom_fact = 1
        self.is_frame = is_frame

        self.original_scale = self.scale
        self.original_position = self.position

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
        
    

