import math
from ursina import *
from .scalable import Scalable

class Ellipse(Scalable):
    def __init__(self, a=1.0, b=0.5, resolution=32, mode='line', thickness=1, **kwargs):
        # Extract mesh-specific kwargs
        mesh_kwargs = {}
        if 'thickness' in kwargs or thickness != 1:
            mesh_kwargs['thickness'] = thickness
        
        # Initialize parent Scalable class
        super().__init__(**kwargs)
        
        self.vertices = []
        
        # Generate vertices using parametric equations
        for i in range(resolution):
            t = (2 * math.pi * i) / resolution
            x = a * math.cos(t)  # Semi-major axis
            y = b * math.sin(t)  # Semi-minor axis
            self.vertices.append((x, 0, y))
        
        # Close the shape for line mode
        if mode == 'line':
            self.vertices.append(self.vertices[0])
        
        # Create mesh as a child entity
        self.mesh_entity = Entity(
            parent=self,
            model=Mesh(vertices=self.vertices, mode=mode, **mesh_kwargs)
        )