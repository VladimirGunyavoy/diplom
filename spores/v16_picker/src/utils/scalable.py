from ursina import Entity
import numpy as np
from typing import Any

class Scalable(Entity):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.real_position: np.ndarray = np.array(self.position)
        self.real_scale: np.ndarray = np.array(self.scale)

    def apply_transform(self, a: float, b: np.ndarray, **kwargs: Any) -> None:
        # 🔍 ОТЛАДКА ТРАНСФОРМАЦИИ
        if hasattr(self, 'id') and self.id and 'tree_spore' in str(self.id):
            print(f"🔧 ТРАНСФОРМАЦИЯ для {self.id}:")
            print(f"   real_position: {self.real_position}")
            print(f"   a (масштаб): {a}")
            print(f"   b (смещение): {b}")
            old_pos = self.position.copy() if hasattr(self.position, 'copy') else self.position
            self.position = self.real_position * a + b
            print(f"   позиция: {old_pos} → {self.position}")
        
        self.position = self.real_position * a + b
        self.scale = self.real_scale * a

    def __str__(self) -> str:
        return f'{self.position}'

    def __repr__(self) -> str:
        return f'{self.position}'
    

class ScalableFrame(Scalable):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

class ScalableFloor(Scalable):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

class Link(Scalable):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        
    

