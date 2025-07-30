from ursina import Entity, Mesh, color
import numpy as np
from typing import Optional

from ..logic.spawn_area import SpawnArea
from ..utils.scalable import Scalable

class SpawnAreaVisualizer(Scalable, Entity):
    """
    Класс для визуализации границы SpawnArea в Ursina.
    Принимает на вход логический объект SpawnArea.
    """
    def __init__(self, 
                 spawn_area: SpawnArea, 
                 resolution: int = 64, 
                 mode: str = 'line', 
                 thickness: int = 2, 
                 **kwargs):
        # цвет извлекается из kwargs, чтобы не передавать его в Entity
        visual_color = kwargs.pop('color', color.white)
        
        super().__init__(**kwargs)
        self.spawn_area: SpawnArea = spawn_area
        self.resolution: int = resolution
        self.mode: str = mode
        self.thickness: int = thickness
        self.color = visual_color # Используем извлеченный цвет

        self._draw_boundary()

    def _draw_boundary(self) -> None:
        """
        Получает точки у логического объекта и создает на их основе
        3D-модель (линию) в Ursina.
        """
        points_2d = self.spawn_area.get_points(n_points=self.resolution)
        
        # Конвертируем 2D точки (x, y) в 3D для Ursina (x, 0, y)
        vertices_3d = [(p[0], 0, p[1]) for p in points_2d]

        # Для режима 'line' нужно замкнуть петлю
        if self.mode == 'line' and len(vertices_3d) > 1:
            vertices_3d.append(vertices_3d[0])

        # Создаем модель
        self.model = Mesh(
            vertices=vertices_3d, 
            mode=self.mode, 
            thickness=self.thickness
        )
        self.color = self.color

    def update_visuals(self, new_spawn_area: Optional[SpawnArea] = None) -> None:
        """
        Обновляет визуализацию, если логический объект изменился.
        """
        if new_spawn_area:
            self.spawn_area = new_spawn_area
        
        # Пересоздаем модель без явного уничтожения
        self._draw_boundary() 