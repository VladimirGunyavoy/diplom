from ursina import Vec3
import numpy as np
from typing import Optional, Dict, Any
from ..utils.scalable import Scalable
from ..core.spore import Spore
from ..managers.color_manager import ColorManager
from ..managers.zoom_manager import ZoomManager

class Link(Scalable):
    def __init__(self, 
                 parent_spore: Spore, 
                 child_spore: Spore, 
                 zoom_manager: ZoomManager,
                 color_manager: Optional[ColorManager] = None, 
                 config: Optional[Dict[str, Any]] = None,
                 link_type: str = 'default',
                 max_length: Optional[float] = None,
                 **kwargs):
        self.config: Dict[str, Any] = config if config is not None else {}
        
        thickness: float
        if link_type == 'angel':
            thickness = self.config.get('angel', {}).get('link_thickness', 1)
        else:
            thickness = self.config.get('link', {}).get('thickness', 1)
        self.thickness = thickness
            
        # Используем переданный ColorManager или создаем новый
        self.color_manager: ColorManager = color_manager if color_manager is not None else ColorManager()
        self.zoom_manager: ZoomManager = zoom_manager

        self.parent_spore: Spore = parent_spore
        self.child_spore: Spore = child_spore

        self.max_length: Optional[float] = max_length

        super().__init__(
            model='models/arrow.obj',
            position=(0, 0, 0),  # Временная позиция, будет обновлена в update_geometry
            color=self.color_manager.get_color('link', 'default'),
            render_queue=0,
            thickness=self.thickness,
            **kwargs
        )
        
        # Используем общий метод для настройки геометрии

        self.update_geometry()

    def set_max_length(self, max_length: Optional[float]) -> None:
        """Устанавливает максимальную визуальную длину линка и сразу обновляет геометрию."""
        self.max_length = max_length
        self.update_geometry()

    def update_geometry(self) -> None:
        """Обновляет позицию, ориентацию и масштаб линка между двумя точками с учетом max_length."""
        ball_size = self.parent_spore.real_scale[0] * self.zoom_manager.spores_scale

        parent_pos = self.parent_spore.real_position
        child_pos = self.child_spore.real_position
        vec = child_pos - parent_pos
        distance = float(np.linalg.norm(vec))

        # Защита от слишком малых расстояний
        min_distance = ball_size * 0.1
        if distance < 1e-12:
            self.visible = False
            return

        # Применяем ограничение максимальной длины, если есть
        effective_length = distance
        if self.max_length is not None:
            effective_length = min(effective_length, float(self.max_length))

        if effective_length < min_distance:
            self.visible = False
            return
        else:
            self.visible = True

        # Направление и точка "усеченной" цели
        direction = vec / distance
        target_pos = parent_pos + direction * effective_length

        # Сдвиг основания стрелки от родителя на ~ ball_size/2, как раньше
        alpha = ball_size / effective_length / 2.0
        real_pos = (1.0 - alpha) * parent_pos + alpha * target_pos

        arrow_width = self.thickness * self.zoom_manager.spores_scale
        # высота = эффективная длина минус диаметр касания с шарами
        real_scale = (arrow_width, max(effective_length - ball_size, min_distance), arrow_width)

        if np.any(np.isnan(real_pos)):
            print(f"WARNING: NaN detected in Link position, hiding link")
            self.visible = False
            return

        # Устанавливаем позицию/ориентацию/масштаб
        self.position = tuple(real_pos)
        self.look_at(Vec3(*target_pos), axis='up')

        self.real_scale = np.array(real_scale)
        self.real_position = real_pos
        
    def apply_transform(self, a: float, b: np.ndarray, **kwargs) -> None:
        spores_scale = kwargs.get('spores_scale', 1.0)
        self.position = self.real_position * a + b
        self.scale = self.real_scale * a