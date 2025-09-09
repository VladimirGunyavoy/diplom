from typing import Tuple
import numpy as np
from ursina import camera, mouse
from ...managers.zoom_manager import ZoomManager
from ...managers.color_manager import ColorManager
from ...logic.pendulum import PendulumSystem


class SharedDependencies:
    """
    Общие зависимости для всех manual creation компонентов.
    Убирает дублирование и централизует общие вычисления.
    """
    
    def __init__(self, 
                 zoom_manager: ZoomManager, 
                 color_manager: ColorManager, 
                 pendulum: PendulumSystem, 
                 config: dict):
        self.zoom_manager = zoom_manager
        self.color_manager = color_manager
        self.pendulum = pendulum
        self.config = config
        
        # Кэшируем границы управления (вычисляются один раз)
        control_bounds = pendulum.get_control_bounds()
        self.min_control = float(control_bounds[0])
        self.max_control = float(control_bounds[1])
        
        # Добавляем spore_manager (будет установлен в ManualSporeManager)
        self.spore_manager = None
        
    def get_dt(self) -> float:
        """Получает текущий dt из конфигурации."""
        return self.config.get('pendulum', {}).get('dt', 0.1)
        
    def get_cursor_position_2d(self) -> np.ndarray:
        """
        ЦЕНТРАЛИЗОВАННОЕ вычисление позиции курсора.
        Переносим логику из ManualSporeManager.get_mouse_world_position()
        """
        try:
            # 1. Получаем точку взгляда камеры (без зума)
            look_point_x, look_point_z = self.zoom_manager.identify_invariant_point()

            # 2. Получаем позицию origin_cube из frame
            frame = getattr(self.zoom_manager.scene_setup, 'frame', None)
            if frame and hasattr(frame, 'origin_cube'):
                origin_pos = frame.origin_cube.position
                if hasattr(origin_pos, 'x'):
                    origin_x, origin_z = origin_pos.x, origin_pos.z
                else:
                    origin_x, origin_z = origin_pos[0], origin_pos[2]
            else:
                origin_x, origin_z = 0.0, 0.0

            # 3. Получаем масштаб трансформации
            transform_scale = getattr(self.zoom_manager, 'a_transformation', 1.0)

            # 4. ПРАВИЛЬНАЯ ФОРМУЛА: (look_point - frame_origin) / scale
            corrected_x = (look_point_x - origin_x) / transform_scale
            corrected_z = (look_point_z - origin_z) / transform_scale

            return np.array([corrected_x, corrected_z], dtype=float)

        except Exception as e:
            print(f"⚠️ Ошибка вычисления позиции курсора: {e}")
            return np.array([0.0, 0.0], dtype=float)
