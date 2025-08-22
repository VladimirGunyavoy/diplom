from typing import Optional, Tuple
import numpy as np
from ...managers.zoom_manager import ZoomManager


class CursorTracker:
    """
    Класс для отслеживания позиции курсора мыши в 3D пространстве.
    Конвертирует 2D координаты курсора в корректные 3D координаты сцены.
    """

    def __init__(self, zoom_manager: ZoomManager):
        """
        Инициализация трекера курсора.

        Args:
            zoom_manager: ZoomManager для получения точки взгляда камеры
        """
        self.zoom_manager = zoom_manager
        self.current_position_2d = np.array([0.0, 0.0], dtype=float)

    def get_mouse_world_position(self) -> Optional[Tuple[float, float]]:
        """
        Получает позицию мыши в мировых координатах.

        Алгоритм:
        1. Определить точку на которую смотрит камера (без зума)
        2. Отнять позицию frame origin_cube
        3. Поделить на common_scale

        Returns:
            (x, z) координаты курсора в правильной системе координат
        """
        try:
            # 1. Получаем точку взгляда камеры (без зума)
            look_point_x, look_point_z = self.zoom_manager.identify_invariant_point()

            # 2. Получаем трансформированную позицию origin_cube из frame
            frame = getattr(self.zoom_manager.scene_setup, 'frame', None)
            if frame and hasattr(frame, 'origin_cube'):
                # Используем обычную position (после трансформаций) а не real_position
                origin_pos = frame.origin_cube.position
                if hasattr(origin_pos, 'x'):
                    origin_x, origin_z = origin_pos.x, origin_pos.z
                else:
                    origin_x, origin_z = origin_pos[0], origin_pos[2]
            else:
                # Fallback если frame не найден
                origin_x, origin_z = 0.0, 0.0

            # 3. Получаем масштаб трансформации
            transform_scale = getattr(self.zoom_manager, 'a_transformation', 1.0)

            # 4. ПРАВИЛЬНАЯ ФОРМУЛА: (look_point - frame_origin_cube) / scale
            corrected_x = (look_point_x - origin_x) / transform_scale
            corrected_z = (look_point_z - origin_z) / transform_scale

            return (corrected_x, corrected_z)

        except Exception as e:
            print(f"Ошибка определения позиции мыши: {e}")
            return None

    def update_position(self) -> bool:
        """
        Обновляет текущую позицию курсора.

        Returns:
            True если позиция успешно обновлена, False при ошибке
        """
        mouse_pos = self.get_mouse_world_position()
        if mouse_pos is None:
            return False

        # Обновляем сохраненную позицию
        self.current_position_2d[0] = mouse_pos[0]
        self.current_position_2d[1] = mouse_pos[1]
        return True

    def get_current_position(self) -> np.ndarray:
        """
        Возвращает текущую позицию курсора.

        Returns:
            np.array([x, z]) - текущая позиция курсора
        """
        return self.current_position_2d.copy()

    def get_current_position_3d(self) -> Tuple[float, float, float]:
        """
        Возвращает текущую позицию курсора в 3D формате (x, y, z).

        Returns:
            (x, 0.0, z) - позиция для создания 3D объектов
        """
        return (self.current_position_2d[0], 0.0, self.current_position_2d[1])
