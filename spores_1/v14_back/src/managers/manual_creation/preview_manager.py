from typing import Optional
import numpy as np
from ursina import Vec3, mouse, camera, raycast

from ...core.spore import Spore
from ...managers.zoom_manager import ZoomManager
from ...managers.color_manager import ColorManager
from ...logic.pendulum import PendulumSystem


class PreviewManager:
    """
    Управляет превью спорой, которая следует за курсором мыши.

    Ответственности:
    - Создание и обновление полупрозрачной превью споры
    - Отслеживание позиции курсора в 2D плоскости
    - Применение трансформаций zoom manager к превью
    """

    def __init__(self,
                 zoom_manager: ZoomManager,
                 pendulum: PendulumSystem,
                 color_manager: ColorManager,
                 config: dict):

        self.zoom_manager = zoom_manager
        self.pendulum = pendulum
        self.color_manager = color_manager
        self.config = config

        # Настройки превью
        self.preview_enabled = True
        self.preview_alpha = 0.5  # Полупрозрачность

        # Превью спора и её позиция
        self.preview_spore: Optional[Spore] = None
        self.preview_position_2d = np.array([0.0, 0.0], dtype=float)

        print("   ✓ Preview Manager создан")

    def update_cursor_position(self) -> None:
        """
        Обновляет позицию превью споры по позиции курсора мыши.
        Вызывается каждый кадр из update manager.
        """
        if not self.preview_enabled:
            return

        try:
            # Делаем raycast от камеры через позицию мыши к плоскости y=0
            ray_result = raycast(
                origin=camera.world_position,
                direction=camera.forward,
                distance=1000,
                ignore=[],
                traverse_target=None
            )

            # Если луч пересек плоскость, обновляем позицию
            if ray_result.hit:
                hit_point = ray_result.world_point
                # Игнорируем Y координату, работаем в плоскости XZ
                self.preview_position_2d = np.array([hit_point.x, hit_point.z], dtype=float)
            else:
                # Fallback: используем экранные координаты мыши
                # Приблизительное преобразование из экранных координат
                mouse_x = mouse.position[0] * 10  # Масштабируем
                mouse_z = mouse.position[1] * 10
                self.preview_position_2d = np.array([mouse_x, mouse_z], dtype=float)

            # Обновляем или создаем превью спору
            self._update_preview_spore()

        except Exception as e:
            # В случае ошибки используем последнюю известную позицию
            print(f"⚠️ Ошибка обновления позиции курсора: {e}")

    def set_preview_enabled(self, enabled: bool) -> None:
        """Включает/выключает показ превью споры."""
        self.preview_enabled = enabled
        if not enabled:
            self._destroy_preview()

    def get_preview_position(self) -> np.ndarray:
        """Возвращает текущую позицию превью в 2D."""
        return self.preview_position_2d.copy()

    def has_preview(self) -> bool:
        """Проверяет, есть ли активная превью спора."""
        return self.preview_spore is not None

    def _update_preview_spore(self) -> None:
        """Создает или обновляет превью спору."""
        if not self.preview_spore:
            self._create_preview_spore()
        else:
            # Обновляем позицию существующей споры
            self.preview_spore.real_position = Vec3(
                self.preview_position_2d[0],
                0.0,
                self.preview_position_2d[1]
            )
            # Применяем трансформации zoom manager
            self.preview_spore.apply_transform(
                self.zoom_manager.a_transformation,
                self.zoom_manager.b_translation,
                spores_scale=self.zoom_manager.spores_scale
            )

    def _create_preview_spore(self) -> None:
        """Создает новую превью спору."""
        try:
            goal_position = self.config.get('spore', {}).get('goal_position', [0, 0])
            spore_config = self.config.get('spore', {})
            pendulum_config = self.config.get('pendulum', {})

            self.preview_spore = Spore(
                pendulum=self.pendulum,
                dt=pendulum_config.get('dt', 0.1),
                goal_position=goal_position,
                scale=spore_config.get('scale', 0.1),
                position=(self.preview_position_2d[0], 0.0, self.preview_position_2d[1]),
                color_manager=self.color_manager,
                is_ghost=True  # Делаем спору-призрак
            )

            base_color = self.color_manager.get_color('spore', 'default')

            # Устанавливаем полупрозрачность
            try:
                self.preview_spore.color = (base_color.r, base_color.g, base_color.b, self.preview_alpha)
            except AttributeError:
                try:
                    self.preview_spore.color = (base_color[0], base_color[1], base_color[2], self.preview_alpha)
                except (TypeError, IndexError):
                    # Fallback на стандартный цвет
                    self.preview_spore.color = (0.6, 0.4, 0.9, self.preview_alpha)  # Фиолетовый
                    print("   ⚠️ Использован fallback цвет для preview spore")

            # Регистрируем в zoom manager
            self.zoom_manager.register_object(self.preview_spore, name='manual_preview')

        except Exception as e:
            print(f"❌ Ошибка создания превью споры: {e}")

    def _destroy_preview(self) -> None:
        """Уничтожает превью спору."""
        if self.preview_spore:
            self.zoom_manager.unregister_object('manual_preview')
            from ursina import destroy
            destroy(self.preview_spore)
            self.preview_spore = None

    def destroy(self) -> None:
        """Очищает все ресурсы менеджера."""
        self._destroy_preview()
        print("   ✓ Preview Manager уничтожен")
