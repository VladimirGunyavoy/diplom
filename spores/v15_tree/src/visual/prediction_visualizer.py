from ursina import destroy
import numpy as np
from typing import Dict, Optional

from ..core.spore import Spore
from ..visual.pillar import Pillar
from ..managers.color_manager import ColorManager
from ..managers.zoom_manager import ZoomManager
from ..logic.cost_function import CostFunction

class PredictionVisualizer:
    """
    Управляет полным набором визуальных элементов для одного
    предсказанного будущего состояния ("призрака").

    Это включает в себя:
    - Саму спору-призрака.
    - "Ангела" (визуализацию стоимости).
    - "Столб" под ангелом.

    Класс позволяет включать и выключать отдельные части визуализации
    через параметры конфигурации.
    """
    ANGEL_HEIGHT_OFFSET_RATIO: float = 0.01

    def __init__(self,
                 parent_spore: Optional[Spore],
                 color_manager: ColorManager,
                 zoom_manager: ZoomManager,
                 cost_function: Optional[CostFunction],
                 config: Dict,
                 spore_id: str):
        """
        Инициализирует визуализатор для одного предсказания.

        Args:
            parent_spore: Реальная спора, от которой делается предсказание, или None для призрачных деревьев.
            color_manager: Менеджер цветов.
            zoom_manager: Менеджер масштабирования.
            cost_function: Функция стоимости для расчета высоты ангелов.
            config: Словарь с конфигурацией, содержащий флаги 'show_ghosts', 'show_angels', 'show_pillars'.
            spore_id: Уникальный ID для этого набора визуальных элементов.
        """
        self.parent_spore: Optional[Spore] = parent_spore
        self.color_manager: ColorManager = color_manager
        self.zoom_manager: ZoomManager = zoom_manager
        self.cost_function: Optional[CostFunction] = cost_function
        self.config: Dict = config
        self.id: str = spore_id

        # Флаги видимости
        self.show_ghost_spore: bool = self.config.get('spore',{}).get('show_ghosts', True)
        self.show_angel: bool = self.config.get('angel', {}).get('show_angels', True)
        self.show_pillar: bool = self.config.get('angel', {}).get('show_pillars', True)

        # Конфигурация столбов
        angel_config = self.config.get('angel', {})
        self.pillar_width: float = angel_config.get('ghost_pillar_width', angel_config.get('pillar_width', 0.05))

        # Сущности Ursina
        self.ghost_spore: Optional[Spore] = None
        self.angel: Optional[Spore] = None
        self.pillar: Optional[Pillar] = None
        
        # Оптимизация: буферы для массивов чтобы избежать np.array() в update()
        self._position_3d_buffer = np.array([0.0, 0.0, 0.0], dtype=float)
        self._angel_offset_buffer = np.array([0.0, 0.0, 0.0], dtype=float)
        self._pillar_offset_buffer = np.array([0.0, 0.0, 0.0], dtype=float)
        self._pillar_scale_buffer = np.array([0.0, 0.0, 0.0], dtype=float)
        
        self._create_entities()

    def _create_entities(self) -> None:
        """Создает все необходимые сущности Ursina."""
        
        # 1. Создаем спору-призрака (базовый элемент)
        if self.show_ghost_spore:
            if self.parent_spore is not None:
                # Обычный случай - клонируем родительскую спору
                self.ghost_spore = self.parent_spore.clone()
                self.ghost_spore.is_ghost = True
                self.ghost_spore.id = f"prediction_ghost_{self.id}"
                self.ghost_spore.color = self.color_manager.get_color('spore', 'ghost')
                self.ghost_spore.set_y_coordinate(0.0)
                # Призрачные споры не регистрируются в ZoomManager - они постоянные
            else:
                # Случай призрачного дерева - создаем новую спору
                from ..core.spore import Spore
                from ..logic.pendulum import PendulumSystem
                
                # Создаем временный маятник для призрака
                temp_pendulum = PendulumSystem()
                
                correct_scale = getattr(self.zoom_manager, 'config', {}).get('spore', {}).get('scale', 0.02)
                self.ghost_spore = Spore(
                    dt=0.1,
                    pendulum=temp_pendulum,
                    goal_position=(0, 0),
                    model='sphere',
                    position=(0, 0, 0),
                    scale=correct_scale,  # 🔧 ИСПРАВЛЕНИЕ: Добавляем правильный scale
                    color_manager=self.color_manager,
                    is_ghost=True
                )
                self.ghost_spore.id = f"prediction_ghost_{self.id}"
                self.ghost_spore.color = self.color_manager.get_color('spore', 'ghost')
                self.ghost_spore.set_y_coordinate(0.0)
                # Призрачные споры не регистрируются в ZoomManager - они постоянные

        # Для ангела и столба нужен ghost_spore, даже если он невидимый, для получения координат
        if not self.ghost_spore and self.parent_spore is not None:
            self.ghost_spore = self.parent_spore.clone()
            self.ghost_spore.enabled = False
            self.ghost_spore.set_y_coordinate(0.0)
        elif not self.ghost_spore:
            # Случай когда нет parent_spore - создаем минимальную призрачную спору
            from ..core.spore import Spore
            from ..logic.pendulum import PendulumSystem
            
            temp_pendulum = PendulumSystem()
            self.ghost_spore = Spore(
                dt=0.1,
                pendulum=temp_pendulum,
                goal_position=(0, 0),
                model='sphere',
                position=(0, 0, 0),
                color_manager=self.color_manager,
                is_ghost=True
            )
            self.ghost_spore.enabled = False
            self.ghost_spore.set_y_coordinate(0.0)

        # 2. Создаем ангела
        if self.show_angel and self.cost_function and self.ghost_spore:
            if hasattr(self.ghost_spore, 'clone'):
                self.angel = self.ghost_spore.clone()
                self.angel.id = f"ghost_angel_{self.id}"
                self.angel.color = self.color_manager.get_color('angel', 'ghost')
                self.zoom_manager.register_object(self.angel, self.angel.id)

        # 3. Создаем столб
        if self.show_pillar and self.cost_function and self.ghost_spore:
            self.pillar = Pillar(
                model='cube',
                color=self.color_manager.get_color('angel', 'ghost_pillar')
            )
            self.pillar.id = f"ghost_pillar_{self.id}"
            self.zoom_manager.register_object(self.pillar, self.pillar.id)


    def update(self, predicted_state_2d: np.ndarray) -> None:
        """
        Обновляет положение и масштаб всех визуальных элементов на основе
        нового предсказанного состояния.

        Args:
            predicted_state_2d (np.array): Предсказанное 2D состояние (угол, угловая скорость).
        """
        if not self.ghost_spore:
            return

        # 1. Обновляем позицию споры-призрака (Y=0 для плоскости XZ)
        # Оптимизация: используем буфер вместо np.array()
        self._position_3d_buffer[0] = predicted_state_2d[0]
        self._position_3d_buffer[1] = 0.0
        self._position_3d_buffer[2] = predicted_state_2d[1]
        self.ghost_spore.real_position = self._position_3d_buffer.copy()
        
        # Обновляем логическую позицию напрямую, не затрагивая Y
        self.ghost_spore.logic.set_position_2d(predicted_state_2d)
        self.ghost_spore.apply_transform(
            self.zoom_manager.a_transformation,
            self.zoom_manager.b_translation,
            spores_scale=self.zoom_manager.spores_scale
        )
        
        if not self.cost_function:
            return

        # 2. Рассчитываем стоимость для высоты
        display_height = self.cost_function.get_cost(predicted_state_2d)
        y_offset = display_height * self.ANGEL_HEIGHT_OFFSET_RATIO

        # 3. Обновляем ангела
        if self.angel:
            # Оптимизация: используем буфер вместо np.array([0, height, 0])
            self._angel_offset_buffer[0] = 0.0
            self._angel_offset_buffer[1] = display_height + y_offset
            self._angel_offset_buffer[2] = 0.0
            self.angel.real_position = self.ghost_spore.real_position + self._angel_offset_buffer
            self.angel.apply_transform(
                self.zoom_manager.a_transformation,
                self.zoom_manager.b_translation,
                spores_scale=self.zoom_manager.spores_scale
            )

        # 4. Обновляем столб
        if self.pillar:
            # Оптимизация: используем буферы вместо np.array()
            pillar_height = (display_height + y_offset) / 2
            self._pillar_offset_buffer[0] = 0.0
            self._pillar_offset_buffer[1] = pillar_height
            self._pillar_offset_buffer[2] = 0.0
            self.pillar.real_position = self.ghost_spore.real_position + self._pillar_offset_buffer
            
            self._pillar_scale_buffer[0] = self.pillar_width
            self._pillar_scale_buffer[1] = display_height + y_offset
            self._pillar_scale_buffer[2] = self.pillar_width
            self.pillar.real_scale = self._pillar_scale_buffer.copy()
            
            self.pillar.apply_transform(
                self.zoom_manager.a_transformation,
                self.zoom_manager.b_translation,
                spores_scale=self.zoom_manager.spores_scale
            )

    def destroy(self) -> None:
        """Уничтожает все связанные с этим предсказанием сущности."""
        if self.ghost_spore:
            destroy(self.ghost_spore)
        if self.angel:
            destroy(self.angel)
        if self.pillar:
            destroy(self.pillar)
        
        self.ghost_spore = None
        self.angel = None
        self.pillar = None
        
    def set_visibility(self, is_visible: bool) -> None:
        """Управляет видимостью всех сущностей."""
        if self.ghost_spore and self.show_ghost_spore:
            self.ghost_spore.enabled = is_visible
        if self.angel and self.show_angel:
            self.angel.enabled = is_visible
        if self.pillar and self.show_pillar:
            self.pillar.enabled = is_visible 