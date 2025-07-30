from ursina import *
from typing import Optional
from ..logic.spawn_area import SpawnArea as SpawnAreaLogic # Используем alias для обратной совместимости
from ..visual.spawn_area_visualizer import SpawnAreaVisualizer
from ..visual.cost_visualizer import CostVisualizer


class SpawnAreaManager:
    """
    Управляет всеми компонентами, связанными с областью спавна:
    - Логика (spawn_area_logic)
    - Визуализация (spawn_area_visualizer)
    - Поверхность стоимости (cost_visualizer)
    
    Предоставляет простой интерфейс для изменения параметров.
    """
    def __init__(self, 
                 spawn_area_logic: SpawnAreaLogic, 
                 spawn_area_visualizer: SpawnAreaVisualizer, 
                 cost_visualizer: Optional[CostVisualizer] = None):
        
        self.logic: SpawnAreaLogic = spawn_area_logic
        self.visualizer: SpawnAreaVisualizer = spawn_area_visualizer
        self.cost_visualizer: Optional[CostVisualizer] = cost_visualizer
        self.eccentricity_step: float = 1.05

    def _update_visuals(self) -> None:
        """Обновляет все связанные визуальные компоненты."""
        if self.visualizer:
            self.visualizer.update_visuals()
        if self.cost_visualizer:
            self.cost_visualizer.generate_surface()

    def increase_eccentricity(self) -> None:
        """Увеличивает эксцентриситет (мультипликативно)."""
        if not self.logic:
            return
        new_ecc = self.logic.eccentricity * self.eccentricity_step
        self.logic.set_eccentricity(new_ecc)
        self._update_visuals()

    def decrease_eccentricity(self) -> None:
        """Уменьшает эксцентриситет (мультипликативно)."""
        if not self.logic:
            return
        new_ecc = self.logic.eccentricity / self.eccentricity_step
        self.logic.set_eccentricity(new_ecc)
        self._update_visuals() 