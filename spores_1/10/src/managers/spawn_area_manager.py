from ursina import *

class SpawnAreaManager:
    """
    Управляет всеми компонентами, связанными с областью спавна:
    - Логика (spawn_area_logic)
    - Визуализация (spawn_area_visualizer)
    - Поверхность стоимости (cost_visualizer)
    
    Предоставляет простой интерфейс для изменения параметров.
    """
    def __init__(self, spawn_area_logic, spawn_area_visualizer, cost_visualizer):
        self.logic = spawn_area_logic
        self.visualizer = spawn_area_visualizer
        self.cost_visualizer = cost_visualizer
        self.eccentricity_step = 1.05

    def _update_visuals(self):
        """Обновляет все связанные визуальные компоненты."""
        if self.visualizer:
            self.visualizer.update_visuals()
        if self.cost_visualizer:
            self.cost_visualizer.generate_surface()

    def increase_eccentricity(self):
        """Увеличивает эксцентриситет (мультипликативно)."""
        if not self.logic:
            return
        new_ecc = self.logic.eccentricity * self.eccentricity_step
        self.logic.set_eccentricity(new_ecc)
        self._update_visuals()

    def decrease_eccentricity(self):
        """Уменьшает эксцентриситет (мультипликативно)."""
        if not self.logic:
            return
        new_ecc = self.logic.eccentricity / self.eccentricity_step
        self.logic.set_eccentricity(new_ecc)
        self._update_visuals() 