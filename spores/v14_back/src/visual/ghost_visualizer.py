from ursina import Entity, destroy
from typing import List, Dict, Optional, Any
from ..visual.spore_visual import SporeVisual

class GhostVisualizer(Entity):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.ghost_entities: Dict[str, SporeVisual] = {}

    def update_ghosts(self, ghost_states: Dict[str, Dict[str, Any]]) -> None:
        """
        Синхронизирует визуальное представление "призраков" с данными из GhostProcessor.
        """
        current_ids = set(self.ghost_entities.keys())
        new_ids = set(ghost_states.keys())

        # Удаляем старых
        for ghost_id in current_ids - new_ids:
            destroy(self.ghost_entities[ghost_id])
            del self.ghost_entities[ghost_id]

        # Создаем или обновляем
        for ghost_id, state in ghost_states.items():
            pos_2d = state['position']
            pos_3d = (pos_2d[0], 0, pos_2d[1])
            
            if ghost_id not in self.ghost_entities:
                self.ghost_entities[ghost_id] = SporeVisual(
                    parent=self, 
                    position=pos_3d
                )
            else:
                self.ghost_entities[ghost_id].position = pos_3d 