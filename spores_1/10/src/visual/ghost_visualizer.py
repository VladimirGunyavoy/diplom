from ursina import Entity, color, destroy
import numpy as np

class GhostVisualizer:
    """
    Класс для управления визуализацией "призраков" (предсказанных состояний).
    """

    def __init__(self, data: dict, model='sphere', base_scale=0.5):
        self.ghost_entities = []
        self.model = model
        self.base_scale = base_scale
        self.enabled = True
        
        if data:
            self.update_ghosts(data)

    def update_ghosts(self, data: dict):
        """
        Обновляет визуализацию призраков на основе новых данных.
        
        Args:
            data (dict): Словарь с ключами 'states', 'costs', 'controls'.
        """
        self.clear_ghosts()
        
        if not self.enabled or not data or 'states' not in data:
            return

        states = data.get('states', [])
        costs = data.get('costs', [])

        if not states:
            return

        min_cost = min(costs) if costs else 0
        max_cost = max(costs) if costs else 1

        for i, pos_2d in enumerate(states):
            cost = costs[i] if i < len(costs) else 0
            
            y_pos = 0
            ghost_color = self._cost_to_color(cost, min_cost, max_cost)

            ghost = Entity(
                model=self.model,
                position=(pos_2d[0], y_pos, pos_2d[1]),
                scale=self.base_scale,
                color=ghost_color
            )
            self.ghost_entities.append(ghost)

    def _cost_to_color(self, cost, min_c, max_c):
        """Определяет цвет на основе стоимости (от зеленого к красному)."""
        norm_cost = (cost - min_c) / (max_c - min_c + 1e-6)
        red = norm_cost
        green = 1 - norm_cost
        return color.rgba(red * 255, green * 255, 0, 150)

    def clear_ghosts(self):
        """Удаляет все текущие объекты-призраки."""
        for ghost in self.ghost_entities:
            destroy(ghost)
        self.ghost_entities = []

    def clear(self):
        """Скрывает всех призраков."""
        for ghost in self.ghost_entities:
            ghost.enabled = False 