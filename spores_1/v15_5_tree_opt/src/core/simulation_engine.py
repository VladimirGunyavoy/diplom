from typing import Dict, Any
import numpy as np
import uuid

from ..logic.pendulum import PendulumSystem
from ..logic.spore_logic import SporeLogic
from ..logic.spawn_area import SpawnArea
from ..logic.ghost_processor import GhostProcessor


class SimulationEngine:
    """
    Центральный движок симуляции, управляющий всей логикой.
    Этот класс не зависит от Ursina и отвечает за:
    - Хранение и управление множеством объектов SporeLogic.
    - Создание новых спор в области спавна.
    - Выполнение шагов симуляции для всех спор.
    - Обработку и предоставление данных о "призраках".
    """

    def __init__(self, pendulum_params: Dict[str, float], spawn_area_params: Dict[str, Any], dt: float):
        """
        Args:
            pendulum_system: Экземпляр системы маятника.
            spawn_area: Логика области спавна.
            ghost_processor: Логика обработки "призраков".
            dt: Временной шаг симуляции.
            goal_position: Целевое 2D-состояние.
        """
        self.pendulum_system = PendulumSystem(**pendulum_params)
        self.spawn_area = SpawnArea(**spawn_area_params)
        self.ghost_processor = GhostProcessor(self.pendulum_system, dt=dt)
        
        self.spores: Dict[str, SporeLogic] = {}
        self._next_spore_id = 0

    def spawn_spore(self, initial_pos_2d: np.ndarray, goal_pos_2d: np.ndarray) -> str:
        """
        Создает новую логическую спору.
        
        Returns:
            Уникальный ID созданной споры.
        """
        spore_id = f"spore_{self._next_spore_id}"
        self._next_spore_id += 1
        
        self.spores[spore_id] = SporeLogic(
            pendulum=self.pendulum_system,
            dt=0.1,  # Пример, dt может быть параметром
            initial_position_2d=initial_pos_2d,
            goal_position_2d=goal_pos_2d
        )
        return spore_id

    def step(self, controls: Dict[str, float]) -> None:
        """
        Выполняет один шаг симуляции для всех спор.
        
        Args:
            controls: Словарь, где ключ - ID споры, значение - управление.
        """
        for spore_id, spore in self.spores.items():
            control = controls.get(spore_id, 0.0)
            spore.evolve(control=control)

    def get_spore_state(self, spore_id: str) -> dict:
        """Возвращает полное состояние одной споры."""
        spore = self.spores[spore_id]
        return {
            'position': spore.get_position_2d(),
            'cost': spore.get_cost()
        }

    def get_all_spore_states(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает состояния всех активных спор."""
        return {
            spore_id: {
                "position": spore.get_position_2d(),
                "cost": spore.get_cost()
            }
            for spore_id, spore in self.spores.items()
        }

    def get_ghosts(self, spore_id: str, num_ghosts: int = 10, control_method: str = 'random') -> dict:
        """
        Возвращает состояния "призраков" для указанной споры.
        """
        spore_logic = self.spores[spore_id]
        
        # Генерируем управления
        controls = spore_logic.sample_controls(N=num_ghosts, method=control_method)
        
        # Рассчитываем будущие состояния
        future_states = self.ghost_processor.process(
            current_state_2d=spore_logic.get_position_2d(),
            controls=controls
        )
        
        # Рассчитываем стоимости для будущих состояний
        future_costs = [spore_logic.calculate_cost(pos) for pos in future_states]
        
        return {
            'states': future_states,
            'costs': future_costs,
            'controls': controls
        }
        
    def remove_spore(self, spore_id: str):
        """Удаляет спору из симуляции."""
        if spore_id in self.spores:
            del self.spores[spore_id]

    def __repr__(self):
        return f"SimulationEngine(spores={len(self.spores)})" 