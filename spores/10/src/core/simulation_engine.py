import numpy as np
import uuid

from src.logic.pendulum import PendulumSystem
from src.logic.spore_logic import SporeLogic
from src.logic.spawn_area import SpawnArea
from src.logic.ghost_processor import GhostProcessor


class SimulationEngine:
    """
    Центральный движок симуляции, управляющий всей логикой.
    Этот класс не зависит от Ursina и отвечает за:
    - Хранение и управление множеством объектов SporeLogic.
    - Создание новых спор в области спавна.
    - Выполнение шагов симуляции для всех спор.
    - Обработку и предоставление данных о "призраках".
    """

    def __init__(self,
                 pendulum_system: PendulumSystem,
                 spawn_area: SpawnArea,
                 ghost_processor: GhostProcessor,
                 dt: float,
                 goal_position: np.ndarray):
        """
        Args:
            pendulum_system: Экземпляр системы маятника.
            spawn_area: Логика области спавна.
            ghost_processor: Логика обработки "призраков".
            dt: Временной шаг симуляции.
            goal_position: Целевое 2D-состояние.
        """
        self.pendulum = pendulum_system
        self.spawn_area = spawn_area
        self.ghost_processor = ghost_processor
        self.dt = dt
        self.goal_position = np.asarray(goal_position, dtype=float)

        self.spores: dict[str, SporeLogic] = {}

    def spawn_spore(self) -> str:
        """
        Создает новую спору в случайной точке области спавна.
        Returns:
            Уникальный ID созданной споры.
        """
        initial_pos = self.spawn_area.sample_random_point()
        
        spore_logic = SporeLogic(
            pendulum=self.pendulum,
            dt=self.dt,
            goal_position_2d=self.goal_position,
            initial_position_2d=initial_pos
        )
        
        spore_id = str(uuid.uuid4())
        self.spores[spore_id] = spore_logic
        
        return spore_id

    def step(self, controls: dict[str, float]):
        """
        Выполняет один шаг симуляции для всех спор.
        Args:
            controls: Словарь {spore_id: control_value}.
                      Если для споры нет управления, используется 0.
        """
        for spore_id, spore in self.spores.items():
            control = controls.get(spore_id, 0.0)
            new_position = spore.evolve(control=control)
            spore.set_position_2d(new_position)

    def get_spore_state(self, spore_id: str) -> dict:
        """Возвращает полное состояние одной споры."""
        spore = self.spores[spore_id]
        return {
            'position': spore.get_position_2d(),
            'cost': spore.get_cost()
        }

    def get_all_spore_states(self) -> dict[str, dict]:
        """Возвращает состояния всех активных спор."""
        return {spore_id: self.get_spore_state(spore_id) for spore_id in self.spores}

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