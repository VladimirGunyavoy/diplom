from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple

from .pendulum import PendulumSystem


class SporeLogic:
    """
    Класс, инкапсулирующий всю математическую логику споры.
    Работает исключительно с 2D numpy массивами.
    """
    
    def __init__(self, pendulum: PendulumSystem, dt: float, 
                 goal_position_2d: np.ndarray, 
                 initial_position_2d: Optional[np.ndarray] = None):
        
        self.pendulum: PendulumSystem = pendulum
        self.dt: float = dt
        self.goal_position_2d: np.ndarray = np.array(goal_position_2d, dtype=float)
        
        if initial_position_2d is None:
            self.position_2d: np.ndarray = np.array([0.0, 0.0], dtype=float)
        else:
            self.position_2d: np.ndarray = np.array(initial_position_2d, dtype=float)
            
        self.cost: float = self.calculate_cost()
        
        # Параметр жизни споры - если dt=0, то спора мертва
        self.alive: bool = True
        
        # Для хранения оптимальных параметров, найденных оптимизатором
        self.optimal_control: Optional[np.ndarray] = None
        self.optimal_dt: Optional[float] = None
        
    def step(self, control: float, dt: float) -> np.ndarray:
        """
        Выполняет один шаг дискретной динамики.
        
        Args:
            control: Управляющее воздействие.
            dt: Временной шаг.
            
        Returns:
            Новое 2D состояние.
        """
        next_state_2d = self.pendulum.discrete_step(self.position_2d, control, dt)
        return next_state_2d

    def evolve(self, control: float = 0, dt: Optional[float] = None) -> np.ndarray:
        """
        Эволюционирует состояние на один шаг и обновляет внутреннее состояние.
        
        Args:
            control: Управляющее воздействие.
            dt: Временной шаг (если None, используется self.dt).
            
        Returns:
            Новое 2D состояние.
        """
        
        # Если dt не передан, используем dt по умолчанию
        current_dt = dt if dt is not None else self.dt
        
        # Получаем следующее состояние от системы маятника
        next_state_2d = self.step(control, current_dt)
        
        # Обновляем состояние
        self.position_2d = next_state_2d
        self.cost = self.calculate_cost() # Обновляем стоимость
        return next_state_2d

    def get_position_2d(self) -> np.ndarray:
        """Возвращает текущую 2D позицию."""
        return self.position_2d.copy()
    
    def set_position_2d(self, position_2d: np.ndarray) -> None:
        """Устанавливает 2D позицию и обновляет стоимость."""
        self.position_2d = np.array(position_2d, dtype=float)
        self.cost = self.calculate_cost()

    def calculate_cost(self, position_2d: Optional[np.ndarray] = None) -> float:
        """
        Рассчитывает стоимость (расстояние до цели).
        Если позиция не передана, используется текущая.
        """
        pos = self.position_2d if position_2d is None else position_2d
        return float(np.linalg.norm(pos - self.goal_position_2d))

    def get_cost(self) -> float:
        """Возвращает текущую стоимость."""
        return self.cost
    
    def set_alive(self, alive: bool) -> None:
        """Устанавливает состояние жизни споры."""
        self.alive = alive
    
    def is_alive(self) -> bool:
        """Проверяет жива ли спора."""
        return self.alive
    
    def check_death(self) -> None:
        """Проверяет условия смерти споры (optimal_dt = 0) и устанавливает alive=False."""
        if self.optimal_dt is not None and self.optimal_dt == 0.0:
            self.alive = False

    def sample_controls(self, N: int, method: str = 'random') -> np.ndarray:
        """
        Генерирует N сэмплов управления.
        
        Args:
            N: Количество сэмплов.
            method: 'random' или 'mesh'.
            
        Returns:
            Массив сэмплов управления.
        """
        min_control, max_control = self.pendulum.get_control_bounds()
        
        if method == 'random':
            return np.random.uniform(min_control, max_control, N)
        elif method == 'mesh':
            return np.linspace(min_control, max_control, N)
        else:
            raise ValueError("Метод должен быть 'random' или 'mesh'")

    def simulate_controls(self, controls: np.ndarray, dt: Optional[float] = None) -> List[np.ndarray]:
        """
        Симулирует несколько управляющих воздействий из текущего состояния.
        Не изменяет внутреннее состояние объекта.
        
        Args:
            controls: Массив управляющих воздействий.
            
        Returns:
            Список будущих 2D состояний.
        """
        # Оптимизация: используем pendulum напрямую без создания временного объекта
        dt_value = dt or self.dt
        return [self.pendulum.discrete_step(self.position_2d, control=c, dt=dt_value) for c in controls]

    def clone(self) -> 'SporeLogic':
        """Создает и возвращает глубокую копию объекта."""
        cloned = SporeLogic(
            pendulum=self.pendulum,
            dt=self.dt,
            goal_position_2d=self.goal_position_2d.copy(),
            initial_position_2d=self.position_2d.copy()
        )
        # Копируем состояние жизни и оптимальные параметры
        cloned.alive = self.alive
        cloned.optimal_control = self.optimal_control
        cloned.optimal_dt = self.optimal_dt
        return cloned 