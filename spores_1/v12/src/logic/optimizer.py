from typing import Dict, Any
import numpy as np
from scipy.optimize import minimize
from .pendulum import PendulumSystem
from ..core.spore import Spore

class SporeOptimizer:
    def __init__(self, pendulum: PendulumSystem, config: Dict[str, Any] = None):
        self.pendulum = pendulum
        self.config = config or {}

    def _objective_function(self, x: np.ndarray, spore: Spore) -> float:
        control, dt = x
        
        # Создаем временную логику для симуляции
        temp_logic = spore.logic.clone()
        next_state = temp_logic.evolve(control=control, dt=dt)
        
        # Стоимость - это расстояние до цели
        return temp_logic.calculate_cost(next_state)

    def find_optimal_step(self, spore: Spore) -> Dict[str, Any]:
        """
        Находит оптимальное управление и временной шаг для минимизации
        стоимости (расстояния до цели).
        """
        control_bounds = self.pendulum.get_control_bounds()
        
        # Получаем границы dt из конфигурации
        optimizer_config = self.config.get('pendulum', {}).get('optimizer', {})
        dt_min = optimizer_config.get('dt_min', 0.01)
        dt_max = optimizer_config.get('dt_max', 0.1)
        dt_bounds = (dt_min, dt_max)
        
        initial_guess = [0.0, np.mean(dt_bounds)] # Начальное предположение

        result = minimize(
            self._objective_function,
            initial_guess,
            args=(spore,),
            method='L-BFGS-B',
            bounds=[control_bounds, dt_bounds]
        )
        return result
    
    def linearized_cost_function(self, u: np.ndarray, A: np.ndarray, B: np.ndarray, current_state: np.ndarray, goal_state: np.ndarray) -> float:
        """
        Линеаризованная функция стоимости для MPC.
        """
        # x_next = Ax + Bu
        predicted_state = A @ current_state + B @ u
        # Стоимость - квадрат расстояния до цели
        return float(np.sum((predicted_state - goal_state)**2))

    def find_optimal_control_mpc(self, spore: Spore, prediction_horizon: int) -> np.ndarray:
        """
        Находит оптимальную последовательность управлений с использованием MPC.
        """
        current_state = spore.logic.get_position_2d()
        goal_state = spore.logic.goal_position_2d
        
        # Получаем линеаризованные матрицы
        A, B = self.pendulum.get_linearized_matrices_at_state(current_state)
        
        # Начальное предположение для последовательности управлений
        u_initial_guess = np.zeros(prediction_horizon)
        
        # Ограничения для каждого шага управления
        control_bounds = [self.pendulum.get_control_bounds()] * prediction_horizon

        # Функция для минимизации
        def mpc_objective(u_sequence: np.ndarray) -> float:
            cost = 0
            state = current_state
            for u in u_sequence:
                state = A @ state + B @ np.array([u])
                cost += np.sum((state - goal_state)**2)
            return float(cost)

        result = minimize(
            mpc_objective,
            u_initial_guess,
            method='L-BFGS-B',
            bounds=control_bounds
        )
        
        return result.x 