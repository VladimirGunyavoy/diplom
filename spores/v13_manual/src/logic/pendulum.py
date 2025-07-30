import numpy as np
from scipy.linalg import expm
from typing import Tuple

class PendulumSystem:
    """
    Класс, описывающий систему маятника.
    Позволяет выполнять линеаризацию и дискретизацию в произвольном состоянии.
    """
    def __init__(self, 
                 g: float = 9.81,
                 l: float = 2.0,
                 m: float = 1.0,
                 damping: float = 0.1, 
                 max_control: float = 1):
        self.g: float = g
        self.l: float = l
        self.m: float = m
        self.damping: float = damping
        self.max_control: float = float(max_control)
        
        # Оптимизация: кэш для избежания повторных матричных вычислений
        self._linearization_cache = {}  # key: (theta_0,), value: (A_cont, B_cont)
        self._discretization_cache = {}  # key: (A_hash, B_hash, dt), value: (A_d, B_d)
        
    def get_control_bounds(self) -> np.ndarray:
        return np.array([-self.max_control, self.max_control])
        
    def get_linearized_matrices_at_state(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Линеаризация нелинейной динамики маятника в произвольном состоянии.
        Возвращает непрерывные матрицы A и B.
        """
        theta_0, _ = state
        
        # Оптимизация: кэширование результатов по округленному theta_0
        cache_key = round(float(theta_0), 6)  # точность до 6 знаков
        
        if cache_key in self._linearization_cache:
            return self._linearization_cache[cache_key]
        
        # Вычисляем матрицы только если их нет в кэше
        A_cont = np.array([
            [0.0, 1.0],
            [-self.g / self.l * np.cos(theta_0), -self.damping]
        ])
        
        B_cont = np.array([
            [0.0],
            [1.0]
        ])
        
        # Сохраняем в кэш
        result = (A_cont, B_cont)
        self._linearization_cache[cache_key] = result
        
        return result

    def discretize(self, A_cont: np.ndarray, B_cont: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Дискретизация непрерывной системы с помощью матричной экспоненты.
        """
        # Оптимизация: кэширование дорогой операции expm()
        A_hash = hash(A_cont.tobytes())
        B_hash = hash(B_cont.tobytes()) 
        dt_rounded = round(float(dt), 8)  # точность dt до 8 знаков
        cache_key = (A_hash, B_hash, dt_rounded)
        
        if cache_key in self._discretization_cache:
            return self._discretization_cache[cache_key]
        
        # Вычисляем только если нет в кэше
        n = A_cont.shape[0]
        m = B_cont.shape[1]
        
        augmented_matrix = np.zeros((n + m, n + m))
        augmented_matrix[0:n, 0:n] = A_cont
        augmented_matrix[0:n, n:n+m] = B_cont
        
        phi = expm(augmented_matrix * dt)  # Дорогая операция!
        
        A_discrete = phi[0:n, 0:n]
        B_discrete = phi[0:n, n:n+m]
        
        # Сохраняем в кэш
        result = (A_discrete, B_discrete)
        self._discretization_cache[cache_key] = result
        
        return result

    def discrete_step(self, state: np.ndarray, control: float, dt: float) -> np.ndarray:
        """
        Выполняет один шаг дискретной динамики.
        """
        A_cont, B_cont = self.get_linearized_matrices_at_state(state)
        A_discrete, B_discrete = self.discretize(A_cont, B_cont, dt)
        
        # Убедимся, что state и control имеют правильную форму
        state = np.asarray(state).reshape(-1, 1)
        control = np.asarray(control).reshape(-1, 1)

        next_state = A_discrete @ state + B_discrete @ control
        return next_state.flatten()