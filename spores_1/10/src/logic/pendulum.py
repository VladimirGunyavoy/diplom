import numpy as np
from scipy.linalg import expm

class PendulumSystem:
    """
    Класс, описывающий систему маятника.
    Позволяет выполнять линеаризацию и дискретизацию в произвольном состоянии.
    """
    def __init__(self, 
                 g=9.81,
                 l=2.0,
                 m=1.0,
                 damping=0.1, 
                 max_control=1):
        self.g = g
        self.l = l
        self.m = m
        self.damping = damping
        self.max_control = max_control
        
    def get_control_bounds(self):
        return np.array([-self.max_control, self.max_control])
        
    def get_linearized_matrices_at_state(self, state):
        """
        Линеаризация нелинейной динамики маятника в произвольном состоянии.
        Возвращает непрерывные матрицы A и B.
        """
        theta_0, _ = state
        
        A_cont = np.array([
            [0.0, 1.0],
            [-self.g / self.l * np.cos(theta_0), -self.damping]
        ])
        
        B_cont = np.array([
            [0.0],
            [1.0]
        ])
        
        return A_cont, B_cont

    def discretize(self, A_cont, B_cont, dt):
        """
        Дискретизация непрерывной системы с помощью матричной экспоненты.
        """
        n = A_cont.shape[0]
        m = B_cont.shape[1]
        
        augmented_matrix = np.zeros((n + m, n + m))
        augmented_matrix[0:n, 0:n] = A_cont
        augmented_matrix[0:n, n:n+m] = B_cont
        
        phi = expm(augmented_matrix * dt)
        
        A_discrete = phi[0:n, 0:n]
        B_discrete = phi[0:n, n:n+m]
        
        return A_discrete, B_discrete

    def discrete_step(self, state, control, dt):
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