import numpy as np

class GhostProcessor:
    """
    Чистый логический класс для расчета позиций "призраков".
    Не зависит от Ursina и других визуальных компонентов.
    """
    def __init__(self, pendulum_system, dt):
        """
        :param pendulum_system: Экземпляр класса PendulumSystem.
        :param dt: Шаг дискретизации по времени.
        """
        self.pendulum = pendulum_system
        self.dt = dt
        
        # Оптимизация: буфер для control массива чтобы избежать np.array([control]) в цикле
        self._control_buffer = np.array([0.0], dtype=float)

    def process(self, current_state_2d, controls):
        """
        Рассчитывает будущие 2D-позиции на основе текущего состояния и набора управлений.

        :param current_state_2d: np.array, текущая 2D-позиция (состояние).
        :param controls: list | np.array, список управляющих воздействий.
        :return: list[np.array], список будущих 2D-позиций "призраков".
        """
        # Оптимизация: заранее выделяем память под результат
        future_states = [None] * len(controls)
        A, B = self.pendulum.get_linearized_matrices_at_state(current_state_2d)
        Ad, Bd = self.pendulum.discretize(A, B, self.dt)

        for i, control in enumerate(controls):
            # Оптимизация: используем буфер вместо создания np.array([control])
            self._control_buffer[0] = control
            # x_k+1 = Ad * x_k + Bd * u_k
            next_state = Ad @ current_state_2d + Bd @ self._control_buffer
            future_states[i] = next_state
            
        return future_states 