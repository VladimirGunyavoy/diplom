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

    def process(self, current_state_2d, controls):
        """
        Рассчитывает будущие 2D-позиции на основе текущего состояния и набора управлений.

        :param current_state_2d: np.array, текущая 2D-позиция (состояние).
        :param controls: list | np.array, список управляющих воздействий.
        :return: list[np.array], список будущих 2D-позиций "призраков".
        """
        future_states = []
        A, B = self.pendulum.get_linearized_matrices_at_state(current_state_2d)
        Ad, Bd = self.pendulum.discretize(A, B, self.dt)

        for control in controls:
            # x_k+1 = Ad * x_k + Bd * u_k
            next_state = Ad @ current_state_2d + Bd @ np.array([control])
            future_states.append(next_state)
            
        return future_states 