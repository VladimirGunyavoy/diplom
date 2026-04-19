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
        Использует scipy RK45 интегрирование для высокой точности.

        :param current_state_2d: np.array, текущая 2D-позиция (состояние).
        :param controls: list | np.array, список управляющих воздействий.
        :return: list[np.array], список будущих 2D-позиций "призраков".
        """
        # Оптимизация: заранее выделяем память под результат
        future_states = [None] * len(controls)

        for i, control in enumerate(controls):
            # Используем RK45 интегрирование для каждого управления
            next_state = self.pendulum.scipy_rk45_step(current_state_2d, control, self.dt)
            future_states[i] = next_state
            
        return future_states
    

    def process_backward(self, current_state_2d, controls):
        """
        Простой расчет призраков назад во времени.
        
        :param current_state_2d: np.array, текущее состояние
        :param controls: list, список управлений
        :return: list[np.array], список прошлых состояний
        """
        past_states = []
        
        for control in controls:
            # Простой шаг назад
            past_state = self.pendulum.scipy_rk45_step_backward(current_state_2d, control, self.dt)
            past_states.append(past_state)
        
        return past_states