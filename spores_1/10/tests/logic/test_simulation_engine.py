import unittest
import numpy as np

from src.core.simulation_engine import SimulationEngine
from src.logic.pendulum import PendulumSystem
from src.logic.cost_function import CostFunction

class TestSimulationEngine(unittest.TestCase):
    
    def setUp(self):
        """Настройка, которая выполняется перед каждым тестом."""
        self.pendulum = PendulumSystem(g=9.81, l=1.0, m=1.0, damping=0.1)
        self.cost_function = CostFunction(goal_position_2d=np.array([0.0, 0.0]))
        self.initial_state = np.array([np.pi / 4, 0.0]) # Начальное состояние [theta, theta_dot]
        self.dt = 0.05
        
        self.engine = SimulationEngine(
            pendulum_system=self.pendulum,
            cost_function=self.cost_function,
            initial_state=self.initial_state,
            dt=self.dt
        )

    def test_initialization(self):
        """Тест правильной инициализации движка."""
        np.testing.assert_array_equal(self.engine.get_state(), self.initial_state)
        self.assertIsInstance(self.engine, SimulationEngine)

    def test_step_changes_state(self):
        """Тест того, что метод step изменяет состояние."""
        initial_state_copy = self.engine.get_state()
        
        # Выполняем шаг с ненулевым управлением
        self.engine.step(control=0.5)
        
        new_state = self.engine.get_state()
        
        # Проверяем, что состояние изменилось
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(new_state, initial_state_copy)
            
    def test_zero_control_at_rest(self):
        """
        Тест: если система в равновесии (0, 0) и управление равно нулю,
        состояние не должно меняться.
        """
        # Устанавливаем начальное состояние в точку равновесия
        zero_state = np.array([0.0, 0.0])
        engine = SimulationEngine(
            pendulum_system=self.pendulum,
            cost_function=self.cost_function,
            initial_state=zero_state,
            dt=self.dt
        )
        
        engine.step(control=0.0)
        
        new_state = engine.get_state()
        
        # Ожидаем, что состояние останется очень близким к нулю
        np.testing.assert_allclose(new_state, zero_state, atol=1e-9)

if __name__ == '__main__':
    unittest.main() 