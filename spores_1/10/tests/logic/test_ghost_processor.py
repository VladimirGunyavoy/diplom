import unittest
import numpy as np
import sys
import os

# Добавляем корневую директорию проекта в sys.path ('diplom/spores/10')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.logic.pendulum import PendulumSystem
from src.logic.ghost_processor import GhostProcessor

class TestGhostProcessor(unittest.TestCase):

    def setUp(self):
        """Настройка объектов для тестов."""
        self.pendulum = PendulumSystem(damping=0.1)
        self.dt = 0.1
        self.processor = GhostProcessor(self.pendulum, self.dt)

    def test_process(self):
        """
        Тест метода process. Проверяет количество и базовую корректность
        возвращаемых состояний.
        """
        current_state = np.array([0.1, 0.0])  # Небольшое отклонение от нуля
        controls = [-1.0, 0.0, 1.0]           # Три разных управляющих воздействия

        future_states = self.processor.process(current_state, controls)

        # 1. Проверяем, что количество будущих состояний равно количеству управлений
        self.assertEqual(len(future_states), len(controls))

        # 2. Проверяем, что каждое будущее состояние - это numpy-массив размерности 2
        for state in future_states:
            self.assertIsInstance(state, np.ndarray)
            self.assertEqual(state.shape, (2,))

        # 3. Проверяем, что будущие состояния не равны исходному (система должна сдвинуться)
        for state in future_states:
            self.assertFalse(np.allclose(state, current_state), 
                             "Будущее состояние не должно быть равно текущему.")

        # 4. Проверяем, что состояния для разных управлений отличаются
        # (если управление не нулевое и система не в точке равновесия)
        self.assertFalse(np.allclose(future_states[0], future_states[1]))
        self.assertFalse(np.allclose(future_states[1], future_states[2]))

if __name__ == '__main__':
    unittest.main() 