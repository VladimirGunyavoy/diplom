import unittest
import numpy as np
import sys
import os

# Добавляем корневую папку проекта в sys.path
# Это необходимо, чтобы можно было импортировать из 'src'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.logic.cost_function import CostFunction

class TestCostFunction(unittest.TestCase):

    def setUp(self):
        """Настройка тестового окружения перед каждым тестом."""
        self.goal_pos_2d = np.array([10.0, 5.0])
        self.cost_func = CostFunction(goal_position_2d=self.goal_pos_2d, k=0.1, c=1.0)

    def test_get_cost_at_goal(self):
        """Тест: стоимость в точке цели должна быть равна 'c'."""
        cost = self.cost_func.get_cost(self.goal_pos_2d)
        self.assertAlmostEqual(cost, 1.0, msg="Стоимость в цели должна быть 1.0")

    def test_get_cost_at_point(self):
        """Тест: стоимость в произвольной точке."""
        point_2d = np.array([13.0, 9.0])
        # Ожидаемое значение: 0.1 * ((13-10)^2 + (9-5)^2) + 1.0
        # = 0.1 * (3^2 + 4^2) + 1.0
        # = 0.1 * (9 + 16) + 1.0
        # = 0.1 * 25 + 1.0 = 2.5 + 1.0 = 3.5
        expected_cost = 3.5
        cost = self.cost_func.get_cost(point_2d)
        self.assertAlmostEqual(cost, expected_cost, msg="Неверный расчет стоимости в точке")

    def test_get_gradient_at_goal(self):
        """Тест: градиент в точке цели должен быть нулевым вектором."""
        grad = self.cost_func.get_gradient(self.goal_pos_2d)
        np.testing.assert_array_almost_equal(grad, np.array([0.0, 0.0]),
                                              err_msg="Градиент в цели должен быть [0, 0]")

    def test_get_gradient_at_point(self):
        """Тест: градиент в произвольной точке."""
        point_2d = np.array([13.0, 9.0])
        # Ожидаемое значение: 2 * 0.1 * ([13, 9] - [10, 5])
        # = 0.2 * [3, 4] = [0.6, 0.8]
        expected_grad = np.array([0.6, 0.8])
        grad = self.cost_func.get_gradient(point_2d)
        np.testing.assert_array_almost_equal(grad, expected_grad,
                                              err_msg="Неверный расчет градиента в точке")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 