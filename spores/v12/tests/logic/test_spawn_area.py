import unittest
import numpy as np
import sys
import os

# Добавляем корневую директорию проекта в sys.path ('diplom/spores/10')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.logic.spawn_area import SpawnArea

class TestSpawnArea(unittest.TestCase):

    def setUp(self):
        """Настройка тестового эллипса для каждого теста."""
        self.focus1 = np.array([-3, 0])
        self.focus2 = np.array([3, 0])
        self.eccentricity = 0.6  # e = c/a => 3 / 5 = 0.6
        self.spawn_area = SpawnArea(self.focus1, self.focus2, self.eccentricity)

    def test_initialization(self):
        """Тест правильности вычисления ключевых параметров эллипса."""
        # c = половина расстояния между фокусами
        self.assertAlmostEqual(self.spawn_area.c, 3.0)
        # a = большая полуось
        self.assertAlmostEqual(self.spawn_area.a, 5.0)
        # b = малая полуось, b = a * sqrt(1-e^2) = 5 * sqrt(1-0.36) = 5 * 0.8 = 4
        self.assertAlmostEqual(self.spawn_area.b, 4.0)
        # Центр должен быть в начале координат
        np.testing.assert_array_almost_equal(self.spawn_area.center, np.array([0, 0]))

    def test_is_inside(self):
        """Тест для метода is_inside()."""
        # Точки, которые точно внутри
        self.assertTrue(self.spawn_area.is_inside([0, 0])) # Центр
        self.assertTrue(self.spawn_area.is_inside([3, 0])) # Фокус
        self.assertTrue(self.spawn_area.is_inside([-4.9, 0])) # Внутри, но близко к границе по большой оси

        # Точки на границе
        self.assertTrue(self.spawn_area.is_inside([5, 0]))  # Вершина на большой оси
        self.assertTrue(self.spawn_area.is_inside([0, 4]))  # Вершина на малой оси

        # Точки снаружи
        self.assertFalse(self.spawn_area.is_inside([5.1, 0]))
        self.assertFalse(self.spawn_area.is_inside([0, -4.1]))
        self.assertFalse(self.spawn_area.is_inside([10, 10]))

    def test_get_points(self):
        """Тест для метода get_points()."""
        num_points = 10
        points = self.spawn_area.get_points(n_points=num_points)
        
        # Проверяем, что количество точек правильное
        self.assertEqual(len(points), num_points)
        # Проверяем, что размерность точек правильная (2D)
        self.assertEqual(points.shape[1], 2)
        
        # Каждая сгенерированная точка должна (почти) удовлетворять условию нахождения на границе.
        # То есть is_inside для них должен быть True, но если проверить чуть дальше, будет False.
        for p in points:
            self.assertTrue(self.spawn_area.is_inside(p))
            # Проверим точку чуть дальше от центра по тому же направлению
            direction_vector = p - self.spawn_area.center
            outside_point = p + direction_vector * 0.01
            self.assertFalse(self.spawn_area.is_inside(outside_point))

if __name__ == '__main__':
    unittest.main() 