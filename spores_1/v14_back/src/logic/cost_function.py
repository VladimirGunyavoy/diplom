import numpy as np

class CostFunction:
    """
    Чистый логический класс для расчета стоимости и ее градиента.
    Не зависит от Ursina и визуальных компонентов.
    """
    def __init__(self, goal_position_2d: np.ndarray, k=0.1, c=1.0):
        """
        Инициализирует функцию стоимости.
        
        Args:
            goal_position_2d (np.ndarray): 2D-координаты цели (x, z).
            k (float): Коэффициент в квадратичной формуле стоимости.
            c (float): Свободный член в формуле стоимости.
        """
        self.goal_position_2d = np.asarray(goal_position_2d, dtype=float)
        self.k = k
        self.c = c

    def get_cost(self, position_2d: np.ndarray) -> float:
        """
        Вычисляет стоимость для 2D-координат (x, z).
        
        Args:
            position_2d (np.ndarray): Текущая 2D-позиция.
            
        Returns:
            float: Значение стоимости.
        """
        distance_sq = np.sum((np.asarray(position_2d) - self.goal_position_2d)**2)
        return self.k * distance_sq + self.c

    def get_gradient(self, position_2d: np.ndarray) -> np.ndarray:
        """
        Вычисляет градиент функции стоимости в 2D-точке.
        
        Args:
            position_2d (np.ndarray): 2D-позиция для расчета градиента.
            
        Returns:
            np.ndarray: 2D-вектор градиента.
        """
        grad_2d = 2 * self.k * (np.asarray(position_2d) - self.goal_position_2d)
        return grad_2d 