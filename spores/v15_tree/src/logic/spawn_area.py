import numpy as np
from ..utils.poisson_disk import PoissonDiskSampler
from typing import List, Callable

class SpawnArea:
    """
    Класс для работы с n-мерным эллипсоидом (областью спавна),
    определенным двумя фокусами и эксцентриситетом.
    Полностью независим от Ursina.
    """
    def __init__(self, focus1: np.ndarray, focus2: np.ndarray, eccentricity: float):
        """
        Инициализирует эллипсоид.

        :param focus1: np.array, координаты первого фокуса.
        :param focus2: np.array, координаты второго фокуса.
        :param eccentricity: float, эксцентриситет (0 < e < 1).
        """
        self.A: np.ndarray = np.asarray(focus1, dtype=float)
        self.B: np.ndarray = np.asarray(focus2, dtype=float)
        self.eccentricity: float = float(eccentricity)

        if self.A.shape != self.B.shape:
            raise ValueError("Координаты фокусов должны иметь одинаковую размерность.")
        if not (0 < self.eccentricity < 1):
            raise ValueError("Эксцентриситет должен быть в диапазоне (0, 1).")
            
        self.n_dim: int = self.A.shape[0]
        
        # Добавляем список колбэков для UI
        self.on_eccentricity_change_callbacks: List[Callable[[float], None]] = []

        self._calculate_parameters()

    def set_eccentricity(self, new_eccentricity: float) -> None:
        """
        Устанавливает новый эксцентриситет и вызывает колбэки.
        """
        if not (0 < new_eccentricity < 1):
            # Можно просто игнорировать или выводить предупреждение
            return

        self.eccentricity = new_eccentricity
        self._calculate_parameters() # Пересчитываем параметры эллипса
        
        # Вызываем все зарегистрированные колбэки
        for callback in self.on_eccentricity_change_callbacks:
            callback(self.eccentricity)

    def _calculate_parameters(self) -> None:
        """
        Вычисляет центр, полуоси и матрицу преобразования для эллипсоида.
        """
        self.center: np.ndarray = (self.A + self.B) / 2
        dist_foci = np.linalg.norm(self.B - self.A)
        self.c: float = float(dist_foci / 2)

        if self.c < 1e-9:
             raise ValueError("Фокусы не могут совпадать для эллипса с e > 0.")

        self.a: float = self.c / self.eccentricity
        self.b: float = self.a * np.sqrt(1 - self.eccentricity**2)

        foci_vec = self.B - self.A
        n = foci_vec / dist_foci
        
        N = np.outer(n, n)
        self.T: np.ndarray = self.a * N + self.b * (np.eye(self.n_dim) - N)

    def get_points(self, n_points: int = 100) -> np.ndarray:
        """
        Возвращает точки на границе эллипсоида.
        """
        if self.n_dim == 2:
            theta = np.linspace(0, 2 * np.pi, n_points)
            points_on_circle = np.vstack((np.cos(theta), np.sin(theta))).T
            boundary_points = points_on_circle @ self.T + self.center
        else:
            points0 = np.random.randn(n_points, self.n_dim)
            points_on_sphere = points0 / np.linalg.norm(points0, axis=1, keepdims=True)
            boundary_points = points_on_sphere @ self.T + self.center
        
        return boundary_points

    def is_inside(self, point: np.ndarray) -> bool:
        """
        Проверяет, находится ли точка внутри эллипсоида.
        """
        point = np.asarray(point, dtype=float)
        centered_point = point - self.center
        
        try:
            T_inv = np.linalg.inv(self.T)
            transformed_point = centered_point @ T_inv.T
            distance_squared = np.sum(transformed_point**2)
            return distance_squared <= 1.0 or np.isclose(distance_squared, 1.0)
        except np.linalg.LinAlgError:
            return False

    def sample_random_point(self) -> np.ndarray:
        """
        Генерирует одну случайную точку внутри n-мерного эллипсоида.
        """
        while True:
            # Генерируем точку в единичной n-мерной сфере
            p = np.random.randn(self.n_dim)
            p /= np.linalg.norm(p)
            r = np.random.rand()**(1/self.n_dim)
            point_in_sphere = p * r
            
            # Преобразуем точку в эллипсоид
            point_in_ellipsoid = point_in_sphere @ self.T + self.center
            
            # Проверяем, что точка действительно внутри (для безопасности)
            if self.is_inside(point_in_ellipsoid):
                return point_in_ellipsoid

    def sample_poisson_disk(self, min_radius: float) -> np.ndarray:
        """
        Генерирует точки с контролируемым минимальным расстоянием внутри эллипсоида.
        """
        if not (0 < min_radius <= 2):
            raise ValueError("min_radius должен быть в диапазоне (0, 2].")

        sampler = PoissonDiskSampler(min_radius=min_radius, n_dim=self.n_dim)
        points_in_sphere = sampler.sample()

        if len(points_in_sphere) == 0:
            return np.array([]).reshape(0, self.n_dim)
            
        points_in_ellipsoid = points_in_sphere @ self.T + self.center
        
        return points_in_ellipsoid 