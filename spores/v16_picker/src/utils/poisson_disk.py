import numpy as np
import itertools
from typing import List, Tuple

class PoissonDiskSampler:
    """
    Реализация алгоритма Бридсона для сэмплирования дисков Пуассона.
    Генерирует точки, минимальное расстояние между которыми не меньше заданного,
    внутри n-мерной единичной гиперсферы (с центром в нуле и радиусом 1).
    """
    def __init__(self, min_radius: float, n_dim: int, k: int = 30):
        """
        :param min_radius: float, минимальное расстояние между точками. Должно быть в (0, 2].
        :param n_dim: int, размерность пространства.
        :param k: int, количество попыток найти точку-кандидата вокруг активной точки.
        """
        self.min_r: float = min_radius
        self.n_dim: int = n_dim
        self.k: int = k
        
        # Размер ячейки фоновой сетки. В каждой ячейке может быть не более одной точки.
        self.cell_size: float = self.min_r / np.sqrt(self.n_dim)
        
        # Размер сетки. Домен [-1, 1], поэтому его размер 2.
        self.grid_size: int = int(np.ceil(2.0 / self.cell_size))
        
        # Инициализируем сетку. -1 означает, что ячейка пуста.
        self.grid: np.ndarray = -np.ones((self.grid_size,) * self.n_dim, dtype=int)
        
        self.points: List[np.ndarray] = []
        self.active_list: List[int] = []

    def _is_in_bounds(self, p: np.ndarray) -> bool:
        """Проверяет, находится ли точка внутри единичной гиперсферы."""
        return bool(np.linalg.norm(p) <= 1.0)

    def _get_grid_coords(self, p: np.ndarray) -> Tuple[int, ...]:
        """Получает целочисленные координаты ячейки сетки для точки p."""
        # Сдвигаем домен с [-1, 1] на [0, 2] и делим на размер ячейки
        return tuple(((p + 1.0) / self.cell_size).astype(int))

    def sample(self) -> np.ndarray:
        """
        Выполняет сэмплирование.
        :return: np.array, массив сгенерированных точек.
        """
        # --- Шаг 1: Инициализация ---
        # Создаем первую точку случайным образом внутри сферы.
        while True:
            p0 = np.random.uniform(-1, 1, self.n_dim)
            if self._is_in_bounds(p0):
                break
        
        self._add_point(p0)

        # --- Шаг 2: Основной цикл ---
        while self.active_list:
            # Выбираем случайную активную точку
            idx = np.random.randint(len(self.active_list))
            p_active_idx = self.active_list[idx]
            p_active = self.points[p_active_idx]
            
            found_candidate = False
            for _ in range(self.k):
                # Генерируем кандидата в кольце вокруг активной точки
                p_cand = self._generate_candidate(p_active)

                # Проверяем, подходит ли кандидат
                if self._is_in_bounds(p_cand) and self._is_valid(p_cand):
                    self._add_point(p_cand)
                    found_candidate = True
                    break
            
            # Если не нашли подходящего кандидата, деактивируем точку
            if not found_candidate:
                self.active_list.pop(idx)

        return np.array(self.points) if self.points else np.array([]).reshape(0, self.n_dim)

    def _add_point(self, p: np.ndarray) -> None:
        """Добавляет точку в результирующий список, активный список и на сетку."""
        point_idx = len(self.points)
        self.points.append(p)
        self.active_list.append(point_idx)
        grid_coords = self._get_grid_coords(p)
        self.grid[grid_coords] = point_idx

    def _generate_candidate(self, p: np.ndarray) -> np.ndarray:
        """Генерирует случайную точку-кандидата вокруг точки p."""
        # Случайный радиус в диапазоне [r, 2r]
        r = np.random.uniform(self.min_r, 2 * self.min_r)
        # Случайное направление (единичный вектор)
        v = np.random.randn(self.n_dim)
        v /= np.linalg.norm(v)
        return p + r * v

    def _is_valid(self, p_cand: np.ndarray) -> bool:
        """Проверяет, не находится ли кандидат слишком близко к существующим точкам."""
        grid_coords = self._get_grid_coords(p_cand)
        
        # Определяем область поиска в сетке (гиперкуб со стороной 5 ячеек)
        search_min = np.maximum(0, np.array(grid_coords) - 2)
        search_max = np.minimum(self.grid_size - 1, np.array(grid_coords) + 2)

        # Создаем итератор по всем соседним ячейкам в n-мерном пространстве
        ranges = [range(s_min, s_max + 1) for s_min, s_max in zip(search_min, search_max)]
        for neighbor_coords in itertools.product(*ranges):
            point_idx = self.grid[neighbor_coords]
            if point_idx != -1: # Если в ячейке есть точка
                dist = np.linalg.norm(p_cand - self.points[int(point_idx)])
                if dist < self.min_r:
                    return False # Нашли слишком близкого соседа
        
        return True 