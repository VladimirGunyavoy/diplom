import numpy as np
from ursina import *
from .scalable import Scalable

class EllipsoidSampler:
    """
    Класс для работы с n-мерным эллипсоидом, определенным двумя фокусами
    и эксцентриситетом. Позволяет получать точки на его границе.
    """
    def __init__(self, focus1, focus2, eccentricity):
        """
        Инициализирует эллипсоид.

        :param focus1: np.array, координаты первого фокуса.
        :param focus2: np.array, координаты второго фокуса.
        :param eccentricity: float, эксцентриситет (0 < e < 1).
        """
        self.A = np.asarray(focus1, dtype=float)
        self.B = np.asarray(focus2, dtype=float)
        self.e = float(eccentricity)

        if self.A.shape != self.B.shape:
            raise ValueError("Координаты фокусов должны иметь одинаковую размерность.")
        if not (0 < self.e < 1):
            raise ValueError("Эксцентриситет должен быть в диапазоне (0, 1).")
            
        self.n_dim = self.A.shape[0]
        
        # Рассчитываем ключевые параметры эллипсоида
        self._calculate_parameters()

    def _calculate_parameters(self):
        """
        Вычисляет центр, полуоси и матрицу преобразования для эллипсоида.
        """
        self.center = (self.A + self.B) / 2
        dist_foci = np.linalg.norm(self.B - self.A)
        self.c = dist_foci / 2

        # Если фокусы совпадают, это сфера.
        if self.c < 1e-9:
             raise ValueError("Фокусы не могут совпадать для эллипса с e > 0.")

        self.a = self.c / self.e  # Большая полуось
        self.b = self.a * np.sqrt(1 - self.e**2)  # Малая полуось (одна для всех остальных осей)

        # Единичный вектор вдоль большой оси
        foci_vec = self.B - self.A
        n = foci_vec / dist_foci
        
        # Матрица проекции на большую ось
        N = np.outer(n, n)

        # Матрица линейного преобразования T.
        # Она преобразует единичную n-сферу в наш эллипсоид.
        # T = a * (проекция на большую ось) + b * (проекция на ортогональное дополнение)
        self.T = self.a * N + self.b * (np.eye(self.n_dim) - N)

    def get_points(self, n_points=100):
        """
        Возвращает точки на границе эллипсоида.
        Для 2D случая генерирует упорядоченные точки для гладкой линии.
        Для n_dim > 2 возвращает случайную выборку с поверхности.
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

class Ellipsoid2(Scalable):
    """
    Класс для визуализации n-мерного эллипсоида в Ursina.
    Наследуется от Scalable для совместимости с трансформациями.
    """
    def __init__(self, focus1, focus2, eccentricity, dimensions=None, resolution=32, mode='triangle', **kwargs):
        """
        Инициализирует и создает 3D-модель эллипсоида.
        :param dimensions: int, принудительно задает размерность (2 или 3). Если None, определяется автоматически.
        """
        super().__init__(**kwargs)

        # --- 1. Обработка входных данных ---
        def _get_coords(focus):
            # Если передан объект Ursina (Entity, Scalable и т.д.), берем его позицию
            if isinstance(focus, Scalable):
                return focus.real_position
            return focus

        coords1 = np.asarray(_get_coords(focus1), dtype=float)
        coords2 = np.asarray(_get_coords(focus2), dtype=float)

        # Определяем целевую размерность
        inferred_dim = len(coords1)
        target_dim = dimensions if dimensions is not None else inferred_dim

        if target_dim not in (2, 3):
            raise ValueError("Параметр dimensions должен быть 2 или 3.")

        # --- 2. Подготовка фокусов для сэмплера ---
        if target_dim == 2:
            # Для 2D эллипса используем X и Y координаты
            f1 = np.array([coords1[0], coords1[2]])
            f2 = np.array([coords2[0], coords2[2]])
        else: # target_dim == 3
            # Убедимся, что фокусы 3-мерные, дополнив нулями, если нужно
            if len(coords1) < 3:
                coords1 = np.append(coords1, [0] * (3 - len(coords1)))
            if len(coords2) < 3:
                coords2 = np.append(coords2, [0] * (3 - len(coords2)))
            f1 = coords1
            f2 = coords2
        
        # --- 3. Использование сэмплера ---
        self.sampler = EllipsoidSampler(f1, f2, eccentricity)
        
        # --- 4. Создание сетки (Mesh) ---
        if self.sampler.n_dim == 2:
            # Для 2D эллипса создаем линию на плоскости XY
            points_2d = self.sampler.get_points(n_points=resolution)
            self.vertices = [(v[0], 0, v[1]) for v in points_2d] # Используем X, Y
            
            if mode == 'line' and len(self.vertices) > 1:
                self.vertices.append(self.vertices[0])
            
            mesh_model = Mesh(vertices=self.vertices, mode=mode, thickness=self.thickness)

        elif self.sampler.n_dim == 3:
            # Для 3D эллипсоида генерируем точки на единичной сфере,
            # вычисляем их триангуляцию и затем трансформируем.
            try:
                from scipy.spatial import ConvexHull
            except ImportError:
                print("Ошибка: для создания 3D сетки требуется библиотека Scipy.")
                print("Пожалуйста, установите ее: pip install scipy")
                return

            # Генерируем точки на поверхности единичной сферы.
            # Используем больше точек для построения качественной сетки.
            n_mesh_points = max(resolution * 4, 64) # resolution - условная детализация
            points_on_sphere = np.random.randn(n_mesh_points, 3)
            points_on_sphere /= np.linalg.norm(points_on_sphere, axis=1, keepdims=True)

            # Вычисляем выпуклую оболочку (триангуляцию для сферы)
            hull = ConvexHull(points_on_sphere)
            
            # Трансформируем вершины, используя матрицу из сэмплера
            transformed_verts = points_on_sphere @ self.sampler.T + self.sampler.center

            # Конвертируем numpy array в list, чтобы избежать ошибки bool-проверки в ursina
            triangles_list = [list(s) for s in hull.simplices]
            
            mesh_model = Mesh(
                vertices=[tuple(v) for v in transformed_verts],
                triangles=triangles_list,
                mode='triangle'
            )
        else:
            raise ValueError("Визуализация поддерживается только для 2D и 3D эллипсоидов.")

        # Создаем дочернюю сущность с нашей сгенерированной сеткой
        self.mesh_entity = Entity(
            parent=self,
            model=mesh_model,
            **kwargs
        ) 