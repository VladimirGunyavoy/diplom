from ursina import *
import numpy as np
from scipy.spatial import Delaunay
from colorsys import hsv_to_rgb
import matplotlib.pyplot as plt
from matplotlib import contour as mpl_contour

from src.utils.scalable import Scalable
from .ellipse2 import EllipsoidSampler

class Cost(Scalable):
    def __init__(self, 
                 goal_position,
                 spawn_area,
                 color_manager=None,
                 config=None,
                 **kwargs):
        super().__init__(**kwargs)
        
        if isinstance(goal_position, Scalable):
            self.goal_position = goal_position.real_position
        else:
            self.goal_position = np.asarray(goal_position, dtype=float)

        self.spawn_area = spawn_area
        self.color_manager = color_manager
        self.config = config
        self.on_rebuild = None
        
        self.mesh_entity = None
        self.wireframe_entity = None
        self.points_entity = None  # Для визуализации точек сетки
        self.edges_entity = None   # Для визуализации ребер триангуляции
        self.contours_entity = None  # Для визуализации изолиний
        self.mesh_kwargs = {}
        self.last_vertices_3d = []  # Сохраняем последние вершины для визуализации
        self.last_triangulation = None  # Сохраняем триангуляцию для ребер
        self.vertices = []
        self.triangles = []
        self.colors = []
        self.cost_function = lambda x: np.sum((x - self.goal_position.calc_2d_pos())**2)

    def get_cost(self, position_2d):
        """Вычисляет стоимость (высоту y) для 2D-координат (x, z)."""
        # self.goal_position - это 3D numpy-массив (x, y, z)
        # извлекаем из него 2D-координаты для расчета стоимости.
        goal_2d = np.array([self.goal_position[0], self.goal_position[2]])
        
        # Формула стоимости
        distance = np.linalg.norm(position_2d - goal_2d)
        k = 1 / 10
        с = 1
        return k * distance**2 + с

    def gradient(self, position):
        if isinstance(position, Scalable):
            position = position.real_position

        # Проекция на плоскость XZ для вычисления градиента
        pos_2d = np.array([position[0], position[2]])
        goal_2d = np.array([self.goal_position[0], self.goal_position[2]])
        
        # Градиент вычисляется от 2D-вектора
        grad_2d = 2 * (1/10) * (pos_2d - goal_2d)
        
        # Возвращаем 3D-вектор градиента, лежащий в плоскости XZ
        return np.array([grad_2d[0], 0, grad_2d[1]])

    def _generate_point_cloud(self, boundary_points=128, interior_min_radius=0.01):
        """
        Создает 2D-облако точек, комбинируя точки с границы эллипса
        и точки, сэмплированные внутри него по методу Poisson-disk.
        """
        sampler = self.spawn_area.sampler
        
        # 1. Точки на границе
        boundary = sampler.get_points(n_points=boundary_points)
        
        # 2. Точки внутри
        interior = sampler.sample_poisson_disk(min_radius=interior_min_radius)
        
        # 3. Объединение
        if interior.size == 0:
            return boundary
        
        point_cloud = np.vstack([boundary, interior])
        return point_cloud

    def _height_to_rainbow_color(self, height, min_height, max_height, alpha=1.0):
        """Конвертирует высоту в радужный цвет с настраиваемой прозрачностью"""
        # Нормализуем высоту в диапазон [0, 1]
        normalized = (height - min_height) / (max_height - min_height + 1e-8)
        
        # Используем HSV для создания радужного спектра
        # Hue от 240° (синий) до 0° (красный) для перехода от низкого к высокому
        hue = (1.0 - normalized) * 240 / 360  # от синего к красному
        saturation = 0.8
        value = 0.9
        
        # Конвертируем HSV в RGB
        r, g, b = hsv_to_rgb(hue, saturation, value)
        return (r, g, b, alpha)

    def generate_mesh(self, alpha=1.0, **kwargs):
        """
        Генерирует 3D-меш поверхности стоимости на основе 2D-области с радужным цветовым кодированием.
        """
        # Извлекаем параметры генерации точек из kwargs или используем значения по умолчанию
        boundary_points = kwargs.pop('boundary_points', 128)
        interior_min_radius = kwargs.pop('interior_min_radius', 0.01)

        # 1. Получаем 2D точки
        points_2d = self._generate_point_cloud(
            boundary_points=boundary_points,
            interior_min_radius=interior_min_radius
        )
        
        # 2. Триангуляция
        if len(points_2d) < 3:
            return None # Невозможно построить сетку
        
        try:
            tri = Delaunay(points_2d)
        except Exception as e:
            print(f"Ошибка триангуляции Делоне: {e}")
            return None

        # 3. Создание 3D-вершин с вычислением высот
        vertices_3d = []
        heights = []
        
        for p in points_2d:
            cost = self.get_cost(p)
            heights.append(cost)
            # Координаты (x, z) из 2D-точки, y - это cost
            vertices_3d.append((p[0], cost, p[1]))

        # 4. Вычисляем диапазон высот для цветового кодирования
        min_height = min(heights)
        max_height = max(heights)
        
        # 5. Создаем цвета для каждой вершины с учетом прозрачности
        vertex_colors = []
        for height in heights:
            color_rgba = self._height_to_rainbow_color(height, min_height, max_height, alpha)
            vertex_colors.append(color_rgba)

        # 6. Создание меша с дублированными треугольниками для двусторонней видимости
        front_faces = tri.simplices.tolist()
        back_faces = [list(reversed(face)) for face in front_faces]
        all_faces = front_faces + back_faces
        
        # Дублируем цвета вершин для обратных треугольников
        all_vertex_colors = vertex_colors + vertex_colors
        all_vertices = vertices_3d + vertices_3d
        
        # Сохраняем исходные вершины для визуализации точек
        self.last_vertices_3d = vertices_3d.copy()
        
        # Сохраняем триангуляцию для визуализации ребер
        self.last_triangulation = tri
        
        mesh = Mesh(
            vertices=all_vertices,
            triangles=all_faces,
            colors=all_vertex_colors,
            mode='triangle'
        )

        # Создаем Entity для меша
        entity_kwargs = self.mesh_kwargs.copy()
        entity_kwargs.update(kwargs)
        
        self.mesh_entity = Entity(parent=self, model=mesh, **entity_kwargs)
        self.mesh_entity.alpha = self.color_manager.get_value('cost_surface', 'alpha')

        # Запускаем колбэк для перерисовки (если он есть)
        if self.on_rebuild:
            self.on_rebuild()
        
        return self.mesh_entity

    def generate_surface(self):
        # --- Параметры из конфига ---
        cost_config = self.config['cost_surface']
        mesh_gen_config = cost_config['mesh_generation']
        
        show_points = cost_config['show_points']
        show_edges = cost_config['show_edges']
        show_contours = cost_config['show_contours']
        
        boundary_points_count = mesh_gen_config['boundary_points']
        min_radius = mesh_gen_config['interior_min_radius']
        
        # --- Генерация точек ---
        # self.generate_points(boundary_points_count, min_radius) # УДАЛЕНО: точки генерируются внутри generate_mesh
        
        # --- Генерация меша ---
        
        # Уничтожаем старые меши, если они есть
        if self.mesh_entity:
            destroy(self.mesh_entity)
        if self.wireframe_entity:
            destroy(self.wireframe_entity)
        if self.points_entity:
            destroy(self.points_entity)
        if self.edges_entity:
            destroy(self.edges_entity)
        if self.contours_entity:
            destroy(self.contours_entity)
        
        # Сохраняем аргументы для перестройки
        self.mesh_kwargs = {}

        # Извлекаем параметры для генерации меша
        surface_alpha = cost_config.get('alpha', 1.0)  # Прозрачность поверхности
        
        mesh_generation_params = {
            'boundary_points': boundary_points_count,
            'interior_min_radius': min_radius
        }

        mesh_model = self.generate_mesh(alpha=surface_alpha, **mesh_generation_params)
        
        if mesh_model:
            # Готовим аргументы для создания Entity, копируя исходные
            entity_kwargs = cost_config.copy()
            # Устанавливаем все флаги для надежной двусторонней видимости
            entity_kwargs['two_sided'] = True
            entity_kwargs['unlit'] = False  # Включаем освещение для лучшего вида
            
            # Удаляем наши служебные параметры, чтобы не передавать их в Entity
            entity_kwargs.pop('mesh_generation', None)
            entity_kwargs.pop('color', None)  # Убираем статический цвет
            entity_kwargs.pop('show_points', None)  # Убираем параметр для точек
            entity_kwargs.pop('point_size', None)  # Убираем размер точек
            entity_kwargs.pop('show_edges', None)  # Убираем параметр для ребер
            entity_kwargs.pop('edge_color', None)  # Убираем цвет ребер
            entity_kwargs.pop('edge_thickness', None)  # Убираем толщину ребер
            entity_kwargs.pop('show_contours', None)  # Убираем параметр для изолиний
            entity_kwargs.pop('contour_levels', None)  # Убираем количество уровней
            entity_kwargs.pop('contour_color', None)  # Убираем цвет изолиний
            entity_kwargs.pop('contour_thickness', None)  # Убираем толщину изолиний
            entity_kwargs.pop('contour_resolution', None)  # Убираем разрешение сетки
            entity_kwargs.pop('alpha', None)  # Убираем прозрачность

            # Создаем основную поверхность
            self.mesh_entity = Entity(
                model=mesh_model,
                parent=self,
                render_queue=-1,
                **entity_kwargs
            )
            
            # Создаем видимую сетку (wireframe) поверх основной модели
            wireframe_color = (self.color_manager.get_color('cost_surface', 'wireframe')
                             if self.color_manager else color.white)
            
            self.wireframe_entity = Entity(
                parent=self.mesh_entity, # Делаем дочерней, чтобы наследовать трансформации
                model=mesh_model,
                mode='line',
                color=wireframe_color,
                thickness=1,
                unlit=True
            )
            
            # Создаем визуализацию точек сетки, если запрошено
            if show_points:
                point_size = entity_kwargs.get('point_size', 0.02)
                self._create_points_visualization(point_size)
            
            # Создаем визуализацию ребер триангуляции, если запрошено
            if show_edges:
                edge_color = entity_kwargs.get('edge_color', color.cyan)
                edge_thickness = entity_kwargs.get('edge_thickness', 1)
                self._create_edges_visualization(edge_color, edge_thickness)
            
            # Создаем изолинии поверхности стоимости, если запрошено
            if show_contours:
                contour_levels = entity_kwargs.get('contour_levels', 8)
                contour_color = entity_kwargs.get('contour_color', None)
                contour_thickness = entity_kwargs.get('contour_thickness', 2)
                contour_resolution = entity_kwargs.get('contour_resolution', 50)
                self._create_contour_lines(contour_levels, contour_color, contour_thickness, contour_resolution)
            
            # Подписываемся на коллбэк, если еще не подписаны
            if self.rebuild_mesh not in self.spawn_area.on_rebuild_callbacks:
                self.spawn_area.on_rebuild_callbacks.append(self.rebuild_mesh)
            return self.mesh_entity
        
        # Если не удалось создать, отписываемся
        if self.rebuild_mesh in self.spawn_area.on_rebuild_callbacks:
             self.spawn_area.on_rebuild_callbacks.remove(self.rebuild_mesh)
        
        self.mesh_entity = None
        self.wireframe_entity = None
        self.points_entity = None
        self.edges_entity = None
        self.contours_entity = None
        return None
    
    def rebuild_mesh(self):
        """
        Перестраивает 3D-меш поверхности. Вызывается по коллбэку от SpawnArea.
        Использует сохраненные kwargs от последнего вызова generate_surface.
        """
        print("Rebuilding cost surface mesh via callback...")
        self.generate_surface()
    
    def _create_points_visualization(self, point_size=0.02):
        """Создает визуализацию точек сетки в виде маленьких сфер"""
        if not self.last_vertices_3d:
            return None
            
        # Уничтожаем старую визуализацию точек
        if self.points_entity:
            destroy(self.points_entity)
        
        # Создаем родительский объект для всех точек
        self.points_entity = Entity(parent=self)
        
        # Получаем цвет точек из color_manager или используем желтый по умолчанию
        points_color = (self.color_manager.get_color('cost_surface', 'mesh_points') 
                       if self.color_manager else color.yellow)
        
        # Создаем маленькие сферы в каждой вершине
        for i, vertex in enumerate(self.last_vertices_3d):
            point_sphere = Entity(
                model='sphere',
                scale=point_size,
                position=vertex,
                color=points_color,
                parent=self.points_entity,
                unlit=True
            )
        
        return self.points_entity
    
    def _create_edges_visualization(self, edge_color=None, edge_thickness=1):
        """Создает визуализацию ребер триангуляции"""
        if not self.last_vertices_3d or not self.last_triangulation:
            return None
            
        # Уничтожаем старую визуализацию ребер
        if self.edges_entity:
            destroy(self.edges_entity)
        
        # Создаем родительский объект для всех ребер
        self.edges_entity = Entity(parent=self)
        
        # Получаем цвет ребер из color_manager или используем переданный/cyan по умолчанию
        if edge_color is None:
            edge_color = (self.color_manager.get_color('cost_surface', 'triangulation_edges')
                         if self.color_manager else color.cyan)
        
        # Получаем все уникальные ребра из триангуляции
        edges = set()
        for triangle in self.last_triangulation.simplices:
            # Добавляем все три ребра треугольника
            edges.add((min(triangle[0], triangle[1]), max(triangle[0], triangle[1])))
            edges.add((min(triangle[1], triangle[2]), max(triangle[1], triangle[2])))
            edges.add((min(triangle[0], triangle[2]), max(triangle[0], triangle[2])))
        
        # Создаем линии для каждого ребра
        for edge in edges:
            vertex1 = self.last_vertices_3d[edge[0]]
            vertex2 = self.last_vertices_3d[edge[1]]
            
            # Создаем линию между двумя вершинами
            edge_line = Entity(
                model=Mesh(
                    vertices=[vertex1, vertex2],
                    mode='line',
                    thickness=edge_thickness
                ),
                color=edge_color,
                parent=self.edges_entity,
                unlit=True
            )
        
        return self.edges_entity
    
    def _create_contour_lines(self, num_levels=8, contour_color=None, contour_thickness=2, grid_resolution=50):
        """Создает изолинии (контурные линии) поверхности стоимости"""
        # Уничтожаем старые изолинии
        if self.contours_entity:
            destroy(self.contours_entity)
        
        # Создаем родительский объект для всех изолиний
        self.contours_entity = Entity(parent=self)
        
        # Получаем цвет изолиний из color_manager или используем по умолчанию
        if contour_color is None:
            contour_color = (self.color_manager.get_color('cost_surface', 'contour_lines')
                           if self.color_manager else color.orange)
        
        # Получаем границы области эллипса
        sampler = self.spawn_area.sampler
        
        # Находим ограничивающий прямоугольник
        boundary_points = sampler.get_points(n_points=100)
        x_min, x_max = boundary_points[:, 0].min(), boundary_points[:, 0].max()
        z_min, z_max = boundary_points[:, 1].min(), boundary_points[:, 1].max()
        
        # Расширяем границы немного
        margin = 0.1
        x_range = x_max - x_min
        z_range = z_max - z_min
        x_min -= margin * x_range
        x_max += margin * x_range
        z_min -= margin * z_range
        z_max += margin * z_range
        
        # Создаем регулярную сетку
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        z_grid = np.linspace(z_min, z_max, grid_resolution)
        X, Z = np.meshgrid(x_grid, z_grid)
        
        # Вычисляем значения функции стоимости на сетке
        Y = np.zeros_like(X)
        mask = np.zeros_like(X, dtype=bool)  # Маска для точек внутри эллипса
        
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                point_2d = np.array([X[i, j], Z[i, j]])
                
                # Проверяем, находится ли точка внутри эллипса
                if sampler.is_inside(point_2d):
                    Y[i, j] = self.get_cost(point_2d)
                    mask[i, j] = True
        
        # Маскируем точки вне эллипса
        Y_masked = np.ma.masked_where(~mask, Y)
        
        # Находим диапазон значений для определения уровней изолиний
        valid_values = Y[mask]
        if len(valid_values) == 0:
            return None
            
        min_val, max_val = valid_values.min(), valid_values.max()
        
        # Создаем уровни для изолиний
        levels = np.linspace(min_val, max_val, num_levels + 2)[1:-1]  # Исключаем крайние значения
        
        # Создаем контурные линии с помощью matplotlib
        try:
            # Создаем фиктивную фигуру для вычисления контуров
            fig, ax = plt.subplots(figsize=(1, 1))
            cs = ax.contour(X, Z, Y_masked, levels=levels)
            plt.close(fig)  # Закрываем фигуру, она нам не нужна
            
            # Извлекаем пути контуров - совместимо с разными версиями matplotlib
            if hasattr(cs, 'allsegs'):
                # Старый API (matplotlib < 3.8)
                for level_idx, level_segs in enumerate(cs.allsegs):
                    for seg in level_segs:
                        if len(seg) < 2:
                            continue
                        
                        # Конвертируем 2D контур в 3D линию на поверхности
                        vertices_3d = []
                        for vertex in seg:
                            x, z = vertex
                            y = self.get_cost(np.array([x, z]))  # Высота на поверхности
                            vertices_3d.append((x, y + 0.01, z))  # Немного приподнимаем над поверхностью
                        
                        # Создаем линию изолинии
                        if len(vertices_3d) > 1:
                            contour_line = Entity(
                                model=Mesh(
                                    vertices=vertices_3d,
                                    mode='line',
                                    thickness=contour_thickness
                                ),
                                color=contour_color,
                                parent=self.contours_entity,
                                unlit=True
                            )
            
            elif hasattr(cs, 'collections'):
                # Средний API (matplotlib 3.8+)
                for level_idx, collection in enumerate(cs.collections):
                    for path in collection.get_paths():
                        vertices = path.vertices
                        if len(vertices) < 2:
                            continue
                        
                        # Конвертируем 2D контур в 3D линию на поверхности
                        vertices_3d = []
                        for vertex in vertices:
                            x, z = vertex
                            y = self.get_cost(np.array([x, z]))  # Высота на поверхности
                            vertices_3d.append((x, y + 0.01, z))  # Немного приподнимаем над поверхностью
                        
                        # Создаем линию изолинии
                        if len(vertices_3d) > 1:
                            contour_line = Entity(
                                model=Mesh(
                                    vertices=vertices_3d,
                                    mode='line',
                                    thickness=contour_thickness
                                ),
                                color=contour_color,
                                parent=self.contours_entity,
                                unlit=True
                            )
            
            else:
                # Новый API (matplotlib 3.9+) - прямой доступ к путям
                for level_idx, level in enumerate(levels):
                    # Получаем пути для конкретного уровня
                    paths = cs.get_paths()
                    if level_idx < len(paths):
                        path = paths[level_idx]
                        vertices = path.vertices
                        if len(vertices) < 2:
                            continue
                        
                        # Конвертируем 2D контур в 3D линию на поверхности
                        vertices_3d = []
                        for vertex in vertices:
                            x, z = vertex
                            y = self.get_cost(np.array([x, z]))  # Высота на поверхности
                            vertices_3d.append((x, y + 0.01, z))  # Немного приподнимаем над поверхностью
                        
                        # Создаем линию изолинии
                        if len(vertices_3d) > 1:
                            contour_line = Entity(
                                model=Mesh(
                                    vertices=vertices_3d,
                                    mode='line',
                                    thickness=contour_thickness
                                ),
                                color=contour_color,
                                parent=self.contours_entity,
                                unlit=True
                            )
        
        except Exception as e:
            print(f"Ошибка при создании изолиний с matplotlib: {e}")
            print("Попытка создания простых изолиний...")
            # Fallback: простые круговые изолинии вокруг цели
            self._create_simple_contours(num_levels, contour_color, contour_thickness)
        
        return self.contours_entity
    
    def _create_simple_contours(self, num_levels=8, contour_color=None, contour_thickness=2):
        """
        Простой fallback метод для создания круговых изолиний вокруг цели.
        Используется, если matplotlib API не работает.
        """
        if contour_color is None:
            contour_color = (self.color_manager.get_color('cost_surface', 'contour_lines')
                           if self.color_manager else color.orange)
        
        # Цель на плоскости XZ
        goal_2d = np.array([self.goal_position[0], self.goal_position[2]])
        
        # Находим размер области
        sampler = self.spawn_area.sampler
        boundary_points = sampler.get_points(n_points=100)
        max_distance = np.max([np.linalg.norm(p - goal_2d) for p in boundary_points])
        
        # Создаем круговые контуры на разных расстояниях от цели
        for i in range(1, num_levels + 1):
            radius = (i / (num_levels + 1)) * max_distance
            
            # Создаем круг из точек
            n_points = max(32, int(radius * 20))  # Больше точек для больших кругов
            angles = np.linspace(0, 2 * np.pi, n_points)
            
            circle_points_2d = []
            for angle in angles:
                x = goal_2d[0] + radius * np.cos(angle)
                z = goal_2d[1] + radius * np.sin(angle)
                point_2d = np.array([x, z])
                
                # Проверяем, что точка внутри эллипса
                if sampler.is_inside(point_2d):
                    circle_points_2d.append(point_2d)
            
            if len(circle_points_2d) < 3:
                continue  # Слишком мало точек для контура
            
            # Конвертируем в 3D точки на поверхности
            vertices_3d = []
            for point_2d in circle_points_2d:
                x, z = point_2d
                y = self.get_cost(point_2d)
                vertices_3d.append((x, y + 0.01, z))  # Приподнимаем над поверхностью
            
            # Замыкаем контур
            if len(vertices_3d) > 0:
                vertices_3d.append(vertices_3d[0])
            
            # Создаем линию контура
            if len(vertices_3d) > 1:
                contour_line = Entity(
                    model=Mesh(
                        vertices=vertices_3d,
                        mode='line',
                        thickness=contour_thickness
                    ),
                    color=contour_color,
                    parent=self.contours_entity,
                    unlit=True
                )
        
        print(f"Создано {num_levels} простых круговых изолиний")
    
    def update_cost_colors(self):
        if not self.vertices:
            return

        # Обновляем цвета вершин в зависимости от значений функции стоимости
        for i, vertex in enumerate(self.vertices):
            cost = self.get_cost(vertex)
            color_rgba = self._height_to_rainbow_color(cost, self.min_height, self.max_height)
            self.colors[i] = color_rgba

        # Обновляем цвета вершин в 3D-меше
        self.mesh_entity.model.colors = self.colors

    def generate_points(self, boundary_points_count, min_radius):
        point_cloud = self._generate_point_cloud(boundary_points_count, min_radius)
        self.vertices = point_cloud
        
        if self.config['cost_surface']['show_points']:
            points_data = []
            for vertex in point_cloud:
                points_data.append(
                    Entity(
                        parent=self.points_entity, 
                        model='sphere', 
                        position=vertex, 
                        scale=self.config['cost_surface']['point_size'], 
                        color=self.color_manager.get_color('cost_surface', 'mesh_points')
                    )
                )
            return points_data
        return []

    def generate_triangulation_edges(self, triangulation):
        edge_vertices = []
        edge_colors = []
        
        for triangle in triangulation.simplices:
            for i in range(3):
                edge_vertices.append(triangulation.points[triangle[i]])
                edge_vertices.append(triangulation.points[triangle[(i + 1) % 3]])
                edge_colors.append(self.color_manager.get_color('cost_surface', 'triangulation_edges'))
                edge_colors.append(self.color_manager.get_color('cost_surface', 'triangulation_edges'))
        
        self.edges_entity.model = Mesh(
            vertices=edge_vertices,
            mode='line',
            thickness=self.config['cost_surface']['edge_thickness'],
            colors=edge_colors
        )

    def generate_contours(self, triangulation, values, levels=10, resolution=200):
        contour_lines_mesh = Mesh(
            vertices=all_segments, 
            mode='line', 
            thickness=self.config['cost_surface']['contour_thickness'], 
            colors=all_colors
        )
        self.contours_entity.model = contour_lines_mesh

    def update_cost_colors(self):
        if not self.vertices:
            return

        # Обновляем цвета вершин в зависимости от значений функции стоимости
        for i, vertex in enumerate(self.vertices):
            cost = self.get_cost(vertex)
            color_rgba = self._height_to_rainbow_color(cost, self.min_height, self.max_height)
            self.colors[i] = color_rgba

        # Обновляем цвета вершин в 3D-меше
        self.mesh_entity.model.colors = self.colors

