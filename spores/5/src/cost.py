from ursina import *
import numpy as np
from scipy.spatial import Delaunay
from colorsys import hsv_to_rgb

from .scalable import Scalable
from .ellipse2 import EllipsoidSampler

class Cost(Scalable):
    def __init__(self, 
                 goal_position,
                 spawn_area,
                 color_manager=None):
        super().__init__()
        
        if isinstance(goal_position, Scalable):
            self.goal_position = goal_position.real_position
        else:
            self.goal_position = np.asarray(goal_position, dtype=float)

        self.spawn_area = spawn_area
        self.color_manager = color_manager
        
        self.mesh_entity = None
        self.wireframe_entity = None
        self.points_entity = None  # Для визуализации точек сетки
        self.edges_entity = None   # Для визуализации ребер триангуляции
        self.mesh_kwargs = {}
        self.last_vertices_3d = []  # Сохраняем последние вершины для визуализации
        self.last_triangulation = None  # Сохраняем триангуляцию для ребер

    def get_cost(self, position_2d):
        """Вычисляет стоимость (высоту y) для 2D-координат (x, z)."""
        # Проекция goal_position на плоскость XZ
        goal_2d = np.array([self.goal_position[0], self.goal_position[2]])
        
        distance = np.linalg.norm(position_2d - goal_2d)
        k = 1 / 10
        с = 1/2
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

    def _generate_point_cloud(self, boundary_points=128, interior_min_radius=0.5):
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
        # 1. Получаем 2D точки
        points_2d = self._generate_point_cloud(**kwargs)
        
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
        return mesh

    def generate_surface(self, **kwargs):
        """
        Явный метод для генерации или перестройки 3D-меша поверхности.
        Этот метод является основной точкой входа для создания поверхности.
        """
        # Уничтожаем старые меши, если они есть
        if self.mesh_entity:
            destroy(self.mesh_entity)
        if self.wireframe_entity:
            destroy(self.wireframe_entity)
        if self.points_entity:
            destroy(self.points_entity)
        if self.edges_entity:
            destroy(self.edges_entity)
        
        # Сохраняем аргументы для перестройки
        self.mesh_kwargs = kwargs

        # Извлекаем параметры для генерации меша
        mesh_gen_params = kwargs.get('mesh_generation', {})
        surface_alpha = kwargs.get('alpha', 1.0)  # Прозрачность поверхности
        
        mesh_model = self.generate_mesh(alpha=surface_alpha, **mesh_gen_params)
        
        if mesh_model:
            # Готовим аргументы для создания Entity, копируя исходные
            entity_kwargs = kwargs.copy()
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
            entity_kwargs.pop('alpha', None)  # Убираем прозрачность

            # Создаем основную поверхность
            self.mesh_entity = Entity(
                model=mesh_model,
                parent=self,
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
            if kwargs.get('show_points', True):
                point_size = kwargs.get('point_size', 0.02)
                self._create_points_visualization(point_size)
            
            # Создаем визуализацию ребер триангуляции, если запрошено
            if kwargs.get('show_edges', False):
                edge_color = kwargs.get('edge_color', color.cyan)
                edge_thickness = kwargs.get('edge_thickness', 1)
                self._create_edges_visualization(edge_color, edge_thickness)
            
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
        return None
    
    def rebuild_mesh(self):
        """
        Перестраивает 3D-меш поверхности. Вызывается по коллбэку от SpawnArea.
        Использует сохраненные kwargs от последнего вызова generate_surface.
        """
        print("Rebuilding cost surface mesh via callback...")
        self.generate_surface(**self.mesh_kwargs)
    
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
    
