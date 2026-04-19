from ursina import *
import numpy as np
from .ellipse2 import Ellipsoid2
from .color_manager import ColorManager
from .scalable import Scalable

class SpawnArea(Ellipsoid2):
    """
    Класс для создания области спавна в виде эллипса.
    
    Управление:
    - Клавиша 3: Уменьшить эксцентриситет (-0.05)
    - Клавиша 4: Увеличить эксцентриситет (+0.05)
    
    Цвета (из colors.json):
    - spawn_area.boundary: Контур эллипса [0.9, 0.9, 0.3, 0.8]
    - spawn_area.fill: Заливка эллипса [0.9, 0.9, 0.3, 0.2]
    - spawn_area.focus_point: Точки фокуса [0.9, 0.9, 0.3, 1.0]
    
    Пример использования:
        spawn_area = SpawnArea(
            focus1=(-2, 0, 0),
            focus2=(2, 0, 0),
            eccentricity=0.5,
            dimensions=2,
            resolution=64,
            mode='line',
            color=color_manager.get_color('spawn_area', 'boundary'),
            color_manager=color_manager
        )
        
        # В главном цикле:
        def input(key):
            spawn_area.input_handler(key)
    """
    def __init__(self, 
                 focus1, 
                 focus2, 
                 eccentricity, 
                 dimensions=None, 
                 resolution=32, 
                 mode='triangle', 
                 color_manager=None, 
                 **kwargs):



        # Используем переданный ColorManager или создаем новый
        if color_manager is None:
            color_manager = ColorManager()
        self.color_manager = color_manager
        
        # Устанавливаем thickness до вызова родительского конструктора
        self.thickness = kwargs.get('thickness', 10)
        
        # Устанавливаем цвет из карты цветов если не передан
        if 'color' not in kwargs:
            kwargs['color'] = self.color_manager.get_color('spawn_area', 'boundary')
        
        super().__init__(focus1, focus2, eccentricity, dimensions, resolution, mode, **kwargs)
        
        # Список для коллбэков, которые будут вызываться при перестройке
        self.on_rebuild_callbacks = []
        self.on_eccentricity_change_callbacks = []
        
        # Сохраняем параметры для возможности изменения эксцентриситета
        self.focus1 = focus1
        self.focus2 = focus2
        self.eccentricity = eccentricity
        self.dimensions = dimensions
        self.resolution = resolution
        self.mode = mode
        self.kwargs = kwargs
        
        # Параметры изменения эксцентриситета
        self.eccentricity_step = 1.05
        self.min_eccentricity = 0.01
        self.max_eccentricity = 0.999
        
        # # Создаем инструкции на экране - ЭТО БУДЕТ ПЕРЕНЕСЕНО В UI_SETUP
        # self.instructions = Text(
        #     text=dedent('''
        #     <white>SPAWN AREA:
        #     3 - decrease eccentricity
        #     4 - increase eccentricity
        #     ''').strip(),
        #     position=(0.55, 0.21),
        #     scale=0.75,
        #     color=self.color_manager.get_color('ui', 'text_primary'),
        #     background=True,
        #     background_color=self.color_manager.get_color('ui', 'background_transparent'),
        # )
        
        # # Текст с информацией о параметрах - ЭТО БУДЕТ ПЕРЕНЕСЕНО В UI_SETUP
        # self.info_text = Text(
        #     text=f'ECCENTRICITY: {self.eccentricity:.3f}',
        #     position=(0.55, 0.3, 0),
        #     scale=0.75,
        #     color=self.color_manager.get_color('ui', 'text_primary'),
        #     background_color=self.color_manager.get_color('ui', 'background_transparent'),
        #     origin=(-0.5, 0.5),
        #     parent=camera.ui,
        #     font='VeraMono.ttf',
        #     font_size=16,
        # )
        # self.info_text.background = True

    def update_eccentricity(self, new_eccentricity):
        """Обновляет эксцентриситет и перестраивает эллипс"""
        # Ограничиваем значение эксцентриситета допустимыми пределами
        self.eccentricity = max(self.min_eccentricity, min(self.max_eccentricity, new_eccentricity))
        
        # Обновляем сэмплер с новым эксцентриситетом
        def _get_coords(focus):
            if isinstance(focus, Scalable):
                return focus.real_position
            return focus

        coords1 = np.asarray(_get_coords(self.focus1), dtype=float)
        coords2 = np.asarray(_get_coords(self.focus2), dtype=float)

        target_dim = self.dimensions if self.dimensions is not None else len(coords1)

        if target_dim == 2:
            f1 = np.array([coords1[0], coords1[2]])
            f2 = np.array([coords2[0], coords2[2]])
        else:
            if len(coords1) < 3:
                coords1 = np.append(coords1, [0] * (3 - len(coords1)))
            if len(coords2) < 3:
                coords2 = np.append(coords2, [0] * (3 - len(coords2)))
            f1 = coords1
            f2 = coords2
        
        # Создаем новый сэмплер
        from .ellipse2 import EllipsoidSampler
        self.sampler = EllipsoidSampler(f1, f2, self.eccentricity)
        
        # Перестраиваем геометрию
        self._rebuild_geometry()
        
        # Вызываем коллбэки
        for callback in self.on_rebuild_callbacks:
            callback()
        
        # Обновляем текст с информацией
        # self.info_text.text = f'ECCENTRICITY: {self.eccentricity:.3f}' # <- REMOVE THIS
        
        # Вызываем коллбэки для изменения эксцентриситета
        for callback in self.on_eccentricity_change_callbacks:
            callback(self.eccentricity)

    def _rebuild_geometry(self):
        """Перестраивает геометрию эллипса"""
        # Удаляем старую геометрию
        if hasattr(self, 'mesh_entity'):
            destroy(self.mesh_entity)
        
        # Создаем новую геометрию
        if self.sampler.n_dim == 2:
            points_2d = self.sampler.get_points(n_points=self.resolution)
            vertices = [(v[0], 0, v[1]) for v in points_2d]
            
            if self.mode == 'line' and len(vertices) > 1:
                vertices.append(vertices[0])
            
            mesh_model = Mesh(vertices=vertices, mode=self.mode, thickness=self.thickness)

        elif self.sampler.n_dim == 3:
            try:
                from scipy.spatial import ConvexHull
            except ImportError:
                print("Ошибка: для создания 3D сетки требуется библиотека Scipy.")
                return

            n_mesh_points = max(self.resolution * 4, 64)
            points_on_sphere = np.random.randn(n_mesh_points, 3)
            points_on_sphere /= np.linalg.norm(points_on_sphere, axis=1, keepdims=True)

            hull = ConvexHull(points_on_sphere)
            transformed_verts = points_on_sphere @ self.sampler.T + self.sampler.center
            triangles_list = [list(s) for s in hull.simplices]
            
            mesh_model = Mesh(
                vertices=[tuple(v) for v in transformed_verts],
                triangles=triangles_list,
                mode='triangle'
            )

        # Создаем новую сущность с геометрией
        self.mesh_entity = Entity(
            parent=self,
            model=mesh_model,
            **self.kwargs
        )

    def update(self):
        pass

    def input_handler(self, key):
        """Обработчик ввода для управления эксцентриситетом"""
        if key == '3':
            # Уменьшить эксцентриситет
            new_eccentricity = self.eccentricity * self.eccentricity_step
            self.update_eccentricity(new_eccentricity)
        elif key == '4':
            # Увеличить эксцентриситет
            new_eccentricity = self.eccentricity / self.eccentricity_step
            self.update_eccentricity(new_eccentricity)
        
        # Вызываем родительский обработчик, если он есть
        if hasattr(super(), 'input_handler'):
            super().input_handler(key)