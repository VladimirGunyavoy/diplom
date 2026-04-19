print("ЗАГРУЖЕН ИСПРАВЛЕННЫЙ VISUALIZER")
from ursina import Entity, Mesh, Vec4, color, destroy, Vec3
import numpy as np
from scipy.spatial import Delaunay
from colorsys import hsv_to_rgb
from typing import List, Optional, Tuple, Dict, Any

import matplotlib
matplotlib.use('Agg') # ВАЖНО: Устанавливаем не-интерактивный бэкенд до импорта pyplot
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

from ..logic.cost_function import CostFunction
from ..utils.scalable import Scalable
from ..logic.spawn_area import SpawnArea
from ..managers.color_manager import ColorManager
from ..utils.debug_output import always_print, debug_print


class CostVisualizer:
    """
    Класс для визуализации поверхности стоимости.
    Принимает на вход логический объект CostFunction и объект,
    определяющий область для генерации точек (spawn_area).
    """
    def __init__(self, 
                 cost_function: CostFunction, 
                 spawn_area: SpawnArea, 
                 parent_entity: Entity, 
                 color_manager: Optional[ColorManager] = None, 
                 config: Optional[Dict[str, Any]] = None, 
                 **kwargs):
        
        self.cost_function: CostFunction = cost_function
        self.spawn_area: SpawnArea = spawn_area
        self.parent_entity: Entity = parent_entity
        self.color_manager: Optional[ColorManager] = color_manager
        self.config: Dict[str, Any] = config if config is not None else {}
        self.mesh_entity: Optional[Entity] = None
        self.edges_entity: Optional[Entity] = None
        self.contours_entity: Optional[Entity] = None
        
        self.last_vertices_3d: List[Tuple[float, float, float]] = []
        self.last_triangulation: Optional[Delaunay] = None
        
        # Проверяем флаг enabled из конфигурации
        self.visible: bool = self.config.get('enabled', True)

        # Генерируем и создаем Entity
        self.generate_surface()
        
        # Если cost surface отключена в конфигурации, скрываем её
        if not self.visible:
            self.hide()
            debug_print("💡 Cost surface отключена в конфигурации - скрыта по умолчанию")

    def _generate_mesh(self) -> Optional[Mesh]:
        """Генерирует и возвращает 3D-меш поверхности стоимости."""
        # Генерация 2D точек
        mesh_gen_config = self.config.get('mesh_generation', {})
        points_2d = self._generate_point_cloud(
            boundary_points=mesh_gen_config.get('boundary_points', 128),
            interior_min_radius=mesh_gen_config.get('interior_min_radius', 0.01)
        )

        if points_2d is None or len(points_2d) < 3:
            return None

        # Триангуляция
        try:
            tri = Delaunay(points_2d)
            self.last_triangulation = tri
        except Exception as e:
            debug_print(f"Ошибка триангуляции Делоне: {e}")
            return None

        # Создание 3D вершин и цветов
        vertices_3d: List[Tuple[float, float, float]] = []
        heights = [self.cost_function.get_cost(p) for p in points_2d]
        min_h, max_h = min(heights), max(heights)
        vertex_colors = [self._height_to_color(h, min_h, max_h) for h in heights]
        for p, h in zip(points_2d, heights):
            vertices_3d.append((p[0], h, p[1]))

        self.last_vertices_3d = vertices_3d

        # Создание меша с дублированием
        front_faces = tri.simplices.tolist()
        back_faces = [list(reversed(face)) for face in front_faces]
        all_vertices = vertices_3d + vertices_3d
        all_vertex_colors = vertex_colors + vertex_colors
        all_faces = front_faces + back_faces
        
        return Mesh(
            vertices=all_vertices,
            triangles=all_faces,
            colors=all_vertex_colors,
            mode='triangle'
        )
        
    def generate_surface(self) -> None:
        """Пересоздает меш и Entity для объекта."""
        if self.mesh_entity:
            destroy(self.mesh_entity)
        if self.edges_entity:
            destroy(self.edges_entity)
        if self.contours_entity:
            destroy(self.contours_entity)

        mesh_model = self._generate_mesh()
        
        if mesh_model:
            self.mesh_entity = Entity(
                parent=self.parent_entity,
                model=mesh_model,
                unlit=False,
                two_sided=True
            )
            
            self._create_edges_visualization()
            self._create_contour_visualization()
            self._create_points_visualization()

    def _generate_point_cloud(self, boundary_points: int, interior_min_radius: float) -> Optional[np.ndarray]:
        """Создает 2D-облако точек на основе spawn_area."""
        boundary = self.spawn_area.get_points(n_points=boundary_points)
        interior = self.spawn_area.sample_poisson_disk(min_radius=interior_min_radius)
        
        if interior.size == 0:
            return boundary
        return np.vstack([boundary, interior])

    def _height_to_color(self, height: float, min_height: float, max_height: float) -> Vec4:
        """Конвертирует высоту в радужный цвет (стабильная версия)."""
        if max_height == min_height:
            normalized = 0.5
        else:
            normalized = (height - min_height) / (max_height - min_height)
        
        # Hue от 240° (синий) до 0° (красный)
        hue = (1.0 - normalized) * 240 / 360
        saturation = 0.8
        value = 0.9
        
        r, g, b = hsv_to_rgb(hue, saturation, value)
        
        alpha = self.config.get('alpha', 1.0)
        return Vec4(r, g, b, float(alpha))

    def _create_points_visualization(self) -> None:
        """Создает визуализацию точек сетки (вершин)."""
        if not self.config.get('show_points', False):
            return

        if not self.last_vertices_3d:
            return

        if not self.color_manager:
            return

        point_color = self.color_manager.get_color('cost_surface', 'mesh_points')
        point_size = self.color_manager.get_value('cost_surface', 'point_size')
        
        points_mesh = Mesh(vertices=self.last_vertices_3d, mode='point', thickness=point_size)
        
        Entity(
            parent=self.mesh_entity, 
            model=points_mesh, 
            color=point_color,
            render_queue=1
        )

    def _create_edges_visualization(self) -> None:
        """Создает визуализацию ребер триангуляции (скелет)."""
        if not self.config.get('show_edges', False):
            return

        if not self.last_triangulation or not self.last_vertices_3d:
            return
            
        if not self.color_manager:
            return

        edge_color = self.color_manager.get_color('cost_surface', 'triangulation_edges')
        edge_thickness = self.config.get('edge_thickness', 1)

        self.edges_entity = Entity(
            parent=self.parent_entity,
            render_queue=-1  # Рисуем ДО основной поверхности
        )
        
        final_color = color.hex(edge_color) if isinstance(edge_color, str) else edge_color
        
        for simplex in self.last_triangulation.simplices:
            for i in range(3):
                start_idx = simplex[i]
                end_idx = simplex[(i + 1) % 3]
                
                Entity(
                    parent=self.edges_entity,
                    model=Mesh(
                        vertices=[self.last_vertices_3d[start_idx], self.last_vertices_3d[end_idx]], 
                        mode='line', 
                        thickness=edge_thickness
                    ),
                    color=final_color
                )

    def _create_contour_visualization(self) -> None:
        """Создает визуализацию линий уровня."""
        if not self.config.get('show_contours', False):
            return

        if not self.color_manager:
            return

        contour_config = self.config.get('contour_visualization', {})
            
        self._generate_contour_lines(
            num_levels=contour_config.get('levels', 10),
            contour_color=self.color_manager.get_color('cost_surface', 'contour_lines'),
            contour_thickness=contour_config.get('thickness', 2),
            grid_resolution=contour_config.get('resolution', 150),
            y_offset=contour_config.get('y_offset', 0.05),
            alpha=float(contour_config.get('alpha', 0.5))
        )

    def _generate_contour_lines(self, num_levels: int, contour_color: Vec4, contour_thickness: int, grid_resolution: int, y_offset: float, alpha: float) -> None:
        if self.contours_entity:
            destroy(self.contours_entity)
        
        if not self.last_vertices_3d or len(self.last_vertices_3d) < 3:
            return

        vertices = np.array(self.last_vertices_3d)
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

        if x.min() == x.max() or z.min() == z.max():
            debug_print("[Contours] WARNING: Cannot generate contours from a 1D point cloud.")
            return

        try:
            debug_print("[Contours] Interpolating grid...")
            grid_x, grid_z = np.meshgrid(
                np.linspace(x.min(), x.max(), grid_resolution),
                np.linspace(z.min(), z.max(), grid_resolution)
            )
            
            # Пробуем 'cubic' для гладкости, с откатом до 'linear' если данных мало
            try:
                grid_y = griddata((x, z), y, (grid_x, grid_z), method='cubic')
            except Exception:
                debug_print("[Contours] WARNING: Cubic interpolation failed, falling back to linear.")
                grid_y = griddata((x, z), y, (grid_x, grid_z), method='linear')

            # --- МАСКИРОВАНИЕ ДАННЫХ ВНЕ КОРПУСА ТРИАНГУЛЯЦИИ (КАК В V9) ---
            if self.last_triangulation:
                interp_points = np.vstack([grid_x.ravel(), grid_z.ravel()]).T
                mask = self.last_triangulation.find_simplex(interp_points) < 0
                grid_y.ravel()[mask] = np.nan

            if np.isnan(grid_y).all():
                debug_print("[Contours] WARNING: Grid interpolation resulted in all NaNs.")
                return
            
            debug_print(f"[Contours] Grid interpolation complete. Grid shape: {grid_y.shape}")

        except Exception as e:
            debug_print(f"Ошибка интерполяции для линий уровня: {e}")
            return

        if np.all(grid_y == grid_y[0, 0]):
            debug_print("[Contours] WARNING: Contour plot not generated for a flat surface.")
            return

        # --- ПОСТРОЕНИЕ ГРАФИКА С MATPLOTLIB ---
        fig, ax = plt.subplots()
        # Используем grid_y.T, чтобы ориентация соответствовала осям Ursina
        contour_set = ax.contour(grid_x, grid_z, grid_y, levels=num_levels)
        
        # --- ПРЕОБРАЗОВАНИЕ В URSINA ENTITIES ---
        self.contours_entity = Entity(parent=self.parent_entity, render_queue=-1)
        
        final_color = color.hex(contour_color) if isinstance(contour_color, str) else contour_color
        final_color = color.rgba(final_color.r, final_color.g, final_color.b, alpha)

        levels = list(contour_set.levels)
        for i, segs_for_level in enumerate(contour_set.allsegs):
            level_height = levels[i] + y_offset
            for seg in segs_for_level:
                if len(seg) < 2:
                    continue
                
                vertices_3d = [Vec3(p[0], level_height, p[1]) for p in seg]
                
                if len(vertices_3d) > 1:
                    Entity(
                        parent=self.contours_entity,
                        model=Mesh(vertices=vertices_3d, mode='line', thickness=contour_thickness),
                        color=final_color
                    )

        # Очистка Matplotlib
        plt.close(fig)

    def set_visibility(self, visible: bool):
        """Устанавливает видимость для всех компонентов визуализатора."""
        self.visible = visible
        if self.mesh_entity:
            self.mesh_entity.enabled = visible
        if self.edges_entity:
            # Видимость ребер зависит и от глобального флага, и от конфига
            self.edges_entity.enabled = visible and self.config.get('show_edges', False)
        if self.contours_entity:
            self.contours_entity.enabled = visible and self.config.get('show_contours', False)
            
        # Также управляем видимостью точек
        if self.mesh_entity:
            for child in self.mesh_entity.children:
                if 'points_mesh' in child.name: # Предполагая, что мы даем имя
                     child.enabled = visible and self.config.get('show_points', False)


    def show(self) -> None:
        """Показывает все компоненты."""
        self.set_visibility(True)

    def hide(self) -> None:
        """Скрывает все компоненты."""
        self.set_visibility(False)
    
    def toggle(self) -> None:
        """Переключает видимость cost surface."""
        self.visible = not self.visible
        self.set_visibility(self.visible)
        status = "включена" if self.visible else "выключена"
        debug_print(f"🗻 Поверхность стоимости {status}") 