print("ЗАГРУЖЕН ИСПРАВЛЕННЫЙ VISUALIZER")
from ursina import *
import numpy as np
from scipy.spatial import Delaunay
from colorsys import hsv_to_rgb

import matplotlib
matplotlib.use('Agg') # ВАЖНО: Устанавливаем не-интерактивный бэкенд до импорта pyplot
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

from src.logic.cost_function import CostFunction
from src.utils.scalable import Scalable

class CostVisualizer:
    """
    Класс для визуализации поверхности стоимости.
    Принимает на вход логический объект CostFunction и объект,
    определяющий область для генерации точек (spawn_area).
    """
    def __init__(self, cost_function: CostFunction, spawn_area, parent_entity, color_manager=None, config=None, **kwargs):
        
        self.cost_function = cost_function
        self.spawn_area = spawn_area
        self.parent_entity = parent_entity
        self.color_manager = color_manager
        self.config = config if config is not None else {}
        self.mesh_entity = None
        self.edges_entity = None
        self.contours_entity = None
        
        self.last_vertices_3d = []
        self.last_triangulation = None
        
        self.visible = True

        # Генерируем и создаем Entity
        self.generate_surface()

    def _generate_mesh(self):
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
            print(f"Ошибка триангуляции Делоне: {e}")
            return None

        # Создание 3D вершин и цветов
        vertices_3d = []
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
        
    def generate_surface(self):
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

    def _generate_point_cloud(self, boundary_points, interior_min_radius):
        """Создает 2D-облако точек на основе spawn_area."""
        boundary = self.spawn_area.get_points(n_points=boundary_points)
        interior = self.spawn_area.sample_poisson_disk(min_radius=interior_min_radius)
        
        if interior.size == 0:
            return boundary
        return np.vstack([boundary, interior])

    def _height_to_color(self, height, min_height, max_height):
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
        return Vec4(r, g, b, alpha)

    def _create_points_visualization(self):
        """Создает визуализацию точек сетки (вершин)."""
        if not self.config.get('show_points', False):
            return

        if not self.last_vertices_3d:
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

    def _create_edges_visualization(self):
        """Создает визуализацию ребер триангуляции (скелет)."""
        if not self.config.get('show_edges', False):
            return

        if not self.last_triangulation or not self.last_vertices_3d:
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

    def _create_contour_visualization(self):
        """Создает визуализацию линий уровня."""
        if not self.config.get('show_contours', False):
            return

        contour_config = self.config.get('contour_visualization', {})
            
        self._generate_contour_lines(
            num_levels=contour_config.get('levels', 10),
            contour_color=self.color_manager.get_color('cost_surface', 'contour_lines'),
            contour_thickness=contour_config.get('thickness', 2),
            grid_resolution=contour_config.get('resolution', 150),
            y_offset=contour_config.get('y_offset', 0.05),
            alpha=contour_config.get('alpha', 0.5)
        )

    def _generate_contour_lines(self, num_levels, contour_color, contour_thickness, grid_resolution, y_offset, alpha):
        if self.contours_entity:
            destroy(self.contours_entity)
        
        if not self.last_vertices_3d or len(self.last_vertices_3d) < 3:
            return

        vertices = np.array(self.last_vertices_3d)
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

        if x.min() == x.max() or z.min() == z.max():
            print("[Contours] WARNING: Cannot generate contours from a 1D point cloud.")
            return

        try:
            print("[Contours] Interpolating grid...")
            grid_x, grid_z = np.meshgrid(
                np.linspace(x.min(), x.max(), grid_resolution),
                np.linspace(z.min(), z.max(), grid_resolution)
            )
            
            # Пробуем 'cubic' для гладкости, с откатом до 'linear' если данных мало
            try:
                grid_y = griddata((x, z), y, (grid_x, grid_z), method='cubic')
            except Exception:
                print("[Contours] WARNING: Cubic interpolation failed, falling back to linear.")
                grid_y = griddata((x, z), y, (grid_x, grid_z), method='linear')

            # --- МАСКИРОВАНИЕ ДАННЫХ ВНЕ КОРПУСА ТРИАНГУЛЯЦИИ (КАК В V9) ---
            if self.last_triangulation:
                interp_points = np.vstack([grid_x.ravel(), grid_z.ravel()]).T
                mask = self.last_triangulation.find_simplex(interp_points) < 0
                grid_y.ravel()[mask] = np.nan

            if np.isnan(grid_y).all():
                print("[Contours] WARNING: Grid interpolation resulted in all NaNs.")
                return
            
            print(f"[Contours] Grid interpolation complete. Grid shape: {grid_y.shape}")

        except Exception as e:
            print(f"Ошибка интерполяции для линий уровня: {e}")
            return

        if np.all(grid_y == grid_y[0, 0]):
            print("[Contours] WARNING: Contour plot not generated for a flat surface.")
            return

        self.contours_entity = Entity(parent=self.parent_entity, name="ContoursContainer")
        
        base_color = color.hex(contour_color) if isinstance(contour_color, str) else contour_color
        # Создаем новый объект цвета с нужной прозрачностью, вместо изменения существующего
        final_color = Color(base_color.r, base_color.g, base_color.b, alpha)
        
        try:
            min_y, max_y = np.min(y), np.max(y)
            if min_y >= max_y:
                print("[Contours] WARNING: Cannot generate levels, data range is zero.")
                return

            levels = np.linspace(min_y, max_y, num_levels)
            print(f"[Contours] Generating contours for {len(levels)} levels...")
            
            cs = plt.contour(grid_x, grid_z, grid_y, levels=levels)
            
            if not cs.allsegs or all(len(s) == 0 for s in cs.allsegs):
                print("[Contours] WARNING: No vertices were generated by matplotlib.contour.")
                plt.close('all')
                return

            print(f"[Contours] Found {sum(len(segs) for segs in cs.allsegs)} line segments.")

            for i, segs_for_level in enumerate(cs.allsegs):
                level_height = cs.levels[i] + y_offset
                for seg in segs_for_level:
                    if len(seg) < 2:
                        continue
                    
                    line_vertices = [Vec3(p[0], level_height, p[1]) for p in seg]
                    
                    Entity(
                        parent=self.contours_entity,
                        model=Mesh(vertices=line_vertices, mode='line', thickness=contour_thickness),
                        color=final_color,
                        render_queue=1,
                        unlit=True,
                        always_on_top=True
                    )

        except Exception as e:
            print(f"Ошибка при создании геометрии контуров: {e}")
        finally:
            plt.close('all')

    def show(self):
        """Делает поверхность видимой."""
        if self.mesh_entity:
            self.mesh_entity.show()

    def hide(self):
        """Скрывает поверхность."""
        if self.mesh_entity:
            self.mesh_entity.hide() 