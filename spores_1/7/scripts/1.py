import sys
import os
from ursina import *
import numpy as np
from colorsys import hsv_to_rgb
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

# --- Настройка путей для импорта ---
# Добавляем корневую папку проекта 'spores/5', чтобы можно было делать импорты от 'src'
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Основные импорты из нашего проекта ---
from src.spawn_area import SpawnArea
from src.cost import Cost
from src.color_manager import ColorManager

# from src.zoom_manager import ZoomManager
from src.scene_setup import SceneSetup
from src.frame import Frame
from src.zoom_manager import ZoomManager
from src.scalable import Scalable, ScalableFrame, ScalableFloor
from src.pendulum import PendulumSystem
from src.window_manager import WindowManager
from src.spore import Spore
from src.spore_manager import SporeManager
from src.param_manager import ParamManager

# create app
app = Ursina()

# create color manager
color_manager = ColorManager()

# create scene setup
scene_setup = SceneSetup(init_position=(0.263, -1.451, 0.116), init_rotation_x=42.750, init_rotation_y=-12.135, color_manager=color_manager)
frame = Frame(color_manager=color_manager)
scene_setup.frame = frame  # Добавляем frame в scene_setup
window_manager = WindowManager(color_manager=color_manager)

settings_param = ParamManager(0, show=False, color_manager=color_manager)

# create pendulum
pendulum = PendulumSystem(dt=0.1, damping=0.3)


zoom_manager = ZoomManager(scene_setup, color_manager=color_manager)

def set_initial_zoom():
    # Set initial zoom from image
    target_a = 1.602
    zoom_manager.a_transformation = target_a
    inv_point_2d = zoom_manager.identify_invariant_point()
    p = np.array([inv_point_2d[0], 0, inv_point_2d[1]], dtype=float)
    b = (1 - target_a) * p
    zoom_manager.b_translation = b
    zoom_manager.update_transform()

invoke(set_initial_zoom, delay=0.02)


spore_manager = SporeManager(pendulum=pendulum, zoom_manager=zoom_manager, settings_param=settings_param, color_manager=color_manager)

# Entity(model='models/arrow.obj', position=(0, 0, 0))

ball_size = 1/50

setup_register = {
    'x_axis': frame.x_axis,
    'y_axis': frame.y_axis,
    'z_axis': frame.z_axis,
    'origin': frame.origin,
    'floor': scene_setup.floor
}

for name, obj in setup_register.items():
    zoom_manager.register_object(obj, name)



dt = 0.05

goal_position = (np.pi, 0)
goal = Spore(pendulum=pendulum,
             dt=dt,
             scale=ball_size,
             position=(goal_position[0], 0, goal_position[1]),
             goal_position=goal_position,
             is_goal=True)

spore = Spore(pendulum=pendulum, dt=dt, 
              goal_position=goal_position,
              scale=ball_size,
              position=(0, 0, 2),
              color_manager=color_manager)
spore.color = color.red

spawn_area = SpawnArea(focus1=spore, 
                       focus2=goal, 
                       eccentricity=0.998, 
                       dimensions=2, 
                       resolution=64, 
                       mode='line', 
                       color_manager=color_manager)

# --- Поверхность стоимости (Cost) ---
# Сначала просто создаем объект, без генерации
cost_surface = Cost(
    goal_position=goal,
    spawn_area=spawn_area,
    color_manager=color_manager
)
# === НАСТРОЙКА РАДУЖНОЙ ПОВЕРХНОСТИ ===
# Полный контроль параметров визуализации поверхности стоимости
# 
# ЦВЕТА управляются через colors.json в разделе "cost_surface":
# - "wireframe": [1.0, 1.0, 1.0, 1.0]           - белые линии сетки треугольников  
# - "mesh_points": [1.0, 1.0, 0.0, 1.0]         - желтые точки вершин сетки
# - "triangulation_edges": [0.0, 1.0, 1.0, 1.0] - голубые ребра триангуляции
# - "contour_lines": [1.0, 0.6, 0.0, 1.0]       - оранжевые изолинии (контуры)
cost_surface.generate_surface(
    mesh_generation={
        # ПЛОТНОСТЬ СЕТКИ:
        'boundary_points': 64,       # Количество точек на границе эллипса (больше = плавнее контур)
        'interior_min_radius': 0.2   # Минимальное расстояние между внутренними точками 
                                    # (МЕНЬШЕ = ПЛОТНЕЕ сетка, больше точек)
                                    # Примеры: 0.1 = очень плотно, 0.5 = средне, 1.0 = редко
    },
    # ВИЗУАЛЬНЫЕ ПАРАМЕТРЫ:
    alpha=0.7,                      # Прозрачность поверхности (0.0 = прозрачная, 1.0 = непрозрачная)
    show_points=True,               # Показывать точки сетки (цвет из cost_surface.mesh_points)
    point_size=0.02,                # Размер точек сетки
    show_edges=True,                # Показывать ребра триангуляции (цвет из cost_surface.triangulation_edges)
    edge_thickness=1,               # Толщина линий ребер
    # ИЗОЛИНИИ (КОНТУРНЫЕ ЛИНИИ):
    show_contours=True,             # Показывать изолинии равной стоимости
    contour_levels=22,              # Количество уровней изолиний
    contour_thickness=3,            # Толщина линий изолиний
    contour_resolution=60           # Разрешение сетки для расчета контуров (больше = точнее, медленнее)
)


spore_manager.add_spore(goal)
spore_manager.add_spore(spore)


zoom_manager.register_object(goal)
zoom_manager.register_object(spore)
zoom_manager.register_object(spawn_area)
zoom_manager.register_object(cost_surface)

# apply initial transform to all registered objects
zoom_manager.update_transform()



# Entity(model='cube', scale=1/10, position=(0, 0, 0), color=color.red)

# N = 10

# Создаем красивую цветовую палитру
# cmap = plt.cm.rainbow  # или можно использовать другие: 'viridis', 'plasma', 'magma', 'inferno'

# def get_color_by_index(index, total):
#     # Получаем цвет из палитры matplotlib
#     color = cmap(index / total)
#     return Color(color[0], color[1], color[2], color[3])


# for i in range(len(spore_manager.objects)):
#     spore_manager.objects[i].color = get_color_by_index(i, N)





def update():
    scene_setup.update(time.dt)
    zoom_manager.identify_invariant_point()
    settings_param.update()
    # spore_manager.update()



def input(key):
    scene_setup.input_handler(key)
    zoom_manager.input_handler(key)
    spore_manager.input_handler(key)
    settings_param.input_handler(key)
    spawn_area.input_handler(key)


if __name__ == '__main__':
    app.run()



