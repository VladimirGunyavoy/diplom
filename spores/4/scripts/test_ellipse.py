from ursina import *
import os
import sys
import numpy as np
from colorsys import hsv_to_rgb
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

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
from src.color_manager import ColorManager
from src.ellipse import Ellipse
from src.ellipse2 import Ellipsoid2
from src.spawn_area import SpawnArea

# create app
app = Ursina()

# create color manager
color_manager = ColorManager()

# create scene setup
scene_setup = SceneSetup(init_position=(0.5, 0.5, -2), init_rotation_x=42.750, init_rotation_y=-12.135, color_manager=color_manager)
frame = Frame(color_manager=color_manager)
scene_setup.frame = frame  # Добавляем frame в scene_setup
window_manager = WindowManager(color_manager=color_manager)

settings_param = ParamManager(0, show=False, color_manager=color_manager)

# create pendulum

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

setup_register = {
    'x_axis': frame.x_axis,
    'y_axis': frame.y_axis,
    'z_axis': frame.z_axis,
    'origin': frame.origin,
    'floor': scene_setup.floor
}

for name, obj in setup_register.items():
    zoom_manager.register_object(obj, name)

bass_size = 1/10
A = Scalable(model='sphere', scale=bass_size, position=(1.5, 0, 0), color=color.red)
zoom_manager.register_object(A, 'A')

B = Scalable(model='sphere', scale=bass_size, position=(1, 0, 2), color=color.blue)
zoom_manager.register_object(B, 'B')


ellipse = Ellipse(a=2.0, b=1.0, 
                 resolution=128, 
                 thickness=5,
                 color=color.yellow)


spawn_area = SpawnArea(focus1=A, 
                      focus2=B, 
                      eccentricity=0.8, 
                      dimensions=2,
                      resolution=128, 
                      thickness=10, 
                      color=color.green,
                      mode='line')

zoom_manager.register_object(ellipse, 'ellipse')
zoom_manager.register_object(spawn_area, 'spawn_area')

# apply initial transform to all registered objects
zoom_manager.update_transform()

def update():
    scene_setup.update(time.dt)
    zoom_manager.identify_invariant_point()
    settings_param.update()
    # spore_manager.update()



def input(key):
    scene_setup.input_handler(key)
    zoom_manager.input_handler(key)
    settings_param.input_handler(key)
    spawn_area.input_handler(key)

app.run()



