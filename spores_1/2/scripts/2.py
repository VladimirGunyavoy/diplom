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

# create app
app = Ursina()

# create scene setup
scene_setup = SceneSetup(init_position=(0.5, -1.5, 0.5), init_rotation_x=39, init_rotation_y=-45)
frame = Frame()
scene_setup.frame = frame  # Добавляем frame в scene_setup
zoom_manager = ZoomManager(scene_setup)
window_manager = WindowManager()

# create pendulum
pendulum = PendulumSystem(dt=0.2, damping=0.1)

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

trajectory = []
trajectory.append(Spore(pendulum=pendulum, dt=0.05, 
                       scale=ball_size,
                       position=(0, 0, 1)))





N = 10

# Создаем красивую цветовую палитру
cmap = plt.cm.rainbow  # или можно использовать другие: 'viridis', 'plasma', 'magma', 'inferno'

def get_color_by_index(index, total):
    # Получаем цвет из палитры matplotlib
    color = cmap(index / total)
    return Color(color[0], color[1], color[2], color[3])

for i in range(N):    
    trajectory.append(trajectory[-1].A_evolve())

for i in range(len(trajectory)):
    trajectory[i].color = get_color_by_index(i, N)

for i in range(len(trajectory)):
    zoom_manager.register_object(trajectory[i])
      




def update():
    scene_setup.update(time.dt)
    zoom_manager.identify_invariant_point()


def input(key):
    scene_setup.input_handler(key)
    zoom_manager.input_handler(key)





app.run()



