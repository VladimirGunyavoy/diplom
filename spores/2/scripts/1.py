from ursina import *
import os
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# from src.zoom_manager import ZoomManager
from src.scene_setup import SceneSetup
from src.frame import Frame
from src.zoom_manager import ZoomManager
from src.scalable import Scalable
from src.pendulum import PendulumSystem

app = Ursina()

scene_setup = SceneSetup()
frame = Frame()
zoom_manager = ZoomManager(scene_setup.player)

# pendulum = PendulumSystem()

zoom_manager.register_object(frame.x_axis)
zoom_manager.register_object(frame.y_axis)
zoom_manager.register_object(frame.z_axis)
zoom_manager.register_object(frame.origin)

trajectory = []
trajectory.append(Scalable(model='sphere',
                           scale=1/10,
                           position=(0, 0, 0)))

                    










def update():
    scene_setup.update(time.dt)
    zoom_manager.identify_invariant_point()


def input(key):
    scene_setup.input_handler(key)
    zoom_manager.input_handler(key)





app.run()



