# --- Start of test setup ---
# This setup is necessary to run this file as a standalone script
# for testing purposes. It adds the project's root directory to the
# Python path to resolve absolute imports like 'from src. ...'.
import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- End of test setup ---

from ursina import *
from src.managers.angel_manager import AngelManager
from src.core.spore import Spore
from src.core.pendulum import PendulumSystem
from src.managers.color_manager import ColorManager
from src.managers.zoom_manager import ZoomManager
from src.scene.scene_setup import SceneSetup
from src.scene.frame import Frame
from src.utils.scalable import ScalableFloor

if __name__ == '__main__':
    app = Ursina()

    # --- Загрузка конфигурации ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    with open(os.path.join(project_root, 'config', 'json', 'config.json'), 'r') as f:
        config = json.load(f)

    # ===== СОЗДАНИЕ МЕНЕДЖЕРОВ И СЦЕНЫ (как в 1.py) =====
    color_manager = ColorManager()
    
    scene_config = config['scene_setup']
    scene_setup = SceneSetup(
        init_position=scene_config['init_position'], 
        init_rotation_x=scene_config['init_rotation_x'], 
        init_rotation_y=scene_config['init_rotation_y'], 
        color_manager=color_manager
    )
    frame = Frame(color_manager=color_manager)
    scene_setup.frame = frame

    zoom_manager = ZoomManager(scene_setup, color_manager=color_manager)
    zoom_manager.register_object(frame, name='frame')
    
    floor = ScalableFloor(
        model='quad',
        scale=(20, 20, 1),
        rotation_x=90,
        color=color_manager.get_color('scene', 'floor'),
        texture='white_cube',
        texture_scale=(40, 40)
    )
    zoom_manager.register_object(floor, name='floor')

    # ===== МЕНЕДЖЕР, КОТОРЫЙ ТЕСТИРУЕМ =====
    am = AngelManager(color_manager=color_manager, zoom_manager=zoom_manager)
    
    # ===== СОЗДАНИЕ СПОР ДЛЯ ТЕСТА =====
    pendulum_system = PendulumSystem()
    goal_position = config['spore']['goal_position']

    spore1 = Spore(
        dt=0.1, pendulum=pendulum_system, goal_position=goal_position,
        position=Vec3(0, 0, 0), color_manager=color_manager
    )
    spore1.logic._cost = 2 
    am.on_spore_created(spore1)
    zoom_manager.register_object(spore1)

    spore2 = Spore(
        dt=0.1, pendulum=pendulum_system, goal_position=goal_position,
        position=Vec3(5, 0, 5), color_manager=color_manager
    )
    spore2.logic._cost = 3
    am.on_spore_created(spore2)
    zoom_manager.register_object(spore2)

    print("AngelManager test finished. 3D scene is running.")
    print("Use WASD, Shift/Space and the mouse to navigate.")

    # ===== ФУНКЦИИ УПРАВЛЕНИЯ КАМЕРОЙ =====
    def update():
        scene_setup.update(time.dt)
    
    def input(key):
        scene_setup.input_handler(key)

    app.run() 