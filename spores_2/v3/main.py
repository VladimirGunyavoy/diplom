"""
PLAYER ZOOM - Camera and Zoom Sandbox
======================================

Controls:
- WASD: movement
- Mouse: look around
- Space/Shift: up/down
- Alt: release/capture cursor
- Escape: exit
- Q/E: zoom out/in
- R: reset zoom
- F11: fullscreen mode
- U: toggle frame visibility
- H: debug info
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np

from ursina import Ursina, Entity
from ursina.models.procedural.circle import Circle


from src.core.scene_manager import SceneManager
from src.core.zoom_manager import ZoomManager
from src.core.window_manager import WindowManager
from src.core.color_manager import ColorManager
from src.core.line_manager import LineManager
from src.core.surface_manager import SurfaceManager
from src.core.scalable import ScalableFloor
from src.core.input_manager import InputManager
from src.core.update_manager import UpdateManager
from src.core.object_manager import ObjectManager
from src.core.screen_manager import ScreenManager, Message
from src.core.shared_context import SharedContext
from src.core.param_manager import ParamManager
from src.math import DoubleIntegrator, Pendulum
from src.spores.spore import GhostSpore
from src.spores.spore_manager import SporeManager
from src.spores.branch_family_manager import BranchFamilyManager

print("=" * 50)
print("PLAYER ZOOM - Sandbox")
print("=" * 50)

app = Ursina()

# ===== MANAGERS =====
color_manager = ColorManager()
window_manager = WindowManager(monitor='left', fullscreen=False)
input_manager = InputManager()
update_manager = UpdateManager()

# ===== SCENE =====
scene_setup = SceneManager(
    init_position=(1.5, -1, -2),
    init_rotation_x=21,
    init_rotation_y=-35,
    color_manager=color_manager,
    input_manager=input_manager,
    update_manager=update_manager
)

zoom_manager = ZoomManager(scene_setup, color_manager=color_manager)
scene_setup.register_frame_in_zoom(zoom_manager)

floor = ScalableFloor(
    model='quad',
    scale=40,
    rotation_x=90,
    color=color_manager.get_color('scene', 'floor'),
    texture='white_cube',
    texture_scale=(40, 40)
)
zoom_manager.register_object(floor, name='floor')

# ===== SHARED CONTEXT =====
shared_context = SharedContext()
shared_context.bind('look_point', lambda: zoom_manager.real_look_point, default=np.zeros(2))

# ===== OBJECT MANAGER =====
object_manager = ObjectManager(zoom_manager, shared_context)
spore_manager = SporeManager(zoom_manager, object_manager)
line_manager = LineManager(zoom_manager)
surface_manager = SurfaceManager(zoom_manager)

# ===== BIND MANAGERS TO CONTEXT =====
shared_context.color_manager = color_manager
shared_context.object_manager = object_manager
shared_context.spore_manager = spore_manager
shared_context.line_manager = line_manager
shared_context.surface_manager = surface_manager

# ===== SCREEN MANAGER =====
screen_manager = ScreenManager()
mx, my = window_manager.get_margin()

screen_manager.add_message(Message(
    name='look_point',
    position=(-0.79 + mx, 0.48 - my),
    offset=(0.0, 0.0),
    getter=lambda: f"Look: [{shared_context.look_point[0]:5.2f} {shared_context.look_point[1]:5.2f}]"
))


# ===== PARAM MANAGER =====
param_manager = ParamManager()
param_manager.add('tau',   1.5, mode='exp',    min_val=0.0)
param_manager.add('a_max', 0.5, mode='linear', step=0.1,  min_val=0.0)
param_manager.add('n_tau', 4,   mode='linear', step=1,    min_val=0)
shared_context.bind('param_manager', lambda: param_manager, default=param_manager)


# ===== DYNAMICS MODEL =====
model = Pendulum(shared_context)   # <- меняй класс, чтобы сменить динамику
shared_context.bind('model', lambda: model, default=model)


# ===== PLAY HERE =====

root_spore = spore_manager.create(GhostSpore, 'ghost_spore_0')

branch_manager = BranchFamilyManager(object_manager)
#                       name         template                    suffix
b_ppp = branch_manager.create('b_ppp', root_spore, ((+1,+1),(-1,+1)), shared_context, color_suffix='ppp')
b_ppm = branch_manager.create('b_ppm', root_spore, ((+1,+1),(-1,-1)), shared_context, color_suffix='ppm')
b_pmp = branch_manager.create('b_pmp', root_spore, ((+1,-1),(-1,+1)), shared_context, color_suffix='pmp')
b_pmm = branch_manager.create('b_pmm', root_spore, ((+1,-1),(-1,-1)), shared_context, color_suffix='pmm')
b_mpp = branch_manager.create('b_mpp', root_spore, ((-1,+1),(+1,+1)), shared_context, color_suffix='mpp')
b_mpm = branch_manager.create('b_mpm', root_spore, ((-1,+1),(+1,-1)), shared_context, color_suffix='mpm')
b_mmp = branch_manager.create('b_mmp', root_spore, ((-1,-1),(+1,+1)), shared_context, color_suffix='mmp')
b_mmm = branch_manager.create('b_mmm', root_spore, ((-1,-1),(+1,-1)), shared_context, color_suffix='mmm')


# ===== BINDINGS =====

def _resize(sign):
    if sign > 0:
        spore_manager.increase_size()
        line_manager.increase_arrow_size()
    else:
        spore_manager.decrease_size()
        line_manager.decrease_arrow_size()

input_manager.bind('1', _resize, mode='scroll', description='size', value_getter=lambda: spore_manager.size)
input_manager.bind('2', lambda sign: param_manager.tweak('tau',   sign), mode='scroll', description='tau',   value_getter=lambda: param_manager.tau)
input_manager.bind('3', lambda sign: param_manager.tweak('a_max', sign), mode='scroll', description='a_max', value_getter=lambda: param_manager.a_max)
input_manager.bind('4', lambda sign: param_manager.tweak('n_tau', sign), mode='scroll', description='n_tau', value_getter=lambda: param_manager.n_tau)

# 1-8: toggle branches on/off (nodes, lines, arrows, surface)
input_manager.bind('1', b_ppp.toggle, description='branch ppp')
input_manager.bind('2', b_ppm.toggle, description='branch ppm')
input_manager.bind('3', b_pmp.toggle, description='branch pmp')
input_manager.bind('4', b_pmm.toggle, description='branch pmm')
input_manager.bind('5', b_mpp.toggle, description='branch mpp')
input_manager.bind('6', b_mpm.toggle, description='branch mpm')
input_manager.bind('7', b_mmp.toggle, description='branch mmp')
input_manager.bind('8', b_mmm.toggle, description='branch mmm')


# ===== BINDINGS HELP =====

screen_manager.add_bindings_help(input_manager, position=(-0.79 + mx, 0.4 - my))

# ===== REGISTER COMPONENTS =====
input_manager.register(
    scene_setup=scene_setup,
    zoom_manager=zoom_manager,
    window_manager=window_manager,
    object_manager=object_manager,
)

update_manager.register(
    input_manager=input_manager,
    scene_setup=scene_setup,
    zoom_manager=zoom_manager,
    object_manager=object_manager,
    screen_manager=screen_manager,
    shared_context=shared_context,
)

# ===== LOOP =====
def update():
    update_manager.update_all()

def input(key):
    input_manager.handle_input(key)

print("Ready. WASD to move, Q/E zoom, Alt cursor, Esc exit.")
print("=" * 50)

if __name__ == '__main__':
    app.run()
