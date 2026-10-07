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
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np

import time
from ursina import held_keys, Ursina, Entity, mouse
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

print("=" * 50)
print("PLAYER ZOOM - Sandbox")
print("=" * 50)

app = Ursina()

# Шрифт с кириллицей (дефолтный OpenSans/VeraMono её не содержит)
from ursina import Text
for _f in ('C:/Windows/Fonts/consola.ttf', 'C:/Windows/Fonts/arial.ttf',
           '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'):
    if os.path.exists(_f):
        from panda3d.core import Filename
        Text.default_font = Filename.from_os_specific(_f).get_fullpath()
        break

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
param_manager.add('n_s',  2,   mode='linear', step=1,    min_val=0)
param_manager.add('r_s',  0.5, mode='exp',    min_val=0.01)
shared_context.bind('param_manager', lambda: param_manager, default=param_manager)


# ===== DYNAMICS MODEL =====
model = DoubleIntegrator(shared_context)   # <- меняй класс, чтобы сменить динамику
# model = Pendulum(shared_context)   # <- меняй класс, чтобы сменить динамику
shared_context.bind('model', lambda: model, default=model)


# ===== v8: ПОШАГОВЫЙ ПРОСМОТР РОСТА КЛЕТОК (PLAN п.49) =====
# N — до следующей паузы, M — до паузы того же или крупнее уровня, C — до конца; G — новая затравка в точке взгляда
# (или клик мыши по полю); 1 — размер маркеров. Пока нет алгоритма b6 (src/algo/growN.py + src/stepper/stepper.py) — DemoStepper (синтетические снимки).
from src.stepper.growview import GrowView
try:
    from src.stepper.live import LiveStepper                 # настоящий алгоритм (b6, src/algo/growN.py SYS=di) под Stepper
    stepper = LiveStepper(seed=None)
except ImportError as e_:
    print('[v8] live algorithm unavailable (%s) - DemoStepper' % e_)
    from src.stepper.demo import DemoStepper
    stepper = DemoStepper(seed=(0., 0.))
growview = GrowView(zoom_manager, ax=(0, 1), spore_manager=spore_manager)
growview.draw(stepper.snapshot)    # the first pause (the algorithm's own first seed) is visible right at start

def _step(k):
    t0 = time.perf_counter(); stepper.key(k); growview.draw(stepper.snapshot)
    print(f"[stepper] {k}: {stepper.snapshot and stepper.snapshot.get('phase')}  {1e3 * (time.perf_counter() - t0):.1f} ms")

_last_seed = [None]

def _seed_here():
    x, v = shared_context.look_point; _last_seed[0] = (float(x), float(v)); stepper.seed_at((float(x), float(v))); growview.draw(None); _step('N')

def _seed_click():
    wp = mouse.world_point
    if wp is None: return
    a, b = zoom_manager.a_transformation, zoom_manager.b_translation
    _last_seed[0] = ((wp.x - b[0]) / a, (wp.z - b[2]) / a)
    stepper.seed_at(_last_seed[0]); growview.draw(None); _step('N')

def _restart():
    """X: drop everything built so far and start over from the last seed (or the algorithm's own seeds if none was set by hand)."""
    growview.draw(None); growview.times = type(growview.times)()
    stepper.seed_at(_last_seed[0]); growview.draw(None); _step('N')

floor.collider = 'box'
screen_manager.add_message(Message(name='growview', position=(-0.79 + mx, -0.40 + my), getter=lambda: growview.caption()))
screen_manager.add_message(Message(name='timetable', position=(0.30 + mx, 0.48 - my), getter=lambda: growview.times.text(getattr(stepper, 'times', {}))))

# ===== BINDINGS =====

def _resize(sign):
    if sign > 0:
        spore_manager.increase_size()
    else:
        spore_manager.decrease_size()

input_manager.bind('1', _resize, mode='scroll', description='size', value_getter=lambda: spore_manager.size)
input_manager.bind('n', lambda: _step('N'), description='step to next pause')
input_manager.bind('m', lambda: _step('M'), description='to pause of same level')
input_manager.bind('c', lambda: _step('C'), description='to the end')
input_manager.bind('x', _restart, description='restart from the last seed')
input_manager.bind('g', _seed_here, description='seed at look point')


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
    if key == 'left mouse down' and not held_keys['alt']:
        _seed_click()

print("Ready. WASD to move, Q/E zoom, Alt cursor, Esc exit.")
print("=" * 50)

if __name__ == '__main__':
    app.run()
