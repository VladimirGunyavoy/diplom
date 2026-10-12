"""
main_pend.py — МАЯТНИК в окне (отдельный скрипт, ДИ-окно main.py не затронуто): те же движковые части (stepper/growview/ursina), SYS=pend.
Рост b3: COVTOL=.003, DELTA=.2, UM=.3, цель — круг RHO=.1 как в ДИ, θ∈[-π,π) периодично, ω ±3.5, слои ±UM, 0, SOLVE8_PLUS=1. Пошагово: ЛКМ/N/M/C как в ДИ-окне; затем V/траектории.
Картинки: projects/spore/img/pend/<дата>/expNNN + last_exp; логи: v8/runs_pend/<время>/. Первый запуск: по одной споре от цели на слой u=±UM (V8_MAXCELLS=1; слой 0 не растим; ЛКМ после этапов +1 спора; V8_MAXCELLS=0 — полный атлас). Запуск: python main_pend.py (offscreen: V8_WINDOW=offscreen).
"""
import os as _os
for _k, _v in dict(SYS='pend', V8_GRID='4', M='5', KF='11', NFAIL='400', DELTA='.2', RMAX='.3', TMAX='8', TQDM_MI='1000', GS='0', GLIM='0', GOALB='0', GOALSHAPE='ball', GSEED='1', RHO='.1', UM='.3', COVTOL='.003', OVL='.05', GROW='2', OVH='.5', FRAC='.5', DTN='.06', NORM='gram', SEL='stop', STOPS='0', SOLVE8_PLUS='1', NORMFRONT='2', ACTIVE='1', MAXIT='8000', V8_PEND='1', V8_MAXCELLS='5', V8_WAIT='600').items(): _os.environ[_k] = _v if _k in ('SYS', 'V8_PEND', 'V8_WAIT') else _os.environ.get(_k, _v)
_os.environ['SYS'] = 'pend'

import sys
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np

import time
try:
    from src.runlog import RunLog; runlog = RunLog(root=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs_pend'))      # журнал запуска: runs/<время>/ (grow.log — дубль консоли, params/atlas/vcalc/traj/keys); запись в фоне
except Exception as e_:
    runlog = None; print('[runlog] off: %r' % (e_,))
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

app = Ursina(window_type=os.environ.get('V8_WINDOW', 'onscreen'))    # V8_WINDOW=offscreen for benchmarks without a visible window

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
    init_position=(0, 9, -12),      # поле x ±3 / буфер x ±6 (ширина 12) влезает целиком (2026-10-12)
    init_rotation_x=38,
    init_rotation_y=0,
    color_manager=color_manager,
    input_manager=input_manager,
    update_manager=update_manager
)

zoom_manager = ZoomManager(scene_setup, color_manager=color_manager)
scene_setup.register_frame_in_zoom(zoom_manager)
scene_setup.frame.toggle_visibility()    # start with the frame (axes + cube at 0) hidden: zero is marked by the goal; U toggles it

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
spore_manager.size = 0.5    # v8: initial point size multiplier (growview discs read it; key '1' changes it)
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
import src.stepper.growview as _gvmod; _gvmod.WRAP_X = 2 * 3.141592653589793 if _os.environ.get("V8_WRAP", "1") != "0" else 0.    # шов θ=±π: отрезки/треугольники режутся и клонируются периодически
_gvmod.SHOW_X = float(_os.environ.get('V8_THETA', '8')) if _gvmod.WRAP_X > 0 else 0.    # периодические копии на θ ∈ [−8, 8] (расчёт по одному периоду)
try:
    from src.stepper.live import LiveStepper                 # настоящий алгоритм (b6, src/algo/growN.py SYS=di) под Stepper
    stepper = LiveStepper(seed=None)
except ImportError as e_:
    print('[v8] live algorithm unavailable (%s) - DemoStepper' % e_)
    from src.stepper.demo import DemoStepper
    stepper = DemoStepper(seed=(0., 0.))
growview = GrowView(zoom_manager, ax=(0, 1), spore_manager=spore_manager); growview.runlog = runlog
try:                                                                  # картинки прогона: папка эксперимента этого запуска (projects/spore/img/<дата>/expNNN + last_exp), печать пути
    from src import pics as _pics
    _exp, growview.pics_dirs = _pics.new_experiment(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'img', 'pend')))
    open(os.path.join(_exp, 'run.txt'), 'w', encoding='utf-8').write('started %s\nenv: %s\n' % (time.strftime('%F %T'), {k: v for k, v in sorted(os.environ.items()) if k in ('SYS', 'M', 'MAXC', 'V8_MAXCELLS', 'V8_ONLY_GOAL', 'XL', 'VL', 'GMX', 'KF', 'DELTA', 'RMAX', 'ETOL', 'COMMIT_EVERY')}))
    print('[pics] experiment folder: %s ; pictures are redrawn here after every finished stage: %s' % (_exp, ', '.join(growview.pics_dirs)), flush=True)
except Exception as _e: print('[pics] disabled: %r' % (_e,), flush=True)
if runlog is not None: runlog.params(extra=dict(layers=list(getattr(stepper, 'layers', ())), stepper=type(stepper).__name__))
growview.draw(stepper.snapshot)    # the first pause (the algorithm's own first seed) is visible right at start

def _add_spore():
    """режим первых N спор: ЛКМ после всех этапов — +1 спора каждому слою (рост заново до N+1, этапы заново; первые N спор те же: rng и очередь затравок детерминированы)"""
    from src.stepper import live as _live; from src.algo import growN as _g
    n0 = sum(len(c) for c in getattr(stepper, 'cells_by_layer', {}).values()); n = int(os.environ.get('V8_MAXCELLS', 1)) + 1
    os.environ['V8_MAXCELLS'] = str(n); os.environ['MAXC'] = str(n); _g.MAXC = n; print('[spores] +1 spore per layer: N = %d, regrow' % n, flush=True)
    growview.draw(None); growview.times = type(growview.times)(); _last_seed[0] = None; stepper.seed_at(None); growview.draw(None); stepper.key('C'); growview.draw(stepper.snapshot)
    n1 = sum(len(c) for c in stepper.cells_by_layer.values()); print('[spores] cells per layer now: %s (total %d, was %d)%s' % ({u: len(c) for u, c in stepper.cells_by_layer.items()}, n1, n0, '' if n1 > n0 else ' - no more spores can grow'), flush=True)
    for u, cs in stepper.cells_by_layer.items():
        for i, c in enumerate(cs): print('[spores]   u=%+g spore %d: centre (%.3f, %.3f), rows %d' % (u, i + 1, c['c'][0], c['c'][1], len(c['G'])), flush=True)


def _step(k):
    if k == 'N' and os.environ.get('V8_ONLY_GOAL') == '1' and getattr(stepper, 'done', False) and not growview.stage_todo(stepper.snapshot) and not (growview._stj is not None and growview._stj['th'].is_alive()): _add_spore(); return
    if k in ('N', 'M', 'C') and getattr(stepper, 'done', False) and growview.stage_todo(stepper.snapshot): growview.stage_next(stepper.snapshot); return      # рост закончен: ЛКМ/N — следующий этап расчётов (V всех вариантов, траектории)
    t0 = time.perf_counter(); stepper.key(k); growview.draw(stepper.snapshot)
    print(f"[stepper] {k}: {stepper.snapshot and stepper.snapshot.get('phase')}  {1e3 * (time.perf_counter() - t0):.1f} ms")

_last_seed = [None]

def _seed_here():
    x, v = shared_context.look_point; _last_seed[0] = (float(x), float(v)); stepper.seed_at((float(x), float(v))); growview.draw(None); _step('N')

def _seed_click():
    wp = mouse.world_point
    if wp is None: return
    a, b = zoom_manager.a_transformation, zoom_manager.b_translation
    pt = ((wp.x - b[0]) / a, (wp.z - b[2]) / a)
    from src.algo import growN as _g
    if bool(_g.ingoal(np.array(pt, float))): print('[v8] click in goal -> goal seeds'); _last_seed[0] = None; stepper.seed_at(None)
    else: _last_seed[0] = pt; stepper.seed_at(pt)
    growview.draw(None); _step('N')

def _finer():
    if hasattr(stepper, 'finer'): t0 = time.perf_counter(); stepper.finer(); growview.draw(stepper.snapshot); print('[stepper] finer: step=%s  %.1f ms' % (stepper.depth_name, 1e3 * (time.perf_counter() - t0)))
    else: _step('N')

def _up():
    if getattr(stepper, 'done', False) and growview.stage_todo(stepper.snapshot): growview.stage_next(stepper.snapshot, all_=True); return      # ПКМ после роста — все оставшиеся этапы подряд
    if hasattr(stepper, 'up'): t0 = time.perf_counter(); stepper.up(); growview.draw(stepper.snapshot); print('[stepper] up: step=%s  %s  %.1f ms' % (stepper.depth_name, stepper.snapshot and stepper.snapshot.get('phase'), 1e3 * (time.perf_counter() - t0)))
    else: _step('C')

def _back():
    if hasattr(stepper, 'back') and stepper.back(): growview.draw(stepper.snapshot); print('[stepper] back: history %s/%s' % stepper.hist_info if stepper.hist_info else '[stepper] back: live')
    else: print('[stepper] back: no earlier snapshot')

def _caption():
    """строка статуса: каждый фрагмент — отдельная строка (дальше — перенос по словам в Message)"""
    h = getattr(stepper, 'hist_info', None)
    sn = stepper.snapshot or {}
    lay = ('layer %d/%d  show[%s]' % (sn['layer_k'] + 1, sn['layers_n'], ''.join(('+' if growview.vis[i] else '-') for i in range(sn['layers_n'])))) if sn.get('layers_n') else ''
    parts = [growview.caption(), growview.field_text(), growview.pics_label(), ('[MODE: first %s spores per layer (layers +1/-1); LMB after the stages adds one more spore per layer; off: delete v8/ONLY_GOAL]' % os.environ.get('V8_MAXCELLS', '1') if os.environ.get('V8_ONLY_GOAL') == '1' else ''), lay, ('step: %s' % stepper.depth_name) if hasattr(stepper, 'depth_name') else '', growview.skeleton_label(), growview.vlabel(), growview.stage_label(), ('[history %d/%d]' % h) if h else '']
    return '\n'.join(p for p in parts if p)

def _restart():
    """X: drop everything built so far (and a hand-placed seed) and start over with the algorithm's own seeds (from the goal)."""
    growview.draw(None); growview.times = type(growview.times)()
    _last_seed[0] = None; stepper.seed_at(None); growview.draw(None); _step('N')

floor.collider = 'box'
screen_manager.add_message(Message(name='growview', position=(-0.79 + mx, -0.46 + my), getter=_caption, bottom=True))
screen_manager.add_message(Message(name='timetable', position=(0.30 + mx, 0.48 - my), getter=lambda: growview.times.text(getattr(stepper, 'times', {}))))

# ===== BINDINGS =====

def _resize(sign):
    if sign > 0:
        spore_manager.increase_size()
    else:
        spore_manager.decrease_size()

input_manager.bind('1', _resize, mode='scroll', description='size', value_getter=lambda: spore_manager.size)
input_manager.bind('n', lambda: _step('N'), description='step to next pause')
input_manager.bind('left mouse down', lambda: None if held_keys['alt'] else _step('N'), description='next step')
input_manager.bind('control+left mouse down', lambda: None if held_keys['alt'] else _finer())    # InputManager turns every key into 'control+<key>' while Ctrl is held; Shift is taken (fly down, SceneManager)
input_manager.add_help('Ctrl+LMB', 'finer + step')    # input() forwards every key to input_manager; Alt = free cursor (mouse off)
input_manager.bind('right mouse down', lambda: None if held_keys['alt'] else _up(), description='up a level (in a spore: finish it; on a spore: end of layer)')
input_manager.bind('middle mouse down', lambda: None if held_keys['alt'] else _seed_click(), description='restart with a seed at the click\n      (in the goal: algorithm seeds)')
input_manager.bind('m', lambda: _step('M'), description='to pause of same level')
input_manager.bind('c', lambda: _step('C'), description='to the end')
input_manager.bind('z', _back, description='step back (history); N/LMB go forward')
input_manager.bind('x', _restart, description='restart: algorithm seeds (drop click)')
input_manager.bind('g', _seed_here, description='seed at look point')
for _i, _key in enumerate('789'): input_manager.bind(_key, lambda i=_i: growview.toggle_layer(i), description='show/hide layer %d' % (_i + 1))
input_manager.bind('k', growview.toggle_skeleton, description='skeleton: normal / time')
input_manager.bind('v', growview.toggle_vsurf, description='V surface per layer (height, colour = V)')
input_manager.bind('y', growview.cycle_vcalc, description='V calc: old / by Q lin / by Q quad')
input_manager.bind('t', growview.toggle_traj, description='trajectories: off / floor+surface / floor / surface')
input_manager.bind('p', growview.live_toggle, description='play the trajectories live (again: off)')
input_manager.bind('o', lambda: growview.live_launch(tuple(shared_context.look_point)), description='new agent from the look point (live)')
input_manager.bind('b', growview.toggle_common, description='common V* layer (min over layers)')
def _toggle_floor(): floor.visible_self = not floor.visible_self
input_manager.bind('f', _toggle_floor, description='toggle floor')


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
    update_manager.update_all(); growview.poll(); growview.live_update()
    growview.refresh()    # spore size ('1') changed -> rebuild the point discs

def input(key):
    t0 = time.perf_counter(); input_manager.handle_input(key)
    if runlog is not None: runlog.key(key, dict(v=growview.vmode, b=growview.cmode, traj=growview.tview if growview.tmode else 'off', calc=growview.vcalc, skel=growview.skel, vis=''.join('+' if x else '-' for x in growview.vis[:3])), dict(ms=round(1e3 * (time.perf_counter() - t0), 1), **growview.dinfo))      # состояние ПОСЛЕ нажатия + время отклика и cache hit/miss последней отрисовки

print("Ready. WASD to move, Q/E zoom, Alt cursor, Esc exit.")
print("=" * 50)

if __name__ == '__main__':
    app.run()
