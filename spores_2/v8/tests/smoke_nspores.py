"""Режим первых N спор: N=1 → этапы → ЛКМ (+1 спора на слой) → этапы (N=2) → ЛКМ (N=3). Windows-питон offscreen; картинки reports/ns_*.png."""
import os, sys, time
os.environ['V8_WINDOW'] = 'offscreen'; os.environ['V8_MAXCELLS'] = '1'; os.chdir(r'C:\GitHub\claude-work\projects\spore\spores_2\v8'); sys.path.insert(0, os.getcwd())
import ursina; _M = type(ursina.mouse); _M.locked = property(lambda s: False, lambda s, v: None); _M.visible = property(lambda s: True, lambda s, v: None)
import main as M
from panda3d.core import Filename
G = M.growview; P = M.scene_setup.player
def shot(name):
    for _ in range(3): M.app.step(); G.poll()
    print('shot', name, ursina.application.base.win.save_screenshot(Filename.from_os_specific(os.path.join(r'C:\GitHub\claude-work\projects\spore\spores_2\v8\reports', name))), flush=True)
def wait():
    t0 = time.time()
    while G._stj is not None and time.time() - t0 < 400: M.app.step(); G.poll(); time.sleep(.05)
for k in (1, 2, 3):
    if k == 1: M._step('C')
    else: M._step('N')
    print('N =', os.environ['V8_MAXCELLS'], 'cells', {u: len(c) for u, c in M.stepper.cells_by_layer.items()}, 'todo', len(G.stage_todo(M.stepper.snapshot)), flush=True)
    M._up(); wait(); M.app.step(); G.poll()
    if G._stj is None and G.stage_todo(M.stepper.snapshot): M._up(); wait()
    G.toggle_common(); G.toggle_traj(); M.app.step(); G.poll(); shot('ns_%d.png' % k); G.toggle_traj(); G.toggle_traj(); G.toggle_traj(); G.toggle_common()
print('SMOKE OK')
