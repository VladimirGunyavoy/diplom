"""W1б: этапы расчёта после роста (ЛКМ — один этап, ПКМ — все). Offscreen, Windows-питон: timeout 290 python.exe tests/smoke_stages.py. Картинки reports/stages_*.png"""
import os, sys, time
os.environ['V8_WINDOW'] = 'offscreen'; os.chdir(r'C:\GitHub\claude-work\projects\spore\spores_2\v8'); sys.path.insert(0, os.getcwd())
import ursina; _M = type(ursina.mouse); _M.locked = property(lambda s: False, lambda s, v: None); _M.visible = property(lambda s: True, lambda s, v: None)
import main as M
from panda3d.core import Filename
def shot(name):
    for _ in range(3): M.app.step(); G.poll()
    print('shot', name, ursina.application.base.win.save_screenshot(Filename.from_os_specific(os.path.join(r'C:\GitHub\claude-work\projects\spore\spores_2\v8\reports', name))), flush=True)
def wait():
    t0 = time.time()
    while G._stj is not None and time.time() - t0 < 560: M.app.step(); G.poll(); time.sleep(.05)
P = M.scene_setup.player; P.cursor.enabled = False; P.position = (0, 6, -9); P.camera_pivot.rotation_x = 32; P.rotation_y = 0
G = M.growview; M._step('C'); print('done:', M.stepper.done, 'todo', [(t['kind'], t['calc']) for t in G.stage_todo(M.stepper.snapshot)]); print('LABEL', G.stage_label())
M._step('N'); print('LMB 1 started'); wait(); M.app.step(); G.poll(); print('LABEL', G.stage_label()); G.toggle_common(); shot('stages_after1.png'); G.toggle_common()
M._up(); wait(); M.app.step(); G.poll(); print('after RMB todo', G.stage_todo(M.stepper.snapshot), 'LABEL', repr(G.stage_label()))
G.toggle_common(); G.toggle_traj(); M.app.step(); G.poll(); print(G.vlabel(), 'tfresh', G._tfresh); shot('stages_all_q_traj.png')
for i in range(2): G.cycle_vcalc(); M.app.step(); G.poll(); print(G.vlabel()[:140], G.dinfo)
print('SMOKE OK')
