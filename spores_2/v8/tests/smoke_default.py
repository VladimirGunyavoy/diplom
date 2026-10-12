"""V и траектории включены по умолчанию: до конца роста ничего не считается, после (оба слоя) — V и траектории появляются без клавиш. V8_MAXCELLS=2. Картинка reports/default_on.png"""
import os, sys, time
os.environ['V8_WINDOW'] = 'offscreen'; os.environ['V8_MAXCELLS'] = '2'; os.chdir(r'C:\GitHub\claude-work\projects\spore\spores_2\v8'); sys.path.insert(0, os.getcwd())
import ursina; _M = type(ursina.mouse); _M.locked = property(lambda s: False, lambda s, v: None); _M.visible = property(lambda s: True, lambda s, v: None)
import main as M
from panda3d.core import Filename
G = M.growview
print('DEFAULTS vmode', G.vmode, 'tmode', G.tmode, 'auto', G._auto)
for _ in range(80):
    M._step('N')
    if M.stepper.snapshot.get('phase') == 'layer' and M.stepper.snapshot.get('layer_k') == 0: break
M.app.step(); G.poll(); print('AFTER LAYER 1: vres', G._vres is not None, 'job', G._stj is not None)
M._step('C'); print('done', M.stepper.done)
t0 = time.time()
while (G._traj is None) and time.time() - t0 < 200: M.app.step(); G.poll(); time.sleep(.05)
print('VLABEL', G.vlabel()[:200], '| vres', G._vres is not None, 'traj', G._traj is not None)
for _ in range(3): M.app.step(); G.poll()
print('shot', ursina.application.base.win.save_screenshot(Filename.from_os_specific(r'C:\GitHub\claude-work\projects\spore\spores_2\v8\reports\default_on.png')))
time.sleep(25)
print('SMOKE OK')
