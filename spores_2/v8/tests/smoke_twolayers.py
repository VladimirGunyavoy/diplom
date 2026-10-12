"""Режим первой споры: V/T до готовности обоих слоёв ничего не считают («waiting»), после — оба слоя. Windows-питон offscreen."""
import os, sys, time
os.environ['V8_WINDOW'] = 'offscreen'; os.environ['V8_ONLY_GOAL'] = '1'; os.chdir(r'C:\GitHub\claude-work\projects\spore\spores_2\v8'); sys.path.insert(0, os.getcwd())
import ursina; _M = type(ursina.mouse); _M.locked = property(lambda s: False, lambda s, v: None); _M.visible = property(lambda s: True, lambda s, v: None)
import main as M
G = M.growview
def ph(): return M.stepper.snapshot.get('phase'), M.stepper.snapshot.get('layer_k')
for _ in range(60):
    M._step('N')
    if ph()[0] == 'layer': break
print('AFTER LAYER 1:', ph(), 'done', M.stepper.done)
G.toggle_common(); G.toggle_traj(); M.app.step(); G.poll(); print('VLABEL1', G.vlabel()[:200], '| vres', G._vres is not None)
for _ in range(60):
    if M.stepper.done: break
    M._step('N')
print('AFTER ALL:', ph(), 'done', M.stepper.done); M.app.step(); G.poll(); print('VLABEL2', G.vlabel()[:260])
t0 = time.time()
while (G._stj is not None or G._traj is None) and time.time() - t0 < 120: M.app.step(); G.poll(); time.sleep(.05)
print('VLABEL3', G.vlabel()[:420]); print('SMOKE OK')
