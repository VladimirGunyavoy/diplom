import os, sys
os.environ['V8_WINDOW'] = 'offscreen'; os.environ['VSURF_TRAJ_N'] = '100'; os.chdir(r'C:\GitHub\claude-work\projects\spore\spores_2\v8'); sys.path.insert(0, os.getcwd())
import ursina; _M = type(ursina.mouse); _M.locked = property(lambda s: False, lambda s, v: None); _M.visible = property(lambda s: True, lambda s, v: None)
import time, main as M
from src.stepper import vsurf as VS
cnt = [0]; _ap = VS.agent_paths; _sl = VS.solve_layers; ns = [0]
def ap(*a, **k): cnt[0] += 1; return _ap(*a, **k)
def sl(*a, **k): ns[0] += 1; return _sl(*a, **k)
VS.agent_paths = ap; VS.solve_layers = sl
P = M.scene_setup.player; P.cursor.enabled = False
G = M.growview; M.stepper.key('C'); G.draw(M.stepper.snapshot)
def wait():
    t0 = time.time()
    while (G._traj is None or G._tjob is not None) and time.time() - t0 < 200: M.app.step(); G.poll(); time.sleep(.05)
M.input('v'); M.input('t'); print('1', G.vlabel()[:90]); wait(); print('after compute', cnt[0], ns[0], G.vlabel()[:110])
for i in range(8): M.input('t'); M.app.step(); G.poll()      # off, both, floor, surface, off, both, floor, surface
print('after T x8: agent_paths calls', cnt[0], 'solve calls', ns[0], G.vlabel()[:110])
G.toggle_traj(); G.cycle_vcalc(); print('mode q', G.vlabel()[:90]); wait(); print('computed q', cnt[0], ns[0])
G.cycle_vcalc(); G.cycle_vcalc(); wait(); print('back to v: calls', cnt[0], ns[0], G.vlabel()[:110])      # из кэша
G.cycle_vcalc(); wait(); print('back to q: calls', cnt[0], ns[0], G.vlabel()[:110])
assert cnt[0] == 3 and ns[0] == 3      # v, q, qquad — по разу; возвраты из кэша, (cnt[0], ns[0]); print('SMOKE OK')
