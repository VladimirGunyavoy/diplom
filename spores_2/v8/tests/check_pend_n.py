import os, sys, time
os.environ["V8_WINDOW"] = "offscreen"; os.chdir(r'C:\GitHub\claude-work\projects\spore\spores_2\v8'); sys.path.insert(0, os.getcwd())
import ursina; _M = type(ursina.mouse); _M.locked = property(lambda s: False, lambda s, v: None); _M.visible = property(lambda s: True, lambda s, v: None)
import main_pend as M, numpy as np
G = M.growview; M.stepper.key('C'); G.draw(M.stepper.snapshot); t0 = time.time()
while (G._traj is None or G._tprog) and time.time() - t0 < 250: M.app.step(); G.poll(); time.sleep(.05)
res, m = G._traj; st = np.asarray(res['status'], object); np.set_printoptions(precision=3, suppress=True, linewidth=200)
import collections
print('CELLS', {u: len(c) for u, c in M.stepper.cells_by_layer.items()} if hasattr(M.stepper, 'cells_by_layer') else '?')
print('N', len(st), 'reached', int((st == 'reached').sum()), collections.Counter(st.tolist()))
for i in np.flatnonzero(st != 'reached'): P = res['paths'][i]; print('FAIL #%d start (%.2f,%.2f) end (%.2f,%.2f) n=%d pre=%s Vend=%s' % (i, P[0, 0], P[0, 1], P[-1, 0], P[-1, 1], len(P), bool(res['prephase'][i]), np.asarray(res['V'][i])[-1]))
print('DBG DONE')
