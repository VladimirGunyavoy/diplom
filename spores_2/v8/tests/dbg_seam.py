import os, sys, time
os.environ["V8_WINDOW"] = "offscreen"; os.chdir(r'C:\GitHub\claude-work\projects\spore\spores_2\v8'); sys.path.insert(0, os.getcwd())
import ursina; _M = type(ursina.mouse); _M.locked = property(lambda s: False, lambda s, v: None); _M.visible = property(lambda s: True, lambda s, v: None)
import main_pend as M, numpy as np
G = M.growview; M.stepper.key('C'); G.draw(M.stepper.snapshot); t0 = time.time()
while (G._traj is None or G._tprog) and time.time() - t0 < 250: M.app.step(); G.poll(); time.sleep(.05)
res, m = G._traj; st = np.asarray(res['status'], object); np.set_printoptions(precision=3, suppress=True, linewidth=200)
for i in np.flatnonzero(st == 'failed')[:4]:
    P = res['paths'][i]; print('#', i, 'start', res['X0'][i], 'npts', len(P), 'T', res['T'][i]); print(' last pts', P[-6:].tolist()); print(' V last', np.asarray(res['V'][i])[-6:], 'U last', np.asarray(res['U'][i])[-6:])
    print(' min/max theta', P[:, 0].min(), P[:, 0].max(), 'n sign flips of theta jump', int((np.abs(np.diff(P[:, 0])) > 3).sum()))
print('DBG DONE')
