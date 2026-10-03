"""какие старты теряет агент бабочек маятника при u·.95 (hub-worker-14). Из spores_2/v7."""
import sys, os; sys.path.insert(0, '.'); sys.argv = ['x']
import numpy as np
os.environ['GAIN'] = '.95'
from src.cells7.butterfly_pend_hybrid_gain import *
B = ButterflyPend(N=5000, m=7, tau=2.0, win=.3); B.V = np.load('/tmp/claude-1000/bp_V4.npy')
rng = np.random.default_rng(0); Q = np.stack([rng.uniform(-np.pi, np.pi, 100), rng.uniform(-2, 2, 100)], 1); ref = np.load('../v5chain/reports/research/pend_ref_T.npy')
Bd = None; T, SW = run(B, Bd, Q, 0.)
lost = np.nonzero(~np.isfinite(T) & (ref > .05))[0]
for i in lost: print('потерян', i, Q[i].round(3), 'ref', round(ref[i], 2), 'V старта', round(B.best(Q[i])[0], 2))
print('--- где умирает агент (последнее состояние)')
for i in lost[1:]:
    y = np.array(Q[i], float); y[0] = wrap(y[0]); t = 0.
    while t < 40:
        J, b = B.best(y)
        if ingoal(y): print(i, 'дошёл?'); break
        if b is None or J >= BIG / 2: print(i, 'умер t', round(t, 2), 'y', y.round(2), 'J', round(float(J), 1)); break
        _, _, ta, u = b; h = min(.06, ta); z = y[None, :]
        for _ in range(4): z = rk4(z, np.array([u * GAIN]), np.array([h / 4])); t += h / 4
        y = z[0].copy(); y[0] = wrap(y[0])
    else: print(i, 'не дошёл за 40 с (бродит)', y.round(2))
