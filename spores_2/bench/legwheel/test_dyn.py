import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lw_dyn as L
from legwheel_robot import env as E, config as C
rng = np.random.default_rng(0); worst = 0
for _ in range(200):
    q = np.array([rng.uniform(-.1, .1), rng.uniform(-.2, .2), C.ALPHA_STAR + rng.uniform(-.2, .2)]); qd = rng.uniform(-2, 2, 3); tau = rng.uniform(-1, 1, 2) * L.UMAX
    ref = np.array(E._f_state(list(q) + list(qd), *tau)); mine = np.array(L.fp()(np.r_[q, qd], tau)).ravel()
    worst = max(worst, np.abs(ref - mine).max() / (np.abs(ref).max() + 1e-9))
print('max rel err', worst)
z = L.from_phys(np.array([.01, .02, C.ALPHA_STAR + .03, .01, .1, .1])); print('z', z, 'back', L.to_phys(z))
print('f_ca ok', np.array(ca_ := L.f_ca(__import__('casadi').DM(z), __import__('casadi').DM([1, -1]))).ravel())
