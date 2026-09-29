import sys; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.dd_atlas import solve_dd, V_at
rng = np.random.default_rng(2); P = [np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)]) for _ in range(200)]
R = {}
for h in (0.5, 0.25, 0.125):
    A = solve_dd(h=h); R[h] = np.array([V_at(A, p) for p in P])
for a, b in ((0.5, 0.125), (0.25, 0.125)):
    d = R[a] - R[b]; print('h=%s vs 0.125: mean|d| %.3f max %.3f mean d %+.3f' % (a, abs(d).mean(), abs(d).max(), d.mean()))
