import sys; sys.path.insert(0,'/home/rl/claude-work/projects/spore/spores_2/v6/src')
import numpy as np
from atlas6.dd_atlas import solve_dd, V_at, LAYERS
from atlas6.dd3 import flow
import atlas6.dd_atlas as da
def run(A, p0, mode, T=20.0, dt=None):
    dt = dt or A['tau']; p = np.array(p0, float); t = 0
    if mode == 'spec':
        g = np.linspace(-1, 1, 21); cand = [(v, w) for v in g for w in g if abs(v) + abs(w) <= 1 + 1e-9] if len(LAYERS) == 4 else [(v, w) for v in g for w in g]
    else: cand = [tuple(l) for l in LAYERS]
    best = 9
    for _ in range(int(T/dt)):
        e = min((flow(p, c, dt) for c in cand), key=lambda e: V_at(A, e)); p = e
        d = (p[2] + np.pi) % (2*np.pi) - np.pi; best = min(best, np.hypot(*p[:2]) + abs(d))
    d = (p[2] + np.pi) % (2*np.pi) - np.pi
    return np.hypot(*p[:2]), abs(d), best
print('LAYERS', LAYERS)
rng = np.random.default_rng(1)
for h in (.25, .1, .05):
    A = solve_dd(h=h, lim=max(1.0, 3*h*6) if h<.2 else 3.0); P0 = [np.r_[rng.uniform(-.6,.6,2), rng.uniform(-1,1)] for _ in range(6)]
    for mode in ('bb', 'spec'):
        r = np.array([run(A, p0, mode) for p0 in P0]); print(f'h={h} {mode}: итог xy {np.median(r[:,0]):.3f} (max {r[:,0].max():.3f}) θ {np.median(r[:,1]):.3f} лучшее за прогон мед {np.median(r[:,2]):.3f}', flush=True)
