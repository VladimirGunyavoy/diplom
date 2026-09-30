"""research hub-research-3: refine vs refine_fast по одним и тем же кандидатам (dd, запрос 2 profile_query). Запуск из v6."""
import sys, time; sys.path.insert(0, '.'); sys.path.insert(0, '../v5chain/reports/research')
import numpy as np
import src.atlas6.corridor_nd as C, refine_fast as RF
from scipy import optimize
from src.atlas6.adaptive_nd import SysN, build_back
from src.atlas6.dd3 import flow
from src.atlas6.dd_atlas import LAYERS
log = []
def mc(*a, **k):
    r = optimize.minimize(*a, **k); log.append((r.nit, r.nfev, r.status)); return r
C.minimize = mc; RF.minimize = mc
NB, NF = 1200, 50; tau = 0.25; R = .25; Rth = .26; rho = 0.08
fl = lambda P, s, t: flow(P, LAYERS[s], t); dth = lambda th: (th + np.pi) % (2 * np.pi) - np.pi
ing = lambda P: (np.hypot(P[..., 0], P[..., 1]) < R) & (np.abs(dth(P[..., 2])) < Rth)
g = lambda x: np.array([R ** 2 - x[0] ** 2 - x[1] ** 2, Rth ** 2 - dth(x[2]) ** 2])
miss = lambda X: np.maximum(np.hypot(X[..., 0], X[..., 1]) - R, 0) + np.maximum(np.abs(dth(X[..., 2])) - Rth, 0)
S = SysN(fl, 4, (1.0, 1.0, np.pi), ing, [(0.0, 0.0, 0.0)], per=(0, 0, 2 * np.pi), ok=lambda p: abs(p[0]) <= 3 and abs(p[1]) <= 3)
rng = np.random.default_rng(7); Q = [np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)]) for _ in range(2)]
back = build_back(S, tau, NB, rho); x = Q[int(sys.argv[1]) if len(sys.argv) > 1 else 1]
Cs, nh = C.candidates(S, x, back, tau, NF, rho, miss)
seen = set()
for hit, sc, p in Cs:
    sq = tuple(C.merge(p)[0])
    if sq in seen or not p: continue
    seen.add(sq)
    if len(seen) > 10: break
    out = []
    for f in (C.refine, RF.refine_fast):
        log.clear(); t = time.perf_counter(); T, _, d, ok = f(fl, x, p, g, tries=3); out.append('%s T %.3f ok %d %.2fs %s' % (f.__name__[:6], T, ok, time.perf_counter() - t, log))
    print(hit, len(sq), ' | '.join(out))
