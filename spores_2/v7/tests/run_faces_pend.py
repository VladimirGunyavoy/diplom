import sys, time; sys.path.insert(0, '.')
import numpy as np
from src.cells7.systems import Sys2
from src.cells7.faces import Faces, lines_uniform, lines_graded
def pend_top(umax=.3, wmax=4.0):
    f = lambda x, u: np.stack([x[..., 1], np.sin(x[..., 0]) + u], -1)      # φ = θ − π: цель (0,0), верх; θ̈ = sin φ + u
    A = lambda x, u: np.array([[0.0, 1.0], [np.cos(x[0]), 0.0]])
    return Sys2('pend_top', (np.pi, wmax), f, A, umax, per=(2.0, 0.0), box=((-1, 1), (-1, 1)))
S = pend_top(); rng = np.random.default_rng(0)
def build(kind, a, ratio, n):
    lx = lines_graded(a, ratio, 200) if kind == 'g' else lines_uniform(n, True)
    return lx
Q = np.stack([rng.uniform(-1, 1, 300), rng.uniform(-.8, .8, 300)], 1)       # старты по всему кольцу, включая низ (x≈±1 — это θ≈0, φ=π)
# эталон: мелкая сетка
t0 = time.time(); lf = lines_graded(.02, 1.0, 400, d0=.015); Ff = Faces(S, lf, lf, m=4, periodic_x=True, tmax=40.0); Ff.solve(); Tf, _ = Ff.rollout(Q); print('эталон: линий', len(lf), 'проб', len(Ff.P), 'дошли', np.isfinite(Tf).mean(), 'T мед', np.nanmedian(Tf[np.isfinite(Tf)]), round(time.time() - t0), flush=True)
for kind, a, ratio, n, d0 in (('g', .02, 1.0, 0, .04), ('g', .02, 1.0, 0, .08), ('g', .02, 1.15, 0, .04), ('g', .02, 1.3, 0, .04), ('g', .02, 1.3, 0, .08)):
    ln = lines_graded(a, ratio, 400, d0=d0); t0 = time.time(); F = Faces(S, ln, ln, m=4, periodic_x=True, tmax=40.0); F.solve(); T, sw = F.rollout(Q); ok = np.isfinite(T) & np.isfinite(Tf); r = T[ok] / Tf[ok]
    print(f'd0={d0} ratio={ratio}: линий {len(ln)} проб {len(F.P)} дошли {np.isfinite(T).mean():.2f} T/T_эталон mean {r.mean():.3f} med {np.median(r):.3f} max {r.max():.2f} сек {time.time()-t0:.1f}', flush=True)
