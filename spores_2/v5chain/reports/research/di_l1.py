"""DI, цена ∫(1+ρ|u|): V на клетках v7 для bang/tri/spec11; сравнение по реальной цене агента (эталон — лучший из трёх по старту)."""
import sys, os, json, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v7_spore_di import Cover
from v7_spore_spec_l1 import solve_U, rollout_U
rng = np.random.default_rng(1); Q = rng.uniform(-1.5, 1.5, (300, 2)); Cv = Cover(tau=.4, r=.1, seeds=6000); dt = Cv.dt0
sets = {'bang': (-1.0, 1.0), 'tri': (-1.0, 0.0, 1.0), 'spec11': tuple(np.linspace(-1, 1, 11))}
for rho in (.5, 2.):
    R = {}
    for nm, U in sets.items(): solve_U(Cv, U, rho); R[nm] = rollout_U(Cv, Q, U, rho, dt)
    best = np.min(np.stack([R[n][0] for n in sets]), 0); ok = np.isfinite(best) & (best > .05)
    for nm in sets:
        J, T, sw, tv = R[nm]; print(json.dumps(dict(sys='DI', cost='L1', rho=rho, U=nm, reach=float(np.isfinite(J).mean()), J_over_best=round(float(np.mean(J[ok] / best[ok])), 4), J_over_best_max=round(float(np.max(J[ok] / best[ok])), 3), sw_med=float(np.median(sw[ok])), tv_med=round(float(np.median(tv[ok])), 2))), flush=True)
