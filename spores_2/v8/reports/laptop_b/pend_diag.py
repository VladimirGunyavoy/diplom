"""b3: диагноз роста слоя pend. UL=<u> MAXC=<n> python3 pend_diag.py  (env как v7-бенч, см. pend_env.sh). Печатает: время, клетки, STOP, r, строки, покрытие поля, клетки у сепаратрис."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S
g = S.load_growN(); u = float(os.environ.get('UL', .3)); tag = os.environ.get('TAG', '-')
t0 = time.perf_counter(); cells, idx = g.build_layer(u, np.random.default_rng(0)); tg = time.perf_counter() - t0
r = np.array([float(c.r[0]) for c in cells]); nr = np.array([c.nf + c.nb for c in cells]); P = np.array([c.p for c in cells])
rq = np.random.default_rng(1); pr = np.stack([rq.uniform(-np.pi, np.pi, 4000), rq.uniform(-3, 3, 4000)], 1); cov = idx.covered(g.wrapy(pr)).mean()
# энергия E = w^2/2 + cos(th) -> сепаратриса E=1 (верх), нижнее равновесие E=-1
Ep = lambda y: y[:, 1] ** 2 / 2 + np.cos(y[:, 0])
sep = np.abs(Ep(P) - 1) < .3
print('RES[%s] u=%+.2f: %.1f с, клеток %d, покрытие поля(|w|<3) %.3f | STOP %s' % (tag, u, tg, len(cells), cov, dict(g.STOP)))
print('RES[%s]   r(мед/p10/p90/min) %.3f/%.3f/%.3f/%.3f | строк (мед/p10/p90/max) %d/%d/%d/%d | клеток у сепаратрисы |E-1|<.3: %d (r мед %.3f), прочих r мед %.3f' % (tag, np.median(r), *np.percentile(r, [10, 90]), r.min(), np.median(nr), *np.percentile(nr, [10, 90]), nr.max(), sep.sum(), np.median(r[sep]) if sep.any() else 0, np.median(r[~sep]) if (~sep).any() else 0))
np.save('diag_%s.npy' % tag, np.c_[P, r, nr])
