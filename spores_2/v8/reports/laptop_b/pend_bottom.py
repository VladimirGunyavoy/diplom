"""b3 №16: клетки у нижнего равновесия слоя u (центр: θ=π+asin(u), ω=0) при посеве по эллипсам. UL=<u> PNG=<файл> python3 pend_bottom.py ; env как main_pend (pend_bottom_env.sh)"""
import os, sys, time, collections
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S
g = S.load_growN(); u = float(os.environ.get('UL', .3)); tag = os.environ.get('TAG', '-'); PI = np.pi
thb = g.wrapy(np.array([PI + np.arcsin(u), 0.]))[0]; cb = np.array([thb, 0.]); log = []; prev = collections.Counter()
def lg(u_, cells):
    cur = g.STOP; d = {k: cur[k] - prev.get(k, 0) for k in cur if cur[k] != prev.get(k, 0)}; prev.clear(); prev.update(cur); log.append(d)
t0 = time.perf_counter(); cells, idx = g.build_layer(u, np.random.default_rng(0), lg); tg = time.perf_counter() - t0
C = np.array([g.wrapy(c.c) for c in cells]); dist = np.linalg.norm(np.c_[g.wrapy(C - cb)[:, 0], C[:, 1]], axis=1); near = dist < float(os.environ.get('NEARR', .8))
r = np.array([float(c.r[0]) for c in cells]); nr = np.array([c.nf + c.nb for c in cells])
print('RES[%s] u=%+.2f: %.1f с, клеток %d; центр слоя (θ=%.3f, ω=0); клеток в .8 от центра: %d (r мед %.3f, строк мед %d), прочих r мед %.3f' % (tag, u, tg, len(cells), thb, near.sum(), np.median(r[near]) if near.any() else 0, np.median(nr[near]) if near.any() else 0, np.median(r[~near])))
st = collections.Counter()
for i in np.flatnonzero(near):
    for k, v in log[i].items(): st[k] += v
print('RES   причины остановок у центра (сумма по клеткам рядом): %s' % dict(st))
# мин. скорость |f| вдоль клетки и обороты
for i in np.flatnonzero(near)[:12]:
    c = cells[i]; Gc = c.G; fm = np.linalg.norm(g.f(Gc.reshape(-1, 2), u), axis=1).min(); print('RES   клетка %3d центр (%.2f,%.2f) |c−центр| %.2f r %.3f строк %d (hb %d hf %d) min|f| %.3f остановки %s' % (i, *C[i], dist[i], r[i], nr[i], c.hb, c.hf, fm, log[i]))
if os.environ.get('PNG'):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(7, 7)); R = float(os.environ.get('PR', 1.2))
    for i in np.flatnonzero(dist < 1.6 * R):
        G_ = g.wrapy(cells[i].G.reshape(-1, 2)).reshape(cells[i].G.shape); 
        for j in range(G_.shape[1]): ax.plot(G_[:, j, 0], G_[:, j, 1], lw=.4, c='#555')
        for k in range(G_.shape[0]): ax.plot(G_[k, :, 0], G_[k, :, 1], lw=.3, c='#c80')
    ax.plot(*cb, 'r*', ms=12); ax.set_xlim(thb - R, thb + R); ax.set_ylim(-R, R); ax.set_aspect('equal'); ax.set_title('u=%+.2f, %s: клеток у центра %d из %d' % (u, tag, near.sum(), len(cells))); fig.tight_layout(); fig.savefig(os.environ['PNG'], dpi=110); print('RES картинка', os.environ['PNG'])
