# research-18: «бусины» — 300 точек входного среза клетки 160, полёт с u клетки 5.8 с; пути 15 из них на всей фазовой плоскости.
import os, sys, numpy as np
sys.path.insert(0, os.environ['CELLS7']); import grow_cells2d as G
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
e_ = np.linspace(-G.RHO, G.RHO, 41); G.BARRIER = cKDTree(np.r_[np.c_[e_, e_ * 0 - G.RHO], np.c_[e_, e_ * 0 + G.RHO], np.c_[e_ * 0 - G.RHO, e_], np.c_[e_ * 0 + G.RHO, e_]])
rng = np.random.default_rng(0); cells = []
for u in G.US: l, _ = G.build_layer(u, rng); cells += l
c = cells[int(os.environ.get('K', 160))]; s = np.linspace(-c.r, c.r, 300); E = np.c_[np.interp(s, c.sn, c.G[0][:, 0]), np.interp(s, c.sn, c.G[0][:, 1])]
P = [E.copy()]; y = E.copy()
for _ in range(c.nb + c.nf): y = G.step(y, c.u); P.append(y.copy())
P = np.array(P); fig, ax = plt.subplots(figsize=(11, 7))
ax.fill(np.r_[c.G[:, 0, 0], c.G[::-1, -1, 0]], np.r_[c.G[:, 0, 1], c.G[::-1, -1, 1]], color='0.85', label='клетка 160, %.1f с' % ((c.nb + c.nf) * G.DTN))
for j in range(0, 300, 20): ax.plot(P[:, j, 0], P[:, j, 1], '-', lw=1, color=plt.cm.plasma(j / 300))
ax.plot(E[:, 0], E[:, 1], 'k-', lw=4, label='вход: длина %.2f' % np.linalg.norm(E[-1] - E[0]))
sc = ax.scatter(P[-1, :, 0], P[-1, :, 1], c=s, cmap='plasma', s=10, zorder=5, label='выход: длина %.2f' % (np.linalg.norm(np.diff(P[-1], axis=0), axis=1).sum()))
ax.add_patch(plt.Rectangle((-G.RHO, -G.RHO), 2 * G.RHO, 2 * G.RHO, fill=False, ec='tab:cyan', lw=2)); ax.text(.12, -.1, 'цель', color='tab:cyan')
ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-2.5, 2.5); ax.set_xlabel('θ (угол)'); ax.set_ylabel('ω (скорость)'); ax.legend(loc='lower left', fontsize=10); plt.colorbar(sc, label='место бусины на нитке')
ax.set_title('Клетка 160: бусины входного среза летят по потоку (u = %+g)' % c.u, fontsize=12); fig.tight_layout(); fig.savefig(os.environ['OUT'], dpi=110); print('готово', flush=True)
