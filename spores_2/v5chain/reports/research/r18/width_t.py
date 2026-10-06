# research-18: ширина среза клетки 160 по локальному времени: w(t), ln w и d ln w/dt (SLAM), пороги SMAX/SLAM. Срез — отрезок посева (±(1+HALO)r по нормали), пронесённый потоком.
import os, sys, numpy as np
sys.path.insert(0, os.environ['CELLS7']); import grow_cells2d as G
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
e_ = np.linspace(-G.RHO, G.RHO, 41); G.BARRIER = cKDTree(np.r_[np.c_[e_, e_ * 0 - G.RHO], np.c_[e_, e_ * 0 + G.RHO], np.c_[e_ * 0 - G.RHO, e_], np.c_[e_ * 0 + G.RHO, e_]])
rng = np.random.default_rng(0); cells = []
for u in G.US: l, _ = G.build_layer(u, rng); cells += l
c = cells[int(os.environ.get('K', 160))]; seg = c.c + np.linspace(-(1 + G.HALO) * c.r, (1 + G.HALO) * c.r, 101)[:, None] * c.n
wl = lambda r_: float(np.linalg.norm(np.diff(r_, axis=0), axis=1).sum())
T, W = [0.], [wl(seg)]; nmx = int(float(os.environ.get('TSPAN', 6.)) / G.DTN)
for sg in (1., -1.):
    y = seg.copy()
    for i in range(1, nmx + 1): y = G.step(y, c.u, sg); T.append(sg * i * G.DTN); W.append(wl(y))
o = np.argsort(T); T, W = np.array(T)[o], np.array(W)[o]; L = np.log(W / W[T == 0][0]); dL = np.gradient(L, T)
fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
axs[0].plot(T, W / W[T == 0][0], 'k-'); axs[0].set_yscale('log'); axs[0].set_ylabel('w(t) / w(0)')
for v, ls in ((2, '--'), (3, ':')): axs[0].axhline(v, color='tab:red', ls=ls, lw=1, label='SMAX %g' % v); axs[0].axhline(1 / v, color='tab:red', ls=ls, lw=1)
axs[1].plot(T, dL, 'k-'); axs[1].set_ylabel('d ln w / dt  [1/с]'); axs[1].set_xlabel('локальное время клетки t, с (0 — центр клетки)')
for v, ls in ((.7, '--'), (.5, ':')): axs[1].axhline(v, color='tab:blue', ls=ls, lw=1, label='SLAM %g' % v); axs[1].axhline(-v, color='tab:blue', ls=ls, lw=1)
for ax in axs:
    ax.axvspan(-c.nb * G.DTN, c.nf * G.DTN, color='0.9', zorder=0, label='клетка (без огранич.)'); ax.axvline(0, color='0.6', lw=.8); ax.legend(fontsize=8, loc='upper left')
axs[0].set_title('клетка %d (u = %+g): ширина среза вдоль потока' % (int(os.environ.get('K', 160)), c.u), fontsize=11)
fig.tight_layout(); fig.savefig(os.environ['OUT'], dpi=110)
def first(mask): k = np.flatnonzero(mask); return round(float(T[k[0]]), 2) if len(k) else None
fw, bw = T >= 0, T <= 0
for lab, m in (('SMAX 2', np.maximum(W / W[T == 0][0], W[T == 0][0] / W) > 2), ('SMAX 3', np.maximum(W / W[T == 0][0], W[T == 0][0] / W) > 3), ('SLAM .7', np.abs(dL) > .7), ('SLAM .5', np.abs(dL) > .5)):
    k1 = np.flatnonzero(m & fw); k2 = np.flatnonzero(m & bw)
    print(lab, 'стоп вперёд t =', round(float(T[k1[0]]), 2) if len(k1) else None, 'назад t =', round(float(T[k2[-1]]), 2) if len(k2) else None)
print('клетка без огранич.: t от', -c.nb * G.DTN, 'до', c.nf * G.DTN, '| w(конец)/w(0) =', round(float(W[np.argmin(abs(T - c.nf * G.DTN))] / W[T == 0][0]), 2), 'w(начало)/w(0) =', round(float(W[np.argmin(abs(T + c.nb * G.DTN))] / W[T == 0][0]), 2), flush=True)
