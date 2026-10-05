"""research-13: одна спора-«бабочка» ДИ крупно — отрезок с узлами, исходящие и входящие дуги (цвет = управление u), цена V на отрезке против точного T*."""
import numpy as np, sys, importlib.util
sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import butterfly_di as BD
from v7_faces_di import tstar_box
sp = importlib.util.spec_from_file_location('bf_v7', '../../../v7/src/cells7/butterfly_di.py'); mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
B = mod.ButterflyFast(N=1500, m=7).solve()
INK, MUT = '#1f2328', '#6b7280'; BX = dict(boxstyle='round,pad=.25', fc='white', ec='#d0d5dd', lw=.6)
cm = matplotlib.colors.LinearSegmentedColormap.from_list('div', ['#1f5fbf', '#9aa3ad', '#d9480f']); nrm = matplotlib.colors.Normalize(-1, 1)
i0 = int(np.argmin(np.linalg.norm(B.C - np.array([1.0, .45]), axis=1))); c = B.C[i0]; r = B.r
k = np.array(B.tree.query_ball_point(c, B.tau * (abs(c[1]) + 1) + r + .4)); k = k[k != i0]
fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.4), dpi=140, gridspec_kw=dict(width_ratios=[1.9, 1])); fig.patch.set_facecolor('white'); a = ax[0]
segs = np.stack([np.c_[B.C[:, 0], B.C[:, 1] - r], np.c_[B.C[:, 0], B.C[:, 1] + r]], 1); a.add_collection(LineCollection(segs, colors='#b8bfc9', linewidths=1.6))
def arc(y, t, u, n=16): tt = np.linspace(0, t, n); return np.c_[y[0] + y[1] * tt + u * tt ** 2 / 2, y[1] + u * tt]
# исходящие: из каждого из 7 узлов — в узлы соседних отрезков
nout = 0; best = None
for j, sv in enumerate(B.sn):
    y = c + np.r_[0, sv]; t, u = B.arcs(y, B.C[k], B.sn)
    for ii in range(len(k)):
        for jj in range(B.m):
            if np.isfinite(t[ii, jj]):
                a.plot(*arc(y, t[ii, jj], u[ii, jj]).T, color=cm(nrm(u[ii, jj])), lw=.7, alpha=.75 if j == B.m // 2 else .22); nout += 1
                J = t[ii, jj] + B.V[k[ii], jj]
                if j == B.m // 2 and (best is None or J < best[0]): best = (J, y, t[ii, jj], u[ii, jj], k[ii])
# входящие: из центров соседей — в узлы нашего отрезка
nin = 0
for kk in k:
    t, u = B.arcs(B.C[kk], c[None], B.sn)
    for jj in range(B.m):
        if np.isfinite(t[0, jj]): a.plot(*arc(B.C[kk], t[0, jj], u[0, jj]).T, color=cm(nrm(u[0, jj])), lw=.7, alpha=.5, ls=(0, (3, 2))); nin += 1
P = arc(best[1], best[2], best[3]); a.plot(*P.T, color=INK, lw=2.6)
a.plot([c[0], c[0]], [c[1] - r, c[1] + r], color=INK, lw=3.2); a.plot(np.full(B.m, c[0]), c[1] + B.sn, 'o', color='white', mec=INK, mew=1.4, ms=6, zorder=5)
a.annotate('спора: центр + отрезок ±0.1 по скорости,\n7 узлов; в каждом хранится цена V', (c[0], c[1] + r), (c[0] - .52, c[1] + .62), color=INK, fontsize=9.5, bbox=BX, arrowprops=dict(arrowstyle='-', color=MUT, lw=.8))
a.annotate('лучшая дуга из центра:\nu = %+.2f, t = %.2f с,\nt + V(приход) = %.2f с' % (best[3], best[2], best[0]), P[len(P) // 2], (c[0] + .42, c[1] - .62), color=INK, fontsize=9.5, bbox=BX, arrowprops=dict(arrowstyle='-', color=MUT, lw=.8))
a.text(c[0] - .56, c[1] - .7, 'пунктир — входящие дуги\n(из центров соседей в узлы этой споры): %d' % nin, color=INK, fontsize=9.5, bbox=BX)
a.text(c[0] + .42, c[1] + .58, 'сплошные — исходящие дуги из 7 узлов\nв узлы соседних отрезков: %d\n(яркие — из центрального узла)' % nout, color=INK, fontsize=9.5, bbox=BX)
a.set_xlim(c[0] - .6, c[0] + 1.0); a.set_ylim(c[1] - .78, c[1] + .78); a.set_xlabel('положение x', color=INK); a.set_ylabel('скорость v', color=INK)
a.set_title('Одна спора-«бабочка» двойного интегратора: дуги с постоянным управлением, t ≤ 0.5 с', color=INK, fontsize=11, loc='left')
cb = fig.colorbar(matplotlib.cm.ScalarMappable(nrm, cm), ax=a, fraction=.03, pad=.015); cb.set_label('управление u на дуге', color=INK); cb.outline.set_visible(False)
# правая: цена на отрезке
b = ax[1]; vv = c[1] + B.sn; Ts = tstar_box(np.full(B.m, c[0]), vv, BD.RHO); b.plot(B.V[i0], vv, '-o', color='#1f5fbf', lw=2, ms=7, mec='white', mew=1.2); b.plot(Ts, vv, '-', color=INK, lw=1.2, ls=(0, (4, 3)))
b.annotate('V в узлах споры\n(между узлами — линейно)', (B.V[i0][-1], vv[-1]), (B.V[i0].min() - .02, vv[-1] + .035), color=INK, fontsize=9.5, bbox=BX)
b.annotate('точное время T*\n(только для сравнения)', (Ts[0], vv[0]), (Ts.min() - .12, vv[0] - .06), color=INK, fontsize=9.5, bbox=BX)
for x_, v_ in zip(B.V[i0], vv): b.text(x_ + .012, v_ - .006, '%.2f' % x_, color=MUT, fontsize=8.5)
b.set_xlabel('время до цели, с', color=INK); b.set_ylabel('скорость v вдоль отрезка споры', color=INK); b.set_title('Цена на отрезке: V/T* = %.3f в центре' % (B.V[i0][B.m // 2] / Ts[B.m // 2]), color=INK, fontsize=11, loc='left')
b.set_xlim(Ts.min() - .15, B.V[i0].max() + .12); b.set_ylim(vv[0] - .09, vv[-1] + .07)
for a_ in ax:
    a_.tick_params(colors=MUT, labelsize=8); a_.grid(color='#eef0f3', lw=.6); a_.set_axisbelow(True)
    for s_ in a_.spines.values(): s_.set_color('#d0d5dd')
fig.tight_layout(); fig.savefig('di_one_spore.png'); print('spore', i0, np.round(c, 3), 'out', nout, 'in', nin, 'best', np.round(best[0], 3), 'V', np.round(B.V[i0], 3), 'T*', np.round(Ts, 3))
