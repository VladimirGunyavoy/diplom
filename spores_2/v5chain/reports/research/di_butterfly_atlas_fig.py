import numpy as np, sys, time, pickle, os
sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import butterfly_di as BD
from v7_faces_di import tstar_box
S = '.'; t0 = time.time()
import importlib.util; sp = importlib.util.spec_from_file_location('bf_v7', '../../../v7/src/cells7/butterfly_di.py'); mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod); BF = mod.ButterflyFast
B = BF(N=1500, m=7).solve(); print('pairs', B.npairs, 'iters', B.n_it)
print('solve sec', round(time.time() - t0), 'BIG', round(float((B.V >= BD.BIG / 2).mean()), 3), flush=True)
INK, MUT = '#1f2328', '#6b7280'; BX = dict(boxstyle='round,pad=.25', fc='white', ec='#d0d5dd', lw=.6)
fig, ax = plt.subplots(1, 2, figsize=(13, 6.2), dpi=140); fig.patch.set_facecolor('white')
# левая: атлас — отрезки спор, цвет = V в центре
Vc = B.V[:, B.m // 2]; fin = Vc < BD.BIG / 2; segs = np.stack([np.c_[B.C[:, 0], B.C[:, 1] - B.r], np.c_[B.C[:, 0], B.C[:, 1] + B.r]], 1)
a = ax[0]; a.add_collection(LineCollection(segs[~fin], colors='#c9ced6', linewidths=1.2))
lc = LineCollection(segs[fin], array=Vc[fin], cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b', plt.cm.Blues_r(np.linspace(0, .72, 64))), linewidths=1.6, clim=(0, np.percentile(Vc[fin], 98))); a.add_collection(lc)
cb = fig.colorbar(lc, ax=a, fraction=.046, pad=.02); cb.set_label('V споры — время до цели, с', color=INK); cb.outline.set_visible(False)
v = np.linspace(-2.5, 2.5, 200); a.plot(-v * np.abs(v) / 2, v, color=INK, lw=1, ls=(0, (4, 3)))
a.annotate('кривая переключения\n(точное решение, для сравнения)', (-1.62, 1.8), (-2.4, -1.75), color=INK, fontsize=9, bbox=BX, arrowprops=dict(arrowstyle='-', color=MUT, lw=.8))
a.add_patch(plt.Rectangle((-.1, -.1), .2, .2, fill=False, ec='#b42318', lw=1.4)); a.annotate('цель ±0.1', (.1, .1), (.9, 1.9), color=INK, fontsize=9, bbox=BX, arrowprops=dict(arrowstyle='-', color=MUT, lw=.8))
a.set_title('Атлас: %d спор × 7 узлов (серые — без пути к цели)' % B.K, color=INK, fontsize=11, loc='left')
# правая: «бабочка» одной споры + пути агента
b = ax[1]; b.add_collection(LineCollection(segs, colors='#c9ced6', linewidths=1.)); b.plot(-v * np.abs(v) / 2, v, color=INK, lw=1, ls=(0, (4, 3)))
i0 = int(np.argmin(np.linalg.norm(B.C - np.array([1.0, .45]), axis=1))); y = B.C[i0]; k = np.array(B.tree.query_ball_point(y, B.tau * (abs(y[1]) + 1) + B.r + .05)); k = k[k != i0]
t, u = B.arcs(y, B.C[k], B.sn); n = 0
for ii in range(len(k)):
    for jj in range(B.m):
        if np.isfinite(t[ii, jj]):
            tt = np.linspace(0, t[ii, jj], 12); b.plot(y[0] + y[1] * tt + u[ii, jj] * tt ** 2 / 2, y[1] + u[ii, jj] * tt, color='#2a78d6', lw=.5, alpha=.45); n += 1
b.plot([y[0], y[0]], [y[1] - B.r, y[1] + B.r], color='#0b3d91', lw=3); b.annotate('одна спора: из её центра\n%d дуг с постоянным u\nв узлы соседних отрезков' % n, y, (.75, -2.2), color=INK, fontsize=9, bbox=BX, arrowprops=dict(arrowstyle='-', color=MUT, lw=.8))
for q in ([-1.3, -.6], [1.4, 1.0], [-.4, 1.3]):
    yy = np.array(q, float); P = [yy.copy()]; tt = 0.
    while tt < 20 and not (abs(yy[0]) <= .1 and abs(yy[1]) <= .1):
        J, bb = B.best(yy)
        if bb is None or J >= BD.BIG / 2: break
        h = min(.06, bb[2]); uu = bb[3]; yy = np.r_[yy[0] + yy[1] * h + uu * h * h / 2, yy[1] + uu * h]; tt += h; P.append(yy.copy())
    P = np.array(P); Ts = float(tstar_box(np.array([q[0]]), np.array([q[1]]), .1)[0]); b.plot(P[:, 0], P[:, 1], color='#d9480f', lw=2); b.plot(*q, 'o', color='#d9480f', ms=6, mec='white', mew=1.2)
    b.annotate('T %.2f с (точное %.2f)' % (tt, Ts), q, (q[0] + .08, q[1] + .14 if q[1] > 0 else q[1] - .22), color=INK, fontsize=9, bbox=BX)
b.add_patch(plt.Rectangle((-.1, -.1), .2, .2, fill=False, ec='#b42318', lw=1.4)); b.set_title('«Бабочка» одной споры и три пути агента (оранжевые)', color=INK, fontsize=11, loc='left')
for a_ in ax:
    a_.set_xlim(-2.5, 2.5); a_.set_ylim(-2.5, 2.5); a_.set_xlabel('положение x', color=INK); a_.set_ylabel('скорость v', color=INK); a_.tick_params(colors=MUT, labelsize=8)
    for sp in a_.spines.values(): sp.set_color('#d0d5dd')
    a_.grid(color='#eef0f3', lw=.6); a_.set_axisbelow(True)
fig.tight_layout(); fig.savefig(S + '/di_butterfly_atlas.png'); print('saved', round(time.time() - t0))
