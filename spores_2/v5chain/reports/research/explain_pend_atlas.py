"""research hub-v5chain-research-9: картинка атласа бабочек маятника для пользователя. Слева — весь атлас (споры-отрезки, цвет = время до цели V)
и путь агента дугами; справа — увеличение: споры с узлами и все дуги из одного узла, лучшая выделена."""
import os; os.environ.setdefault('TL', '2')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from butterfly_pend import ButterflyPend, BIG, RHO, flow
from pend_cost_grid import step, wrap, UM
B = ButterflyPend(N=2000, tau=.3, r=.1).build().solve()
def path(x, w, u, t, n=40):
    ts = np.linspace(0, t, n); P = [(x, w)]
    for a, b in zip(ts[:-1], ts[1:]): x, w = step(x, w, u, b - a); P.append((x, w))
    return np.array(P)
# путь агента: держит u всю дугу, потом выбирает следующую
y = np.array([-2.2, .6]); arcs = []; tt = 0.
for _ in range(60):
    if abs(wrap(y[0])) <= RHO and abs(y[1]) <= RHO: break
    J, ta, u = B.best(y[None]); P = path(y[0], y[1], u[0], ta[0]); arcs.append((P, u[0])); y = P[-1]; tt += ta[0]
ink, mute, acc, grid = '#1f2328', '#6e7781', '#d4600a', '#d0d7de'
cm = plt.get_cmap('Blues_r'); Vmax = 8.
fig, ax = plt.subplots(1, 2, figsize=(15, 6.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
a = ax[0]; Vm = np.where(B.V >= BIG / 2, np.nan, B.V).mean(1)
seg = np.stack([B.C - .1 * B.n, B.C + .1 * B.n], 1); col = cm(np.clip(np.nan_to_num(Vm, nan=Vmax) / Vmax, 0, 1) * .85)
a.add_collection(LineCollection(seg, colors=col, linewidths=1.6))
for P, u in arcs:
    Pw = P.copy(); Pw[:, 0] = wrap(Pw[:, 0]); br = np.flatnonzero(np.abs(np.diff(Pw[:, 0])) > 3) + 1
    for q in np.split(Pw, br): a.plot(q[:, 0], q[:, 1], color=acc, lw=2.2, solid_capstyle='round')
    a.plot(wrap(P[0, 0]), P[0, 1], 'o', ms=4, color=acc)
a.plot(-2.2, .6, 'o', ms=9, color=acc, mec='white', mew=1.5); a.annotate('старт', (-2.2, .6), (-2.75, 1.05), color=ink, fontsize=11)
a.add_patch(plt.Rectangle((-RHO, -RHO), 2 * RHO, 2 * RHO, fill=False, ec=ink, lw=2)); a.annotate('цель (верх)', (0, .1), (.25, .55), color=ink, fontsize=11, arrowprops=dict(arrowstyle='-', color=mute))
a.set_xlim(-np.pi, np.pi); a.set_ylim(-3.5, 3.5); a.set_xlabel('угол φ (0 — верх, ±π — низ)', color=ink); a.set_ylabel('скорость ω', color=ink)
a.set_title('Атлас: 2009 спор-отрезков; цвет — время до цели V\nоранжевый — путь агента: %d дуг, %.1f с' % (len(arcs), tt), color=ink, fontsize=12, loc='left')
sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(cm(np.linspace(0, .85, 256))), norm=plt.Normalize(0, Vmax)); cb = fig.colorbar(sm, ax=a, fraction=.035, pad=.01); cb.set_label('V, с (светлее — дольше)', color=ink)
# увеличение: узел на пути агента
P0 = arcs[len(arcs) // 2][0][0]; d = np.hypot(wrap(B.P[..., 0] - P0[0]), B.P[..., 1] - P0[1]); ki, ji = np.unravel_index(np.argmin(d), d.shape); y0 = B.P[ki, ji]
b = ax[1]; L = .32
near = np.flatnonzero((np.abs(wrap(B.C[:, 0] - y0[0])) < L + .1) & (np.abs(B.C[:, 1] - y0[1]) < L + .1))
iy, k, q, t, u = B.pairs(y0[None], own=np.array([ki])); iy, k, q = iy.astype(int), k.astype(int), q.astype(int)
j0, aw = B.qj0[q], B.qa[q]; val = t + (1 - aw) * B.V[k, j0] + aw * B.V[k, j0 + 1]; bi = int(np.argmin(val))
tgt = set(k.tolist())
for i in range(len(t)):
    P = path(y0[0], y0[1], u[i], t[i]); b.plot(P[:, 0], P[:, 1], color='#57606a', lw=1.6, ls=(0, (4, 2)), zorder=2)
P = path(y0[0], y0[1], u[bi], t[bi]); b.plot(P[:, 0], P[:, 1], color=acc, lw=2.6)
for c in near:
    s0, s1 = B.C[c] - .1 * B.n[c], B.C[c] + .1 * B.n[c]; xs = np.array([s0[0], s1[0]]); xs = y0[0] + wrap(xs - y0[0])
    hot = c in tgt or c == ki; b.plot(xs, [s0[1], s1[1]], color=ink if hot else grid, lw=2 if hot else 1.2, zorder=1)
    if not hot: continue
    Px = y0[0] + wrap(B.P[c, :, 0] - y0[0]); Vc = np.where(B.V[c] >= BIG / 2, Vmax, B.V[c]); b.scatter(Px, B.P[c, :, 1], s=22, c=cm(np.clip(Vc / Vmax, 0, 1) * .85), edgecolors=ink, linewidths=.4, zorder=3)
b.plot(*y0, 'o', ms=11, color=acc, mec='white', mew=1.5, zorder=4)
b.annotate('узел, из которого\nсчитаем V', y0, (y0[0] + .04, y0[1] - .09), color=ink, fontsize=10, arrowprops=dict(arrowstyle='->', color=mute))
pe = P[-1]; b.annotate('лучшая дуга: u = %+.2f, t = %.2f с\nV(узла) = t + V(точки прихода) = %.2f с' % (u[bi], t[bi], val[bi]), pe, (pe[0] + .04, pe[1] + .05), color=ink, fontsize=10, arrowprops=dict(arrowstyle='->', color=mute))
yc = (y0 + pe) / 2; b.set_aspect('equal', adjustable='box'); b.set_xlim(yc[0] - .2, yc[0] + .3); b.set_ylim(yc[1] - .25, yc[1] + .25); b.set_xlabel('угол φ', color=ink); b.set_ylabel('скорость ω', color=ink)
b.set_title('Увеличение: спора = отрезок с 7 узлами (цвет — V)\nпунктир — все %d дуг из узла (u постоянно, |u| ≤ 0.3);\nчёрные — споры, куда они ведут, бледные — остальные' % len(t), color=ink, fontsize=12, loc='left')
for a_ in ax:
    a_.grid(color=grid, lw=.6); a_.set_facecolor('white')
    for sp in a_.spines.values(): sp.set_color(grid)
    a_.tick_params(colors=mute)
plt.tight_layout(); os.makedirs('figs', exist_ok=True); plt.savefig('figs/explain_pend_atlas.png', dpi=110, facecolor='white'); print('ok', len(arcs), round(tt, 2), len(t))
