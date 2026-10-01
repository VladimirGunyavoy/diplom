"""Картинка для пользователя (hub-research-4): атлас дифдрайва (адаптивный, v6 adaptive_nd): обратное дерево цели NB=1200 в (x, y, θ),
и коридор для двух запросов: боковой сдвиг (0, 0.3, 0) и (1.5, 1, π/2). Постановка как tests/check_corridor_nd_dd.py."""
import sys; sys.path.insert(0, '../v6')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.atlas6.adaptive_nd import SysN, build_back, _tree
from src.atlas6.corridor_nd import corridor_query
from src.atlas6.dd3 import flow
from src.atlas6.dd_atlas import LAYERS
B, O, INK, GR = '#2a78d6', '#eb6834', '#222222', '#1baf7a'
NB, NF = 1200, 50; tau = 0.25; R = .25; Rth = .26; rho = 0.08
fl = lambda P, s, t: flow(P, LAYERS[s], t)
dth = lambda th: (th + np.pi) % (2 * np.pi) - np.pi
ing = lambda P: (np.hypot(P[..., 0], P[..., 1]) < R) & (np.abs(dth(P[..., 2])) < Rth)
g = lambda x: np.array([R ** 2 - x[0] ** 2 - x[1] ** 2, Rth ** 2 - dth(x[2]) ** 2])
miss = lambda X: np.maximum(np.hypot(X[..., 0], X[..., 1]) - R, 0) + np.maximum(np.abs(dth(X[..., 2])) - Rth, 0)
S = SysN(fl, 4, (1.0, 1.0, np.pi), ing, [(0.0, 0.0, 0.0)], per=(0, 0, 2 * np.pi), ok=lambda p: abs(p[0]) <= 3 and abs(p[1]) <= 3)
print('LAYERS', LAYERS)
back = build_back(S, tau, NB, rho); Pb, Gb, parb, layb = back[:4]; Pb = np.array(Pb)
kind = lambda s: 'G' if abs(LAYERS[s][1]) < 1e-9 else 'T'
fig = plt.figure(figsize=(19, 6.4))
a = fig.add_subplot(1, 3, 1)
for i in range(len(Pb)):
    col = B if (parb[i] < 0 or kind(layb[i]) == 'G') else O
    x, y, th = Pb[i]; a.plot([x, x + 0.07 * np.cos(th)], [y, y + 0.07 * np.sin(th)], color=col, lw=0.8); a.plot(x, y, '.', color=col, ms=2)
a.add_patch(plt.Circle((0, 0), R, fill=False, color=GR)); a.set_aspect('equal'); a.set_xlim(-2.2, 2.2); a.set_ylim(-2.2, 2.2)
a.set_title('1. Обратное дерево цели (NB = %d) в проекции (x, y)\nчёрточка — курс θ' % len(Pb), loc='left')
a.plot([], [], color=B, label='ребро G (прямая)'); a.plot([], [], color=O, label='ребро T (поворот на месте)'); a.legend(loc='lower left', fontsize=9)
a = fig.add_subplot(1, 3, 2, projection='3d')
cols = [B if (parb[i] < 0 or kind(layb[i]) == 'G') else O for i in range(len(Pb))]
a.scatter(Pb[:, 0], Pb[:, 1], Pb[:, 2], c=cols, s=3, depthshade=False); a.set_xlabel('x'); a.set_ylabel('y'); a.set_zlabel('θ')
a.set_title('2. То же в 3D (x, y, θ): споры заполняют\n«трубки» вокруг цели, θ периодичен', loc='left')
a = fig.add_subplot(1, 3, 3)
def path_pts(x0, seq, dts):
    P = [np.array(x0, float)]
    for s, d in zip(seq, dts):
        for k in range(1, 21): P.append(fl(P[-1], s, d / 20))
    return np.array(P)
for x0, col in (((0.0, 1.0, 0.0), B), ((1.5, 1.0, np.pi / 2), O)):
    b = corridor_query(S, fl, np.array(x0), back, tau, NF, rho, g, miss)
    if b is None: print('нет коридора', x0); continue
    P = path_pts(x0, b[1], b[2]); a.plot(P[:, 0], P[:, 1], color=col, lw=2.2)
    for k in range(0, len(P), 10): a.plot([P[k, 0], P[k, 0] + 0.06 * np.cos(P[k, 2])], [P[k, 1], P[k, 1] + 0.06 * np.sin(P[k, 2])], color=INK, lw=0.8)
    segs = ' '.join(('%s%.2f' % (kind(s) + ('+' if (LAYERS[s][0] + LAYERS[s][1]) > 0 else '−'), d)) for s, d in zip(b[1], b[2]))
    a.text(x0[0] - (0.45 if x0[0] < 1 else -0.05), x0[1] + (0.12 if x0[0] < 1 else -0.35), 'старт %s\nT = %.2f\n%s' % (tuple(round(v, 2) for v in x0), b[0], segs.replace(' ', '\n')), fontsize=7.5, color=INK, bbox=dict(fc='white', ec='none', alpha=0.85))
    print(x0, b[0], segs)
a.add_patch(plt.Circle((0, 0), R, fill=False, color=GR)); a.plot(0, 0, '*', color=GR, ms=12)
a.set_aspect('equal'); a.set_xlim(-0.6, 2.2); a.set_ylim(-0.5, 2.0); a.set_title('3. Коридор (дерево → топология → оптимизатор)\nчёрточки — курс робота вдоль пути', loc='left')
plt.tight_layout(); plt.savefig('reports/research/figs/explain_dd_atlas.png', dpi=100)
