"""Картинка для пользователя (hub-research-4): клетки адаптивного атласа DI по слоям (NB300 общий + 12 запросов × NF40, как explain_di_multi.py).
В коде (adaptive_nd) клетка явно не хранится: спора + дуга слоя за τ + ядро (квадрат ρ, дедупликация). Здесь клетка нарисована по source_doc §3:
отрезок ±r поперёк потока через спору, протянутый по потоку на τ (прямое дерево — вперёд, обратное — назад), r = ρ/2."""
import sys; sys.path.insert(0, '../v6')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.atlas6.adaptive_nd import SysN, _tree, build_back
B, O, INK, GR = '#2a78d6', '#eb6834', '#222222', '#1baf7a'
U = (1.0, -1.0)
def fl(P, s, t):
    P = np.asarray(P, float); u = U[s]; x, v = P[..., 0], P[..., 1]
    return np.stack([x + v * t + u * t * t / 2, v + u * t], -1)
Rg = 0.2; S = SysN(fl, 2, (1.0, 1.0), lambda P: np.hypot(np.asarray(P)[..., 0], np.asarray(P)[..., 1]) < Rg, [(0.0, 0.0)])
tau, rho = 0.25, 0.15; r = rho / 2
rng = np.random.default_rng(1); X0 = rng.uniform(-2, 2, (12, 2))
Pb, Gb, parb, layb = build_back(S, tau, 300, rho)[:4]
cells = []                                  # (спора, слой, знак времени, дерево)
for i in range(len(Pb)):
    if parb[i] >= 0: cells.append((Pb[i], layb[i], -1, 'b'))
for x0 in X0:
    Pf, Gf, parf, layf = _tree(S, tau, [x0], 40, rho, 10.0, True)
    for i in range(len(Pf)):
        for s in range(2): cells.append((Pf[i], s, +1, 'f')) if parf[i] < 0 else None
        if parf[i] >= 0: cells.append((Pf[parf[i]], layf[i], +1, 'f'))
def poly(c, s, sg):
    F = np.array([c[1], U[s]]); n = np.array([-F[1], F[0]]) / np.hypot(*F); t = np.linspace(0, sg * tau, 15)
    L = [fl(c + q * n, s, t) for q in (-r, r)]; return np.vstack([L[0], L[1][::-1], L[0][:1]])
fig, ax = plt.subplots(2, 2, figsize=(15, 13))
for a, s, col in ((ax[0, 0], 0, B), (ax[0, 1], 1, O)):
    nb = nf = 0
    for c, ls, sg, tr in cells:
        if ls != s: continue
        P = poly(np.asarray(c), ls, sg); a.fill(P[:, 0], P[:, 1], color=col, alpha=0.18 if tr == 'f' else 0.30, lw=0)
        a.plot(P[:, 0], P[:, 1], color=col, lw=0.5, ls='-' if tr == 'f' else '--'); nb += tr == 'b'; nf += tr == 'f'
    a.set_title('Слой a = %s: %d клеток (обратное дерево, пунктир — %d; прямые деревья — %d)' % ('+1' if s == 0 else '−1', nb + nf, nb, nf), loc='left', fontsize=11)
    a.set_xlim(-4, 4); a.set_ylim(-3, 3)
a = ax[1, 0]; z = (-1.4, -0.2, 0.4, 1.6)
for c, ls, sg, tr in cells:
    c = np.asarray(c)
    if not (z[0] - 0.5 < c[0] < z[1] + 0.5 and z[2] - 0.5 < c[1] < z[3] + 0.5): continue
    col = B if ls == 0 else O; P = poly(c, ls, sg); a.fill(P[:, 0], P[:, 1], color=col, alpha=0.2, lw=0); a.plot(P[:, 0], P[:, 1], color=col, lw=0.9, ls='-' if tr == 'f' else '--')
    a.plot(*c, 'o' if tr == 'f' else 's', color=INK, ms=3)
a.set_xlim(z[0], z[1]); a.set_ylim(z[2], z[3]); a.set_title('Увеличено (оба слоя): клетки — короткие изогнутые полоски\nширина 2r = ρ = %.2f, длина — путь за τ = %.2f; точки — споры' % (rho, tau), loc='left', fontsize=11)
a = ax[1, 1]
for c, ls, sg, tr in cells:
    c = np.asarray(c)
    if not (z[0] - 0.5 < c[0] < z[1] + 0.5 and z[2] - 0.5 < c[1] < z[3] + 0.5): continue
    col = B if ls == 0 else O
    k = np.floor(c / rho) * rho; a.add_patch(Rectangle(k, rho, rho, fc=col, ec=col, alpha=0.25, lw=0.8))
    a.plot(*c, 'o' if tr == 'f' else 's', color=INK, ms=3)
a.set_xlim(z[0], z[1]); a.set_ylim(z[2], z[3]); a.set_title('Тот же участок — ЯДРА (как в коде): квадрат ρ × ρ на слой;\nвторая спора того же слоя в занятое ядро не ставится', loc='left', fontsize=11)
for a in ax.flat:
    vv = np.linspace(-3, 3, 100); a.plot(-vv * np.abs(vv) / 2, vv, color='#999999', lw=1, ls=':')
    a.plot(0, 0, '*', color=GR, ms=14); a.set_aspect('equal'); a.set_xlabel('x'); a.set_ylabel('v')
    for sp in ('top', 'right'): a.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig('reports/research/figs/explain_di_cells.png', dpi=95)
