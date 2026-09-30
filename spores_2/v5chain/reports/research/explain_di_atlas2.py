"""Картинка для пользователя (hub-research-4): атлас + и − отдельно; какой слой выбрал Беллман в каждой споре; знак ∂V/∂v у агента."""
import sys; sys.path.insert(0, '../v6')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.atlas6.cell import flow
from src.atlas6.agent import T_star, make_cells
from src.atlas6.value import solve_V, V_interp
B, O, INK = '#2a78d6', '#eb6834', '#222222'
fig, ax = plt.subplots(2, 2, figsize=(13, 13)); vv = np.linspace(-2.6, 2.6, 100); sw = lambda a: a.plot(-vv * np.abs(vv) / 2, vv, color=INK, lw=1.2, ls='--', label='кривая переключения (точная)')
h = 0.5; r = tau = h / 2
for a, u, col, nm in ((ax[0, 0], 1, B, '+1'), (ax[0, 1], -1, O, '−1')):
    for x0 in np.arange(-2, 2.01, h):
        for v0 in np.arange(-2, 2.01, h):
            F = np.array([v0, u], float); n = np.array([-F[1], F[0]]) / np.hypot(*F); c = np.array([x0, v0]); t = np.linspace(-tau, tau, 20)
            L = [flow(c + s * n, u, t) for s in (-r, r)]; poly = np.vstack([L[0], L[1][::-1]])
            a.fill(poly[:, 0], poly[:, 1], color=col, alpha=0.12, lw=0); a.plot(*np.vstack([poly, poly[:1]]).T, color=col, lw=0.7)
            e = flow(c, u, tau); a.annotate('', e, c, arrowprops=dict(arrowstyle='->', color=INK, lw=0.6)); a.plot(*c, 'o', color=INK, ms=3)
    sw(a); a.set_xlim(-2.6, 2.6); a.set_ylim(-2.6, 2.6); a.set_aspect('equal'); a.set_xlabel('x'); a.set_ylabel('v'); a.legend(loc='lower left', fontsize=9)
    a.set_title('Атлас слоя a = %s (h = 0.5, r = τ = 0.25)\nстрелка: спора → выход клетки через τ' % nm, loc='left')
h, tau = 0.2, 0.4; xs, V, _ = solve_V(h, tau=tau); X, Vv = np.meshgrid(xs, xs, indexing='ij'); P = np.stack([X, Vv], -1)
Q = [tau + np.vectorize(lambda x, v: V_interp(xs, V, x, v))(*np.moveaxis(flow(P, s, tau), -1, 0)) for s in (+1, -1)]
W = np.where(Q[0] <= Q[1], 1.0, -1.0); W[np.hypot(X, Vv) < 2 * h] = np.nan
cm = ListedColormap([O, B]); a = ax[1, 0]
a.pcolormesh(X, Vv, W, cmap=cm, shading='nearest', vmin=-1, vmax=1, alpha=0.8); sw(a)
a.set_title('Беллман: какой слой дал min в каждой споре\nсиний: τ+V(выход +1) меньше; оранжевый: −1 (h = 0.2, τ = 0.4)', loc='left')
cells = make_cells(h, V=lambda x, v: V_interp(xs, V, x, v)); C = np.array([c.cell.c for c in cells]); G = np.array([c.grad for c in cells])
S = np.full(X.shape, np.nan); idx = {(round(c[0], 6), round(c[1], 6)): k for k, c in enumerate(C)}
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        k = idx.get((round(X[i, j], 6), round(Vv[i, j], 6)))
        if k is not None and np.hypot(X[i, j], Vv[i, j]) >= 2 * h: S[i, j] = -np.sign(G[k, 1]) or 1.0
a = ax[1, 1]; a.pcolormesh(X, Vv, S, cmap=cm, shading='nearest', vmin=-1, vmax=1, alpha=0.8); sw(a)
a.set_title('Агент: a = −sign(∂V/∂v), наклон по 5 точкам клетки\nсиний: a = +1; оранжевый: a = −1', loc='left')
for a in ax[1]:
    a.add_patch(plt.Circle((0, 0), 2 * h, fill=False, color=INK)); a.set_xlim(-2.6, 2.6); a.set_ylim(-2.6, 2.6); a.set_aspect('equal'); a.set_xlabel('x'); a.set_ylabel('v'); a.legend(loc='lower left', fontsize=9)
plt.tight_layout(); plt.savefig('reports/research/figs/explain_di_atlas2.png', dpi=95)
m = np.hypot(X, Vv) < 2.6; ok = ~np.isnan(W) & ~np.isnan(S) & m
print('совпадение выбора Беллмана и агента: %.3f' % np.mean(W[ok] == S[ok]))
ex = np.where(X + Vv * np.abs(Vv) / 2 > 0, -1.0, 1.0); print('Беллман = точное правило: %.3f; агент = точное: %.3f' % (np.mean(W[ok] == ex[ok]), np.mean(S[ok] == ex[ok])))
