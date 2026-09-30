"""Картинка для пользователя (hub-research-4): атлас DI для многих запросов — одно общее обратное дерево (NB) + прямые деревья (NF на запрос)."""
import sys; sys.path.insert(0, '../v6')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.atlas6.adaptive_nd import SysN, _tree, build_back, replay_value
from src.atlas6.agent import T_star
def T_win(x0, R):
    """точное время до окна |(x,v)|<R: перебор bang-bang с ≤1 переключением (для DI оптимум к выпуклой цели такой), шаг .004"""
    t = np.arange(0, 6, 0.004); best = np.inf
    for u in (1.0, -1.0):
        x1 = x0[0] + x0[1] * t + u * t * t / 2; v1 = x0[1] + u * t
        X = x1[:, None] + v1[:, None] * t[None] - u * t[None] ** 2 / 2; Vv = v1[:, None] - u * t[None]
        ok = np.hypot(X, Vv) < R; tot = np.where(ok, t[:, None] + t[None], np.inf); best = min(best, tot.min())
    return best
B, O, INK, GR = '#2a78d6', '#eb6834', '#222222', '#1baf7a'
U = (1.0, -1.0)
def fl(P, s, t):
    P = np.asarray(P, float); u = U[s]; x, v = P[..., 0], P[..., 1]
    return np.stack([x + v * t + u * t * t / 2, v + u * t], -1)
Rg = 0.2; S = SysN(fl, 2, (1.0, 1.0), lambda P: np.hypot(np.asarray(P)[..., 0], np.asarray(P)[..., 1]) < Rg, [(0.0, 0.0)])
tau, rho = 0.25, 0.15; NB, NF = int(sys.argv[1]) if len(sys.argv) > 1 else 300, int(sys.argv[2]) if len(sys.argv) > 2 else 40
rng = np.random.default_rng(1); X0 = rng.uniform(-2, 2, (12, 2))
back = build_back(S, tau, NB, rho); Pb, Gb, parb, layb = back[:4]
fig, ax = plt.subplots(1, 2, figsize=(16, 7.6))
def arcs(a, P, par, lay, sgn, col=None, alpha=0.6, lw=0.8):
    for i in range(len(P)):
        if par[i] < 0: continue
        Q = fl(P[par[i]], lay[i], np.linspace(0, sgn * tau, 10)); a.plot(Q[:, 0], Q[:, 1], color=col or (B if lay[i] == 0 else O), lw=lw, alpha=alpha)
a = ax[0]; arcs(a, Pb, parb, layb, -1); a.plot(*np.array(Pb).T, 's', color=GR, ms=2.5)
a.set_title('Общее обратное дерево от цели: NB = %d спор\n(строится один раз для всех запросов)' % len(Pb), loc='left')
a = ax[1]; arcs(a, Pb, parb, layb, -1, col='#bbbbbb', alpha=0.5)
res = []
for x0 in X0:
    Pf, Gf, parf, layf = _tree(S, tau, [x0], NF, rho, 10.0, True); arcs(a, Pf, parf, layf, 1, alpha=0.5)
    V, nb, nf, path = replay_value(S, tau, x0, NB, NF, rho, rho, back=back); r = V / T_win(x0, Rg); res.append(r)
    if path is not None:
        p = np.array(x0, float); Q = [p]
        for s, d in path:
            for tt in np.linspace(0, d, 8)[1:]: Q.append(fl(p, s, tt))
            p = fl(p, s, d)
        Q = np.array(Q); a.plot(Q[:, 0], Q[:, 1], color=INK, lw=1.8, ls='--')
    a.plot(*x0, 'o', color=INK, ms=6); a.text(x0[0] + 0.05, x0[1] + 0.08, '%.2f' % r, fontsize=8, color=INK, bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.5))
res = np.array(res); ok = np.isfinite(res)
a.set_title('12 запросов: у каждого своё прямое дерево NF = %d спор (цветное),\nобратное — серым; число у старта = V / T*(до окна); найдено %d/12, mean %.3f' % (NF, ok.sum(), res[ok].mean()), loc='left')
for a in ax:
    vv = np.linspace(-3, 3, 100); a.plot(-vv * np.abs(vv) / 2, vv, color='#999999', lw=1, ls=':')
    a.plot(0, 0, '*', color=GR, ms=15); a.add_patch(plt.Circle((0, 0), Rg, fill=False, color=GR)); a.set_xlim(-4, 4); a.set_ylim(-3, 3); a.set_aspect('equal'); a.set_xlabel('x'); a.set_ylabel('v')
    for sp in ('top', 'right'): a.spines[sp].set_visible(False)
ax[0].plot([], [], color=B, label='дуга слоя +1'); ax[0].plot([], [], color=O, label='дуга слоя −1'); ax[0].plot([], [], color=INK, ls='--', label='найденный путь'); ax[0].legend(loc='lower left', fontsize=9)
plt.tight_layout(); plt.savefig('reports/research/figs/explain_di_multi.png', dpi=100); print(np.round(res, 3), 'всего спор', len(Pb) + 12 * NF)
