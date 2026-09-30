"""Картинка для пользователя (hub-research-4): адаптивный атлас DI = два дерева цепочек (от старта и от цели) + стык проигрышем (adaptive_nd)."""
import sys; sys.path.insert(0, '../v6')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.atlas6.adaptive_nd import SysN, _tree, build_back, replay_value
from src.atlas6.agent import T_star
B, O, INK, GR = '#2a78d6', '#eb6834', '#222222', '#1baf7a'
U = (1.0, -1.0)
def fl(P, s, t):
    P = np.asarray(P, float); u = U[s]; x, v = P[..., 0], P[..., 1]
    return np.stack([x + v * t + u * t * t / 2, v + u * t], -1)
Rg = 0.2; ing = lambda P: np.hypot(np.asarray(P)[..., 0], np.asarray(P)[..., 1]) < Rg
S = SysN(fl, 2, (1.0, 1.0), ing, [(0.0, 0.0)])
tau, rho = 0.25, 0.15; x0 = np.array([-2.0, 0.5])
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
for a, (NB, NF) in zip(ax, ((20, 20), (60, 60))):
    back = build_back(S, tau, NB, rho); Pb, Gb, parb, layb = back[:4]
    Pf, Gf, parf, layf = _tree(S, tau, [x0], NF, rho, 10.0, True)
    def arcs(P, par, lay, sgn, alpha):
        for i in range(len(P)):
            if par[i] < 0: continue
            t = np.linspace(0, sgn * tau, 12); Q = fl(P[par[i]], lay[i], t)
            a.plot(Q[:, 0], Q[:, 1], color=B if lay[i] == 0 else O, lw=1.1, alpha=alpha)
    arcs(Pf, parf, layf, 1, 0.9); arcs(Pb, parb, layb, -1, 0.9)
    a.plot(*np.array(Pf).T, 'o', color=INK, ms=3); a.plot(*np.array(Pb).T, 's', color=GR, ms=3.5)
    V, nb, nf, path = replay_value(S, tau, x0, NB, NF, rho, rho, back=back)
    p = x0.copy(); Q = [p]
    for s, d in path:
        for tt in np.linspace(0, d, 8)[1:]: Q.append(fl(p, s, tt))
        p = fl(p, s, d)
    Q = np.array(Q); a.plot(Q[:, 0], Q[:, 1], color=INK, lw=2.4, ls='--')
    vv = np.linspace(-2, 2, 100); a.plot(-vv * np.abs(vv) / 2, vv, color='#999999', lw=1, ls=':')
    a.plot(*x0, 'o', color=INK, ms=9); a.plot(0, 0, '*', color=GR, ms=16); a.add_patch(plt.Circle((0, 0), Rg, fill=False, color=GR))
    a.set_title('Споры: %d от старта (чёрные точки) + %d от цели (зелёные)\nV (реальная траектория, пунктир) = %.2f, T* = %.2f, V/T* = %.3f' % (nf, nb, V, T_star(*x0), V / T_star(*x0)), loc='left')
    a.set_xlim(-3.2, 1.2); a.set_ylim(-1.6, 2.2); a.set_aspect('equal'); a.set_xlabel('x'); a.set_ylabel('v')
ax[0].plot([], [], color=B, label='дуга слоя +1'); ax[0].plot([], [], color=O, label='дуга слоя −1'); ax[0].plot([], [], color=INK, ls='--', lw=2, label='найденный путь'); ax[0].plot([], [], color='#999999', ls=':', label='кривая переключения')
ax[0].legend(loc='lower left', fontsize=9)
for s in ax:
    for sp in ('top', 'right'): s.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig('reports/research/figs/explain_di_tree.png', dpi=100)
