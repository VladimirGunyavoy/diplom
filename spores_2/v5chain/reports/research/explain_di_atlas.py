"""Картинка для пользователя (hub-research-4): атлас DI v6 на решётке, V из графа, агент и коридор. Запуск из v5chain: python3 reports/research/explain_di_atlas.py"""
import sys; sys.path.insert(0, '../v6')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.atlas6.cell import flow
from src.atlas6.agent import T_star, make_cells
from src.atlas6.value import solve_V, V_interp
from src.atlas6.corridor import agent_controls, compress, optimize
B, O, INK, MUT = '#2a78d6', '#eb6834', '#222222', '#6b6b66'
fig, ax = plt.subplots(1, 3, figsize=(18, 6.2))
# 1: атлас — клетки обоих слоёв на решётке h = 0.5, r = τ = h/2
a = ax[0]; h = 0.5; r = tau = h / 2
for x0 in np.arange(-2, 2.01, h):
    for v0 in np.arange(-2, 2.01, h):
        for u, col in ((1, B), (-1, O)):
            F = np.array([v0, u], float); n = np.array([-F[1], F[0]]) / np.hypot(*F); c = np.array([x0, v0])
            t = np.linspace(-tau, tau, 20)
            L = [flow(c + s * n, u, t) for s in (-r, r)]
            poly = np.vstack([L[0][:, :], L[1][::-1, :]])
            a.fill(poly[:, 0], poly[:, 1], color=col, alpha=0.10, lw=0); a.plot(*np.vstack([poly, poly[:1]]).T, color=col, lw=0.5, alpha=0.8)
        a.plot(x0, v0, 'o', color=INK, ms=2.5)
vv = np.linspace(-2.6, 2.6, 100); a.plot(-vv * np.abs(vv) / 2, vv, color=INK, lw=1, ls='--', label='кривая переключения')
a.plot([], [], color=B, lw=2, label='клетки слоя a=+1'); a.plot([], [], color=O, lw=2, label='клетки слоя a=−1')
a.set_xlim(-2.6, 2.6); a.set_ylim(-2.6, 2.6); a.set_aspect('equal'); a.legend(loc='lower left', fontsize=9)
a.set_title('1. Атлас: у каждой споры (точка) две клетки,\nh = 0.5, r = τ = 0.25; клетки перекрываются', loc='left'); a.set_xlabel('x'); a.set_ylabel('v')
# 2–3: V из графа (h = 0.2, τ = 0.4, как в tests/test_atlas6_corridor.py), агент и коридор
h, tau = 0.2, 0.4; xs, V, it = solve_V(h, tau=tau); X, Vv = np.meshgrid(xs, xs, indexing='ij'); Ts = np.vectorize(T_star)(X, Vv)
a = ax[1]; m = (np.abs(X) <= 2.6) & (np.abs(Vv) <= 2.6)
cs = a.pcolormesh(X, Vv, V, cmap='Blues', shading='auto', vmin=0, vmax=7); fig.colorbar(cs, ax=a, label='V из графа (с)')
a.contour(X, Vv, V, levels=[1, 2, 3, 4, 5], colors='white', linewidths=0.6); a.plot(-vv * np.abs(vv) / 2, vv, color=INK, lw=1, ls='--')
cells = make_cells(h, V=lambda x, v: V_interp(xs, V, x, v))
for st in [(-2.0, 0.0), (1.5, 1.0)]:
    us, ok = agent_controls(*st, cells, 2 * h); p = np.array(st, float); P = [p]
    for u in us: p = flow(p, u, 0.01); P.append(p)
    P = np.array(P); a.plot(P[:, 0], P[:, 1], color=O, lw=1.6)
    cor, T, res = optimize(*st, compress(us)); p = np.array(st, float); Q = [p]
    for d, u in cor:
        for tt in np.linspace(0, d, 40)[1:]: Q.append(flow(p, u, tt))
        p = flow(p, u, d)
    Q = np.array(Q); a.plot(Q[:, 0], Q[:, 1], color=INK, lw=1.6, ls='--'); a.plot(*st, 'o', color=INK, ms=7)
    a.text(st[0] + 0.1, st[1] + 0.15, 'T*=%.2f\nагент %.2f\nкоридор %.2f' % (T_star(*st), 0.01 * len(us), T), fontsize=9, color=INK, bbox=dict(fc='white', ec='none', alpha=0.8))
a.plot([], [], color=O, lw=2, label='агент (по наклону V)'); a.plot([], [], color=INK, lw=2, ls='--', label='коридор после оптимизатора')
a.plot(0, 0, '*', color=INK, ms=12); a.set_xlim(-2.6, 2.6); a.set_ylim(-2.6, 2.6); a.set_aspect('equal'); a.legend(loc='lower left', fontsize=9)
a.set_title('2. V, посчитанная Беллманом по клеткам\n(h = 0.2, τ = 0.4), и две поездки', loc='left'); a.set_xlabel('x'); a.set_ylabel('v')
a = ax[2]; R = np.where(Ts > 0.3, V / np.maximum(Ts, 1e-9), np.nan)
cs = a.pcolormesh(X, Vv, R, cmap='RdBu_r', shading='auto', vmin=0.5, vmax=1.5); fig.colorbar(cs, ax=a, label='V / T*  (красное — V завышена)')
a.plot(-vv * np.abs(vv) / 2, vv, color=INK, lw=1, ls='--'); a.add_patch(plt.Circle((0, 0), 2 * h, fill=False, color=INK, lw=1))
a.text(0.45, -0.1, 'клетка цели\n(там T* точно)', fontsize=8, color=INK)
sel = m & (Ts > 0.3); a.set_xlim(-2.6, 2.6); a.set_ylim(-2.6, 2.6); a.set_aspect('equal'); a.set_xlabel('x'); a.set_ylabel('v')
a.set_title('3. Где V врёт: V / T*\nсреднее %.2f, min %.2f, max %.2f' % (np.nanmean(R[sel]), np.nanmin(R[sel]), np.nanmax(R[sel])), loc='left')
for s in ax:
    for sp in ('top', 'right'): s.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig('reports/research/figs/explain_di_atlas.png', dpi=105); print('итераций', it)
