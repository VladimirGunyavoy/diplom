"""Картинка для объяснения пользователю (hub-research-4): T*, клетки и интерполяция V, накопление ошибки. → reports/research/figs/explain_di_v.png"""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
B, O, INK, MUT, GRID = '#2a78d6', '#eb6834', '#222222', '#6b6b66', '#d8d7d0'
def T_star(x, v):
    s = x + v * abs(v) / 2
    if abs(s) < 1e-12: return abs(v)
    sg = 1.0 if s > 0 else -1.0
    return sg * v + 2 * np.sqrt(max(v * v / 2 + sg * x, 0.0))
fl = lambda x, v, u, t: (x + v * t + u * t * t / 2, v + u * t)
fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
plt.rcParams.update({'font.size': 11})
# 1: карта T*
a = ax[0]; xs = np.linspace(-3, 3, 301); X, V = np.meshgrid(xs, xs); T = np.vectorize(T_star)(X, V)
cs = a.pcolormesh(X, V, T, cmap='Blues', shading='auto', vmin=0, vmax=7); a.contour(X, V, T, levels=[1, 2, 3, 4, 5, 6], colors='white', linewidths=0.6); fig.colorbar(cs, ax=a, label='T*  (секунд до цели)')
vv = np.linspace(-3, 3, 200); a.plot(-vv * np.abs(vv) / 2, vv, color=O, lw=2, label='кривая переключения x = −v|v|/2')
t1 = np.sqrt(2.0); tt = np.linspace(0, t1, 50); x1, v1 = fl(-2, 0, 1, tt); a.plot(x1, v1, color=INK, lw=2)
x2, v2 = fl(x1[-1], v1[-1], -1, np.linspace(0, v1[-1], 50)); a.plot(x2, v2, color=INK, lw=2, ls='--')
a.plot(-2, 0, 'o', color=INK, ms=8); a.plot(0, 0, '*', color=INK, ms=14)
a.annotate('старт (−2, 0)\nгаз a=+1', (-2, 0), (-2.9, -1.2), color=INK, arrowprops=dict(arrowstyle='->', color=MUT))
a.annotate('переключение,\nдальше тормоз a=−1', (x1[-1], v1[-1]), (0.3, 2.2), color=INK, arrowprops=dict(arrowstyle='->', color=MUT))
a.annotate('цель (0,0)', (0, 0), (0.6, -0.9), color=INK, arrowprops=dict(arrowstyle='->', color=MUT))
a.set_xlim(-3, 3); a.set_title('1. T*(x, v) — точное минимальное время\nиз (−2, 0): T* = 2.83 с', loc='left'); a.set_xlabel('x (положение)'); a.set_ylabel('v (скорость)'); a.legend(loc='lower left', fontsize=9)
# 2: решётка спор, две клетки, выход между спорами
a = ax[1]; h = 0.5; g = np.arange(-0.5, 2.01, h)
for x in g:
    for v in np.arange(-0.5, 2.51, h): a.plot(x, v, 'o', color=MUT, ms=4)
c = (0.0, 1.0); tau = 0.6; r = 0.12
for u, col, name in ((+1, B, 'клетка слоя a=+1'), (-1, O, 'клетка слоя a=−1')):
    F = np.array([c[1], u]); n = np.array([-F[1], F[0]]) / np.hypot(*F)
    for s in (-r, r):
        p0 = np.array(c) + s * n; t = np.linspace(-tau, tau, 60); w = fl(p0[0], p0[1], u, t); a.plot(w[0], w[1], color=col, lw=1.5)
    for t in (-tau, tau):
        ends = [fl(*(np.array(c) + s * n), u, t) for s in (-r, r)]; a.plot([e[0] for e in ends], [e[1] for e in ends], color=col, lw=1.5)
    tt = np.linspace(0, tau, 40); w = fl(c[0], c[1], u, tt); a.plot(w[0], w[1], color=col, lw=2, ls=':')
    ex = fl(c[0], c[1], u, tau); a.plot(*ex, 's', color=col, ms=9, label=name)
    i0, j0 = np.floor(ex[0] / h) * h, np.floor(ex[1] / h) * h
    for dx in (0, h):
        for dv in (0, h): a.plot([ex[0], i0 + dx], [ex[1], j0 + dv], color=col, lw=0.8, ls='--')
a.plot(*c, 'o', color=INK, ms=10); a.annotate('спора', c, (-0.45, 1.25), color=INK, arrowprops=dict(arrowstyle='->', color=MUT))
a.annotate('выход через τ:\nпопал МЕЖДУ спорами →\nV там = смесь 4 соседей\n(интерполяция, с ошибкой)', fl(0, 1, 1, tau), (0.95, 0.35), color=INK, fontsize=9, bbox=dict(fc='white', ec='none'), arrowprops=dict(arrowstyle='->', color=MUT))
a.annotate('', (0, 2.25), (0.5, 2.25), arrowprops=dict(arrowstyle='<->', color=INK)); a.text(0.25, 2.33, 'h', ha='center', color=INK)
a.set_title('2. V(спора) = min по слоям [ τ + V(выход) ]\nсеро — споры решётки шага h = 0.5', loc='left'); a.set_xlabel('x'); a.set_ylabel('v'); a.set_aspect('equal'); a.legend(loc='lower left', fontsize=9)
# 3: накопление ошибки (reports/research/v6_value_tau.py)
a = ax[2]; H = [0.4, 0.2, 0.1]
a.axhline(0, color=GRID, lw=1); a.plot(H, [-0.81, -1.11, -1.34], 'o-', color=O, lw=2, ms=8, label='τ = h/2 (клетка короче)')
a.plot(H, [0.15, -0.04, -0.08], 'o-', color=B, lw=2, ms=8, label='τ = √h (клетка длиннее)')
for x, y in zip(H, [-0.81, -1.11, -1.34]): a.text(x, y - 0.12, '%.2f' % y, ha='center', color=INK, fontsize=9)
for x, y in zip(H, [0.15, -0.04, -0.08]): a.text(x, y + 0.08, '%+.2f' % y, ha='center', color=INK, fontsize=9)
a.invert_xaxis(); a.set_xticks(H); a.set_xlabel('h — шаг решётки спор (мельче →)'); a.set_ylabel('средняя ошибка V − T*  (секунды)')
a.text(0.39, -0.45, 'ноль = V совпадает с T*', color=MUT, fontsize=9)
a.set_title('3. Мельчим решётку: с короткими клетками\nошибка РАСТЁТ, с длинными — уходит к 0', loc='left'); a.legend(loc='lower left', fontsize=9); a.set_ylim(-1.6, 0.45)
for s in ax:
    for sp in ('top', 'right'): s.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig('reports/research/figs/explain_di_v.png', dpi=110)
