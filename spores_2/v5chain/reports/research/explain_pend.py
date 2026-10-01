"""Картинка для пользователя (hub-research-4): маятник θ̈ = −sin θ + u, |u| ≤ 0.3 — фазовый портрет, раскачка, два слоя."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
B, O, INK, MUT = '#2a78d6', '#eb6834', '#222222', '#8a8a85'; U = 0.3
th = np.linspace(-np.pi, np.pi, 300); w = np.linspace(-2.6, 2.6, 260); TH, W = np.meshgrid(th, w)
f = lambda x, u: np.array([x[1], -np.sin(x[0]) + u])
def rk4(x, u, h):
    k1 = f(x, u); k2 = f(x + h / 2 * k1, u); k3 = f(x + h / 2 * k2, u); k4 = f(x + h * k3, u); return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
fig, ax = plt.subplots(1, 3, figsize=(19, 6.2))
a = ax[0]; E = W ** 2 / 2 - np.cos(TH)
a.contour(TH, W, E, levels=np.linspace(-0.95, 2.5, 16), colors=MUT, linewidths=0.6)
a.contour(TH, W, E, levels=[1.0], colors=INK, linewidths=1.8)
x = np.array([0.0, 0.0]); P = [x]; t = 0; sw = 0; up = None; h = 0.005
while x[1] ** 2 / 2 - np.cos(x[0]) < 1.0 and t < 30:
    u = U if x[1] >= 0 else -U
    if up is not None and u != up: sw += 1
    up = u; x = rk4(x, u, h); t += h; P.append(x.copy())
P = np.array(P); Pw = P.copy(); Pw[:, 0] = (Pw[:, 0] + np.pi) % (2 * np.pi) - np.pi
br = np.where(np.abs(np.diff(Pw[:, 0])) > np.pi)[0] + 1
for seg in np.split(Pw, br): a.plot(seg[:, 0], seg[:, 1], color=O, lw=1.8)
a.plot(0, 0, 'o', color=INK, ms=8); a.plot([-np.pi, np.pi], [0, 0], '*', color='#1baf7a', ms=16, clip_on=False)
a.annotate('низ, покой\n(старт)', (0, 0), (0.35, -1.2), color=INK, arrowprops=dict(arrowstyle='->', color=MUT))
a.annotate('верх (цель)\nθ = ±π — одна точка', (np.pi, 0), (1.25, 1.75), color=INK, arrowprops=dict(arrowstyle='->', color=MUT))
a.annotate('сепаратриса:\nровно энергии\nхватает до верха', (-1.6, 1.17), (-3.05, 2.05), color=INK, fontsize=9, arrowprops=dict(arrowstyle='->', color=MUT))
a.set_title('1. Без мотора — орбиты (серые); u = 0.3·sign(ω) — раскачка\n(оранжевое): до энергии верха %.1f с, %d переключения' % (t, sw), loc='left')
for a, u, col, nm in ((ax[1], U, B, '+0.3'), (ax[2], -U, O, '−0.3')):
    DTH = W; DW = -np.sin(TH) + u
    a.streamplot(TH, W, DTH, DW, color=col, density=1.3, linewidth=0.8, arrowsize=0.8)
    s1 = np.arcsin(u); s2 = np.pi - s1 if u > 0 else -np.pi - s1
    a.plot(s1, 0, 'o', color=INK, ms=8); a.plot(s2, 0, 'X', color=INK, ms=9)
    a.text(s1 + 0.1, -0.35, 'устойчиво', fontsize=9, color=INK, bbox=dict(fc='white', ec='none', alpha=0.8))
    a.text(s2 - (1.2 if u > 0 else -0.1), 0.25, 'седло', fontsize=9, color=INK, bbox=dict(fc='white', ec='none', alpha=0.8))
    a.set_title('%d. Слой u = %s: поток при постоянном моторе\nравновесия сдвинуты: sin θ = %s' % (2 if u > 0 else 3, nm, nm), loc='left')
for a in ax:
    a.set_xlim(-np.pi, np.pi); a.set_ylim(-2.6, 2.6); a.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]); a.set_xticklabels(['−π', '−π/2', '0', 'π/2', 'π'])
    a.set_xlabel('θ (угол; 0 — низ, ±π — верх)'); a.set_ylabel('ω (угловая скорость)')
    for sp in ('top', 'right'): a.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig('reports/research/figs/explain_pend.png', dpi=100); print(t, sw)
