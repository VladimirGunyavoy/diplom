"""research hub-v5chain-research-8: картинка для пользователя — карта клетки ДИ (x, v; a = ±1) двумя способами:
(1) плоское сечение по нормали к линии динамики своего слоя (у каждого a — своя нормаль); (2) криволинейное сечение = кривая переключения."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(14, 6.2))
c = np.array([-.6, .4]); r, tau = .25, .8
for a, col in ((1., 'tab:red'), (-1., 'tab:blue')):
    rho = np.hypot(c[1], a); f = np.array([c[1], a]) / rho; n = np.array([-a, c[1]]) / rho          # поле и нормаль слоя a
    for s in np.linspace(-r, r, 7):                                                                 # линии s = const (траектории слоя)
        t = np.linspace(0, tau, 50); y = c + s * n; ax[0].plot(y[0] + y[1] * t + a * t * t / 2, y[1] + a * t, color=col, lw=.8)
    for t in np.linspace(0, tau, 5):                                                                # линии t = const (сдвинутое сечение)
        s = np.linspace(-r, r, 20); Y = c[:, None] + np.outer(n, s); ax[0].plot(Y[0] + Y[1] * t + a * t * t / 2, Y[1] + a * t, color=col, lw=.8, ls=':')
    ax[0].annotate('', c + .35 * f, c, arrowprops=dict(arrowstyle='->', color=col, lw=2)); ax[0].annotate('', c + .3 * n, c, arrowprops=dict(arrowstyle='->', color=col, lw=2, ls='--'))
    ax[0].text(*(c + .38 * f), f'f (a={a:+.0f})', color=col); ax[0].text(*(c + .33 * n), f'нормаль a={a:+.0f}', color=col)
ax[0].plot(*c, 'ko'); ax[0].set_title('(1) плоское сечение: спора c, у каждого a своя нормаль n = (−a, v)/|f|\nсплошные — s = const, пунктир — t = const')
ax[0].set_xlabel('x'); ax[0].set_ylabel('v'); ax[0].set_aspect('equal'); ax[0].grid(alpha=.3)
# (2) сечение = кривая переключения Γ: (−s²/2, s), s>0 (дуга a=−1 в цель); клетки слоя a=+1 ВВЕРХ по потоку: Ψ(s,t) = поток a=+1 назад на t
S = np.linspace(0, 1.6, 200); ax[1].plot(-S**2 / 2, S, 'k', lw=3, label='Γ: x = −v²/2 (дуга a=−1 в цель)')
for s in np.linspace(.1, 1.5, 8):
    t = np.linspace(0, 2, 60); ax[1].plot(-s * s / 2 - s * t + t * t / 2, s - t, color='tab:red', lw=.8)
for t in np.linspace(.25, 2, 8):
    s = np.linspace(0, 1.6, 60); ax[1].plot(-s * s / 2 - s * t + t * t / 2, s - t, color='tab:red', lw=.8, ls=':')
X, V = np.meshgrid(np.linspace(-2.5, 1.5, 300), np.linspace(-2, 2, 300)); below = X < -V * np.abs(V) / 2
Tq = np.where(below, -V + 2 * np.sqrt(np.maximum(V * V / 2 - X, 0)), np.nan); cs = ax[1].contour(X, V, Tq, levels=[.5, 1, 1.5, 2, 2.5, 3], colors='gray', linewidths=.7); ax[1].clabel(cs, fontsize=8)
ax[1].plot(0, 0, 'k*', ms=14); ax[1].set_title('(2) сечение = кривая переключения Γ: координаты (s, t)\ns — скорость в момент переключения, t — время до него;  T = t + s')
ax[1].set_xlabel('x'); ax[1].set_ylabel('v'); ax[1].set_xlim(-2.5, 1.5); ax[1].set_ylim(-2, 2); ax[1].legend(loc='lower right'); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig('figs/explain_di_chart.png', dpi=110)
