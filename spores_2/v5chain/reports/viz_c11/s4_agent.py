"""Сюжет 4: агент со старта (трудный старт) — слева старая сшивка (ближайший+KV·d), справа Делоне. На каждом шаге веер дуг и выбранная."""
import sys; from vizlib import *
i0 = int(sys.argv[1]) if len(sys.argv) > 1 else 10
S = get_S(); rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-PI, PI, 100), rq.uniform(-2, 2, 100)], 1); x0 = Q[i0]
R = {m: rollout(S, m, x0) for m in ('nn', 'dl')}; G = {m: grid_V(S, m) for m in ('nn', 'dl')}
names = {'nn': 'Старая сшивка: ближайший узел + KV·d', 'dl': 'Новая сшивка: треугольник Делоне, веса λ'}
nfr = max(len(R['nn'][2]), len(R['dl'][2])); frames = []; fig, axs = plt.subplots(1, 2, figsize=(14, 7.2)); fig.subplots_adjust(left=.05, right=.99, top=.86, bottom=.1, wspace=.14)
for k in range(nfr + 1):
    for ax, m in zip(axs, ('nn', 'dl')):
        ax.clear(); tr, acts, info, T = R[m]; base_ax(ax, names[m]); heat(ax, *G[m]); n = len(info); kk = min(k, n)
        plot_path(ax, tr[:kk + 1], color='white', lw=5, zorder=4); plot_path(ax, tr[:kk + 1], color='k', lw=2.5, zorder=5)
        if kk < n:
            x = info[kk]['x']; ch = info[kk]['s']
            for s in range(3):
                ok = np.isfinite(info[kk]['c'][s]) and info[kk]['c'][s] < 1e8
                plot_arc(ax, x, s, color=COL[s], lw=4.5 if s == ch else 2, alpha=1, zorder=7 if s == ch else 6, ls='-' if ok else ':')
                e = info[kk]['E'][s]; ax.plot(e[0], e[1], 'o', color=COL[s], ms=9 if s == ch else 6, mec='w', zorder=8)
            ax.plot(*wrap(x), 'o', color='w', mec='k', ms=11, zorder=9)
            xw = wrap(x); w = .5; ia = ax.inset_axes([.60, .60, .38, .38]); ia.set_facecolor('#f4f4f4'); ia.set_xlim(xw[0] - w, xw[0] + w); ia.set_ylim(xw[1] - w, xw[1] + w); ia.set_xticks([]); ia.set_yticks([]); ia.text(.03, .03, 'лупа ×3', transform=ia.transAxes, fontsize=10)
            for s in range(3):
                ok = np.isfinite(info[kk]['c'][s]) and info[kk]['c'][s] < 1e8
                plot_arc(ia, x, s, color=COL[s], lw=4 if s == ch else 2, zorder=5 if s == ch else 4, ls='-' if ok else ':'); e = wrap(info[kk]['E'][s]); ia.plot(e[0], e[1], 'o', color=COL[s], ms=10 if s == ch else 7, mec='k', zorder=6)
            ia.plot(*xw, 'o', color='w', mec='k', ms=11, zorder=7)
            st = 'шаг %d  t = %.1f' % (kk + 1, 0.2 * kk)
        else:
            ax.plot(*wrap(tr[-1]), 'o', color='w', mec='k', ms=11, zorder=9); st = 'ДОШЁЛ: %d шагов, T = %.1f с' % (n, T)
        ax.plot(*wrap(x0), 'D', color='magenta', mec='k', ms=11, zorder=9); ax.text(.02, .96, st, transform=ax.transAxes, va='top', fontsize=15, bbox=dict(fc='w', ec='k', alpha=.9), zorder=20)
    if k == 0:
        for s in range(3): axs[0].plot([], [], color=COL[s], lw=3, label=UNAME[s] + ' (дуга τ = 0.2 с)')
        axs[0].plot([], [], 'D', color='magenta', mec='k', label='старт'); axs[0].legend(loc='lower left', fontsize=11, framealpha=.9).set_zorder(30)
    else:
        axs[0].legend(handles=[plt.Line2D([], [], color=COL[s], lw=3, label=UNAME[s]) for s in range(3)], loc='lower left', fontsize=11, framealpha=.9).set_zorder(30)
    fig.suptitle('Агент пробует 3 дуги, берёт ту, где τ + V(конец) меньше.  Фон = V: светлое — близко, тёмное — далеко, белое — V не определена', fontsize=13, y=.97)
    frames.append(fig_to_img(fig))
frames[0].save('/tmp/c11_s4_first_%d.png' % i0)
save_gif(frames, '/tmp/c11_method_agent_%d.gif' % i0, fps=5, hold_last=12)
