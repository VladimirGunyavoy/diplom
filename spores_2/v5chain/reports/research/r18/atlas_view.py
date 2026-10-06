# research-19: атлас прогона (cells.pkl + value.npz) — 3 слоя в ряд, сверху всё поле, снизу окрестность цели; зелёным — где V конечна (значение дошло от цели).
import os, sys, pickle, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})
D = sys.argv[1]; OUT = sys.argv[2]; TITLE = sys.argv[3]; RHO = .1; WL = 3.5
cells = pickle.load(open(os.path.join(D, 'cells.pkl'), 'rb')); v = np.load(os.path.join(D, 'value.npz')); ok = (v['V'] < 500).astype(float); US = sorted({c['u'] for c in cells})
fig, axs = plt.subplots(2, 3, figsize=(27, 14), constrained_layout=True)
for r, (xl, yl) in enumerate((((-5., 5.), (-WL, WL)), ((-1.3, 1.3), (-1., 1.)))):
    for ax, u in zip(axs[r], US):
        for c in cells:
            if c['u'] != u: continue
            g = c['G']; px, py = np.r_[g[:, 0, 0], g[::-1, -1, 0]], np.r_[g[:, 0, 1], g[::-1, -1, 1]]
            for sh in (0., 2 * np.pi, -2 * np.pi, 4 * np.pi, -4 * np.pi):
                if px.min() + sh > xl[1] or px.max() + sh < xl[0]: continue
                ax.fill(px + sh, py, fc=(.45, .65, .9, .35), ec='0.15', lw=.7 if r == 0 else 1.1)
        [ax.contourf(v['gx'] + sh, v['gw'], ok, levels=[.5, 1.5], colors=['tab:green'], alpha=.45) for sh in (0., 2 * np.pi, -2 * np.pi)]
        ax.add_patch(plt.Rectangle((-RHO, -RHO), 2 * RHO, 2 * RHO, fill=False, ec='tab:red', lw=2.5, zorder=6)); ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_xlabel('θ')
        ax.set_title(('u = %+.1f  (%d клеток)' % (u, sum(c['u'] == u for c in cells))) if r == 0 else 'u = %+.1f — у цели' % u)
    axs[r][0].set_ylabel('ω')
fig.suptitle(TITLE + '   ·   зелёное — V конечна (общая для слоёв), красный квадрат — цель', fontsize=14); fig.savefig(OUT, dpi=90); print('готово', OUT, 'V конечна на', round(float(ok.mean()), 3), 'поля')
