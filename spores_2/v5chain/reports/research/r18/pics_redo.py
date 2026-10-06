# research-18: читаемые картинки. MODE=map — карта растяжения (3 слоя друг под другом); MODE=cell — клетка 160: бусины + сепаратриса и нитки по времени.
import os, sys, numpy as np
sys.path.insert(0, os.environ['CELLS7']); import grow_cells2d as G
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.colors import LogNorm
plt.rcParams.update({'font.size': 11})
e_ = np.linspace(-G.RHO, G.RHO, 41); G.BARRIER = cKDTree(np.r_[np.c_[e_, e_ * 0 - G.RHO], np.c_[e_, e_ * 0 + G.RHO], np.c_[e_ * 0 - G.RHO, e_], np.c_[e_ * 0 + G.RHO, e_]])
rng = np.random.default_rng(0); cells = []
for u in G.US: l, _ = G.build_layer(u, rng); cells += l
wl = lambda r_: float(np.linalg.norm(np.diff(r_, axis=0), axis=1).sum()) + 1e-12
P = os.environ['PICS']; BAR = np.load(os.environ['BARRIER']); th = np.linspace(-np.pi, np.pi, 400)
if os.environ['MODE'] == 'map':
    st = np.array([max(wl(c.G[-1]) / wl(c.G[0]), wl(c.G[0]) / wl(c.G[-1])) for c in cells]); nrm = LogNorm(1, 15); cm = plt.cm.inferno
    fig, axs = plt.subplots(3, 1, figsize=(10, 17), sharex=True)
    for ax, u in zip(axs, G.US):
        for c, sv in zip(cells, st):
            if c.u == u: ax.fill(np.r_[c.G[:, 0, 0], c.G[::-1, -1, 0]], np.r_[c.G[:, 0, 1], c.G[::-1, -1, 1]], color=cm(nrm(sv)), lw=.3, ec='k')
        ax.plot(BAR[:, 0], BAR[:, 1], '.', ms=2, color='tab:cyan'); ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-G.WL, G.WL); ax.set_ylabel('ω')
        ax.set_title('управление u = %+.1f  (%d клеток)' % (u, sum(c.u == u for c in cells)))
    axs[-1].set_xlabel('θ'); fig.suptitle(os.environ.get('TITLE', 'растяжение среза клетки'), fontsize=13)
    fig.colorbar(plt.cm.ScalarMappable(nrm, cm), ax=axs, shrink=.5, label='во сколько раз растянут срез'); fig.savefig(os.path.join(P, os.environ['OUT']), dpi=95, bbox_inches='tight')
    print('готово', os.environ['OUT'], 'растяжение кв.', np.quantile(st, [.5, .9, .99, 1]).round(1).tolist(), '>3:', int((st > 3).sum()), 'клеток', len(cells), flush=True)
else:
    c = cells[160]; s = np.linspace(-c.r, c.r, 300); E = np.c_[np.interp(s, c.sn, c.G[0][:, 0]), np.interp(s, c.sn, c.G[0][:, 1])]
    Y = [E.copy()]; y = E.copy()
    for _ in range(c.nb + c.nf): y = G.step(y, c.u); Y.append(y.copy())
    Y = np.array(Y); Tc = (c.nb + c.nf) * G.DTN
    # 12: бусины (легенда снизу)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.fill(np.r_[c.G[:, 0, 0], c.G[::-1, -1, 0]], np.r_[c.G[:, 0, 1], c.G[::-1, -1, 1]], color='0.85', label='клетка 160 (u = 0, длина %.1f с)' % Tc)
    for j in range(0, 300, 20): ax.plot(Y[:, j, 0], Y[:, j, 1], '-', lw=1, color=plt.cm.plasma(j / 300))
    ax.plot(E[:, 0], E[:, 1], 'k-', lw=5, label='нитка на входе: длина %.2f' % wl(E)); ax.scatter(Y[-1, :, 0], Y[-1, :, 1], c=s, cmap='plasma', s=10, zorder=5, label='нитка на выходе: длина %.2f' % wl(Y[-1]))
    ax.add_patch(plt.Rectangle((-G.RHO, -G.RHO), 2 * G.RHO, 2 * G.RHO, fill=False, ec='tab:cyan', lw=2, label='цель'))
    ax.set_xlim(-np.pi, 1.3); ax.set_ylim(-1, 2.4); ax.set_xlabel('θ'); ax.set_ylabel('ω'); ax.set_title('Бусины: откуда и куда (клетка 160)')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.1), ncol=2); fig.savefig(os.path.join(P, '12_beads_cell160.png'), dpi=100, bbox_inches='tight')
    # 13: сепаратриса + нитки через 0.5 с
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(th, -2 * np.sin(th / 2), '-', color='tab:green', lw=2.5, label='сепаратриса ВХОДЯЩАЯ в седло (ω = −2 sin θ/2)')
    ax.plot(th, 2 * np.sin(th / 2), '--', color='tab:red', lw=2.5, label='сепаратриса ВЫХОДЯЩАЯ из седла (ω = +2 sin θ/2)')
    ax.fill(np.r_[c.G[:, 0, 0], c.G[::-1, -1, 0]], np.r_[c.G[:, 0, 1], c.G[::-1, -1, 1]], color='0.88', label='клетка 160 (u = 0)')
    ks = list(range(0, len(Y), 5)) + [len(Y) - 1]
    for k in ks:
        ax.plot(Y[k, :, 0], Y[k, :, 1], '-', lw=3, color=plt.cm.viridis(k / (len(Y) - 1)))
        m = Y[k, 150]; ax.annotate('t=%.1f\nw=%.2f' % (k * G.DTN, wl(Y[k])), m, xytext=(8, 8), textcoords='offset points', fontsize=8.5)
    ax.add_patch(plt.Rectangle((-G.RHO, -G.RHO), 2 * G.RHO, 2 * G.RHO, fill=False, ec='tab:cyan', lw=2)); ax.annotate('цель = седло\n(маятник вверх)', (0, 0), xytext=(.25, -.6), textcoords='data', arrowprops=dict(arrowstyle='->'))
    ax.set_xlim(-np.pi, 1.3); ax.set_ylim(-1, 2.4); ax.set_xlabel('θ'); ax.set_ylabel('ω'); ax.set_title('Почему ширина среза сначала падает, потом растёт (клетка 160, нитки через 0.5 с)')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.08), ncol=1); fig.savefig(os.path.join(P, '13_cell160_separatrix.png'), dpi=100, bbox_inches='tight')
    print('готово 12, 13; ширины:', [(round(k * G.DTN, 1), round(wl(Y[k]), 2)) for k in ks], flush=True)
