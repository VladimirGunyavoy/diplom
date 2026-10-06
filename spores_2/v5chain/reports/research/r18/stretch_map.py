# research-18: картинка — растяжение клеток (w выхода / w входа) на поле + стена CUTR. Атлас прохода 0 (SEED, OWN как у cutr_s0) или с SMAX/SLAM.
import os, sys, numpy as np
sys.path.insert(0, os.environ['CELLS7']); import grow_cells2d as G
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.colors import LogNorm
e_ = np.linspace(-G.RHO, G.RHO, 41); G.BARRIER = cKDTree(np.r_[np.c_[e_, e_ * 0 - G.RHO], np.c_[e_, e_ * 0 + G.RHO], np.c_[e_ * 0 - G.RHO, e_], np.c_[e_ * 0 + G.RHO, e_]])
rng = np.random.default_rng(int(os.environ.get('SEED', 0))); cells = []
for u in G.US: l, _ = G.build_layer(u, rng); cells += l
wl = lambda r_: float(np.linalg.norm(np.diff(r_, axis=0), axis=1).sum()) + 1e-12
st = np.array([max(wl(c.G[-1]) / wl(c.G[0]), wl(c.G[0]) / wl(c.G[-1])) for c in cells]); BAR = np.load(os.environ['BARRIER'])
fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True); cm = plt.cm.inferno; nrm = LogNorm(1, max(20, st.max()))
for j, u in enumerate(G.US):
    ax = axs[j]
    for c, sv in zip(cells, st):
        if c.u != u: continue
        ax.fill(np.r_[c.G[:, 0, 0], c.G[::-1, -1, 0]], np.r_[c.G[:, 0, 1], c.G[::-1, -1, 1]], color=cm(nrm(sv)), alpha=.85, lw=.3, ec='k')
    ax.plot(BAR[:, 0], BAR[:, 1], '.', ms=1.5, color='tab:cyan'); ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-G.WL, G.WL); ax.set_title('слой u %+g: %d клеток; цвет — растяжение среза max(w_вых/w_вх, обратное)' % (u, sum(c.u == u for c in cells)), fontsize=9); ax.set_xlabel('θ')
axs[0].set_ylabel('ω'); fig.colorbar(plt.cm.ScalarMappable(nrm, cm), ax=axs, label='растяжение (лог)'); fn = os.environ['OUT']; fig.savefig(fn, dpi=100)
print('готово', fn, 'клеток', len(cells), 'растяжение кв.', np.quantile(st, [.5, .9, .99, 1]).round(1).tolist(), '>3:', int((st > 3).sum()), '>10:', int((st > 10).sum()), flush=True)
