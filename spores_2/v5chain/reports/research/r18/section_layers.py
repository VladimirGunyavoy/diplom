# research-18 (идея пользователя, разбор 2): срез споры → куда ведёт КАЖДЫЙ слой управления + ландшафт V̂ там, куда приезжают. Одно зерно маятника.
import os, sys, numpy as np
sys.path.insert(0, os.environ['CELLS7']); import grow_cells2d as G
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = os.environ.get('PICS', '.'); os.makedirs(OUT, exist_ok=True)
e_ = np.linspace(-G.RHO, G.RHO, 41); G.BARRIER = cKDTree(np.r_[np.c_[e_, e_ * 0 - G.RHO], np.c_[e_, e_ * 0 + G.RHO], np.c_[e_ * 0 - G.RHO, e_], np.c_[e_ * 0 + G.RHO, e_]])
rng = np.random.default_rng(int(os.environ.get('SEED', 0))); A = G.Atlas.__new__(G.Atlas); A.layers = []; A.idx = []
for u in G.US: l, ix = G.build_layer(u, rng); A.layers.append(l); A.idx.append(ix)
A.finish(); A.solve(); cells = A.cells; cof = np.concatenate([np.full(len(c.G) * c.m, k) for k, c in enumerate(cells)])
BAR = np.load(os.environ['BARRIER']) if os.environ.get('BARRIER') else None; NS = 300; COL = {-.3: 'tab:blue', 0.: 'tab:green', .3: 'tab:red'}
def recv(Y, u):
    """шаг управлением u из Y: V̂ после шага (+DTN) и клетка-приёмник (та, что дала min V̂)"""
    tg = A.tgoal(Y, u); pi, IDX, W = A.stencils(G.step(Y, u)); v = A.interp(W, A.V[IDX]) + G.DTN
    best = np.full(len(Y), np.inf); cc = np.full(len(Y), -1)
    for i, vv, ii in zip(pi, v, IDX[:, 0]):
        if vv < best[i]: best[i] = vv; cc[i] = cof[ii]
    best = np.where(np.isfinite(tg), tg, best); cc = np.where(np.isfinite(tg), -2, cc); return best, cc
def outline(ax, c, **kw): ax.plot(np.r_[c.G[:, 0, 0], c.G[::-1, -1, 0], c.G[0, 0, 0]], np.r_[c.G[:, 0, 1], c.G[::-1, -1, 1], c.G[0, 0, 1]], **kw)
for tag, k in [t.split(':') for t in os.environ.get('CELLS', '160:160,64:64').split(',')]:
    k = int(k); c = cells[k]; s = np.linspace(-c.r, c.r, NS); E = np.c_[np.interp(s, c.sn, c.G[0][:, 0]), np.interp(s, c.sn, c.G[0][:, 1])]; y = E.copy(); nst = c.nb + c.nf
    for _ in range(nst): y = G.step(y, c.u)
    y[:, 0] = G.wrap(y[:, 0]); tc = nst * G.DTN; R = {u: recv(y, u) for u in G.US}
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    for u in G.US:                                                                                  # (1) J по слоям
        b, cc = R[u]; axs[0].plot(s, np.where(b < G.BIG / 2, tc + b, np.nan), '.', ms=2.5, color=COL[float(u)], label='ход u %+g' % u)
    jb = np.min([R[u][0] for u in G.US], 0); axs[0].plot(s, tc + jb, 'k-', lw=.8, label='лучший'); axs[0].legend(); axs[0].set_xlabel('s на входном срезе'); axs[0].set_ylabel('t в клетке + V̂ после хода')
    axs[0].set_title('клетка %d (u %+g, r %.3f, %.1f с): J по каждому слою' % (k, c.u, c.r, tc))
    for j, u in enumerate(G.US):                                                                    # (2) приёмник по слоям: номер по порядку появления вдоль s
        cc = R[u][1]; seen = {}; lab = np.array([seen.setdefault(v, len(seen)) for v in cc]); axs[1].plot(s, lab + .12 * (j - 1), '.', ms=2.5, color=COL[float(u)], label='u %+g: %d спор' % (u, len(seen)))
    axs[1].legend(); axs[1].set_xlabel('s'); axs[1].set_ylabel('спора-приёмник (№ по порядку вдоль s; −2 → цель)'); axs[1].set_title('куда ведёт каждый слой (ступенька = смена споры)')
    pad = .25; x0, x1 = y[:, 0].min() - pad, y[:, 0].max() + pad; w0, w1 = y[:, 1].min() - pad, y[:, 1].max() + pad   # (3) ландшафт V̂ у выхода
    gx, gw = np.meshgrid(np.linspace(x0, x1, 160), np.linspace(w0, w1, 160)); VG = A.vstar(np.c_[G.wrap(gx.ravel()), gw.ravel()]).reshape(gx.shape); VG[VG > G.BIG / 2] = np.nan
    im = axs[2].pcolormesh(gx, gw, VG, cmap='viridis', shading='auto'); plt.colorbar(im, ax=axs[2], label='V̂*')
    rc = set(); [rc.update(R[u][1].tolist()) for u in G.US]
    for q in rc:
        if q >= 0: outline(axs[2], cells[q], color=COL[float(cells[q].u)], lw=.7)
    axs[2].scatter(y[:, 0], y[:, 1], c=s, cmap='plasma', s=6, zorder=5); axs[2].plot(E[:, 0], E[:, 1], 'k-', lw=2)
    if BAR is not None: axs[2].plot(BAR[:, 0], BAR[:, 1], 'c.', ms=3, label='стена CUTR')
    axs[2].set_xlim(x0, x1); axs[2].set_ylim(w0, w1); axs[2].set_xlabel('θ'); axs[2].set_ylabel('ω'); axs[2].set_title('ландшафт V̂ у выхода; точки выхода (цвет s), контуры спор-приёмников (цвет слоя)')
    fig.tight_layout(); fn = os.path.join(OUT, '%s.png' % tag); fig.savefig(fn, dpi=110); print('готово', fn, {float(u): len(set(R[u][1].tolist())) for u in G.US}, flush=True)
