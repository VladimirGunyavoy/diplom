# research-18 (идея пользователя): спора знает, какие точки её НОРМАЛЬНОГО СРЕЗА в какие споры приезжают. Один случай, одно зерно.
# Проход 0 (как cutr_s0: SEED 0, OWN 1, GOALB), для каждой клетки 200 точек входного среза → поток своей u до выхода → следующая спора (лучший ход) и J = t_клетки + V̂.
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.environ['CELLS7'])
import grow_cells2d as G
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
e_ = np.linspace(-G.RHO, G.RHO, 41); GB = np.r_[np.c_[e_, e_ * 0 - G.RHO], np.c_[e_, e_ * 0 + G.RHO], np.c_[e_ * 0 - G.RHO, e_], np.c_[e_ * 0 + G.RHO, e_]]; G.BARRIER = cKDTree(GB)
rng = np.random.default_rng(int(os.environ.get('SEED', 0))); A = G.Atlas.__new__(G.Atlas); A.layers = []; A.idx = []
for u in G.US: l, ix = G.build_layer(u, rng); A.layers.append(l); A.idx.append(ix)
A.finish(); A.solve(); cells = A.cells; NS = 200
cof = np.concatenate([np.full(len(c.G) * c.m, k) for k, c in enumerate(cells)])            # узел → клетка
def nxt(Y):
    """лучший ход из точек Y: (J без t_клетки, клетка-приёмник, u)"""
    best = np.full(len(Y), np.inf); bc = np.full(len(Y), -1); bu = np.full(len(Y), np.nan)
    for u in G.US:
        tg = A.tgoal(Y, u); Ys = G.step(Y, u); pi, IDX, W = A.stencils(Ys); v = A.interp(W, A.V[IDX]) + G.DTN
        cand = np.full(len(Y), np.inf); cc = np.full(len(Y), -1)
        for i, vv, ii in zip(pi, v, IDX[:, 0]):
            if vv < cand[i]: cand[i] = vv; cc[i] = cof[ii]
        cand = np.where(np.isfinite(tg), tg, cand); cc = np.where(np.isfinite(tg), -2, cc)
        m = cand < best; best[m] = cand[m]; bc[m] = cc[m]; bu[m] = u
    return best, bc, bu
res = []
for k, c in enumerate(cells):
    s = np.linspace(-c.r, c.r, NS); sn = c.sn; g0 = c.G[0]
    E = np.c_[np.interp(s, sn, g0[:, 0]), np.interp(s, sn, g0[:, 1])]; y = E.copy(); nst = c.nb + c.nf
    for _ in range(nst): y = G.step(y, c.u)
    y[:, 0] = G.wrap(y[:, 0]); J0, bc, bu = nxt(y); J = nst * G.DTN + J0; ok = J < G.BIG / 2
    jmp = np.abs(np.diff(np.where(ok, J, np.nan))); jm = float(np.nanmax(jmp)) if np.isfinite(jmp).any() else 0.
    res.append(dict(k=k, u=float(c.u), c=c.c.tolist(), r=float(c.r), jump=jm, nnext=int(len(set(bc[ok].tolist()))), spread=float(np.nanmax(J[ok]) - np.nanmin(J[ok])) if ok.any() else 0., okf=float(ok.mean())))
    c.case = (s, E, y, J, bc, bu)
jumps = np.array([r_['jump'] for r_ in res]); order = np.argsort(-jumps)
bp = os.environ.get('BARRIER'); BAR = np.load(bp) if bp and os.path.exists(bp) else None
print('клеток', len(cells), '| скачок J между соседними точками среза > 1:', int((jumps > 1).sum()), '> .5:', int((jumps > .5).sum()), flush=True)
for i in order[:8]: print(json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in res[i].items() if kk != 'c'}), 'центр', np.round(res[i]['c'], 2).tolist())
# 1) поле: клетки по скачку J + стена CUTR того же зерна
fig, ax = plt.subplots(1, 1, figsize=(11, 6))
for k, c in enumerate(cells):
    col = plt.cm.inferno(min(jumps[k] / 2, 1.)); P = c.G[[0, -1]]
    ax.fill(np.r_[c.G[:, 0, 0], c.G[::-1, -1, 0]], np.r_[c.G[:, 0, 1], c.G[::-1, -1, 1]], color=col, alpha=.35 if jumps[k] > .5 else .08, lw=0)
if BAR is not None: ax.plot(BAR[:, 0], BAR[:, 1], '.', ms=1.5, color='tab:cyan', label='стена CUTR (зерно 0)')
for i in order[:3]: ax.plot(*cells[i].c, 'o', mfc='none', mec='lime', ms=12)
ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-G.WL, G.WL); ax.set_xlabel('θ'); ax.set_ylabel('ω'); ax.set_title('маятник u .3, проход 0: клетки по max скачку J вдоль входного среза (ярче = больше; кольца — топ-3)'); ax.legend(loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'section_field.png'), dpi=110)
# 2) один случай: топ-1 клетка
i = order[int(os.environ.get('PICK', 0))]; c = cells[i]; s, E, y, J, bc, bu = c.case
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
axs[0].plot(s, J, '.-', ms=3); axs[0].set_xlabel('координата на входном срезе s'); axs[0].set_ylabel('J = t в клетке + V̂ выхода'); axs[0].set_title('клетка %d (u %+g), скачок %.2f' % (i, c.u, jumps[i]))
ids = {v: n for n, v in enumerate(sorted(set(bc.tolist())))}; axs[1].scatter(s, [ids[v] for v in bc], c=bu, cmap='coolwarm', s=8); axs[1].set_xlabel('s'); axs[1].set_ylabel('следующая спора (номер в списке)'); axs[1].set_title('куда приезжает точка среза (цвет — u хода)')
L = A.rollout(y, tmax=3.)[2]
for j in range(0, NS, 5): axs[2].plot(L[:, j, 0], L[:, j, 1], '-', lw=.6, color=plt.cm.viridis(j / NS))
axs[2].plot(c.G[..., 0].ravel(), c.G[..., 1].ravel(), 'k.', ms=1); axs[2].set_title('выход + 3 с жадного агента (цвет — s)'); axs[2].set_xlabel('θ'); axs[2].set_ylabel('ω')
fig.tight_layout(); fig.savefig(os.path.join(HERE, os.environ.get('OUTN', 'section_case.png')), dpi=110); print('картинки готовы', flush=True)
