"""Д1 п.(а),(в): где лежат работающие клетки u=0; чьи старты дают хвост T/T_ref (база 249 кл. и без неработающих). Рисует поле, клетки u=0 (работающие — красным), старты (цвет — T/T_ref)."""
import sys, os, json, pickle; sys.path.insert(0, os.path.join(os.getcwd(), '..', 'v7', 'reports', 'big'))
import big_lib as B, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
G = B.G; A = B.make_atlas(); B.solve(A); OUT = os.environ['BIGOUT']; d = np.load(OUT + '/pend_d1_cells.npz'); frac, cu = d['frac'], d['cu']
Qs, ref = G.starts_ref(); T, sw, path = A.rollout(Qs); r = T / ref; W = np.flatnonzero((cu == 1) & (frac > .1))
cent = [(i, A.cells[i].G[..., 0].mean(), A.cells[i].G[..., 1].mean(), float(np.abs(G.wrap(A.cells[i].G[..., 0])).min()), float(np.hypot(G.wrap(A.cells[i].G.reshape(-1, 2)[:, 0]), A.cells[i].G.reshape(-1, 2)[:, 1]).min()), float(frac[i])) for i in W]
print('u=0 работающие: (idx, θ̄, ω̄, minθ, мин. расстояние до цели, доля)', [tuple(round(x, 3) for x in c) for c in cent])
worst = np.argsort(-np.nan_to_num(r, nan=9, posinf=9))[:10]; print('хвост: старт, θ, ω, T/Tref, T, Tref', [(int(i), round(float(Qs[i, 0]), 2), round(float(Qs[i, 1]), 2), round(float(r[i]), 3), round(float(T[i]), 2), round(float(ref[i]), 2)) for i in worst])
us_used = None
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
for a_, ttl in zip(ax, ('клетки u=0 (серые — не работают, красные — работают)', 'старты: цвет T/T_ref; траектории худших 5')):
    for i, c in enumerate(A.cells):
        if cu[i] != 1: continue
        X = c.G.reshape(-1, 2); col = 'tab:red' if frac[i] > .1 else '0.8'
        if a_ is ax[0]: a_.scatter(G.wrap(X[:, 0]), X[:, 1], s=.3, c=col)
    a_.set_title(ttl); a_.set_xlim(-np.pi, np.pi); a_.set_ylim(-3.5, 3.5); a_.set_xlabel('θ'); a_.set_ylabel('ω'); a_.plot([0], [0], 'k*', ms=10)
sc = ax[1].scatter(Qs[:, 0], Qs[:, 1], c=np.clip(r, 1, 1.5), cmap='viridis', s=18); plt.colorbar(sc, ax=ax[1])
for i in worst[:5]:
    p = path[:, i]; ax[1].plot(G.wrap(p[:, 0]), p[:, 1], '.', ms=1.5); ax[1].annotate(str(int(i)), Qs[i])
plt.tight_layout(); plt.savefig(OUT + '/../pics/pend_u0_tail.png', dpi=100)
json.dump(dict(u0_working=[dict(idx=int(c[0]), theta_mean=c[1], omega_mean=c[2], min_dist_goal=c[4], frac=c[5]) for c in cent], tail=[dict(i=int(i), theta=float(Qs[i, 0]), omega=float(Qs[i, 1]), ratio=float(r[i]), T=float(T[i]), Tref=float(ref[i])) for i in worst]), open(OUT + '/pend_u0_tail.json', 'w'), indent=1)
