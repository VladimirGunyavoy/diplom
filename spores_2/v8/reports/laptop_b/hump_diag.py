"""laptop-B: разбор 5 трасс с наибольшим горбом (SUBK=K, MAXC=3): по t — слой/клетка/строка/столбец/локальные координаты ячейки, Q слоёв, откуда min. Из v8 (aida): MAXC=3 SUBK=3 python3 reports/laptop_b/hump_diag.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S, ref_di_disk as R
import hump as H
TG = H.TG; g = S.load_growN(); atlas, _ = S.build_atlas([1., -1., 0.], 0, par=False); G = S.make_graph(g, atlas, 'N'); rho = G.rho; G.set_calc('q'); G.subk = int(os.environ.get('SUBK', 3)); G.solve(verbose=False)
X0 = S.starts(g, 600, 1); mx = S.path_extent(X0, rho); Xz = X0[mx <= float(g.XLV[0])]; U0, T1, TT = H.opt_switch(Xz, rho)
ids = np.flatnonzero(np.isfinite(T1) & (T1 >= .8) & (TT - T1 >= .3)); res = []
for i in ids:
    t = T1[i] + TG; a = np.minimum(t, T1[i]); b = np.maximum(t - T1[i], 0.); u = U0[i]; x1 = Xz[i, 0] + Xz[i, 1] * a + u * a * a / 2; v1 = Xz[i, 1] + u * a; P = np.stack([x1 + v1 * b - u * b * b / 2, v1 - u * b], 1); e = G.Vlayers(P).min(1) - (TT[i] - t)
    if not np.isfinite(e).all(): continue
    win = (TG >= -.5) & (TG <= 0); res.append((e[win].max() - e[0], i, P, e))
res.sort(key=lambda r: -r[0]); hs = np.array([r[0] for r in res]); print('RES трасс %d, горб мед %.3f p90 %.3f, топ-5: %s' % (len(res), np.median(hs), np.percentile(hs, 90), np.round(hs[:5], 3)))
np.set_printoptions(precision=3, suppress=True, linewidth=220)
for h, i, P, e in res[:5]:
    print('RES === трасса %d: старт (%.2f,%.2f), u0=%+g, t_sw=%.2f, T=%.2f, горб %.3f' % (i, *Xz[i], U0[i], T1[i], TT[i], h))
    pk = int(np.argmax(np.where((TG >= -.5) & (TG <= 0), e, -9))); show = [k for k in range(len(TG)) if abs(TG[k] - TG[pk]) <= .1 or TG[k] in (-.8, 0.)]
    for k in show:
        row = []
        for li, u in enumerate(G.us):
            pi, verts, w, rank = G.locate(u, P[k:k + 1]); info = 'нет ячейки'
            for r in range(len(pi)):
                vv = verts[r]; Qv = G.V[vv]
                if (Qv < S.BIG / 2).all(): info = 'кл%d ряд%d/%d кол%d sc=(%.2f,%.2f) ранг%d Q=%.3f верш %s' % (G.nodecell[vv[0]] - G.gcell[u][0], G.rowid[vv[0]], len(atlas[u][G.nodecell[vv[0]] - G.gcell[u][0]].G) if False else 0, (vv[0] - G.base[u][G.nodecell[vv[0]] - G.gcell[u][0]]) % G.M, w[r, 2] + w[r, 3], w[r, 1] + w[r, 3], rank[r], float((w[r] * Qv).sum()), np.round(Qv, 2)); break
            row.append('u=%+g: %s' % (u, info))
        print('RES  t−t_sw=%+.2f (%.2f,%.2f) e=%+.3f | %s' % (TG[k], *P[k], e[k], ' || '.join(row)))
# картинка худшей трассы + разброс вершин ячейки, ответившей на запрос, по всем трассам
def spread(u, Pk):
    pi, verts, w, rank = G.locate(u, Pk); out = np.full(len(Pk), np.nan); done = np.zeros(len(Pk), bool)
    for r in range(len(pi)):
        Qv = G.V[verts[r]]
        if (Qv < S.BIG / 2).all() and not done[pi[r]]: out[pi[r]] = Qv.max() - Qv.min(); done[pi[r]] = True
    return out
sp = []; 
for h, i, P, e in res: sp.append(np.nanmax([np.nanmax(spread(u, P[(TG >= -.5) & (TG <= 0)])) for u in G.us if u != 0]))
sp = np.array(sp); hs = np.array([r[0] for r in res]); print('RES разброс Q по вершинам ответившей ячейки (max на [−.5,0]): мед %.2f p90 %.2f; корреляция с горбом %.2f; горб>.1 у %d трасс, из них разброс>.5 у %d' % (np.median(sp), np.percentile(sp, 90), np.corrcoef(sp, hs)[0, 1], (hs > .1).sum(), ((hs > .1) & (sp > .5)).sum()))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
h, i, P, e = res[0]; Tr_t = TT[i] - (T1[i] + TG); Vl = G.Vlayers(P); fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for li, u in enumerate(G.us):
    if u == 0: continue
    ax[0].plot(TG, Vl[:, li] - Tr_t, label='Q u=%+g − T_ref' % u); ax[1].plot(TG, spread(u, P), label='разброс вершин u=%+g' % u)
ax[0].plot(TG, e, 'k', lw=2, label='V* − T_ref'); ax[0].axvline(0, color='gray'); ax[0].axhline(0, color='gray'); ax[0].set_xlabel('t − t_sw'); ax[0].legend(fontsize=8); ax[0].set_title('трасса %d: горб %.3f (MAXC=3)' % (i, h)); ax[1].legend(fontsize=8); ax[1].set_xlabel('t − t_sw'); ax[1].set_title('Q_max−Q_min по 4 вершинам ячейки')
for u, c in ((1., 'C0'), (-1., 'C1')):
    for cl in atlas[u]: gg = np.asarray(cl.G); ax[2].plot(gg[:, :, 0].T, gg[:, :, 1].T, c, lw=.2) if False else ax[2].plot(gg[:, 0, 0], gg[:, 0, 1], c, lw=.4, alpha=.5)
ax[2].plot(P[:, 0], P[:, 1], 'k.-'); ax[2].add_patch(plt.Circle((0, 0), rho, fill=False)); ax[2].set_xlim(-1.2, 1.2); ax[2].set_ylim(-1, 1); ax[2].set_title('оптимальная трасса (чёрная) и крайние столбцы клеток'); plt.tight_layout(); plt.savefig(os.environ.get('OUT', 'hump_diag') + '.png', dpi=90)
