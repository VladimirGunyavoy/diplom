# research-20 (слово пользователя): картинка прогона — 3 слоя u (цвет клетки — число клонов) + панель «фон V* градиентом + пути агента, цвет отрезка = управление на шаге»;
# вызывается из compute.py в конце каждого прогона → pics/auto/<прогон>.png; отдельно: python3 atlas_pic.py <npz> <out>
import os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.collections import LineCollection
RHO = .1; SH = [k * 2 * np.pi for k in range(-2, 3)]
MC = {5: '#9ecae1', 9: '#a1d99b', 17: '#fdd0a2', 33: '#fc9272'}; EC = {5: '#3182bd', 9: '#31a354', 17: '#e6550d', 33: '#de2d26'}
UC = {-0.3: '#c51b7d', 0.0: '#111111', 0.3: '#ff7f00'}                                          # пурпурный / чёрный / оранжевый — не сливаются с фоном YlGnBu
def contour(G): return np.r_[G[:, 0], G[-1, 1:-1], G[::-1, -1], G[0, -2:0:-1]]
def controls(P, US, step, wrap):
    UA = np.array(US); U = np.full(P.shape[:2], np.nan)
    for k in range(P.shape[0] - 1):
        mv = np.any(P[k + 1] != P[k], 1)
        if not mv.any(): continue
        d = np.stack([np.hypot(wrap(step(P[k][mv], u)[:, 0] - P[k + 1][mv, 0]), step(P[k][mv], u)[:, 1] - P[k + 1][mv, 1]) for u in US], 1); U[k, mv] = UA[d.argmin(1)]
    return U
def draw(cells, gx, gw, VG, P, U, T, title, out, XR=6., WL=3.5):
    """cells: список dict(u, G) ; VG: V* на сетке gx × gw ; P: (шаги+1, n, 2) ; U: (шаги+1, n)"""
    plt.rcParams.update({'font.size': 12}); VG = np.where(VG > 500, np.nan, VG)
    US = sorted(set(round(float(c['u']), 3) for c in cells))
    fig = plt.figure(figsize=(30, 14)); gs = fig.add_gridspec(2, 3); a0 = fig.add_subplot(gs[0, 0])
    axs = [a0, fig.add_subplot(gs[0, 1], sharex=a0, sharey=a0), fig.add_subplot(gs[0, 2], sharex=a0, sharey=a0), fig.add_subplot(gs[1, 1:], sharex=a0, sharey=a0), fig.add_subplot(gs[1, 0], sharex=a0, sharey=a0)]
    ax = axs[4]; sd = {}                                                                          # research-20 (слово пользователя): посев спор — только затравки: цвет = порядок посева, размер = во скольких слоях затравка дала клетку
    for i, c in enumerate(cells):
        if c.get('p0') is None: continue
        k = int(c['seq']) if c.get('seq', -1) is not None and int(c.get('seq', -1)) >= 0 else i; sd.setdefault(k, [np.array(c['p0'], float), 0]); sd[k][1] += 1
    if sd:
        ks = sorted(sd); Pq = np.array([sd[k][0] for k in ks]); nl = np.array([sd[k][1] for k in ks]); oc = np.arange(len(ks))
        for sh in SH: sc = ax.scatter(Pq[:, 0] + sh, Pq[:, 1], c=oc, cmap='viridis', s=12 + 22 * nl, alpha=.85, edgecolors='k', linewidths=.4)
        fig.colorbar(sc, ax=ax, pad=.01, shrink=.9, label='порядок посева')
    ax.set_title('посев спор: %d затравок → %d клеток (размер точки — сколько слоёв дала затравка)' % (len(sd), sum(v[1] for v in sd.values())))
    for ax, u in zip(axs, US):
        cs = [c for c in cells if round(float(c['u']), 3) == u]
        for c in cs:
            G = np.array(c['G'], float); G[..., 0] = np.unwrap(G[..., 0], axis=0); m = G.shape[1]; pg = contour(G)
            for sh in SH:
                if pg[:, 0].max() + sh < -XR or pg[:, 0].min() + sh > XR: continue
                ax.fill(pg[:, 0] + sh, pg[:, 1], color=MC.get(m, '#ddd'), alpha=.45, ec=EC.get(m, 'k'), lw=.7)
                if m > 5:
                    for j in range(0, m, max(1, (m - 1) // 8)): ax.plot(G[:, j, 0] + sh, G[:, j, 1], '-', color=EC.get(m, 'k'), lw=.3, alpha=.6)
        ax.set_title('слой u = %+.1f — %d клеток' % (u, len(cs)))
    ax = axs[3]; vmax = np.nanquantile(VG, .99)
    for sh in SH:
        im = ax.pcolormesh(gx + sh, gw, VG.T, cmap='YlGnBu', shading='auto', vmin=0, vmax=vmax, alpha=.85, rasterized=True)
        ax.contour(gx + sh, gw, VG.T, levels=np.arange(1, vmax, 1.), colors='w', linewidths=.6, alpha=.7)
    segs = {u: [] for u in UC}
    for i in range(P.shape[1]):
        p = P[:, i]; mv = np.r_[True, np.any(np.diff(p, axis=0) != 0, 1)]; k = np.flatnonzero(mv)[-1] + 1; p = p[:k].copy(); p[:, 0] = np.unwrap(p[:, 0])
        for j in range(k - 1):
            if np.isnan(U[j, i]): continue
            for sh in SH: segs.setdefault(round(float(U[j, i]), 1), []).append(p[j:j + 2] + [sh, 0])
    for u, sg in segs.items(): ax.add_collection(LineCollection(sg, colors=UC.get(u, 'k'), linewidths=1.3, alpha=.25))
    for sh in SH: ax.plot(P[0, :, 0] + sh, P[0, :, 1], 'o', ms=3, color='k', alpha=.6)
    ax.set_title('V* (фон, белые линии через 1 с) + пути агента из %d стартов, дошли %d — цвет = управление на шаге' % (P.shape[1], int(np.isfinite(T).sum())))
    fig.colorbar(im, ax=axs[3], pad=.01, shrink=.9, label='V*, с')
    for ax in axs:
        for sh in SH: ax.add_patch(plt.Rectangle((-RHO + sh, -RHO), 2 * RHO, 2 * RHO, fill=False, ec='k', lw=2))
        ax.set_xlim(-XR, XR); ax.set_ylim(-WL, WL); ax.grid(alpha=.25)
        for v in (-np.pi, np.pi): ax.axvline(v, color='0.4', ls=':', lw=1)
    for ax in axs[3:]: ax.set_xlabel('θ  (повтор через 2π; пунктир — ±π)')
    for ax in (axs[0], axs[4]): ax.set_ylabel('ω')
    cnt = {k: sum(np.shape(c['G'])[1] == k for c in cells) for k in MC}
    h = [plt.Rectangle((0, 0), 1, 1, fc=MC[k], ec=EC[k], alpha=.6) for k in MC] + [plt.Line2D([], [], color=c, lw=4) for c in UC.values()]
    fig.legend(h, ['%d клонов (%d кл.)' % (k, cnt[k]) for k in MC] + ['u = %+.1f' % u for u in UC], loc='lower center', ncol=7, fontsize=12, frameon=False)
    fig.suptitle(title, fontsize=14); fig.subplots_adjust(left=.03, right=.99, top=.94, bottom=.07, wspace=.06, hspace=.12)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True); fig.savefig(out, dpi=80); plt.close(fig); return out
if __name__ == '__main__':
    import sys, pickle, json
    Z = np.load(sys.argv[1]); D = sys.argv[3]; C = pickle.load(open(os.path.join(D, 'cells.pkl'), 'rb')); S = json.load(open(os.path.join(D, 'status.json'))); q = S.get('q_ms') or {}
    t = '%s — %d клеток, %d узлов; T/эталон ср. %.4f, max %.2f; атлас %.0f с, solve %.0f с, запрос мед. %.2f с' % (os.path.basename(os.path.dirname(os.path.abspath(D))), len(C), S.get('nodes', 0), S['T_mean'], S['T_max'], S.get('t_build', 0), S.get('t_solve', 0), q.get('med', 0) / 1e3)
    print(draw(C, Z['gx'], Z['gw'], Z['VG'], Z['P'], Z['U'], Z['T'], t, sys.argv[2]))
