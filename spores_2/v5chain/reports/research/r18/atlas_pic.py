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
def overlap(cells, n=12000):
    """доля площади слоя, покрытой ≥ 2 клетками этого слоя (по случайным точкам; контур клетки — боковины + торцы)."""
    from matplotlib.path import Path
    X = np.c_[np.random.default_rng(0).uniform(-np.pi, np.pi, n), np.random.default_rng(1).uniform(-3.5, 3.5, n)]; res = {}
    for u in sorted(set(round(float(c['u']), 3) for c in cells)):
        cnt = np.zeros(n, int)
        for c in cells:
            if round(float(c['u']), 3) != u: continue
            G = np.array(c['G'], float); G[..., 0] = np.unwrap(G[..., 0], axis=0); Pp = Path(contour(G)); ins = np.zeros(n, bool)
            for sh in (-2 * np.pi, 0., 2 * np.pi): ins |= Pp.contains_points(X + [sh, 0])
            cnt += ins
        res[u] = float((cnt >= 2)[cnt > 0].mean()) if (cnt > 0).any() else 0.
    return res
def stats_panel(ax, cells, st, ratio, T):
    ax.axis('off'); US = sorted(set(round(float(c['u']), 3) for c in cells)); ov = overlap(cells); q = st.get('q_ms') or {}
    nl = {u: sum(round(float(c['u']), 3) == u for c in cells) for u in US}; nn = {u: sum(int(np.prod(np.shape(c['G'])[:2])) for c in cells if round(float(c['u']), 3) == u) for u in US}
    cl = {k: sum(np.shape(c['G'])[1] == k for c in cells) for k in (5, 9, 17, 33)}
    r = np.asarray(ratio, float) if ratio is not None else None; rf = r[np.isfinite(r)] if r is not None else np.zeros(0)
    npar = len(set((int(c.get('seq', -1)), round(float(c['u']), 3)) for c in cells)) if all(c.get('seq', -1) not in (None, -1) for c in cells) else len(cells)
    rows = [('клеток / подклеток', '%d / %d' % (npar, len(cells))),
            ('подклеток по слоям ' + ' / '.join('%+.1f' % u for u in US), ' / '.join(str(nl[u]) for u in US)),
            ('узлов', '%d  (%s)' % (sum(nn.values()), ' / '.join('%dk' % round(nn[u] / 1e3) for u in US))),
            ('клонов 5/9/17/33 (клеток)', ' / '.join(str(cl[k]) for k in cl)),
            ('наложение: площадь под ≥2 кл.', ' / '.join('%d%%' % round(100 * ov[u]) for u in US)),
            ('дошли', '%d / %d' % (int(np.isfinite(T).sum()), len(T))),
            ('T/эталон ср. / мед.', '%.4f / %.4f' % (rf.mean(), np.median(rf)) if len(rf) else '—'),
            ('T/эталон p90 / max', '%.3f / %.3f' % (np.quantile(rf, .9), rf.max()) if len(rf) else '—'),
            ('переключений (мед.)', '%g' % st.get('sw_med', np.nan)),
            ('атлас / solve', '%.0f с / %.0f с' % (st.get('t_build', np.nan), st.get('t_solve', np.nan))),
            ('запрос мед. / p90 / max', '%.2f / %.2f / %.2f с' % (q.get('med', np.nan) / 1e3, q.get('p90', np.nan) / 1e3, q.get('max', np.nan) / 1e3) if q else '—'),
            ('итераций solve, узлов V=BIG', '%s, %s%%' % (st.get('iters', '—'), round(100 * st.get('big_nodes', np.nan), 1)))]
    tb = ax.table(cellText=[[a, b] for a, b in rows], colWidths=[.55, .45], loc='upper center', cellLoc='left', bbox=[0, .42, 1, .58]); tb.auto_set_font_size(False); tb.set_fontsize(13)
    for (i, j), cell in tb.get_celld().items(): cell.set_edgecolor('0.8'); cell.set_facecolor('#f4f4f4' if i % 2 else 'white')
    if len(rf):
        ah = ax.inset_axes([.08, .03, .88, .32]); ah.hist(rf, bins=np.linspace(min(.98, rf.min()), max(1.2, min(rf.max(), 1.6)), 40), color='#3182bd', alpha=.8)
        ah.axvline(1, color='k', lw=1); ah.axvline(rf.mean(), color='#de2d26', lw=1.5, label='среднее %.4f' % rf.mean()); ah.axvline(np.median(rf), color='#31a354', lw=1.5, ls='--', label='медиана %.4f' % np.median(rf)); ah.set_xlabel('T/эталон по стартам'); ah.set_ylabel('стартов'); ah.legend(frameon=False)
    ax.set_title('статистика прогона')
def draw(cells, gx, gw, VG, P, U, T, title, out, XR=6., WL=3.5, st=None, ratio=None):
    """cells: список dict(u, G) ; VG: V* на сетке gx × gw ; P: (шаги+1, n, 2) ; U: (шаги+1, n)"""
    plt.rcParams.update({'font.size': 12}); VG = np.where(VG > 500, np.nan, VG)
    US = sorted(set(round(float(c['u']), 3) for c in cells))
    fig = plt.figure(figsize=(30, 14)); gs = fig.add_gridspec(2, 3); a0 = fig.add_subplot(gs[0, 0])
    axs = [a0, fig.add_subplot(gs[0, 1], sharex=a0, sharey=a0), fig.add_subplot(gs[0, 2], sharex=a0, sharey=a0), fig.add_subplot(gs[1, 1], sharex=a0, sharey=a0), fig.add_subplot(gs[1, 0], sharex=a0, sharey=a0)]; axS = fig.add_subplot(gs[1, 2])   # research-20 (слово пользователя): пути — в одну колонку, справа статистика
    ax = axs[4]; sd = {}                                                                          # research-20 (слово пользователя): посев спор — только затравки: цвет = порядок посева, размер = во скольких слоях затравка дала клетку
    for i, c in enumerate(cells):
        if c.get('p0') is None: continue
        k = int(c['seq']) if c.get('seq', -1) is not None and int(c.get('seq', -1)) >= 0 else i; sd.setdefault(k, [np.array(c['p0'], float), 0.]); pg_ = contour(np.array(c['G'], float)); sd[k][1] += .5 * abs(np.dot(pg_[:, 0], np.roll(pg_[:, 1], 1)) - np.dot(pg_[:, 1], np.roll(pg_[:, 0], 1)))   # слово пользователя: размер точки — площадь выросших из затравки клеток (сумма по слоям и подклеткам)
    if sd:
        ks = sorted(sd); Pq = np.array([sd[k][0] for k in ks]); ar = np.array([sd[k][1] for k in ks]); oc = np.arange(len(ks))
        for sh in SH: sc = ax.scatter(Pq[:, 0] + sh, Pq[:, 1], c=oc, cmap='viridis', s=8 + 600 * ar / max(ar.max(), 1e-9), alpha=.85, edgecolors='k', linewidths=.4)
        fig.colorbar(sc, ax=ax, pad=.01, shrink=.9, label='порядок посева')
    ax.set_title('посев: %d затравок (размер точки — площадь выросших клеток, цвет — порядок)' % len(sd), fontsize=11)
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
    ax.set_title('V* (фон) + пути агента из %d стартов; цвет = управление' % P.shape[1])
    fig.colorbar(im, ax=axs[3], pad=.01, shrink=.9, label='V*, с')
    stats_panel(axS, cells, st or {}, ratio, T)
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
    A_ = np.load(os.path.join(D, 'agent.npz')); ra = A_['T'] / A_['ref'] if 'ref' in A_ else None
    print(draw(C, Z['gx'], Z['gw'], Z['VG'], Z['P'], Z['U'], Z['T'], t, sys.argv[2], st=S, ratio=ra))
