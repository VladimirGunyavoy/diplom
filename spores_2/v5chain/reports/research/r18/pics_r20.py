# research-20: атлас из сохранённого прогона (cells.pkl, paths.pkl): 3 слоя u + пути агента; цвет клетки — число клонов; θ ∈ [−XR, XR] с повторами
import os, sys, json, pickle, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
D = sys.argv[1]; OUT = sys.argv[2]; XR = float(os.environ.get('XR', 6.)); RHO = .1; WL = 3.5
C = pickle.load(open(os.path.join(D, 'cells.pkl'), 'rb')); P = pickle.load(open(os.path.join(D, 'paths.pkl'), 'rb')); S = json.load(open(os.path.join(D, 'status.json')))
US = sorted(set(round(c['u'], 3) for c in C)); MC = {5: '#9ecae1', 9: '#a1d99b', 17: '#fdd0a2', 33: '#fc9272'}; EC = {5: '#3182bd', 9: '#31a354', 17: '#e6550d', 33: '#de2d26'}
SH = [k * 2 * np.pi for k in range(-2, 3)]
def contour(G): return np.r_[G[:, 0], G[-1, 1:-1], G[::-1, -1], G[0, -2:0:-1]]
def unwrap_line(ax, p, **kw):
    x = np.unwrap(p[:, 0]); y = p[:, 1]
    for sh in SH: ax.plot(x + sh, y, **kw)
fig, axs = plt.subplots(2, 2, figsize=(22, 13), sharex=True, sharey=True); axs = axs.ravel()
for ax, u in zip(axs, US):
    cs = [c for c in C if round(c['u'], 3) == u]
    for c in cs:
        G = c['G'].copy(); G[..., 0] = np.unwrap(G[..., 0], axis=0); m = G.shape[1]; pg = contour(G)
        for sh in SH:
            if pg[:, 0].max() + sh < -XR or pg[:, 0].min() + sh > XR: continue
            ax.fill(pg[:, 0] + sh, pg[:, 1], color=MC.get(m, '#ddd'), alpha=.45, ec=EC.get(m, 'k'), lw=.7)
            if m > 5:
                for j in range(0, m, max(1, (m - 1) // 8)): ax.plot(G[:, j, 0] + sh, G[:, j, 1], '-', color=EC.get(m, 'k'), lw=.3, alpha=.6)
    ax.set_title('слой u = %+.1f — %d клеток' % (u, len(cs)))
ax = axs[3]
for c in C:
    G = c['G'].copy(); G[..., 0] = np.unwrap(G[..., 0], axis=0); pg = contour(G)
    for sh in SH: ax.fill(pg[:, 0] + sh, pg[:, 1], color='0.85', alpha=.25, ec='0.6', lw=.3)
for i, p in enumerate(P): unwrap_line(ax, p['p'], color=plt.cm.tab10(i % 10), lw=1.8)
for i, p in enumerate(P): ax.plot(p['p'][0, 0], p['p'][0, 1], 'o', color=plt.cm.tab10(i % 10), ms=6)
ax.set_title('все слои + пути агента из %d стартов' % len(P))
for ax in axs:
    for sh in SH: ax.add_patch(plt.Rectangle((-RHO + sh, -RHO), 2 * RHO, 2 * RHO, fill=False, ec='k', lw=2))
    ax.set_xlim(-XR, XR); ax.set_ylim(-WL, WL); ax.grid(alpha=.25)
    for v in (-np.pi, np.pi): ax.axvline(v, color='0.5', ls=':', lw=1)
for ax in axs[2:]: ax.set_xlabel('θ  (повтор через 2π; пунктир — ±π)')
for ax in axs[::2]: ax.set_ylabel('ω')
h = [plt.Rectangle((0, 0), 1, 1, fc=MC[k], ec=EC[k], alpha=.6) for k in MC]
cnt = {k: sum(c['G'].shape[1] == k for c in C) for k in MC}
fig.legend(h, ['%d клонов (%d кл.)' % (k, cnt[k]) for k in MC], loc='lower center', ncol=4, fontsize=13, frameon=False)
q = S.get('q_ms') or {}
fig.suptitle('Маятник u ≤ .3: жирные клетки-кольца + MADAPT .001 (прогон 56, зерно 0) — %d клеток, %d узлов; T/эталон ср. %.4f, max %.2f; атлас %.0f с, solve %.0f с, запрос (LOOK 20) мед. %.1f с' %
             (len(C), S.get('nodes', 0), S['T_mean'], S['T_max'], S.get('t_build', 0), S.get('t_solve', 0), q.get('med', 0) / 1e3), fontsize=14)
fig.subplots_adjust(left=.04, right=.99, top=.94, bottom=.07, wspace=.04, hspace=.1); fig.savefig(OUT, dpi=80); print('готово', OUT, cnt)
