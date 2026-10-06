# research-19: MODE=map — карта растяжения, 3 слоя В РЯД (1×3); MODE=whole — клетка K целиком в развёрнутом θ (без обрыва на ±π) + w(t) теми же цветами.
import os, sys, numpy as np
sys.path.insert(0, os.environ['CELLS7']); import grow_cells2d as G
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.colors import LogNorm, Normalize
plt.rcParams.update({'font.size': 11})
e_ = np.linspace(-G.RHO, G.RHO, 41); G.BARRIER = cKDTree(np.r_[np.c_[e_, e_ * 0 - G.RHO], np.c_[e_, e_ * 0 + G.RHO], np.c_[e_ * 0 - G.RHO, e_], np.c_[e_ * 0 + G.RHO, e_]])
rng = np.random.default_rng(int(os.environ.get('SEED', 0))); cells = []
for u in G.US: l, _ = G.build_layer(u, rng); cells += l
wl = lambda r_: float(np.linalg.norm(np.diff(r_, axis=0), axis=1).sum()) + 1e-12
OUT = os.environ['OUT']; XR = float(os.environ.get('XR', 5.)); SHS = (0., 2 * np.pi, -2 * np.pi, 4 * np.pi, -4 * np.pi)   # XR: показывать θ ∈ [−XR, XR] с повторами через период (слово пользователя)
if os.environ['MODE'] == 'map':
    BAR = np.load(os.environ['BARRIER']); st = np.array([max(wl(c.G[-1]) / wl(c.G[0]), wl(c.G[0]) / wl(c.G[-1])) for c in cells]); nrm = LogNorm(1, 15); cm = plt.cm.YlOrRd; from matplotlib.path import Path
    gq = np.stack(np.meshgrid(np.linspace(-np.pi, np.pi, 161), np.linspace(-G.WL, G.WL, 141)), -1).reshape(-1, 2); print('θ клеток: min %.2f max %.2f' % (min(c.G[..., 0].min() for c in cells), max(c.G[..., 0].max() for c in cells)))
    fig, axs = plt.subplots(1, 3, figsize=(27, 7), sharey=True, constrained_layout=True)
    for ax, u in zip(axs, G.US):
        cov = np.zeros(len(gq), bool)
        for c, sv in zip(cells, st):
            if c.u != u: continue
            px, py = np.r_[c.G[:, 0, 0], c.G[::-1, -1, 0]], np.r_[c.G[:, 0, 1], c.G[::-1, -1, 1]]
            for sh in SHS:                                                                                                   # копии через период: клетка, ушедшая за ±π, рисуется и с другой стороны
                if px.min() + sh > XR or px.max() + sh < -XR: continue
                ax.fill(px + sh, py, fc=cm(nrm(sv))[:3] + (.55,), lw=.8, ec='0.2'); cov = cov | Path(np.c_[px + sh, py]).contains_points(gq)
        print('слой u %+g: закрашено %.3f поля' % (u, cov.mean()))
        [ax.plot(BAR[:, 0] + sh, BAR[:, 1], '.', ms=2.5, color='tab:blue') for sh in SHS[:3]]; [ax.axvline(x, color='0.4', ls=':', lw=1) for x in (-np.pi, np.pi)]; ax.set_xlim(-XR, XR); ax.set_ylim(-G.WL, G.WL); ax.set_xlabel('θ')
        ax.set_title('u = %+.1f  (%d клеток)' % (u, sum(c.u == u for c in cells)))
    axs[0].set_ylabel('ω'); fig.suptitle(os.environ.get('TITLE', 'Растяжение среза клетки'), fontsize=14)
    fig.colorbar(plt.cm.ScalarMappable(nrm, cm), ax=axs, shrink=.8, pad=.01, label='во сколько раз растянут срез'); fig.savefig(OUT, dpi=95)
    print('готово', OUT, 'растяжение кв.', np.quantile(st, [.5, .9, .99, 1]).round(1).tolist(), '>3:', int((st > 3).sum()), 'клеток', len(cells), flush=True)
elif os.environ['MODE'] == 'dead':                                                                # research-19: где мёртвые узлы NORMFRONT (столбец не пересёк нормаль через центр строки)
    fig, axs = plt.subplots(2, 3, figsize=(27, 14), constrained_layout=True); nd = 0; rep = []
    for r, (xl, yl) in enumerate((((-XR, XR), (-G.WL, G.WL)), ((-1.3, 1.3), (-1., 1.)))):
        for ax, u in zip(axs[r], G.US):
            for k, c in enumerate(cells):
                if c.u != u: continue
                D = getattr(c, 'DEAD', None); has = D is not None and D.any(); px, py = np.r_[c.G[:, 0, 0], c.G[::-1, -1, 0]], np.r_[c.G[:, 0, 1], c.G[::-1, -1, 1]]
                for sh in SHS:
                    if px.min() + sh > xl[1] or px.max() + sh < xl[0]: continue
                    ax.fill(px + sh, py, fc=(1., .75, .3, .55) if has else (.45, .65, .9, .3), ec='0.2', lw=.8)
                    if has: ax.plot(c.G[..., 0][D] + sh, c.G[..., 1][D], 'o', ms=3 if r == 0 else 6, color='tab:red', zorder=5)
                    if has and r == 1: ax.plot(c.G[..., 0][~D] + sh, c.G[..., 1][~D], '.', ms=3, color='0.3', zorder=4)
                if has and r == 0: nd += int(D.sum()); rep.append((k, u, int(D.sum()), int(D.size), np.round(c.c, 2).tolist()))
            ax.add_patch(plt.Rectangle((-G.RHO, -G.RHO), 2 * G.RHO, 2 * G.RHO, fill=False, ec='tab:green', lw=2.5, zorder=6)); ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_xlabel('θ')
            ax.set_title(('u = %+.1f  (%d клеток)' % (u, sum(c.u == u for c in cells))) if r == 0 else 'u = %+.1f — у цели' % u)
        axs[r][0].set_ylabel('ω')
    fig.suptitle(os.environ.get('TITLE', 'Мёртвые узлы') + '   ·   красные точки — мёртвые узлы, оранжевые клетки — с мёртвыми узлами, зелёный квадрат — цель', fontsize=14); fig.savefig(OUT, dpi=85)
    print('готово', OUT, 'клеток', len(cells), 'с мёртвыми узлами', len(rep), 'мёртвых узлов', nd); [print('  клетка %d u %+g: мёртвых %d из %d, центр %s' % t) for t in rep]
else:
    K = int(os.environ.get('K', 160)); c = cells[K]; seg = c.c + np.linspace(-(1 + G.HALO) * c.r, (1 + G.HALO) * c.r, 101)[:, None] * c.n
    nmx = int(float(os.environ.get('TSPAN', 6.)) / G.DTN); T, Y = [0.], [seg.copy()]
    for sg in (1., -1.):
        y = seg.copy()
        for i in range(1, nmx + 1): y = G.step(y, c.u, sg); T.append(sg * i * G.DTN); Y.append(y.copy())                 # step не заворачивает θ — срезы в развёрнутом угле
    o = np.argsort(T); T = np.array(T)[o]; Y = np.array(Y)[o]; W = np.array([wl(y) for y in Y]); i0 = int(np.argmin(abs(T))); Wn = W / W[i0]
    ctr = Y[:, 50]; sp = np.array([np.linalg.norm(G.f(p, c.u)) for p in ctr]); t0, t1 = -c.nb * G.DTN, c.nf * G.DTN; inc = (T >= t0 - 1e-9) & (T <= t1 + 1e-9)
    cm = plt.cm.turbo; nrm = Normalize(T[0], T[-1]); ks = [k for k in range(len(T)) if abs(T[k] / .5 - round(T[k] / .5)) < 1e-6]
    fig, axs = plt.subplots(2, 1, figsize=(15, 11), gridspec_kw=dict(height_ratios=[1.25, 1]), constrained_layout=True); ax = axs[0]
    th = np.linspace(ctr[:, 0].min() - .6, ctr[:, 0].max() + .6, 1200)
    for sgn in (1, -1): ax.plot(th, sgn * 2 * np.abs(np.sin(th / 2)), '-', color='0.55', lw=1.2, label='сепаратрисы (u = 0)' if sgn == 1 else None)
    ax.fill(np.r_[Y[:, 0, 0], Y[::-1, -1, 0]], np.r_[Y[:, 0, 1], Y[::-1, -1, 1]], color='0.93', label='тот же срез до и после клетки (±%g с)' % (nmx * G.DTN))
    ax.fill(np.r_[Y[inc, 0, 0], Y[inc][::-1, -1, 0]], np.r_[Y[inc, 0, 1], Y[inc][::-1, -1, 1]], color='0.72', label='клетка %d целиком (%.1f с)' % (K, t1 - t0))
    for k in ks: ax.plot(Y[k, :, 0], Y[k, :, 1], '-', lw=3 if inc[k] else 1.5, color=cm(nrm(T[k])))
    ax.plot(seg[:, 0], seg[:, 1], 'k-', lw=4, label='срез посева, t = 0')
    for m in range(int(np.floor(th[0] / np.pi)), int(np.ceil(th[-1] / np.pi)) + 1):
        x = m * np.pi
        if not th[0] <= x <= th[-1]: continue
        if m % 2: ax.axvline(x, color='tab:blue', ls=':', lw=1.5); ax.text(x, -1.75, ' НИЗ\n маятника', color='tab:blue', ha='left', va='bottom')
        else: ax.add_patch(plt.Rectangle((x - G.RHO, -G.RHO), 2 * G.RHO, 2 * G.RHO, fill=False, ec='tab:red', lw=2)); ax.text(x, -.35, 'ВЕРХ (седло, цель)', color='tab:red', ha='center', va='top')
    ax.set_xlim(th[0], th[-1]); ax.set_ylim(-1.8, 2.6); ax.set_xlabel('θ, развёрнутый (без обрыва на ±π); верх маятника — θ = 0 и θ = −2π — одна и та же точка'); ax.set_ylabel('ω')
    ax.set_title('Клетка %d (u = %+g) целиком: срезы через 1.5 с, цвет — время' % (K, c.u)); ax.legend(loc='upper right', fontsize=10)
    ax = axs[1]; ax.axvspan(t0, t1, color='0.88', zorder=0, label='клетка'); ax.plot(T, Wn, 'k-', lw=2, label='длина среза w(t) / w(0)')
    ax.plot(T, sp[i0] / sp, '--', color='tab:purple', lw=2, label='1 / скорость потока (норм. к t = 0)'); ax.scatter(T[ks], Wn[ks], c=T[ks], cmap=cm, norm=nrm, s=70, zorder=5, ec='k', lw=.5)
    cr = np.flatnonzero(np.diff(np.floor((ctr[:, 0] + np.pi) / (2 * np.pi))) != 0)                                      # центр проходит θ = π + 2πm — низ маятника
    for j, k in enumerate(cr): ax.axvline(T[k], color='tab:blue', ls=':', lw=1.5); ax.text(T[k], Wn.max(), ' проход НИЗА', color='tab:blue', va='top')
    tb = T[cr[0]] if len(cr) else np.nan
    ax.set_yscale('log'); ax.set_xlabel('локальное время клетки t, с'); ax.set_ylabel('w(t) / w(0)'); ax.legend(loc='lower right', fontsize=10); ax.grid(alpha=.3)
    fig.colorbar(plt.cm.ScalarMappable(nrm, cm), ax=axs, shrink=.6, pad=.01, label='t, с'); fig.savefig(OUT, dpi=95)
    print('готово', OUT, 'клетка t', t0, t1, '| min w при t =', round(float(T[np.argmin(W)]), 2), 'θ центра там', round(float(ctr[np.argmin(W), 0]), 2), '| макс. скорость при t =', round(float(tb), 2), 'θ', round(float(ctr[np.argmax(sp), 0]), 2),
          '| θ центра: t0', round(float(ctr[inc][0, 0]), 2), 't=0', round(float(ctr[i0, 0]), 2), 't1', round(float(ctr[inc][-1, 0]), 2), flush=True)
