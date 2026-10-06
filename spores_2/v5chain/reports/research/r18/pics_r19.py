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
def poly(c): g = c.G; return np.r_[g[:, 0], g[-1, 1:-1], g[::-1, -1], g[0, -2:0:-1]]                 # контур клетки: боковины + торцы ломаной по всем узлам
def curv(c):                                                                                         # кривизна торцов: max по строкам (отклонение узлов от хорды) / длина хорды
    g = c.G; a_, b_ = g[:, 0], g[:, -1]; ch = b_ - a_; L = np.linalg.norm(ch, axis=1) + 1e-12; nrm_ = np.c_[-ch[:, 1], ch[:, 0]] / L[:, None]
    return float((np.abs(((g - a_[:, None]) * nrm_[:, None]).sum(-1)).max(1) / L).max())
OUT = os.environ['OUT']; XR = float(os.environ.get('XR', 5.)); SHS = (0., 2 * np.pi, -2 * np.pi, 4 * np.pi, -4 * np.pi)   # XR: показывать θ ∈ [−XR, XR] с повторами через период (слово пользователя)
if os.environ['MODE'] == 'map':
    BAR = np.load(os.environ['BARRIER']); CV = os.environ.get('COLOR', 'stretch') == 'curv'; st = np.array([curv(c) if CV else max(wl(c.G[-1]) / wl(c.G[0]), wl(c.G[0]) / wl(c.G[-1])) for c in cells]); nrm = Normalize(0, .15) if CV else LogNorm(1, 15); cm = plt.cm.YlOrRd; from matplotlib.path import Path
    gq = np.stack(np.meshgrid(np.linspace(-np.pi, np.pi, 161), np.linspace(-G.WL, G.WL, 141)), -1).reshape(-1, 2); print('θ клеток: min %.2f max %.2f' % (min(c.G[..., 0].min() for c in cells), max(c.G[..., 0].max() for c in cells)))
    fig, axs = plt.subplots(1, 3, figsize=(27, 7), sharey=True, constrained_layout=True)
    for ax, u in zip(axs, G.US):
        cov = np.zeros(len(gq), bool)
        for c, sv in zip(cells, st):
            if c.u != u: continue
            px, py = poly(c).T
            for sh in SHS:                                                                                                   # копии через период: клетка, ушедшая за ±π, рисуется и с другой стороны
                if px.min() + sh > XR or px.max() + sh < -XR: continue
                ax.fill(px + sh, py, fc=cm(nrm(sv))[:3] + (.55,), lw=.8, ec='0.2'); cov = cov | Path(np.c_[px + sh, py]).contains_points(gq)
        print('слой u %+g: закрашено %.3f поля' % (u, cov.mean()))
        [ax.plot(BAR[:, 0] + sh, BAR[:, 1], '.', ms=2.5, color='tab:blue') for sh in SHS[:3]]; [ax.axvline(x, color='0.4', ls=':', lw=1) for x in (-np.pi, np.pi)]; ax.set_xlim(-XR, XR); ax.set_ylim(-G.WL, G.WL); ax.set_xlabel('θ')
        ax.set_title('u = %+.1f  (%d клеток)' % (u, sum(c.u == u for c in cells)))
    axs[0].set_ylabel('ω'); fig.suptitle(os.environ.get('TITLE', 'Растяжение среза клетки'), fontsize=14)
    fig.colorbar(plt.cm.ScalarMappable(nrm, cm), ax=axs, shrink=.8, pad=.01, label='кривизна торца: прогиб / хорда' if CV else 'во сколько раз растянут срез'); fig.savefig(OUT, dpi=95)
    print('готово', OUT, 'метрика кв.', np.quantile(st, [.5, .9, .99, 1]).round(3).tolist(), 'клеток', len(cells), 'TOP', ','.join(str(k) for k in np.argsort(-st)[:int(os.environ.get('NTOP', 6))]), np.round(np.sort(st)[::-1][:6], 3).tolist(), flush=True)
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
elif os.environ['MODE'] == 'clones':                                                              # research-19: одна клетка NORMFRONT подробно — каждый клон (столбец), центральная линия, нормали строк, узлы живые/мёртвые
    for K in [int(x) for x in os.environ['K'].split(',')]:
        c = cells[K]; u = c.u; nr, nc = c.G.shape[:2]; offs = c.off + c.sn; D = getattr(c, 'DEAD', np.zeros((nr, nc), bool)); sub = 16; h = G.DTN / sub
        X0 = np.vstack([c.p0 + offs[:, None] * c.n, c.p0[None]]); ext = int(float(os.environ.get('EXT', 1.5)) / G.DTN)
        def traj(sg, n):
            X = [X0]
            for _ in range(n * sub): X.append(G.rk4(X[-1], u, sg * h, 1))
            return np.array(X)
        F = traj(1., c.nf + ext); B = traj(-1., c.nb + ext); TR = np.concatenate([B[::-1], F[1:]]); tt = (np.arange(len(TR)) - (len(B) - 1)) * h; inc = (tt >= -c.nb * G.DTN - 1e-9) & (tt <= c.nf * G.DTN + 1e-9)
        ths = np.arcsin(-u); eq = np.array([ths, 0.]); lam = np.sqrt(np.cos(ths)); seps = []                                      # седло слоя: sin θ + u = 0; λ = ±√cos θ*
        for sg, ev in ((1., (1, lam)), (1., (-1, -lam)), (-1., (1, -lam)), (-1., (-1, lam))):
            y = eq + 1e-3 * np.array(ev); P = [y]
            for _ in range(int(8 / h)): P.append(G.rk4(P[-1], u, sg * h, 1))
            seps.append((sg, np.array(P)))
        dj = np.flatnonzero(D.any(0)); xs, ys = c.G[..., 0], c.G[..., 1]; pad = .25
        boxes = [(xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad)]
        if D.any(): boxes.append((xs[D].min() - .2, xs[D].max() + .2, ys[D].min() - .2, ys[D].max() + .2))
        fig, axs = plt.subplots(1, len(boxes), figsize=(13 * len(boxes), 11), constrained_layout=True); axs = np.atleast_1d(axs); cmc = plt.cm.coolwarm
        for a_, (ax, bx) in enumerate(zip(axs, boxes)):
            for sg, P in seps: ax.plot(P[:, 0], P[:, 1], '-', color='tab:green' if sg < 0 else 'tab:purple', lw=1.6, alpha=.8)
            ax.plot([], [], '-', color='tab:green', label='сепаратриса, ВХОДЯЩАЯ в седло слоя'); ax.plot([], [], '-', color='tab:purple', label='сепаратриса, ВЫХОДЯЩАЯ из седла слоя')
            ax.plot(*eq, '*', ms=22, color='gold', mec='k', zorder=9, label='седло слоя u = %+g (θ = %.2f)' % (u, ths))
            for j in range(nc):
                col = cmc(j / max(1, nc - 1)); ax.plot(TR[:, j, 0], TR[:, j, 1], '-', lw=.7, color=col, alpha=.5); ax.plot(TR[inc, j, 0], TR[inc, j, 1], '-', lw=1.8, color=col)
            ax.plot(TR[:, -1, 0], TR[:, -1, 1], 'k-', lw=1, alpha=.5); ax.plot(TR[inc, -1, 0], TR[inc, -1, 1], 'k-', lw=4, label='ЦЕНТРАЛЬНАЯ линия (в пределах клетки — жирно)', zorder=6)
            L = 1.3 * np.abs(offs).max() * max(2., float(os.environ.get('NL', 3.)))
            for i in range(-c.nb, c.nf + 1):
                ci = TR[(len(B) - 1) + i * sub, -1]; nn = G.normal(ci, u)
                if G.NORMFRONT == 2: ax.plot(c.G[i + c.nb, :, 0], c.G[i + c.nb, :, 1], '-', color='0.3', lw=1.1, zorder=2)
                else: ax.plot([ci[0] - L * nn[0], ci[0] + L * nn[0]], [ci[1] - L * nn[1], ci[1] + L * nn[1]], '-', color='0.45', lw=.6, zorder=2)
                ax.plot(*ci, 'ko', ms=5, zorder=7)
            ax.plot([], [], '-', color='0.45', lw=.8, label='торец/строка клетки' + (' — ломаная по нормалям клонов' if G.NORMFRONT == 2 else ' — нормаль к потоку в точке центра'))
            ax.plot(xs[~D], ys[~D], 'o', ms=5, mfc='w', mec='k', zorder=8, label='узел клетки (клон пересёк нормаль)'); ax.plot(xs[D], ys[D], 'X', ms=10, color='tab:red', mec='k', zorder=9, label='МЁРТВЫЙ узел (клон нормаль не пересёк)')
            ax.plot(X0[:-1, 0], X0[:-1, 1], 's', ms=7, color='k', zorder=8, label='посев: старт клонов (t = 0)')
            ax.add_patch(plt.Rectangle((-G.RHO, -G.RHO), 2 * G.RHO, 2 * G.RHO, fill=False, ec='tab:cyan', lw=2.5, label='цель'))
            ax.set_xlim(bx[0], bx[1]); ax.set_ylim(bx[2], bx[3]); ax.set_xlabel('θ'); ax.set_ylabel('ω'); ax.set_title('вся клетка' if a_ == 0 else 'крупно: где клоны не пересекают нормаль')
            if a_ == 0: ax.legend(loc='best', fontsize=10)
        fig.suptitle('Клетка %d (u = %+g): %d клонов (цвет — номер клона, синий → красный), %d строк; мёртвых узлов %d, в клонах %s' % (K, u, nc, nr, int(D.sum()), dj.tolist()), fontsize=14); fig.savefig(OUT.replace('KK', str(K)), dpi=85); plt.close(fig)
        print('готово', OUT.replace('KK', str(K)), 'кривизна', round(curv(c), 3), 'G', c.G.shape, 'nb nf', c.nb, c.nf, 'p0', np.round(c.p0, 3).tolist(), 'offs', np.round(offs, 3).tolist(), 'седло', round(float(ths), 3)); print('мёртвые по клонам:', D.sum(0).tolist()); print('мёртвые по строкам:', D.sum(1).tolist())
        print('по какую сторону от входящей сепаратрисы клоны (энергия относительно седла, знак):', np.round([.5 * p[1] ** 2 + np.cos(p[0]) - u * p[0] - (np.cos(ths) - u * ths) for p in X0], 4).tolist())
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
