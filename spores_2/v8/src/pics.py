"""Картинки прогона (v8; слово пользователя 2026-10-12: «такие же, как в прошлых версиях, в папку система/дата/эксперимент/…/последний запуск, сама обновляется»).
Образцы: v5chain/reports/research/r18/atlas_pic.py (v7: клетки слоёв, фон V* + пути агента цветом управления, статистика, гистограмма T/T_эталон), v5chain/reports/figures/T_phase.png, error_map.png.
Чистый numpy + matplotlib (Agg), без Ursina: вызывается из calc_worker (подпроцесс), окно не морозит.
Папки: <claude-work>/projects/spore/img/<дата>/exp<NNN>/ (запуск окна) и .../<дата>/last_exp/ (копия последнего запуска); этапы этого запуска перерисовывают файлы.
Файлы: cells.png (клетки слоёв: контуры, строки, узлы, цель, рамки зоны и буфера), error_<calc>.png (карта V*−T_ref), traj_<calc>.png (V* фоном + пути агента, цвет по управлению), hist_<calc>.png (T/T_ref: гистограмма, мед/p90/max, доля дошедших)."""
import os, glob, shutil, datetime
import numpy as np

UCOL = {1: '#e63030', -1: '#b04dff', 0: '#1aa82a'}            # управление: +1 красный, −1 фиолетовый, 0 зелёный (как в окне)


def new_experiment(root):
    """папка запуска окна: <root>/<дата>/exp<NNN> (номер — счётчик запусков за день) и <root>/<дата>/last_exp — копия последнего запуска (НЕ очищается: файлы перезаписываются на месте; было: очищается). root = claude-work/projects/spore/img (слово пользователя 2026-10-12). → (exp_dir, [exp, last_exp])"""
    day = os.path.join(root, datetime.date.today().isoformat()); os.makedirs(day, exist_ok=True)
    k = 1 + max([int(os.path.basename(p)[3:]) for p in glob.glob(os.path.join(day, 'exp[0-9][0-9][0-9]'))] or [0]); exp = os.path.join(day, 'exp%03d' % k); last = os.path.join(day, 'last_exp')
    os.makedirs(exp, exist_ok=True)
    os.makedirs(last, exist_ok=True); return exp, [exp, last]


def _plt():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; return plt


def _save(fig, name, dirs):
    import io
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=110); data = buf.getvalue()
    for d in dirs:                                                                          # перезапись НА МЕСТЕ (тот же файл/inode): не удалять и не переименовывать — открытая в просмотрщике картинка не закрывается
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name), 'r+b' if os.path.exists(os.path.join(d, name)) else 'wb') as f: f.write(data); f.truncate()
    _plt().close(fig)


def _frames(ax, G):
    from matplotlib.patches import Rectangle, Circle
    xl = np.asarray(G.g.XLV, float); gm = np.broadcast_to(np.asarray(getattr(G.g, 'GMV', 0.), float), xl.shape)
    ax.add_patch(Rectangle((-xl[0], -xl[1]), 2 * xl[0], 2 * xl[1], fill=False, ec='#2a9d3a', ls='--', lw=1.3, label='work zone ±(%g, %g)' % (xl[0], xl[1])))
    ax.add_patch(Rectangle((-xl[0] - gm[0], -xl[1] - gm[1]), 2 * (xl[0] + gm[0]), 2 * (xl[1] + gm[1]), fill=False, ec='#777777', ls=':', lw=1.3, label='growth limit ±(%g, %g)' % (xl[0] + gm[0], xl[1] + gm[1])))
    ax.add_patch(Circle((0, 0), G.rho, fill=False, ec='k', lw=1.8)); ax.set_xlabel('x'); ax.set_ylabel('v'); ax.grid(alpha=.2)
    return xl, xl + gm


def contour(Gc):
    return np.r_[Gc[:, 0], Gc[-1, 1:-1], Gc[::-1, -1], Gc[0, -2:0:-1]]


def cells_png(G, cl, dirs, title=''):
    """клетки каждого слоя на фазовой плоскости: контур, строки, узлы, затравка (центр), цель, рамки"""
    plt = _plt(); us = [u for u in cl if len(cl[u])]; fig, axs = plt.subplots(max(len(us), 1), 1, figsize=(8.5, 5.0 * max(len(us), 1)), squeeze=False)
    for ax, u in zip(axs[:, 0], us):                                   # слои столбиком: ширина 8.5" = 935 px (было в одну строку 7.2"×n, 1584 px)
        n_nodes = 0
        for k, c in enumerate(cl[u]):
            Gc = np.asarray(c['G'], float); n_nodes += Gc.shape[0] * Gc.shape[1]; ct = contour(Gc); ax.fill(ct[:, 0], ct[:, 1], color=plt.cm.tab10(k % 10), alpha=.22, lw=0); ax.plot(ct[:, 0], ct[:, 1], color=plt.cm.tab10(k % 10), lw=1.2)
            for i in range(0, Gc.shape[0], 2): ax.plot(Gc[i, :, 0], Gc[i, :, 1], color='#4477aa', lw=.35, alpha=.7)
            ax.plot(Gc[..., 0].ravel(), Gc[..., 1].ravel(), '.', ms=1.6, color='#223355'); ax.plot(*np.asarray(c['c'], float)[:2], '*', ms=11, color=plt.cm.tab10(k % 10), mec='k', mew=.6); ax.annotate('%d' % (k + 1), np.asarray(c['c'], float)[:2], xytext=(4, 4), textcoords='offset points', fontsize=8)
        xl, xb = _frames(ax, G); ax.set_xlim(-xb[0] - .3, xb[0] + .3); ax.set_ylim(-xb[1] - .3, xb[1] + .3); ax.set_title('layer u = %+g: %d cells, %d nodes' % (u, len(cl[u]), n_nodes), fontsize=10)
    axs[0][0].legend(fontsize=7, loc='lower left'); fig.suptitle('cells (spores) per layer  ' + title, fontsize=11); fig.tight_layout(); _save(fig, 'cells.png', dirs)


TREF_SMALL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', '_tref_cache')
TREF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tref_cache')      # кэш T_ref на диске (в git): T_ref зависит только от точек и RHO, не от графа


def t_ref_cached(X):
    """solve8.t_ref(X)[0] с кэшем по хэшу (X, RHO, константы ref_di_disk): сетка карты ошибки фиксирована — T_ref (~6 мин на 16 тыс. точек) считается один раз; значения те же побитно"""
    import hashlib
    from .algo import solve8, ref_di_disk as R
    X = np.ascontiguousarray(X, float); rho = float(os.environ.get('RHO', .2)); key = hashlib.md5(X.tobytes() + repr((X.shape, rho, R.TMAX1, R.GRID0, R.ZOOM_N, R.ZOOM_ROUNDS)).encode()).hexdigest()[:20]; f = os.path.join(TREF_DIR if len(X) >= 1000 else TREF_SMALL, '%d_%s.npy' % (len(X), key))                # большие (сетка карты ошибки) — в git; малые (старты траекторий) — runs/ (не в git)
    try: return np.load(f)
    except Exception: pass
    T = np.asarray(solve8.t_ref(X)[0], float)
    try: os.makedirs(os.path.dirname(f), exist_ok=True); np.save(f, T)
    except Exception: pass
    return T


def _grid(G, n=140):
    xl = np.asarray(G.g.XLV, float); xs = np.linspace(-xl[0], xl[0], n); vs = np.linspace(-xl[1], xl[1], int(n * xl[1] / xl[0]) + 1); XX, VV = np.meshgrid(xs, vs); X = np.stack([XX.ravel(), VV.ravel()], 1)
    Vs = G.Vlayers(X).min(1); Vs = np.where(np.isfinite(Vs), Vs, np.nan); return xs, vs, X, Vs.reshape(XX.shape)


def error_png(G, dirs, calc_name, extra=''):
    """карта V*−T_ref по рабочей зоне"""
    from . algo import solve8
    plt = _plt(); xs, vs, X, V = _grid(G); Tr = t_ref_cached(X).reshape(V.shape); e = V - Tr; fin = np.isfinite(e) & (np.linalg.norm(X, axis=1).reshape(V.shape) > G.rho)
    fig, ax = plt.subplots(figsize=(8.5, 5.6)); p = ax.pcolormesh(xs, vs, e, cmap='RdBu_r', vmin=-.3, vmax=.3, shading='auto'); plt.colorbar(p, ax=ax, label='V* − T_ref, s (blue: under, red: over)'); _frames(ax, G)
    ax.set_title('%s: V* − T_ref; covered %.0f%%, under (<-.03) %.1f%%, over (>+.03) %.1f%%, median %.3f %s' % (calc_name, 100 * fin.sum() / max((np.linalg.norm(X, axis=1) > G.rho).sum(), 1), 100 * (e[fin] < -.03).mean() if fin.any() else 0, 100 * (e[fin] > .03).mean() if fin.any() else 0, np.nanmedian(e[fin]) if fin.any() else 0, extra), fontsize=9); ax.set_aspect('equal'); fig.tight_layout()
    return fig, ax


def _name(calc): return {'v': 'V old', 'q': 'Q lin', 'qquad': 'Q quad'}.get(calc, calc)


def error_map(G, dirs, calc):
    fig, ax = error_png(G, dirs, _name(calc)); _save(fig, 'error_%s.png' % calc, dirs)


def traj_png(G, res, dirs, calc, stat=''):
    """V* фоном + пути агента, цвет по управлению (+1 красный, −1 фиолетовый, 0 зелёный), старты точками"""
    from matplotlib.collections import LineCollection
    plt = _plt(); xs, vs, X, V = _grid(G); fig, ax = plt.subplots(figsize=(8.8, 5.8)); p = ax.pcolormesh(xs, vs, V, cmap='YlGnBu_r', shading='auto', alpha=.8); plt.colorbar(p, ax=ax, label='V*, s'); segs, cols = [], []
    for P, U in zip(res['paths'], res['U']):
        if len(P) < 2: continue
        U = np.asarray(U, float).copy(); U[0] = U[1]; col = np.where(U > 1e-9, 0, np.where(U < -1e-9, 1, 2)); cc = [UCOL[1], UCOL[-1], UCOL[0]]
        segs.append(np.stack([P[:-1, :2], P[1:, :2]], 1)); cols += [cc[k] for k in col[1:]]
    if segs: ax.add_collection(LineCollection(np.concatenate(segs), colors=cols, lw=.8, alpha=.75))
    X0 = np.asarray(res['X0']); st = np.asarray(res['status'], object); ax.plot(X0[st == 'reached', 0], X0[st == 'reached', 1], '.', ms=3, color='k'); ax.plot(X0[st != 'reached', 0], X0[st != 'reached', 1], 'x', ms=6, color='#ff00aa', label='not reached')
    xl, xb = _frames(ax, G); ax.set_xlim(-xb[0] - .2, xb[0] + .2); ax.set_ylim(-xb[1] - .2, xb[1] + .2); ax.set_title('%s: agent paths from %d starts (u=+1 purple, -1 red, 0 green)%s' % (_name(calc), len(X0), ('; ' + stat) if stat else ''), fontsize=9); ax.legend(fontsize=7, loc='lower left'); fig.tight_layout(); _save(fig, 'traj_%s.png' % calc, dirs)


def hist_png(G, res, dirs, calc):
    """распределение T/T_ref по стартам + мед/p90/max, доля дошедших"""
    from .algo import solve8
    plt = _plt(); X0 = np.asarray(res['X0'], float); T = np.asarray(res['T'], float); Tr = np.asarray(res['Tref'], float) if 'Tref' in res else t_ref_cached(X0); ok = np.isfinite(T); r = T[ok] / Tr[ok]; fig, ax = plt.subplots(figsize=(7.5, 4.6))
    if len(r):
        lo, hi = min(float(r.min()), 1.) - .02, float(r.max()) + .02; ax.hist(r, bins=np.linspace(lo, hi, 63), color='#4a90c8', density=True); ax.set_yscale('log'); ax.set_xlim(lo, hi);       # шкала X по данным (слово пользователя 2026-10-12), Y логарифмическая
        [ax.axvline(v, color=c, ls=l, label='%s %.3f' % (n, v)) for v, c, l, n in ((np.mean(r), '#d62728', '-', 'mean'), (np.median(r), '#2ca02c', '--', 'median'), (np.percentile(r, 90), '#ff7f0e', ':', 'p90'), (r.max(), 'k', ':', 'max'))]
    ax.set_xlabel('T / T_ref (starts)'); ax.set_ylabel('density (log)'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    ax.set_title('%s: reached %d / %d (%.1f%%), switches mean %.1f' % (_name(calc), int(ok.sum()), len(T), 100 * ok.mean() if len(T) else 0, float(np.mean(res['sw'])) if len(T) else 0), fontsize=10); fig.tight_layout(); _save(fig, 'hist_%s.png' % calc, dirs)
