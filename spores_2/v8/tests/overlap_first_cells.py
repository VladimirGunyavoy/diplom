"""Замер заезда клеток друг на друга (утилита, не тест): слои u=+1/−1 (seed 0, env как live.py), клетки 1..K по порядку построения.
Клетка 1 — спора на цели (центр цели). Для пар (i, j): доли площади ядра/ядра+гало клетки i, накрытые ядром / ядром+гало клетки j; плюс доля ядра k, накрытая объединением ядер/гало ВСЕХ прежних клеток.
Площадь — точечно по квадам сетки узлов G (билинейно), веса = площадь квада. Плюс из какой затравки выросла клетка, какие стороны и почему встали (pause 'stop'), сколько затравок отброшено.
Запуск: python tests/overlap_first_cells.py [K=5] [OUT=reports/overlap_first_cells] → .json, .md, .png (контуры клеток 1 и 2)"""
import os, sys, json, collections
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..'); sys.path.insert(0, ROOT)
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
ENV = dict(SYS='di', M='3', KF='21', MAXC='60', NFAIL='400', DELTA='.03', RMAX='.5', TMAX='3', TQDM_MI='1000', GS='0', GLIM='0', GSEED='1', GOALB='0', GOALSHAPE='ball', RHO='.2')   # как live.py
for k, v in ENV.items(): os.environ.setdefault(k, v)
import numpy as np
from src.algo import growN as g
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5; OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'reports', 'overlap_first_cells'); NS = 24   # NS — точек на квад по каждой оси


def bil(G, s, t):
    """билинейная точка сетки узлов G (nt, M, 2): s ∈ [0, nt−1] — строка, t ∈ [0, M−1] — боковой узел"""
    i = np.clip(np.floor(s).astype(int), 0, G.shape[0] - 2); j = np.clip(np.floor(t).astype(int), 0, G.shape[1] - 2); a = (s - i)[..., None]; b = (t - j)[..., None]
    return (1 - a) * (1 - b) * G[i, j] + a * (1 - b) * G[i + 1, j] + (1 - a) * b * G[i, j + 1] + a * b * G[i + 1, j + 1]


def tcore(c):
    """боковой параметр t границы ядра: узлы стоят на ±(1+HALO)·r, ядро — |l| ≤ r"""
    mid = (g.M - 1) / 2; return mid - mid / (1 + g.HALO), mid + mid / (1 + g.HALO)


def samples(c):
    """точки по всей клетке (ядро+гало): (P, вес, в_ядре)"""
    nt, Mm = c.G.shape[0], c.G.shape[1]; q = (np.arange(NS) + .5) / NS
    si, tj = np.meshgrid(np.arange(nt - 1), np.arange(Mm - 1), indexing='ij'); si, tj = si.ravel(), tj.ravel()
    s = (si[:, None, None] + q[None, :, None]) * np.ones((1, 1, NS)); t = (tj[:, None, None] + q[None, None, :]) * np.ones((1, NS, 1))
    ar = .5 * np.abs(np.cross(c.G[si + 1, tj + 1] - c.G[si, tj], c.G[si + 1, tj] - c.G[si, tj + 1])) / NS ** 2                       # площадь квада / число точек
    P = bil(c.G, s.ravel(), t.ravel()); w = np.repeat(ar, NS * NS); t0, t1 = tcore(c)
    core = (s.ravel() >= c.hb - 1e-9) & (s.ravel() <= c.hb + c.nb + c.nf + 1e-9) & (t.ravel() >= t0 - 1e-9) & (t.ravel() <= t1 + 1e-9)
    return P, w, core


def only(c): ix = g.HexIdx(); ix.add(c); return ix


def run(u, seed=0):
    ev = []; cur = dict(rej=collections.Counter(), stops=[], seed=None)
    def pause(ph, lvl, **kw):
        if ph == 'reject': cur['rej'][kw['reason']] += 1
        elif ph == 'seed': cur['seed'] = np.array(kw['seed'])
        elif ph == 'stop': cur['stops'].append((str(kw['dir']), kw['reason']))
        elif ph == 'cell': ev.append(dict(seed=cur['seed'], rej=dict(cur['rej']), stops=list(cur['stops']))); cur.update(rej=collections.Counter(), stops=[])
    g.pause = pause; g.MAXC = K
    cells, idx = g.build_layer(u, np.random.default_rng(seed))
    cells = cells[:K]; S = [samples(c) for c in cells]; I1 = [only(c) for c in cells]; res = dict(u=u, cells=[], pairs=[], ev=[])
    for i, c in enumerate(cells):
        P, w, core = S[i]; wc = w[core].sum(); wa = w.sum()
        r0 = int(np.argmin(np.abs(c.G[:, g.M // 2, 1]))); t0, t1 = tcore(c); lo, hi = bil(c.G, np.array([r0, r0]), np.array([0., g.M - 1.])); clo, chi = bil(c.G, np.array([r0, r0]), np.array([t0, t1]))
        res['cells'].append(dict(n=i + 1, x_row_v0=dict(core=[float(clo[0]), float(chi[0])], halo=[float(lo[0]), float(hi[0])], v=float(c.G[r0, g.M // 2, 1])), c=c.c.round(3).tolist(), r=float(c.r[0]), rows=int(c.nb + c.nf), area_core=float(wc), area_all=float(wa), seed=ev[i]['seed'].round(3).tolist(), rej=ev[i]['rej'], stops=ev[i]['stops']))
        # накрыто объединением всех прежних
        if i:
            U = g.HexIdx()
            for c2 in cells[:i]: U.add(c2)
            cv, ins = U.both(P); res['cells'][-1].update(core_by_prev_core=float(w[core & cv].sum() / wc), core_by_prev_all=float(w[core & ins].sum() / wc), full_by_prev_core=float(w[cv].sum() / wa), full_by_prev_all=float(w[ins].sum() / wa))
        for j in range(len(cells)):
            if i == j: continue
            cv, ins = I1[j].both(P)
            res['pairs'].append(dict(i=i + 1, j=j + 1, core_in_core=float(w[core & cv].sum() / wc), core_in_all=float(w[core & ins].sum() / wc), full_in_core=float(w[cv].sum() / wa), full_in_all=float(w[ins].sum() / wa)))
    return res, cells


def outline(c, full):
    """контур клетки: ядро (full=False) или ядро+гало (full=True) — замкнутая ломаная по границе параметров"""
    t0, t1 = tcore(c); s0, s1 = (0, c.G.shape[0] - 1) if full else (c.hb, c.hb + c.nb + c.nf); a0, a1 = (0, g.M - 1) if full else (t0, t1); n = 40
    s = np.r_[np.linspace(s0, s1, n), np.full(n, s1), np.linspace(s1, s0, n), np.full(n, s0)]; t = np.r_[np.full(n, a0), np.linspace(a0, a1, n), np.full(n, a1), np.linspace(a1, a0, n)]
    return bil(c.G, s, t)


def plot(R, cells_by_u, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(cells_by_u), figsize=(7 * len(cells_by_u), 7)); axs = np.atleast_1d(axs); col = {0: '#1f77b4', 1: '#d62728'}
    for ax, (u, cells), res in zip(axs, cells_by_u.items(), R):
        for n, c in enumerate(cells):
            if n < 2:
                for full, ls, lw in ((True, '--', 1.2), (False, '-', 2.2)):
                    o = outline(c, full); ax.plot(o[:, 0], o[:, 1], ls, color=col[n], lw=lw, label=('клетка %d %s' % (n + 1, 'ядро+гало' if full else 'ядро')) if True else None)
                    if not full: ax.fill(o[:, 0], o[:, 1], color=col[n], alpha=.12)
                ax.plot(*np.array(res['cells'][n]['seed']), 'k*' if n == 0 else 'kx', ms=9)
            else:
                o = outline(c, False); ax.plot(o[:, 0], o[:, 1], '-', color='.6', lw=1); ax.text(c.c[0], c.c[1], str(n + 1), color='.4', ha='center')
        th = np.linspace(0, 2 * np.pi, 100); ax.plot(g.RHOV[0] * np.cos(th), g.RHOV[0] * np.sin(th), 'g:', lw=1.5, label='цель')
        p = [q for q in res['pairs'] if q['i'] == 2 and q['j'] == 1][0]; p2 = [q for q in res['pairs'] if q['i'] == 1 and q['j'] == 2][0]
        ax.set_title('u=%+g seed 0: 2 в 1 (ядро в ядре %.0f%%, ядро в ядре+гало %.0f%%); 1 в 2 (%.0f%% / %.0f%%)' % (u, 100 * p['core_in_core'], 100 * p['core_in_all'], 100 * p2['core_in_core'], 100 * p2['core_in_all']), fontsize=9)
        ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlabel('x'); ax.set_ylabel('v')
        allp = np.vstack([outline(cells[n], True) for n in range(min(2, len(cells)))]); mn, mx = allp.min(0), allp.max(0); pad = .25 * (mx - mn).max() + .1; ax.set_xlim(mn[0] - pad, mx[0] + pad); ax.set_ylim(mn[1] - pad, mx[1] + pad)
    axs[0].legend(fontsize=8, loc='best'); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == '__main__':
    R = []; CB = {}
    for u in g.ULAYERS[:2] if hasattr(g, 'ULAYERS') else (1., -1.):
        r, cells = run(u); R.append(r); CB[u] = cells
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(R, open(OUT + '.json', 'w'), indent=1, default=float); plot(R, CB, OUT + '.png')
    L = ['# Заезд клеток друг на друга (v8, ДИ, seed 0, env как live.py)', '']
    for r in R:
        L += ['## u=%+g' % r['u'], '', '| клетка | затравка | r | строк | площадь ядра | ядро в ядрах прежних | ядро в ядре+гало прежних | ядро+гало в ядрах прежних | ядро+гало в ядре+гало прежних | отброшено затравок | почему встали стороны | x на строке v≈0: ядро / ядро+гало |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for c in r['cells']:
            f = lambda k: '%.1f%%' % (100 * c[k]) if k in c else '–'
            L.append('| %d | %s | %.3f | %d | %.4f | %s | %s | %s | %s | %s | %s | [%.3f, %.3f] / [%.3f, %.3f] |' % (c['n'], c['seed'], c['r'], c['rows'], c['area_core'], f('core_by_prev_core'), f('core_by_prev_all'), f('full_by_prev_core'), f('full_by_prev_all'), c['rej'] or '–', ', '.join('%s: %s' % x for x in c['stops']), *c['x_row_v0']['core'], *c['x_row_v0']['halo']))
        L += ['', 'Пары (i накрыта j): доля площади i, лежащая в ядре / в ядре+гало j', '', '| i | j | ядро i в ядре j | ядро i в ядре+гало j | (ядро+гало) i в ядре j | (ядро+гало) i в ядре+гало j |', '|---|---|---|---|---|---|']
        for q in r['pairs']:
            if q['i'] <= 2 or q['j'] <= 2 or True: L.append('| %d | %d | %.1f%% | %.1f%% | %.1f%% | %.1f%% |' % (q['i'], q['j'], 100 * q['core_in_core'], 100 * q['core_in_all'], 100 * q['full_in_core'], 100 * q['full_in_all']))
        L.append('')
    open(OUT + '.md', 'w', encoding='utf-8').write('\n'.join(L)); print('\n'.join(L))
