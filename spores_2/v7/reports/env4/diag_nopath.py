"""ENV4 (2) диагностика стыка: для каждого из 20 стартов мин. расстояние (метрика S.scale, как в tree_query) между листьями прямого дерева NF и узлами обратного NB,
где лежит пара (x,y,зазор до круга/эллипса), и 5 ближайших пар. Запуск (aida, bench/): NB NF RHOB RHOF PAD из env как у spores_drone.py:
python diag_nopath.py 6000 400 5 → data/diag_nopath.json"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); BENCH = os.path.expanduser(os.environ.get('BENCH', '~/spore_v5/a31/spores_2/bench'))
src = open(os.path.join(BENCH, 'spores_drone.py')).read(); cut = src.index('t0 = time.perf_counter(); back = build_back')
g_ = {'__name__': 'sd', '__file__': os.path.join(BENCH, 'spores_drone.py')}; sys.argv = ['x'] + sys.argv[1:4]
exec(compile(src[:cut], 'spores_drone_head', 'exec'), g_)
import numpy as np
from src.atlas6.adaptive_nd import _tree, build_back
S, NB, NF, X0, clr, CC, CR = g_['S'], g_['NB'], g_['NF'], g_['X0'], g_['clr'], g_['CC'], g_['CR']
RHOB, RHOF = g_['RHOB'], g_['RHOF']; sc = S.scale
t0 = time.perf_counter(); back = build_back(S, .4, NB, RHOB); Pb = np.array(back[0]); Gb = np.array(back[1]); print('back %.1f с' % (time.perf_counter() - t0), flush=True)
def wrap_d(a, b):
    d = (a - b) / sc; 
    for k in range(S.d):
        if S.per[k] > 0: p = S.per[k] / sc[k]; d[..., k] = (d[..., k] + p / 2) % p - p / 2
    return d
from scipy.spatial import cKDTree
tr = cKDTree(Pb / sc)  # без периода θ: для θ берём тоже ближайшие с обёрткой отдельно ниже
def work(i):
    Pf, Gf, parf, layf = _tree(S, .4, [X0[i]], NF, RHOF, 10.0, True); Pf = np.array(Pf); best = []
    for kk in range(len(Pf)):
        for sh in (-2 * np.pi, 0, 2 * np.pi):
            q = Pf[kk].copy(); q[2] += sh; d, j = tr.query(q / sc, k=1); best.append((float(d), kk, int(j)))
    best.sort(); b = best[0]; out = []
    for d, kk, j in best[:5]:
        out.append(dict(d=d, pf=Pf[kk].tolist(), pb=Pb[j].tolist(), Gf=float(Gf[kk]), Gb=float(Gb[j]), clr_f=float(clr(Pf[kk])), clr_b=float(clr(Pb[j]))))
    return dict(i=i, nleaf=int(len(Pf)), dmin=b[0], pairs=out, x_ref_dist_circle=float(np.hypot(*(Pf[b[1]][:2] - CC)) - CR))
import multiprocessing as mp
if __name__ == '__main__':
    with mp.get_context('fork').Pool(int(os.environ.get('PROCS', 10))) as p: res = p.map(work, range(len(X0)), chunksize=1)
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True); json.dump(res, open(os.path.join(HERE, 'data', 'diag_nopath.json'), 'w'), indent=1)
    for r in res: print(r['i'], 'nleaf', r['nleaf'], 'dmin %.3f' % r['dmin'], 'пара f xy', np.round(r['pairs'][0]['pf'][:2], 2), 'b xy', np.round(r['pairs'][0]['pb'][:2], 2), 'зазор f/b %.2f/%.2f' % (r['pairs'][0]['clr_f'], r['pairs'][0]['clr_b']), flush=True)
