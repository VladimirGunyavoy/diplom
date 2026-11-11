"""b18: дрон, старт 3 (единственный настоящий no_path): мин. расстояние лист NF ↔ узел NB в метрике S.scale при разных NB/NF. env: NBS, NFS, STARTS.
Запуск (aida, ~/spore_v5/b17/spores_2/bench): EULER=1 QUANT=1 CLRDT=.05 PAD=.1 RHOB=.3 RHOF=.15 python diag3_b.py 6000 400 5"""
import os, sys, time
BENCH = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(BENCH, 'spores_drone_b.py')).read(); cut = src.index("t0 = time.perf_counter(); back = build_back(S, .4, NB, RHOB)")
g_ = {'__name__': 'sd', '__file__': os.path.join(BENCH, 'spores_drone_b.py')}; sys.argv = ['x'] + sys.argv[1:4]
exec(compile(src[:cut], 'spores_drone_head', 'exec'), g_)
import numpy as np
from scipy.spatial import cKDTree
from src.atlas6.adaptive_nd import _tree, build_back
S, X0, clr, CC, CR, RHOB, RHOF = g_['S'], g_['X0'], g_['clr'], g_['CC'], g_['CR'], g_['RHOB'], g_['RHOF']; sc = S.scale
STARTS = [int(a) for a in os.environ.get('STARTS', '3').split(',')]
for NBv in [int(a) for a in os.environ.get('NBS', '6000').split(',')]:
    t0 = time.perf_counter(); back = build_back(S, .4, NBv, RHOB); Pb = np.array(back[0]); Gb = np.array(back[1]); tr = cKDTree(Pb / sc)
    print('back NB %d: %.1f с' % (NBv, time.perf_counter() - t0), flush=True)
    for NFv in [int(a) for a in os.environ.get('NFS', '400').split(',')]:
        for i in STARTS:
            Pf, Gf, parf, layf = _tree(S, .4, [X0[i]], NFv, RHOF, 10.0, True); Pf = np.array(Pf); best = []
            for kk in range(len(Pf)):
                for sh in (-2 * np.pi, 0, 2 * np.pi):
                    q = Pf[kk].copy(); q[2] += sh; d, j = tr.query(q / sc, k=1); best.append((float(d), kk, int(j)))
            best.sort(); d0, kk, j = best[0]
            print('NB %d NF %d старт %d: dmin %.3f (листьев %d), f xy %s, b xy %s, T_f %.2f T_b %.2f, f до круга %.2f' % (NBv, NFv, i, d0, len(Pf), np.round(Pf[kk][:2], 2), np.round(Pb[j][:2], 2), Gf[kk], Gb[j], np.hypot(*(Pf[kk][:2] - CC)) - CR), flush=True)
