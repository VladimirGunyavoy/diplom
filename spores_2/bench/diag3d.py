"""b18: дрон, старт 3: почему no_path за 1.1 с — какие кандидаты-топологии есть и что делает refine (с зазором / без зазора / больше попыток)."""
import os, sys, time
BENCH = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(BENCH, 'spores_drone_b.py')).read(); cut = src.index("t0 = time.perf_counter(); back = build_back(S, .4, NB, RHOB)")
g_ = {'__name__': 'sd', '__file__': os.path.join(BENCH, 'spores_drone_b.py')}; sys.argv = ['x'] + sys.argv[1:4]
exec(compile(src[:cut], 'spores_drone_head', 'exec'), g_)
import numpy as np
from src.atlas6.adaptive_nd import build_back
from src.atlas6.corridor_nd import candidates, refine, merge
S, X0, clr, PAD, flow, g, miss, RHOB, RHOF, verify = [g_[k] for k in ('S', 'X0', 'clr', 'PAD', 'flow', 'g', 'miss', 'RHOB', 'RHOF', 'verify')]
NB, NF = g_['NB'], g_['NF']
back = build_back(S, .4, NB, RHOB)
from src.atlas6.corridor_nd import corridor_query
CLR = g_['clr']; RHOF_ = RHOF
for i in [int(a) for a in os.environ.get('STARTS', '3').split(',')]:
    for K in [int(a) for a in os.environ.get('KS', '10,20,40').split(',')]:
        t1 = time.perf_counter(); b = corridor_query(S, flow, X0[i], back, .4, NF, RHOF_, g, miss, K=K, tries=3, kn=8, neigh=2, nbest=1, clear=lambda P: CLR(P) - PAD, wlim=None); dt = time.perf_counter() - t1
        if b is None: print('старт %d K %d: no_path, %.1f с' % (i, K, dt), flush=True); continue
        Tv, why, tch = verify(X0[i], list(b[1]), list(b[2])); print('старт %d K %d: путь T_agent %.2f сегм %d, verify T %s %s касание %s, %.1f с' % (i, K, b[0], len(b[1]), None if Tv is None else round(Tv, 2), why, tch, dt), flush=True)
