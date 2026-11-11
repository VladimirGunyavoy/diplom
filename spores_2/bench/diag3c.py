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
for i in [int(a) for a in os.environ.get('STARTS', '3,5').split(',')]:
    x0 = X0[i]; C, nh = candidates(S, x0, back, .4, NF, RHOF, miss, 8)
    print('старт %d: x0 %s; кандидатов-попаданий %d, промахов %d' % (i, np.round(x0, 2), nh, len(C) - nh), flush=True)
    for name, part in (('hit', C[:nh]), ('miss', C[nh:])):
        seen = set(); k = 0
        for h, sc, p in part:
            if not p: continue
            sq = tuple(merge(p)[0])
            if sq in seen: continue
            seen.add(sq); k += 1
            res = {}
            for lab, kw in (('clr+3', dict(tries=3, clear=lambda P: clr(P) - PAD)), ('clr+8', dict(tries=8, clear=lambda P: clr(P) - PAD)), ('noclr+3', dict(tries=3, clear=None))):
                t1 = time.perf_counter(); T, sqr, d, ok = refine(flow, x0, p, g, **kw); res[lab] = (bool(ok), None if T is None else round(float(T), 2), round(time.perf_counter() - t1, 1))
            print('  %s #%d: оценка %.2f сегм %d | %s' % (name, k, sc, len(sq), res), flush=True)
            if k >= 6: break
