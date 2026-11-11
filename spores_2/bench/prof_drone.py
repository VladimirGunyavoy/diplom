"""b18: профиль запроса дрона (corridor_query): cProfile + счётчики refine/candidates. env STARTS=4,7; запуск как spores_drone_b.py: NB NF K."""
import os, sys, time, cProfile, pstats, io
BENCH = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(BENCH, 'spores_drone_b.py')).read(); cut = src.index("t0 = time.perf_counter(); back = build_back(S, .4, NB, RHOB)")
g_ = {'__name__': 'sd', '__file__': os.path.join(BENCH, 'spores_drone_b.py')}; sys.argv = ['x'] + sys.argv[1:4]
exec(compile(src[:cut], 'spores_drone_head', 'exec'), g_)
import numpy as np
from src.atlas6.adaptive_nd import build_back
import src.atlas6.corridor_nd as cn
S, X0, clr, PAD, flow, g, miss, RHOB, RHOF = [g_[k] for k in ('S', 'X0', 'clr', 'PAD', 'flow', 'g', 'miss', 'RHOB', 'RHOF')]
NF, K = g_['NF'], g_['K']
back = build_back(S, .4, g_['NB'], RHOB)
LOG = []; _ref = cn.refine; _cand = cn.candidates; TC = []
def refine_w(flow_, x0, p, g_f, **kw):
    t1 = time.perf_counter(); r = _ref(flow_, x0, p, g_f, **kw); LOG.append((time.perf_counter() - t1, bool(r[3]), len(p), kw.get('tries'), kw.get('clear') is not None)); return r
def cand_w(*a, **k):
    t1 = time.perf_counter(); r = _cand(*a, **k); TC.append(time.perf_counter() - t1); return r
cn.refine = refine_w; cn.candidates = cand_w
for i in [int(a) for a in os.environ.get('STARTS', '4,7').split(',')]:
    LOG.clear(); TC.clear(); pr = cProfile.Profile(); t1 = time.perf_counter(); pr.enable()
    b = cn.corridor_query(S, flow, X0[i], back, .4, NF, RHOF, g, miss, K=K, tries=3, kn=8, neigh=2, nbest=1, clear=lambda P: clr(P) - PAD, wlim=None)
    pr.disable(); tot = time.perf_counter() - t1
    tr = sum(l[0] for l in LOG); n_ok = sum(l[1] for l in LOG); tr_ok = sum(l[0] for l in LOG if l[1]); tr_fail = tr - tr_ok
    print('=== старт %d: запрос %.1f с, путь %s; candidates %.1f с (%.0f%%); refine: %d вызовов (ok %d) %.1f с (%.0f%%), из них неудачных %.1f с, удачных %.1f с; прочее %.1f с' % (i, tot, 'есть' if b else 'нет', sum(TC), 100 * sum(TC) / tot, len(LOG), n_ok, tr, 100 * tr / tot, tr_fail, tr_ok, tot - sum(TC) - tr), flush=True)
    st = [l for l in LOG]; print('   refine по этапам: первые %d (K-топологии) %.1f с; остальные (соседи neigh) %d вызовов %.1f с' % (min(len(st), 2 * K), sum(l[0] for l in st[:2 * K]), max(0, len(st) - 2 * K), sum(l[0] for l in st[2 * K:])), flush=True)
    s = io.StringIO(); ps = pstats.Stats(pr, stream=s).sort_stats('tottime'); ps.print_stats(14); print('\n'.join(l[:170] for l in s.getvalue().split('\n')[:40] if l.strip()), flush=True)
