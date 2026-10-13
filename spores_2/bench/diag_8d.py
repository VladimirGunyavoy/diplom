"""ДИАГНОСТИКА 8D старты 13,16 (w30): кандидаты/попадания/промахи по NF; без подгонки. Копия spores_8d.py. линия A: споры 8D (4 звена + g, коридор v6: обратное дерево NB + прямое NF + SLSQP, эвристика back_heuristic) на m8. Заготовка = build_back, запрос = corridor_query (1 ядро/запрос). python spores_8d.py m8 NB NF K [tag] [wlim]"""

import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, common, envs_6d
sys.path.insert(0, os.path.join(HERE, '..', 'v6'))                  # после common: тот добавляет v8 (тоже пакет src)
from src.atlas6.adaptive_nd import SysN, build_back, back_heuristic
from src.atlas6.corridor_nd import corridor_query
import multiprocessing as mp
name = 'm8'; NB, NF, K = [int(a) for a in sys.argv[2:5]]; tag = sys.argv[5] if len(sys.argv) > 5 else ''; WL = len(sys.argv) > 6 and sys.argv[6] == 'wlim'
e = common.make(name); n = 4; tau = .4; Rq, Rw, WM = .3, .6, 3.; rho = .15; C = np.array(e.C3, float)
import manip3g_dyn as G3
fl = lambda P, s, t: G3.flow(P, s, t, dt_max=.02, n=n, g=e.G)
US_ = np.array([[1.0 if (s >> i) & 1 else -1.0 for i in range(n)] for s in range(2 ** n)])
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
ing = lambda P: (np.max(np.abs(wr(P[..., :n] - C[:n])), -1) < Rq) & (np.max(np.abs(P[..., n:]), -1) < Rw)
g = lambda x: np.concatenate([Rq ** 2 - wr(x[:n] - C[:n]) ** 2, Rw ** 2 - x[n:] ** 2])
miss = lambda X: np.maximum(np.max(np.abs(wr(X[..., :n] - C[:n])), -1) - Rq, 0) + np.maximum(np.max(np.abs(X[..., n:]), -1) - Rw, 0)
rng = np.random.default_rng(0); seeds = []; R8 = np.array([Rq] * n + [Rw] * n)
for _ in range(32):
    v = rng.uniform(-1, 1, 2 * n); v = v / np.max(np.abs(v)) * 0.99; seeds.append(C + v * R8)
S = SysN(fl, 2 ** n, (np.pi,) * n + (WM,) * n, ing, seeds, per=(2 * np.pi,) * n + (0,) * n, ok=lambda p: np.max(np.abs(p[n:])) <= WM)
if os.environ.get('CAREF'):
    import refine_ca; t0 = time.perf_counter(); CAR = refine_ca.install(name, ms=int(os.environ.get('CAMS', 10)), maxit=int(os.environ.get('CAIT', 60)), prebuild=range(1, int(os.environ.get('CAPRE', 12))), wl=WL); print('refine_ca: сборка солверов %.1f с' % (time.perf_counter() - t0), flush=True)
if os.environ.get('TREEFAST'):
    import tree_fast; tree_fast.install(); t0 = time.perf_counter(); S.flow_all = tree_fast.make_flow_all(name, US_); print('tree_fast: компиляция потока %.1f с' % (time.perf_counter() - t0), flush=True)
t0 = time.perf_counter(); back = build_back(S, tau, NB, rho); HF = back_heuristic(S, back, wh=None); t_build = time.perf_counter() - t0; print('build_back NB %d: %.1f с' % (NB, t_build), flush=True)
from src.atlas6 import corridor_nd as _CN
_c0 = _CN.candidates; DG = {}
def _cw(*a, **kw):
    C, nh = _c0(*a, **kw); DG['cand'] = len(C); DG['hits'] = nh; DG['top'] = [(int(c[0]), round(float(c[1]), 3), len(c[2])) for c in C[:6]]; return C, nh
_CN.candidates = _cw
_r0 = _CN.refine
def _rw(*a, **kw):
    r = _r0(*a, **kw); DG.setdefault('ref', []).append((None if r[0] is None else round(float(r[0]), 3), bool(r[3]))); return r
_CN.refine = _rw
if __name__ == '__main__':
    Tr = json.load(open(os.path.join(common.DATA, 'm8_ref.json')))
    for i in [int(v) for v in os.environ.get('STARTS', '13,16').split(',')]:
        for nf in [int(v) for v in os.environ.get('NFS', '15000,40000,100000').split(',')]:
            DG.clear(); t1 = time.perf_counter(); b = corridor_query(S, fl, e.X0[i], back, tau, nf, rho, g, miss, K=K, tries=3, kn=8, neigh=1, hfun=HF, wlim=(tuple(range(n, 2 * n)), WM) if WL else None); dt = time.perf_counter() - t1
            Tv = None
            if b is not None: Tv, why = common.verify(e, e.X0[i], [US_[s] for s in b[1]], list(b[2]))
            print('старт %d NF %d: T_ref %.3f | кандидатов %s, попаданий %s, топ(hit,score,шагов) %s | путь %s T_agent %s T_verify %s | refine(T,ok) %s | %.0f с' % (i, nf, Tr[i], DG.get('cand'), DG.get('hits'), DG.get('top'), None if b is None else len(b[1]), None if b is None else round(float(b[0]), 3), Tv, DG.get('ref'), dt), flush=True)
