"""HYB-8D (w30): споры (конфиг _tf3) + полный OCP fast (свободное T, N50) с тёплым стартом от пути спор; при провале — путь спор. Копия spores_8d.py. Было: линия A: споры 8D (4 звена + g, коридор v6: обратное дерево NB + прямое NF + SLSQP, эвристика back_heuristic) на m8. Заготовка = build_back, запрос = corridor_query (1 ядро/запрос). python spores_8d.py m8 NB NF K [tag] [wlim]"""

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
import ocp_m8_lib as OL; t0h = time.perf_counter(); HS = OL.make_solver(50, 4); print('HYB: решатель собран за %.1f с' % (time.perf_counter() - t0h), flush=True)
def hyb(x0, seq, dts, N=50):
    S_, lbx, ubx, lbg, ubg = HS; T = float(sum(dts)); ts = np.linspace(0, T, N + 1); ce = np.cumsum(dts); x = np.array(x0, float); X = [x.copy()]; seg = 0; t = 0.
    for i in range(1, N + 1):
        while t < ts[i] - 1e-12:
            step = min(ts[i], ce[seg]) - t; x = fl(x, US_[seq[seg]], step); t += step
            if t >= ce[seg] - 1e-12 and seg < len(seq) - 1: seg += 1
        X.append(x.copy())
    X = np.array(X); tm = (ts[:-1] + ts[1:]) / 2; Ug = np.array([US_[seq[min(int(np.searchsorted(ce, v)), len(seq) - 1)]] for v in tm], float)
    tgt = C[:n] + 2 * np.pi * np.round((X[-1][:n] - C[:n]) / (2 * np.pi)); P = np.concatenate([x0, tgt])
    r = S_(x0=np.concatenate([X.ravel(), Ug.ravel(), [T]]), p=P, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg); st = S_.stats()['return_status']
    if st not in ('Solve_Succeeded', 'Solved_To_Acceptable_Level'): return None, st
    wv = np.array(r['x']).ravel(); return (wv[8 * (N + 1):8 * (N + 1) + 4 * N].reshape(N, 4), wv[-1]), st
t0 = time.perf_counter(); back = build_back(S, tau, NB, rho); HF = back_heuristic(S, back, wh=None); t_build = time.perf_counter() - t0; print('build_back NB %d: %.1f с' % (NB, t_build), flush=True)
def work(i):
    e.nf = 0; t1 = time.perf_counter(); b = corridor_query(S, fl, e.X0[i], back, tau, NF, rho, g, miss, K=K, tries=3, kn=8, neigh=int(os.environ.get('NEIGH', 2)), hfun=HF, wlim=(tuple(range(n, 2 * n)), WM) if WL else None); dt = time.perf_counter() - t1
    if b is None and os.environ.get('NFRETRY'):       # не нашли путь — один повтор с большим прямым деревом (только провалы платят)
        b = corridor_query(S, fl, e.X0[i], back, tau, int(os.environ['NFRETRY']), rho, g, miss, K=K, tries=3, kn=8, neigh=int(os.environ.get('NEIGH', 2)), hfun=HF, wlim=(tuple(range(n, 2 * n)), WM) if WL else None); dt = time.perf_counter() - t1
    if b is None: return dict(i=i, T_agent=None, t_query=dt, T_verify=None, why='no_path')
    us = [US_[s] for s in b[1]]; Tv, why = common.verify(e, e.X0[i], us, list(b[2])); th = time.perf_counter(); Th = None; whyh = 'skip'
    if Tv is not None:
        try:
            h, st = hyb(e.X0[i], list(b[1]), list(b[2]))
            if h is not None: Th, whyh = common.verify(e, e.X0[i], [u for u in h[0]], [h[1] / len(h[0])] * len(h[0]))
            else: whyh = st
        except Exception as ex: whyh = 'exc ' + str(ex)[:60]
    th = time.perf_counter() - th; ok = Th is not None and Th <= Tv + 1e-9
    return dict(i=i, T_agent=float(b[0]), t_query=dt, T_spores=Tv, T_verify=(Th if ok else Tv), why=why, whyh=whyh, hyb_used=bool(ok), t_hyb=th, nseg=len(us))
if __name__ == '__main__':
    t0 = time.perf_counter()
    with mp.get_context('fork').Pool(min(int(os.environ.get('PROCS', 16)), len(e.X0))) as p: res = p.map(work, range(len(e.X0)), chunksize=1)
    wall = time.perf_counter() - t0; Ts = [r['T_verify'] for r in res]; tq = np.array([r['t_query'] + r['t_hyb'] for r in res]); s = common.summarize(e, Ts)
    s.update(env=name, method='HYB-8D споры+IPOPT NB%d NF%d K%d%s' % (NB, NF, K, ' wlim' if WL else ''), t_build=t_build, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), t_query_sum=float(tq.sum()), t_wall_parallel=wall,
             hw='CPU 1 ядро/запрос, numpy+math-поток', per_start=res, Tref=e.Tref.tolist(), weight=common.deep_weight(back), nodes_back=int(len(back[0])))
    os.makedirs(common.DATA, exist_ok=True); json.dump(s, open(os.path.join(common.DATA, '%s_hyb8d%s.json' % (name, tag)), 'w'), indent=1, default=float)
    print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')}, flush=True)
