"""В3 (multi_control_cell.md §В3), версия «lite», маятник u .3: (1) грубый атлас 2 слоя (главная точка DTN .12/Mmax 33/MADAPT .001) → V; (2) u*(x) = argmin_u [tgoal, DTN + V*(step(x,u))];
(3) ЕДИНСТВЕННЫЙ слой: тот же посев/рост, но управление клетки = u*(семени) (build_layer с u = ufun(p)); БЕЗ стоп-причины «смена режима» (lite); (4) solve + eval100 против грубого атласа.
Env как у sweep_dens (DTN, MCAP, MADAPT, BSEED, BIGOUT). Запуск из bench/: DTN=.12 MCAP=33 MADAPT=.001 BIGOUT=… python3 ../v7/reports/big/06_b3/b3.py"""
import os, sys, json, time, inspect, re; sys.path.insert(0, os.path.join(os.getcwd(), '..', 'v7', 'reports', 'big'))
import big_lib as B, numpy as np
G = B.G; G.US = (-G.UM, G.UM)
src = inspect.getsource(G.build_layer).replace('def build_layer(u, rng, log=None):', 'def build_b3(ufun, rng, log=None):', 1)
src = src.replace("        p = np.array([wrap(p[0]), p[1]])\n", "        p = np.array([wrap(p[0]), p[1]]); u = float(ufun(p))\n", 1)
src = src.replace("bar = tqdm(desc='layer u=%+g cells' % u,", "bar = tqdm(desc='B3 single layer cells',", 1)
assert 'u = float(ufun(p))' in src and 'B3 single' in src; exec(src, G.__dict__)
def ustar(A):
    UA = np.array(G.US)
    def f(p):
        Y = np.asarray(p, float)[None]; tg = A.tgoal3(Y); Ys = A.step3(Y); J = np.minimum(tg, G.DTN + A.vstar(Ys).reshape(len(G.US), 1).T); return UA[int(np.argmin(J[0]))]
    return f
seed = int(os.environ.get('BSEED', 0)); tag = os.environ.get('TAG', 'b3_s%d_dtn%s_mcap%s' % (seed, os.environ.get('DTN', '.12'), os.environ.get('MCAP', '33')))
t0 = time.perf_counter(); A0 = G.Atlas(seed=seed); tg0 = time.perf_counter() - t0; ts0 = B.solve(A0); s0, _ = B.eval100(A0)
print('грубый:', len(A0.cells), A0.N, 't_grow %.1f t_solve %.1f' % (tg0, ts0), s0, flush=True)
NIT = int(os.environ.get('B3IT', 1)); Ap = A0; R_ = []
for it in range(NIT):
    t1 = time.perf_counter(); rng = np.random.default_rng(seed + 100 * (it + 1)); cells, idx = G.build_b3(ustar(Ap), rng); tg1 = time.perf_counter() - t1
    A1 = G.Atlas.__new__(G.Atlas); A1.layers = [cells]; A1.idx = [idx]; A1.finish(); ts1 = B.solve(A1); s1, _ = B.eval100(A1)
    nu = {float(u): sum(1 for c in cells if c.u == u) for u in G.US}
    R_.append(dict(iter=it + 1, cells=len(cells), nodes=int(A1.N), t_grow=tg1, t_solve=ts1, goalh_used=getattr(A1, 'goalh_used', False), cells_by_u=nu, V_fin=float((A1.V < G.BIG / 2).mean()), **s1)); print('B3 iter', it + 1, R_[-1], flush=True); Ap = A1
r = dict(tag=tag, coarse=dict(cells=len(A0.cells), nodes=int(A0.N), t_grow=tg0, t_solve=ts0, **s0), b3=R_[0], iters=R_)
print(json.dumps(r, default=float), flush=True); json.dump(r, open(os.path.join(os.environ['BIGOUT'], 'b3_%s.json' % tag), 'w'), indent=1, default=float)
