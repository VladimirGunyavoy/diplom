"""C2: споры ДИ — v8 solve8 + агент commit (Q lin, слои 1/−1/0, M 5). Время раздельно: рост, solve (V, затем Q), запрос (на старт, 1 старт за вызов). Траектории → (u_k,dt_k) → единый проверяльщик.
Запуск (aida): OMP_NUM_THREADS=1 M=5 python3 spores_di.py"""
import os, sys, time, json, numpy as np
os.environ.setdefault('M', '5'); os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from src.algo import solve8 as S
e = common.make('di'); g = S.load_growN(); layers = [1., -1., 0.]
t0 = time.perf_counter(); atlas, tg = S.build_atlas(layers, 0, par=False); t_grow = time.perf_counter() - t0
t0 = time.perf_counter(); G = S.make_graph(g, atlas, 'N'); G.solve(verbose=False); t_v = time.perf_counter() - t0
t0 = time.perf_counter(); G.set_calc('q'); G.solve(verbose=False); t_q = time.perf_counter() - t0
res = []; Ts = []; tq = []; tmax = 3 * e.Tref + 2
for i, x0 in enumerate(e.X0):
    path = []; t1 = time.perf_counter(); T, sw = G.run_agent(x0[None].copy(), tmax[i:i + 1], dt=.01, path=path, rec=1, commit=True); dtq = time.perf_counter() - t1
    X = [p[1][0] for p in path if len(p[0]) == 1]; X = np.array(X)                          # состояния на шагах dt .01 (+ точка попадания)
    us = np.diff(X[:, 1]) / .01 if len(X) > 1 else np.zeros(0); dts = np.full(len(us), .01)
    Tv, why = common.verify(e, x0, us[:, None], dts); tq.append(dtq); Ts.append(Tv)
    res.append(dict(i=i, T_agent=None if not np.isfinite(T[0]) else float(T[0]), T_verify=Tv, why=why, t_query=dtq, switches=int(sw[0])))
t1 = time.perf_counter(); G.run_agent(e.X0.copy(), tmax, dt=.01, commit=True); t_batch = time.perf_counter() - t1                     # пакет 20 стартов одним вызовом (пропускная способность)
tq5 = []; Ts5 = []
for i, x0 in enumerate(e.X0):                                                                                                           # агент с dt .05 (= шаг решений PPO)
    path = []; t1 = time.perf_counter(); G.run_agent(x0[None].copy(), tmax[i:i + 1], dt=.05, path=path, rec=1, commit=True); tq5.append(time.perf_counter() - t1)
    X = np.array([p[1][0] for p in path if len(p[0]) == 1]); us = np.diff(X[:, 1]) / .05; Ts5.append(common.verify(e, x0, us[:, None], np.full(len(us), .05))[0])
s5 = common.summarize(e, Ts5); s = common.summarize(e, Ts); s.update(t_batch20=t_batch, dt05=dict(s5, t_query_med=float(np.median(tq5)), t_query_sum=float(np.sum(tq5)))); s.update(env='di', method='spores v8 commit Q-lin M5 layers 1,-1,0', t_grow=t_grow, t_solve_V=t_v, t_solve_Q=t_q, t_build=t_grow + t_v + t_q, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), t_query_sum=float(np.sum(tq)),
                                       hw='CPU 1 proc', cells={str(u): G.layer_counts[u] for u in G.us}, per_start=res, Tref=e.Tref.tolist())
os.makedirs(common.DATA, exist_ok=True); json.dump(s, open(os.path.join(common.DATA, 'di_spores.json'), 'w'), indent=1, default=float)
print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')})
