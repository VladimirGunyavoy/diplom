"""Парето по запросу на атласе 56 (маятник): один атлас, агент с LOOK 0/5/20 (G.LOOK меняется в рантайме). Запуск (aida): VARIANT=56 python3 spores_pend_pareto.py → data/pend_spores_pareto.json"""
import os, sys, time, json
os.environ['VARIANT'] = '56'
import numpy as np, common
import spores_pend as P
G = P.G
if __name__ == '__main__':
    e = common.make('pend'); t0 = time.perf_counter(); A = G.Atlas(); t_grow = time.perf_counter() - t0; t0 = time.perf_counter(); A.solve(); t_solve = time.perf_counter() - t0
    out = dict(env='pend', t_grow=t_grow, t_solve=t_solve, t_build=t_grow + t_solve, cells=[len(l) for l in A.layers], nodes=int(A.N), weight=common.deep_weight(A), hw='CPU (1 proc; numpy)', points=[])
    for look in (0, 5, 20):
        G.LOOK = look; Q = e.X0; Ts = []; tq = []
        for i in range(len(Q)):
            t1 = time.perf_counter(); T, rec = P.rollout_rec(A, Q[i:i + 1]); tq.append(time.perf_counter() - t1)
            us = [[r[0][0]] for r in rec if r[2][0]]; dts = [float(r[1][0]) for r in rec if r[2][0]]; Ts.append(common.verify(e, Q[i], us, dts)[0])
        s = common.summarize(e, Ts); s.update(look=look, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), t_query_sum=float(np.sum(tq)), per_start_T=Ts); out['points'].append(s)
        print(look, {k: v for k, v in s.items() if k != 'per_start_T'}, flush=True)
    json.dump(out, open(os.path.join(common.DATA, 'pend_spores_pareto.json'), 'w'), indent=1, default=float)
