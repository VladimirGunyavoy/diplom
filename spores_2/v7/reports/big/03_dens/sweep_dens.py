"""BIG: свип ПЛОТНОСТИ с построения, 2 слоя (−.3, .3), маятник u .3, штатный рост, 100 стартов. Параметры env: DTN, MCAP (потолок клонов 5/9/17/33), MADAPT (допуск ломаной), VARIANT=56|84.
Запуск из bench/: DTN=.06 MCAP=33 MADAPT=.001 BIGOUT=… python3 ../v7/reports/big/03_dens/sweep_dens.py"""
import os, sys, json, time; sys.path.insert(0, os.path.join(os.getcwd(), '..', 'v7', 'reports', 'big'))
import big_lib as B, numpy as np
G = B.G; tag = os.environ.get('TAG', 'v%s_dtn%s_mcap%s_ma%s' % (os.environ.get('VARIANT', '56'), os.environ.get('DTN', '.06'), os.environ.get('MCAP', '33'), os.environ.get('MADAPT', 'def')))
if not os.environ.get('THREE'): G.US = (-G.UM, G.UM)
t0 = time.perf_counter(); A = G.Atlas(); tg = time.perf_counter() - t0; ts = B.solve(A); t1 = time.perf_counter(); s, _ = B.eval100(A); tq = (time.perf_counter() - t1) / s['n']
r = dict(tag=tag, DTN=G.DTN, MCAP=os.environ.get('MCAP'), MADAPT=os.environ.get('MADAPT'), variant=os.environ.get('VARIANT', '56'), layers=len(G.US), cells=len(A.cells), nodes=int(A.N), m={int(k): sum(1 for c in A.cells if c.m == k) for k in set(c.m for c in A.cells)}, t_grow=tg, t_solve=ts, t_query_100=tq, **s)
print(json.dumps(r), flush=True); json.dump(r, open(os.path.join(os.environ['BIGOUT'], 'dens_%s.json' % tag), 'w'), indent=1, default=float)
