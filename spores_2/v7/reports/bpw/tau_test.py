"""Худший старт маятника (hub-v5chain-worker-15, п.4): ПЛАН-дуга эталона 2.48 с > tau=2 спор → V завышена на ~1.9 вдоль пути. Тест: tau=3 (дуги до 3 с), V(старт) на 100 стартах против pend_ref_T.npy.
Запуск из spores_2/v7: TAU=3 python3 reports/bpw/tau_test.py."""
import sys, os, time, json; sys.path.insert(0, '.')
import numpy as np
from src.cells7.butterfly_pend import ButterflyPend, BIG
TAU = float(os.environ.get('TAU', 3.)); N = int(os.environ.get('N', 5000)); t0 = time.time()
B = ButterflyPend(N=N, m=7, tau=TAU, win=.3); B.build_pairs(); B.solve(); print('готово', round(time.time() - t0), flush=True)
rng = np.random.default_rng(0); Q = np.stack([rng.uniform(-np.pi, np.pi, 100), rng.uniform(-2, 2, 100)], 1); ref = np.load('../v5chain/reports/research/pend_ref_T.npy')
V = np.array([B.best(q)[0] for q in Q]); f = (ref > .05) & (V < BIG / 2); r = V[f] / ref[f]
print(json.dumps(dict(TAU=TAU, N=N, V_finite=round(float(f.sum() / (ref > .05).sum()), 3), V_over_ref=dict(mean=round(float(r.mean()), 3), med=round(float(np.median(r)), 3), p90=round(float(np.percentile(r, 90)), 3), max=round(float(r.max()), 3)), V63=round(float(V[63]), 3), sec=round(time.time() - t0))), flush=True)
np.save('/tmp/claude-1000/bpw_V_tau%g.npy' % TAU, V)
