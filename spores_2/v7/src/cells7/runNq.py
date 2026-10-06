"""w23 (PLAN п.29в): прогон growN с отчётом по времени — построение, solve, один запрос (мед./p90/max по стартам, мс). growN — как модуль, не правится.
Запуск (di4, aida): SYS=di4 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 RHO=.35 PESS=1 MAXC=400 DUMPL=... python3 src/cells7/runNq.py"""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import growN as G
E = os.environ.get; t0 = time.time(); A = G.Atlas(); tb = time.time() - t0
print('построено', [len(l) for l in A.layers], 'узлов', A.N, round(tb), 'с', flush=True)
if E('DUMPL'):
    import pickle; pickle.dump(dict(layers=A.layers), open(E('DUMPL'), 'wb')); print('слои сохранены', E('DUMPL'), flush=True)
t1 = time.time(); A.solve(); ts = time.time() - t1
if E('DUMP'):
    import pickle; pickle.dump(dict(layers=A.layers, V=A.V), open(E('DUMP'), 'wb'))
Q, ref = G.starts_ref(); T = np.full(len(Q), np.inf); sw = np.zeros(len(Q), int); qms = []
for i in range(len(Q)):
    tq = time.time(); Ti, swi, _ = A.rollout(Q[i:i + 1]); qms.append(1000 * (time.time() - tq)); T[i] = Ti[0]; sw[i] = swi[0]
fz = np.isfinite(T) & np.isfinite(ref); r = T[fz] / ref[fz]; qms = np.array(qms)
print(json.dumps(dict(SYS=G.SYS, cells=[len(l) for l in A.layers], nodes=int(A.N), iters=int(A.n_it), big_nodes=round(float((A.V >= G.BIG / 2).mean()), 3), reach=round(float(np.isfinite(T).mean()), 3),
      T_over_ref=dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), max=round(float(r.max()), 3), min=round(float(r.min()), 3)) if fz.any() else None, sw_max=int(sw.max()), sw_med=float(np.median(sw)),
      sec_build=round(tb, 1), sec_solve=round(ts, 1), q_ms=dict(med=round(float(np.median(qms)), 1), p90=round(float(np.quantile(qms, .9)), 1), max=round(float(qms.max()), 1)), sec=round(time.time() - t0, 1))), flush=True)
