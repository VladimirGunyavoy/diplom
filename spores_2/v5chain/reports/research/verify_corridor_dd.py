"""Независимая проверка решений corridor_topk_dd_miss.json: кинематика эталонного модуля (dd_rhombus_ref.pose), конец в окне, T = Σdt;
для T/ref < 0.99 — эталон TGTGT в окно с 200 стартами (вместо 40): не промах ли эталона."""
import os, json, importlib.util, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location('ref', os.path.join(HERE, 'dd_rhombus_ref.py')); ref = importlib.util.module_from_spec(sp); sp.loader.exec_module(ref)
sw = importlib.util.spec_from_file_location('w', os.path.join(HERE, 'dd_window_ref.py')); w = importlib.util.module_from_spec(sw); sw.loader.exec_module(w)
D = json.load(open(os.path.join(HERE, 'corridor_topk_dd_miss.json'))); E = json.load(open(os.path.join(HERE, 'multiquery_dd_ref.json')))
rng = np.random.default_rng(7); Q = [np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)]) for _ in range(len(E))]
A = {0: ('G', 1), 1: ('G', -1), 2: ('T', 1), 3: ('T', -1)}; R, Rth = .25, .26; bad = 0; out = []
for s in D['sol']:
    if s is None: continue
    q = s['q']; p = tuple(Q[q])
    for l, d in zip(s['seq'], s['dts']): p = ref.pose(p, (A[l][0], A[l][1] * d))
    rin = np.hypot(p[0], p[1]); thin = abs(ref.wrap(p[2])); T = sum(s['dts']); ok = rin <= R + 1e-5 and thin <= Rth + 1e-5; bad += not ok
    row = dict(q=q, T_ref=T / E[q], r_end=rin, th_end=thin, ok=bool(ok), seq=s['seq'])
    if T / E[q] < 0.99: e2 = w.tgtgt_window(Q[q], starts=200); row['T_tgtgt200'] = T / e2 if np.isfinite(e2) else None
    out.append(row); print(row, flush=True)
print('невалидных', bad, 'из', len(out))
json.dump(out, open(os.path.join(HERE, 'verify_corridor_dd.json'), 'w'), indent=1)
