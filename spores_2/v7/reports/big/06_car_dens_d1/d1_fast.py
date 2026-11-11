"""w34: Д1-выброс на carfree-атласе кара БЕЗ отдельного анализа d1_car: argmin_u берётся из одного прохода GPU-solve (ARGMIN=1, growCar), потом обрезанный атлас решается тёплым стартом от V полного (NOLATCH=1).
Запуск (aida, из v7/, env как в tp.sh + DTN): LAYERS=…/d2_L.pkl OUT=… THR=.1 [D1REF=…/car_d1.json] flock /tmp/gpu.lock python ../reports/big/06_car_dens_d1/d1_fast.py
Выход: OUT/pruned.pkl (layers+V обрезанного, для envs_agentA.py LOAD=), OUT/d1_fast.json (времена, узлы, проходы, совпадение с D1REF)."""
import os, sys, json, pickle, time
os.environ.setdefault('SYS', 'car'); os.environ['ARGMIN'] = '1'; os.environ['NOLATCH'] = '1'
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'cells7')); import numpy as np, growCar as G, __main__
for n_ in dir(G):
    if n_[0].isupper() and isinstance(getattr(G, n_), type): setattr(__main__, n_, getattr(G, n_))
THR = float(os.environ.get('THR', .1)); OUT = os.path.expanduser(os.environ['OUT']); res = {}
layers = pickle.load(open(os.path.expanduser(os.environ['LAYERS']), 'rb'))['layers']
A = G.Atlas.__new__(G.Atlas); A.layers = layers; A.finish(); res['nodes_full'] = int(A.N)
t = time.time(); A.solve(); res['t_solve_full'] = time.time() - t; res['it_full'] = int(A.n_it); print('full solve %.0f с, проходов %d' % (res['t_solve_full'], A.n_it), flush=True)
am = A.amin_u; fin = A.V < G.BIG / 2; cell_layer = np.concatenate([[k] * len(l) for k, l in enumerate(layers)])
t = time.time(); rows = []; keep_cells = []; keep_idx = []
for ci, c in enumerate(A.cells):
    n = c.G.shape[0] * G.M ** G.m_; sl = slice(c.o, c.o + n); f = fin[sl] & (am[sl] != 127)
    own = float((am[sl][f] == cell_layer[ci]).mean()) if f.any() else 0.; rows.append(dict(ci=ci, own=own, nodes=int(n)))
    if own > THR: keep_cells.append(ci); keep_idx.append(np.arange(c.o, c.o + n))
res['t_prune_select'] = time.time() - t; keep_idx = np.concatenate(keep_idx); ks = set(keep_cells); L2 = []; ci = 0
for l in layers: L2.append([c for k, c in enumerate(l, ci) if k in ks]); ci += len(l)
if os.environ.get('D1REF'):
    ref = {r['ci']: r['own'] for r in json.load(open(os.path.expanduser(os.environ['D1REF'])))['cells']}
    a = np.array([r['own'] > THR for r in rows]); b = np.array([ref[r['ci']] > THR for r in rows]); res['agree_keep'] = float((a == b).mean()); res['corr_own'] = float(np.corrcoef([r['own'] for r in rows], [ref[r['ci']] for r in rows])[0, 1])
A2 = G.Atlas.__new__(G.Atlas); A2.layers = L2; A2.finish(); res['cells_kept'] = len(A2.cells); res['nodes_kept'] = int(A2.N); print('оставлено', res['cells_kept'], 'клеток', A2.N, 'узлов', flush=True)
if int(os.environ.get('WARM', 0)):      # тёплый старт от V полного: НЕ годится (921 проход против 168 холодного, V расходится до 3.7 с) — по умолчанию выключен
    A2.V = np.minimum(A.V[keep_idx], G.BIG); A2.V[A2.goal] = 0.; t = time.time(); A2.solve(); res['t_solve_warm'] = time.time() - t; res['it_warm'] = int(A2.n_it); print('warm solve %.0f с, проходов %d' % (res['t_solve_warm'], A2.n_it), flush=True); Vw = A2.V.copy()
A2.V = np.full(A2.N, G.BIG); A2.V[A2.goal] = 0.; t = time.time(); A2.solve(); res['t_solve_cold'] = time.time() - t; res['it_cold'] = int(A2.n_it); print('cold', res['t_solve_cold'], res['it_cold'], flush=True)
if int(os.environ.get('WARM', 0)): m = (Vw < G.BIG / 2) & (A2.V < G.BIG / 2); res['maxdiff_warm_cold'] = float(np.abs(Vw[m] - A2.V[m]).max())
pickle.dump(dict(layers=L2, V=A2.V), open(os.path.join(OUT, 'pruned.pkl'), 'wb')); json.dump(res, open(os.path.join(OUT, 'd1_fast.json'), 'w')); print(res)
