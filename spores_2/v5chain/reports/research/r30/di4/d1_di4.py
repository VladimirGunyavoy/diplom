"""research-30: Д1 на ДИ 4D (multi_control_cell.md §Д1). Узел клетки слоя k «работает», если argmin_u J_u = u_k, J_u = min(tgoal(P,u), DTN + V*(step(P,u)))
— тот же выбор, что у агента growN.rollout. Клетка работает при доле таких узлов > THR. Пишет таблицу и слои без неработающих клеток (для LOADL).
Запуск (из v7/, env как solve_b.sh): LOAD=<dump с V> OUT=<папка> python src/.../d1_di4.py"""
import os, sys, json, pickle, time, numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'cells7')); import growN as G
d = pickle.load(open(os.environ['LOAD'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']; O = os.environ['OUT']
US = list(G.US); N = A.N; A._init_qx(); t0 = time.time()
def work(rng):
    a, b = rng; P = A.P[a:b]; J = np.stack([np.minimum(A.tgoal(P, u), G.DTN + A.vstar(G.step(P, u))) for u in US], 1); return a, J.argmin(1).astype(np.int8), J.min(1)
from multiprocessing import get_context
ch = 20000; R = [(a, min(a + ch, N)) for a in range(0, N, ch)]; arg = np.empty(N, np.int8); jm = np.empty(N)
with get_context('fork').Pool(int(os.environ.get('NP', 24))) as pool:
    for i, (a, g, j) in enumerate(pool.imap_unordered(work, R)):
        arg[a:a + len(g)] = g; jm[a:a + len(g)] = j
        if i % 20 == 0: print('d1 %d/%d t=%.0f' % (i, len(R), time.time() - t0), flush=True)
cl = np.concatenate([[k] * len(l) for k, l in enumerate(A.layers)]); off = np.array([c.o for c in A.cells] + [N]); own = np.repeat(cl, np.diff(off))
ok = (A.V < G.BIG / 2) & ~A.goal; wn = (arg == own) & ok
frac = np.array([wn[off[i]:off[i + 1]].mean() for i in range(len(A.cells))]); nn = np.diff(off)
rows = []
for k, u in enumerate(US):
    ci = np.flatnonzero(cl == k)
    rows.append(dict(u=list(map(float, np.atleast_1d(u))), cells=len(ci), working10=int((frac[ci] > .1).sum()), working1=int((frac[ci] > .01).sum()), nodes=int(nn[ci].sum()), nodes_working10=int(nn[ci][frac[ci] > .1].sum()), node_share_argmin=float((arg[ok] == k).mean())))
res = dict(rows=rows, cells=len(A.cells), nodes=int(N), t=time.time() - t0, frac_hist=np.histogram(frac, bins=10, range=(0, 1))[0].tolist())
print(json.dumps(res), flush=True); json.dump(res, open(O + '/d1_di4.json', 'w'), indent=1); np.save(O + '/d1_frac.npy', frac)
for thr in (.1, .01):
    lay = [[c for c, f in zip(l, frac[cl == k]) if f > thr] for k, l in enumerate(A.layers)]
    pickle.dump(dict(layers=lay), open(O + '/L_d1_%g.pkl' % thr, 'wb')); print('слои d1 thr', thr, [len(l) for l in lay], flush=True)
