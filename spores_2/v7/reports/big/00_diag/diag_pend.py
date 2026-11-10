"""BIG0 диагноз маятника u .3 (VARIANT=56, как строка бенча по T). Запуск из bench/ на aida: python3 diag_pend.py
Выход: data/pend_cells.json (по клетке: u, r, nb, nf, узлов, объём, why), data/pend_drop' + os.environ.get('SFX','') + '.json (drop q% самых мелких)."""
import os, sys, json, time, copy
os.environ['VARIANT'] = '56'; HERE = os.getcwd(); sys.path.insert(0, HERE)
import spores_pend as SP   # ставит env как строка бенча по T, импортирует grow_cells2d
import numpy as np, common; G = SP.G
OUT = os.environ.get('BIGOUT', os.path.join(HERE, '..', 'v7', 'reports', 'big', '00_diag', 'data'))
e = common.make('pend'); Q = e.X0
def evalA(A):
    Ts = []
    for i in range(len(Q)):
        T, rec = SP.rollout_rec(A, Q[i:i+1]); us = [[r[0][0]] for r in rec if r[2][0]]; dts = [float(r[1][0]) for r in rec if r[2][0]]
        Tv, why = common.verify(e, Q[i], us, dts); Ts.append(Tv)
    return common.summarize(e, Ts)
t0 = time.perf_counter(); A = G.Atlas(); tg = time.perf_counter() - t0
rows = []
for c in A.cells:
    nt = c.G.shape[0]; L = nt * G.DTN; w = 2 * c.r
    rows.append(dict(u=c.u, r=float(c.r), w=float(w), nrows=int(nt), nb=int(c.nb), nf=int(c.nf), m=int(c.m), nodes=int(nt * c.m), L=float(L), vol=float(w * L), why={k: str(v) for k, v in getattr(c, 'why', {}).items()}))
json.dump(dict(t_grow=tg, cells=rows, nodes=int(A.N)), open(OUT + '/pend_cells.json', 'w'))
print('cells', len(rows), 'nodes', A.N, 'why sample', rows[0]['why'], flush=True)
vol = np.array([r['vol'] for r in rows]); order = np.argsort(vol)
res = {}
for q in [int(x) for x in os.environ.get('QS', '0,10,25,50').split(',')]:
    kill = set(order[:int(len(order) * q / 100)].tolist()); A2 = copy.copy(A); A2.layers = []; n = 0; ci = 0
    for l in A.layers:
        ll = []
        for c in l:
            if ci not in kill: ll.append(c)
            ci += 1
        A2.layers.append(ll)
    for a in ('qx',): 
        if hasattr(A2, a): delattr(A2, a)
    A2.finish(); t1 = time.perf_counter(); A2.solve(); ts = time.perf_counter() - t1
    s = evalA(A2); res[q] = dict(cells=len(A2.cells), nodes=int(A2.N), t_solve=ts, reached=s['reached'], ratio_med=s['ratio_med'], ratio_p90=s['ratio_p90'], ratio_max=s['ratio_max']); print('drop', q, res[q], flush=True)
    json.dump(res, open(OUT + '/pend_drop' + os.environ.get('SFX','') + '.json', 'w'), indent=1, default=float)
