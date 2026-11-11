"""Д1 на каре (worker-32): по готовому c1600.pkl (слои + V) для каждого узла — значение DTN + interp(V) при каждом из 6 управлений (стенсилы шага узла, как в solve), argmin_u; клетка «работает», если >10% её узлов имеют argmin = её u (слой). Без пересборки.
Запуск (aida, из bench/): LOAD=~/spore_v5/w24/v7/reports/car/c1600.pkl OUT=… SYS=car STGPU=1 python3 ../v7/reports/big/05_car_d1/d1_car.py"""
import os, sys, json, pickle, time; os.environ.setdefault('SYS', 'car'); os.environ.setdefault('STGPU', '1'); os.environ.setdefault('NOLATCH', '1')
sys.path.insert(0, os.path.join(os.getcwd(), '..', 'v7', 'src', 'cells7')); import numpy as np, growCar as G, __main__
from tqdm import tqdm
for n_ in dir(G):
    if n_[0].isupper() and isinstance(getattr(G, n_), type): setattr(__main__, n_, getattr(G, n_))
d = pickle.load(open(os.path.expanduser(os.environ['LOAD']), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); V = d['V']; N = A.N; BIG = G.BIG
print('nodes', N, 'layers', [len(l) for l in A.layers], 'US', G.US, flush=True)
best = np.full((len(G.US), N), np.inf, np.float32)
CH = int(os.environ.get('CH', 1000000))   # узлы порциями: стенсилы 4.6M×(4 idx + 4 w) целиком не влезают в RAM при загрузке aida
for ui, u in enumerate(tqdm(G.US, desc='u', mininterval=10)):
    for s0 in range(0, N, CH):
        P = A.P[s0:s0 + CH]; a, b, c = A.stencils(G.step(P, u)); a = a + s0; vi = V[b]; ok = vi < BIG / 2; w = c * ok; sm = w.sum(1); mx = np.where(ok, vi, -np.inf).max(1)
        v = (w * np.where(ok, vi, 0)).sum(1) + np.nan_to_num((1 - sm) * (mx + max(G.PESS, 0.)), nan=0., posinf=0., neginf=0.); v[sm < G.WTHR] = BIG; val = (G.DTN + v).astype(np.float32)
        o = np.argsort(a, kind='stable'); a, val = a[o], val[o]; st = np.flatnonzero(np.r_[True, a[1:] != a[:-1]]); best[ui, a[st]] = np.minimum.reduceat(val, st); del a, b, c, vi, ok, w, v, val, o
        best[ui, s0:s0 + CH] = np.minimum(best[ui, s0:s0 + CH], A.tgoal(P, u).astype(np.float32))
    print('u', u, 'готово', flush=True)
am = best.argmin(0); mn = best.min(0); fin = mn < BIG / 2
cell_layer = np.concatenate([[k] * len(l) for k, l in enumerate(A.layers)]); rows = []
for ci, c in enumerate(A.cells):
    n = c.G.shape[0] * G.M ** G.m_; sl = slice(c.o, c.o + n); k = cell_layer[ci]; f = fin[sl]
    rows.append(dict(ci=ci, layer=int(k), nodes=int(n), fin=float(f.mean()), own=float((am[sl][f] == k).mean()) if f.any() else 0., own_tie=float((best[k, sl][f] <= mn[sl][f] + 1e-3).mean()) if f.any() else 0.))
json.dump(dict(US=[list(map(float, u)) for u in G.US], cells=rows, layer_arg_frac=[[float((am[fin] == k2).mean()) for k2 in range(len(G.US))]]), open(os.path.join(os.environ.get('OUT', '.'), 'car_d1.json'), 'w'))
R = rows
for k in range(len(G.US)):
    rk = [r for r in R if r['layer'] == k]; wk = [r for r in rk if r['own'] > .1]; wt = [r for r in rk if r['own_tie'] > .1]
    print('layer', k, G.US[k], 'клеток', len(rk), 'узлов', sum(r['nodes'] for r in rk), '| работают (argmin) %d кл. %d узлов' % (len(wk), sum(r['nodes'] for r in wk)), '| работают (own≈best) %d кл. %d узлов' % (len(wt), sum(r['nodes'] for r in wt)), flush=True)
print('доля узлов V<inf', float(fin.mean()))
