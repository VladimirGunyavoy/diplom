"""w24 п.36: диагностика 20 стартов car по готовому атласу (LOAD=pkl): V*(старт), T агента, дошёл ли."""
import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src/cells7')); import growCar as G
import __main__
for n_ in dir(G):
    if n_[0].isupper() and isinstance(getattr(G, n_), type): setattr(__main__, n_, getattr(G, n_))
d = pickle.load(open(os.environ['LOAD'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']; Q, ref = G.starts_ref()
V0 = A.vstar(G.wrapy(Q)); T, sw, pth = A.rollout(Q); import time
t = time.time(); A.rollout(Q[:1]); tq = time.time() - t
for i in range(len(Q)): print(i, 'V*', round(float(V0[i]), 2), 'ref', round(float(ref[i]), 2), 'T', round(float(T[i]), 2), 'sw', int(sw[i]), 'финал', np.round(pth[-1, i], 2).tolist(), 'dobs', round(float(G.dobs(pth[-1, i])), 2))
qt = []
for i in range(len(Q)):
    t = time.time(); A.rollout(Q[i:i + 1]); qt.append(time.time() - t)
print('q_ms мед./p90/max:', [round(float(x) * 1000) for x in (np.median(qt), np.percentile(qt, 90), max(qt))])
print('rollout 1 старт, с:', round(tq, 2))
