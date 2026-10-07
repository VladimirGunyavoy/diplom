import os, sys, pickle, numpy as np, __main__
sys.path.insert(0, os.path.expanduser('~/spore_v5/w25/v7/src/cells7')); import growN as G
for n in dir(G):
    if n[0].isupper() and isinstance(getattr(G, n), type): setattr(__main__, n, getattr(G, n))
d = pickle.load(open(os.environ['LF'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = np.load(os.environ['LV'])
Q, ref = G.starts_ref(); ref = np.load(os.environ['DI4REF_ABS'])[:, 6]
C = A.vstar(G.wrapy(Q)); T = A.rollout(G.wrapy(Q))[0]
print('V*==inf', int((C >= G.BIG / 2).sum()), 'T inf', int((~np.isfinite(T)).sum()))
for i in range(60): print(i, round(float(ref[i]), 3), round(float(C[i]), 3), round(float(T[i]), 3))
