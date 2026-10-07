"""w23 п.29а2: solve_wl против эталонной V (A.solve). LAYERS, VREF — pkl; тот же env, что у прогона."""
import os, sys, time, pickle, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import growNq as Q, __main__
G = Q.G
for n in dir(G):
    if n[0].isupper() and isinstance(getattr(G, n), type): setattr(__main__, n, getattr(G, n))
d = pickle.load(open(os.environ['LAYERS'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); Vr = pickle.load(open(os.environ['VREF'], 'rb'))['V']
for r in Q.solve_wl(A, tols=tuple(float(x) for x in os.environ.get('TOLS', '1e-3,1e-4').split(','))):
    V = r.pop('V'); f = (Vr < G.BIG / 2) & (V < G.BIG / 2); r.update(maxdiff=float(np.abs(Vr[f] - V[f]).max()), meandiff=float(np.abs(Vr[f] - V[f]).mean()), finite_ref=int((Vr < G.BIG / 2).sum()), finite_wl=int((V < G.BIG / 2).sum())); print(json.dumps(r), flush=True)
