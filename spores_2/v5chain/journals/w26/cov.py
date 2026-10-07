import os, sys, pickle, __main__
import numpy as np
sys.path.insert(0, os.path.expanduser('~/spore_v5/w25/v7/src/cells7')); import growN as G
for n in dir(G):
    if n[0].isupper() and isinstance(getattr(G, n), type): setattr(__main__, n, getattr(G, n))
d = pickle.load(open(os.path.expanduser('~/spore_v5/w25/dp1_L3000.pkl'), 'rb'))
print(type(d), list(d.keys()) if isinstance(d, dict) else '')
L = d['layers'][0]; c = L[0]
print(type(c), [k for k in dir(c) if not k.startswith('_')][:30] if not isinstance(c, dict) else list(c))
for u, L in enumerate(d['layers']):
    r = np.array([getattr(x, 'r', np.nan) if not isinstance(x, dict) else x.get('r', np.nan) for x in L])
    for K in (500, 1000, 1500, 2000, 3000):
        rr = r[:K]; print(u, K, 'sum r^4 %.3g' % np.nansum(rr**4), 'r<.05: %d' % (rr < .05).sum(), 'med r %.3f' % np.nanmedian(rr))
