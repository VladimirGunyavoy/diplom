"""кар: tgoal numpy vs GPU (TGGPU), тождественность и время. Из ~/spore_v5/a31/spores_2/v7."""
import os, sys, time, pickle, numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'cells7')); G = __import__('growCar')
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
d = pickle.load(open(os.environ['LOADL'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish()
for u in G.US[:2]:
    os.environ['TGGPU'] = '0'; t = time.time(); a = A.tgoal(A.P, u); ta = time.time() - t
    os.environ['TGGPU'] = '1'; A.tgoal(A.P[:200000], u); t = time.time(); b = G._tgoal_gpu(A.P, u); tb = time.time() - t
    f = np.isfinite(a) & np.isfinite(b); print(u, 'numpy %.1f с, gpu %.2f с' % (ta, tb), 'inf-расхождений', int((np.isfinite(a) != np.isfinite(b)).sum()), 'max|Δ|', float(np.abs(a[f] - b[f]).max()) if f.any() else None, flush=True)
