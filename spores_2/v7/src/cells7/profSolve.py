"""w23 п.29а2: профиль solve di4 на сохранённых слоях: стенсилы отдельно, затем IT итераций. SYS=di4 ... PESS=1 RHO=.35 LAYERS=reports/di4/L400_rho35.pkl IT=8 python3 src/cells7/profSolve.py"""
import os, sys, time, pickle, cProfile, pstats, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import growN as G, __main__
for n in dir(G):
    if n[0].isupper() and isinstance(getattr(G, n), type): setattr(__main__, n, getattr(G, n))
d = pickle.load(open(os.environ['LAYERS'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); print('узлов', A.N, flush=True)
pr = cProfile.Profile(); t0 = time.time(); pr.enable(); A.solve(it=int(os.environ.get('IT', 8))); pr.disable(); print('solve(it=%s) всего %.1f с' % (os.environ.get('IT', 8), time.time() - t0), 'рёбер', A.edges, flush=True)
pstats.Stats(pr).sort_stats('cumulative').print_stats(14)
