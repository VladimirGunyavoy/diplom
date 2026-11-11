import os, sys, cProfile, pstats, pickle, inspect, __main__
os.environ.setdefault('SYS', 'car'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'v7', 'src', 'cells7'))
import numpy as np, growCar as G, common, envs_agentA as EA
for nm, o in vars(G).items():
    if inspect.isclass(o): setattr(__main__, nm, o)
d = pickle.load(open(os.path.expanduser(os.environ['LOAD']), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']
ag = EA.AgentA(G, A, pad=0., static=True); x = common.make('car').X0[8]; haz = EA.StaticHaz()
A.stencils(np.asarray(x, float)[None]); import time; t0=time.perf_counter(); A.stencils(np.asarray(x, float)[None]); print('прогрев готов')
pr = cProfile.Profile(); pr.enable(); r = EA.episode(ag, haz, x, tmax=3.); pr.disable()
print('шагов', r['steps'], 'мс/шаг', r['ms']); pstats.Stats(pr).sort_stats('cumulative').print_stats(14)
