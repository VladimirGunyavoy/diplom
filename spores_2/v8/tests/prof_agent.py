import os, sys, time, cProfile, pstats, io
os.environ.setdefault('V8_MAXCELLS', '0'); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.stepper import live
import numpy as np
from src.algo import growN as g
from src.stepper import vsurf as VS
cl = {u: [g.cell_dict(c) for c in g.build_layer(u, np.random.default_rng(k))[0]] for k, u in enumerate(live.LAYERS)}
Vs, info = VS.solve_layers(cl, calc='q'); G = info.pop('G'); G.build_table()
pr = cProfile.Profile(); pr.enable(); VS.agent_paths(G, cl=cl); pr.disable(); s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(14); print('\n'.join(l[:140] for l in s.getvalue().splitlines() if l.strip())[-1800:])
