import os, sys, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import envs2026 as E, growCar as G, common, __main__, inspect
for nm, o in vars(G).items():
    if inspect.isclass(o): setattr(__main__, nm, o)
d = pickle.load(open(os.path.expanduser(os.environ['LOAD']), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']
e = common.make('car'); i = int(os.environ.get('I', 0)); y = e.X0[i].copy(); n = 0
while n < 100:
    tgs = np.array([A.tgoal(y[None], u)[0] for u in G.US]); vs = np.array([A.vstar(G.step(y[None], u))[0] for u in G.US]); J = np.minimum(tgs, G.DTN + vs)
    r = E.plan(G, A, y)
    print(n, np.round(y, 2), 'dist', round(float(np.linalg.norm(y[:2] - E.GOAL)), 2), 'Jmin', round(float(J.min()), 2), 'plan', None if r is None else (round(r[2], 2), len(r[0])), flush=True)
    if r is None or np.linalg.norm(y[:2] - E.GOAL) <= E.GR: break
    us, dts, _ = r; cum = np.cumsum(dts); k = min(int(np.searchsorted(cum, .04)), len(us) - 1); y = E.their_step(y, us[k]); n += 1
