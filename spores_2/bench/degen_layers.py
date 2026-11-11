"""b14: вырождение клеток по v из дампа слоёв (DUMPL): на слой — число клеток, доля с полушириной по v < 1e-2 (rv = Σ|r_k·e_k[3]|), размах v по узлам. Запуск: python degen_layers.py layers.pkl out.json"""
import os, sys, pickle, json
os.environ.setdefault('SYS', 'car'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'v7', 'src', 'cells7'))
import numpy as np, growCar as G, __main__, inspect
for nm, o in vars(G).items():
    if inspect.isclass(o): setattr(__main__, nm, o)
L = pickle.load(open(sys.argv[1], 'rb'))['layers']; out = []
for u, cells in zip(G.US, L):
    rv = np.array([float(np.sum(np.abs(np.asarray(c.r) * np.asarray(c.e)[:, 3]))) for c in cells]); sp = np.array([float(np.ptp(c.G[..., 3])) for c in cells]); vm = np.array([float(c.G[..., 3].mean()) for c in cells])
    top = vm > 1.9
    out.append(dict(u=list(u), cells=len(cells), frac_rv_lt_1e2=float((rv < 1e-2).mean()), frac_span_lt_1e2=float((sp < 1e-2).mean()), n_top=int(top.sum()), top_frac_rv_lt_1e2=float((rv[top] < 1e-2).mean()) if top.any() else None, top_span_med=float(np.median(sp[top])) if top.any() else None, rv_med=float(np.median(rv)), nodes=int(sum(c.G.shape[0] * 9 for c in cells))))
    print(out[-1], flush=True)
json.dump(out, open(sys.argv[2], 'w'))
