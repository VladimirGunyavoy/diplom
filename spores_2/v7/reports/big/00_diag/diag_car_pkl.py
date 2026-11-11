"""BIG0 кар: статистика клеток из готового дампа слоёв L1600.pkl (growCar). Без пересборки; Cell — заглушка (атрибуты из pkl)."""
import pickle, sys, json, numpy as np
class Cell: pass
d = pickle.load(open(sys.argv[1], 'rb')); layers = d['layers']; out = []
DTN = float(sys.argv[3]) if len(sys.argv) > 3 else None
c0 = layers[0][0]; print('attrs', sorted(c0.__dict__.keys()), 'r', c0.r, 'G', c0.G.shape, flush=True)
for li, l in enumerate(layers):
    for c in l:
        nt = c.G.shape[0]; side = np.prod(2 * np.asarray(c.r))
        out.append(dict(layer=li, r=[float(x) for x in np.atleast_1d(c.r)], nrows=int(nt), nodes=int(c.G.shape[0] * np.prod(c.G.shape[1:-1])), nb=int(c.nb), nf=int(c.nf), side=float(side), vol_rel=float(side * nt), why=getattr(c, 'why', None)))
json.dump(dict(cells=out, layers=len(layers)), open(sys.argv[2], 'w'))
print(len(out), 'cells', sum(o['nodes'] for o in out), 'nodes')
