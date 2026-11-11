"""b14 ENV1-клип: вырождение клеток слоя по v. Строит один слой (UL=a,w) до MAXC клеток под текущим VSAT; для каждой клетки: полуширина по v (сумма |r_k·e_k[3]|, ящик в боковых осях), размах v по узлам, время. Запуск: VSAT=0|1 UL=1.5,0 MAXC=60 python vsat_degen.py out.json"""
import os, sys, json, time
os.environ.setdefault('SYS', 'car')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'v7', 'src', 'cells7'))
import numpy as np, growCar as G
u = tuple(float(v) for v in os.environ['UL'].split(',')); t0 = time.time(); cells = G.build_layer(u, np.random.default_rng(0), None)[0]; dt = time.time() - t0
rv = np.array([float(np.sum(np.abs(np.asarray(c.r) * np.asarray(c.e)[:, 3]))) for c in cells]); span = np.array([float(np.ptp(c.G[..., 3])) for c in cells]); nrow = np.array([c.G.shape[0] for c in cells])
out = dict(u=u, vsat=int(os.environ.get('VSAT', 0)), cells=len(cells), sec=dt, sec_per_cell=dt / max(len(cells), 1), rv_med=float(np.median(rv)), frac_rv_lt_1e2=float((rv < 1e-2).mean()), span_med=float(np.median(span)), frac_span_lt_1e2=float((span < 1e-2).mean()),
           rows_med=float(np.median(nrow)), rv=rv.tolist(), span=span.tolist(), v_mean=[float(c.G[..., 3].mean()) for c in cells])
json.dump(out, open(sys.argv[1], 'w')); print({k: v for k, v in out.items() if k not in ('rv', 'span', 'v_mean')})
