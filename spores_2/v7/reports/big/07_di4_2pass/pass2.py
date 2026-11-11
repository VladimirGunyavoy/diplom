"""Двухпроходная ДИ 4D, проход 2: рост слоёв (DTN .2 M 3, как база) с фильтром затравок по полю u* прохода 1: слой u принимает затравку p, только если u*(ближайший узел поля) = u или зазор поля < TAU (полоса у переключений);
затем solve и прокатка 60 стартов (как r30/di4/run_seed.py). growN не правится: build_layer_par подменяется копией с проверкой. Env: FIELD, TAU, OUTJ, SEED."""
import os, sys, json, time, inspect, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'cells7')); import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
f = np.load(os.environ['FIELD']); TREE = cKDTree(f['P']); ARG = f['arg']; GAP = f['gap']; TAU = float(os.environ.get('TAU', .3)); UI = {tuple(np.atleast_1d(np.round(u, 6)).tolist()): k for k, u in enumerate(G.US)}
def allowed(p, u): d, i = TREE.query(p); return bool(ARG[i] == UI[tuple(np.atleast_1d(np.round(u, 6)).tolist())] or GAP[i] < TAU)
src = inspect.getsource(G.build_layer_par); key = "            if ingoal(p): continue\n"; assert key in src
exec(src.replace(key, key + "            if not allowed(p, u): continue\n", 1), G.__dict__); G.allowed = allowed
t0 = time.time(); A = G.Atlas(seed=int(os.environ.get('SEED', 0))); tb = time.time() - t0; t1 = time.time(); A.solve(); ts = time.time() - t1
Q, ref = G.starts_ref(); t2 = time.time(); T, sw, _ = A.rollout(Q); tq = time.time() - t2; fz = np.isfinite(T) & np.isfinite(ref); r = T[fz] / ref[fz]
out = dict(TAU=TAU, DTN=G.DTN, M=G.M, cells=[len(l) for l in A.layers], nodes=int(A.N), t_build=tb, t_solve=ts, t_query60=tq, reach=float(np.isfinite(T).mean()), mean=float(r.mean()), med=float(np.median(r)), p90=float(np.percentile(r, 90)), max=float(r.max()), T=[float(x) for x in T])
print(json.dumps({k: v for k, v in out.items() if k != 'T'}), flush=True); json.dump(out, open(os.environ['OUTJ'], 'w'), indent=1)
