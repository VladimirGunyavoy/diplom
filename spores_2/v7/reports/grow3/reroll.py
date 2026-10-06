"""rerun the agent (with shooting finish, VF env) on a pickled atlas: python3 reports/grow3/reroll.py reports/grow3/atlas5.pkl"""
import sys, os, pickle, json, numpy as np
sys.path.insert(0, 'src/cells7'); import grow3 as G; import __main__; __main__.Cell = G.Cell; __main__.HexIdx = G.HexIdx
d = pickle.load(open(sys.argv[1], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.idx = []; A.finish(); A.V = d['V']; Q, ref = d['Q'], d['ref']
T, sw, _ = A.rollout(Q); f = np.isfinite(T); rb = np.load(os.path.join('..', 'v5chain', 'reports', 'research', 'dd_refbox_60.npy')) if os.path.exists(os.path.join('..', 'v5chain', 'reports', 'research', 'dd_refbox_60.npy')) and not G.OBST else ref; r = T[f] / rb[f]  # vs the box reference
print(json.dumps(dict(VF=G.VF, NA=G.NA, reach=round(float(f.mean()), 3), fin=A.n_fin, mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), max=round(float(r.max()), 3), min=round(float(r.min()), 3))), flush=True)
