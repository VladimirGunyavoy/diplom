"""w17, PLAN п.6: бабочки дд + 2 диска (v7 butterfly_dd) против ЭТАЛОНА С ДИСКАМИ (dd_obst_ref_KB60_NG21.npy: x, y, θ, ref; диски ref — [[1,.3,.45],[-.8,-.9,.4]]).
Запуск из spores_2/v7: RW=.9 python3 reports/bdd/obst_ref.py N"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, '.')
from src.cells7 import butterfly_dd as BD
DISKS = [(1.0, 0.3, .45), (-0.8, -0.9, .4)]
ref = np.load('/home/rl/claude-work/projects/spore/spores_2/v5chain/reports/research/dd_obst_ref_KB60_NG21.npy'); Q, R0 = ref[:, :3], ref[:, 3]
free = np.array([all(np.hypot(q[0] - cx, q[1] - cy) > r + .1 for cx, cy, r in DISKS) for q in Q]); Q, R0 = Q[free], R0[free]
N = int(sys.argv[1]); t0 = time.time(); B = BD.ButterflyDDObs(N=N, obs=DISKS).solve(); print(json.dumps(dict(N=N, RW=BD.M.RW, спор=B.K, пар=B.npairs, сек=round(time.time() - t0))), flush=True)
R = [BD.run(B, q, 0.) for q in Q]; T = np.array([a for a, _ in R]); S = np.array([b for _, b in R]); f = np.isfinite(T); r = T[f] / R0[f]
print(json.dumps(dict(n=len(Q), reach=round(float(f.mean()), 3), T_over_ref_obst=dict(mean=round(float(r.mean()), 3), med=round(float(np.median(r)), 3), p90=round(float(np.percentile(r, 90)), 3), max=round(float(r.max()), 3), min=round(float(r.min()), 3)), segs_med=float(np.median(S[f])), сек=round(time.time() - t0))), flush=True)
