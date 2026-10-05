"""w18 (PLAN п.13): несколько путей на одном атласе (старт из разных точек s отрезка стартовой споры × FORCEPLAN 0/1) → доводка каждого → лучший.
Запуск (aida, cwd v7, PYTHONPATH=.:~/spore_v5/r5/pylib, G=2 WCHK=1 WFILT=1): python3 reports/bdp/multipath.py atlas.npz [NS=5]"""
import sys, os, json, time
import numpy as np
os.environ['TRAJALL'] = '1'
from src.cells7 import butterfly_dp as B, refine as RF
A = B.load(sys.argv[1]); B.refresh(A); A.V[A.ingoal] = 0.; A.solve(); A.E = B.node_edges(A)
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 5; res = []
for s in np.linspace(-.8 * B.R, .8 * B.R, NS):
    for fp in (0, 1):
        B.FORCEPLAN = fp; B.TRAJ.clear(); B.PEND[:] = []; T, arcs, wm = B.rollout_edges(A, B.KS, float(s))
        if not np.isfinite(T) or not len(B.TRAJ): print(json.dumps(dict(s=round(float(s), 3), fp=fp, T=None)), flush=True); continue
        Y = np.array([t[0] for t in B.TRAJ]); U = np.array([[t[1], t[2]] for t in B.TRAJ]); H = np.array([t[3] for t in B.TRAJ]); H[-1] -= H.sum() - T
        Tr, _, _, wr, ok, st, sec = RF.refine(Y, U, H); res.append(Tr)
        print(json.dumps(dict(s=round(float(s), 3), fp=fp, T_path=round(float(T), 3), arcs=len(H), T_ref=round(Tr, 4), ratio=round(Tr / RF.OCP, 4), ok=ok, wmax=round(wr, 3), sec=round(sec))), flush=True)
r = np.array(res); print('BEST', json.dumps(dict(best=round(float(r.min()), 4), ratio=round(float(r.min() / RF.OCP), 4), median=round(float(np.median(r[np.isfinite(r)])), 4), n=len(r), nok=int(np.isfinite(r).sum()))), flush=True)
