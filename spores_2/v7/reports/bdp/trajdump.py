"""w18 (PLAN п.12): путь агента по атласу (TRAJALL=1: мини-дерево + ход по рёбрам) → npz (Y старты дуг, U, H, T) для src.cells7.refine. Запуск как fp.py (cwd v7, PYTHONPATH=.)."""
import sys, os, json
import numpy as np
os.environ['TRAJALL'] = '1'
from src.cells7 import butterfly_dp as B
A = B.load(sys.argv[1]); B.refresh(A); A.V[A.ingoal] = 0.; A.solve(); A.E = B.node_edges(A); B.TRAJ.clear()
T, arcs, wm = B.rollout_edges(A, B.KS, 0.)
Y = np.array([t[0] for t in B.TRAJ]); U = np.array([[t[1], t[2]] for t in B.TRAJ]); H = np.array([t[3] for t in B.TRAJ])
if np.isfinite(T) and len(H): H[-1] -= H.sum() - T
np.savez(sys.argv[2], Y=Y, U=U, H=H, T=T)
print(json.dumps(dict(atlas=sys.argv[1].split('/')[-1], FORCEPLAN=B.FORCEPLAN, T=round(float(T), 3), T_over_OCP=round(float(T / B.OCP), 4), arcs=len(H), sumH=round(float(H.sum()), 3), wmax=round(float(wm), 3))), flush=True)
