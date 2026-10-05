"""research-12: путь агента (все дуги: мини-дерево + ходьба по рёбрам, TRAJALL=1) → npz (Y старты дуг, U, H) для тёплого старта ocp_arcs.py (WARM=)."""
import numpy as np, sys, os, json
os.environ['TRAJALL'] = '1'; sys.path.insert(0, '.')
import butterfly_dp_query as Q, butterfly_dp_ellipse as E
A = Q.load(sys.argv[1]); E.refresh(A); Q.TRAJ.clear(); A.E = Q.node_edges(A); T, arcs, wm = Q.rollout_edges(A, 81, 0.)
Y = np.array([t[0] for t in Q.TRAJ]); U = np.array([[t[1], t[2]] for t in Q.TRAJ]); H = np.array([t[3] for t in Q.TRAJ])
if np.isfinite(T): H[-1] -= H.sum() - T                                                      # последняя дуга — до входа в цель
np.savez(sys.argv[2], Y=Y, U=U, H=H, T=T)
print(json.dumps(dict(atlas=sys.argv[1], T=round(float(T), 3), T_over_OCP=round(float(T / 7.636), 4), arcs=len(H), sumH=round(float(H.sum()), 3), wmax=round(float(wm), 3))), flush=True)
