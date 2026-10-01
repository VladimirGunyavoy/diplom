"""research hub-research-7: маятник faces — достижимость vs размер цели (гипотеза: крошечная цель у седла; LQR-клетка расширяет цель)."""
import sys, json; sys.path.insert(0, '.'); sys.argv = ['x']
exec(open('pend_faces_diag.py').read().split("d0 = float")[0])
ln = lines_graded(.02, 1.0, 400, d0=.05)
for gx, gv in ((None, None), (.03, .03), (.05, .05), (.08, .08)):
    S = pend_top(); Q = Qphys / np.array([np.pi, 4.0]); F = Faces(S, ln, ln, m=4, periodic_x=True, tmax=40.0)
    if gx: F.ax, F.av = gx, gv
    V = F.solve(); T, _ = F.rollout(Q)
    print(json.dumps(dict(goal_phi_rad=round(float(F.ax * np.pi), 3), goal_w=round(float(F.av * 4), 3), finV=round(float(np.isfinite(V).mean()), 3), reach=round(float(np.isfinite(T).mean()), 3))), flush=True)
