"""где дребезг у маятника при ρ=0 (агент по сетке V, набор spec11)."""
import numpy as np, json
from pend_cost_grid import *
rng = np.random.default_rng(0); Q = np.stack([rng.uniform(-np.pi, np.pi, 60), rng.uniform(-2, 2, 60)], 1)
S = Grid(); U = np.linspace(-1, 1, 11); S.solve(tuple(U), 0.)
x, w = Q[:, 0].copy(), Q[:, 1].copy(); done = S.goal(x, w); pu = np.full(60, np.nan); E, D, Tt = [], [], []; t = 0.
for _ in range(int(60 / S.dt)):
    a = np.nonzero(~done)[0]
    if not len(a): break
    c = np.stack([S.dt + S.Vq(*step(x[a], w[a], u * UM, S.dt)) for u in U], 1); u = U[np.argmin(c, 1)]; Vh = np.min(c, 1)
    p = pu[a]; jmp = np.isfinite(p) & (np.abs(u - np.where(np.isfinite(p), p, u)) > .5)
    for i in np.nonzero(jmp)[0]: E.append(w[a[i]]**2 / 2 + np.cos(x[a[i]])); D.append(np.hypot(wrap(x[a[i]]), w[a[i]])); Tt.append(Vh[i])
    pu[a] = u; x[a], w[a] = step(x[a], w[a], u * UM, S.dt); done[a[S.goal(x[a], w[a])]] = True
E, D, Tt = map(np.array, (E, D, Tt))
print(json.dumps(dict(switches=len(E), per_traj=round(len(E) / 60, 1), share_near_goal_D_lt_05=round(float((D < .5).mean()), 3), share_T_to_go_lt_1=round(float((Tt < 1).mean()), 3),
      E_quantiles=np.round(np.percentile(E, [10, 50, 90]), 3).tolist(), share_E_near_1=round(float((np.abs(E - 1) < .05).mean()), 3))))
