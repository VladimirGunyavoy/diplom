"""Сходимость дифдрайв-решётки по шагу h: без дуг (Дейкстра) и с дугами (arc_sweeps), поле [−3,3]², общие узлы. -> diffdrive_convergence.json"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.atlas_dd.lattice import solve, arc_sweeps

R, F = {}, {}
for h in (1.0, 0.5, 0.25):
    n = int(round(3 / h)); t0 = time.time()
    nodes, e, T, p, g = solve(n, h)
    T2, it, _ = arc_sweeps(nodes, T, n, h=h, goal=g)
    F[h] = (T, T2); R[str(h)] = dict(nodes=len(nodes), arc_iters=it, sec=round(time.time() - t0, 1))
    print(h, R[str(h)], flush=True)
def stat(a, b, s):      # a — грубое (шаг h_a), b — тонкое; s — отношение шагов
    d = [a[(i, j, k)] - b[(i * s, j * s, k)] for (i, j, k) in a]
    return dict(mean=float(np.mean(d)), max=float(np.max(d)), min=float(np.min(d)))
for name, idx in (("no_arcs", 0), ("arcs", 1)):
    R[name] = {"1->0.5": stat(F[1.0][idx], F[0.5][idx], 2), "0.5->0.25": stat(F[0.5][idx], F[0.25][idx], 2)}
d = [F[0.25][0][k] - F[0.25][1][k] for k in F[0.25][0]]
R["arcs_gain_at_h0.25"] = dict(mean=float(np.mean(d)), max=float(np.max(d)))
json.dump(R, open(os.path.join(os.path.dirname(__file__), "diffdrive_convergence.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
