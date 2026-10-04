"""research-11: фильтр рёбер атласа по |w| ≤ 3 ВДОЛЬ дуги (при построении предел проверялся только в концах — путь по графу давал wmax 3.09, а агент
с проверкой вставал). Для каждого ребра дуга восстанавливается (solve_arcs), скорость меряется в NF точках; плохие рёбра удаляются, V пересчитывается.
Запуск: G=2 WIN=1.0 python3 dp_wfilter.py atlas.npz  →  atlas_wf.npz"""
import numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from butterfly_dp import flow, solve_arcs, BIG, WMAX
from butterfly_dp_atlas import R, MN
import butterfly_dp_query as Q, butterfly_dp_ellipse as E
A = Q.load(sys.argv[1]); E.refresh(A); iy, k, j0, a, t = A.e; Y = A.P.reshape(-1, 4); t0 = time.time(); NF = int(os.environ.get('NF', 8)); keep = np.zeros(len(iy), bool); wm = np.zeros(len(iy))
for c0 in range(0, len(iy), 20000):
    sl = slice(c0, c0 + 20000); y = Y[iy[sl]].T; tt, u1, u2, sv, ok = solve_arcs(y, A.C[k[sl]].T, A.n[k[sl]].T); w = np.zeros(len(tt))
    for fr in np.arange(1, NF + 1) / NF: z = flow(y, u1, u2, tt * fr); w = np.maximum(w, np.nan_to_num(np.abs(z[2:]).max(0), nan=99.))
    keep[sl] = ok & (w <= WMAX); wm[sl] = w
V0 = float(A.V[81, MN // 2]); A.e = [x[keep] for x in A.e]; A.V[:] = BIG; A.V[A.ingoal] = 0.; A.solve()
print(json.dumps(dict(edges=len(iy), dropped=int((~keep).sum()), over_w=int((wm > WMAX).sum()), V_before=round(V0, 3), V_after=round(float(A.V[81, MN // 2]), 3), sec=round(time.time() - t0))), flush=True)
np.savez(sys.argv[1].replace('.npz', '_wf.npz'), C=A.C, n=A.n, V=A.V, e=np.array(A.e, dtype=object), allow_pickle=True)
