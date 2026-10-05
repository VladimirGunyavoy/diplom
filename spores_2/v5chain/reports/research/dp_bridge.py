"""research-12: связь деревьев ДОВОДКОЙ вместо мостиков. Атлас, где старт не связан с целью (V старта = BIG):
прямое дерево (g от старта, Дейкстра с предками) до узла i → разрыв i → j (ближайший узел обратного дерева, V < BIG) → по рёбрам атласа (лучшее по V) до цели.
Цепочка состояний → дуги solve_arcs (не сошлось — t по формуле ДИ, u = 0) → npz (Y, U, H) для `ocp_arcs.py` WARM= (IPOPT закрывает разрыв).
Кандидаты (i, j): минимум g_i + V_j + LAM·|emb_i − emb_j|; берутся NB лучших с разными j. Запуск: python3 dp_bridge.py atl.npz метка [NB]"""
import numpy as np, sys, os, json
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
sys.path.insert(0, '.')
import butterfly_dp_query as Q, butterfly_dp_ellipse as E
from butterfly_dp_atlas import MN, R
from butterfly_dp_grow import emb
from butterfly_dp import solve_arcs, BIG, wrap, ingoal

LAM = float(os.environ.get('LAM', 3.)); NB = int(sys.argv[3]) if len(sys.argv) > 3 else 4
A = Q.load(sys.argv[1]); E.refresh(A); A.V[A.ingoal] = 0.; A.solve(); KS = 81
iy, k, j0, a, t = A.e; to = k * MN + np.clip(np.round(j0 + a).astype(int), 0, MN - 1); n = A.K * MN
g, pred = dijkstra(csr_matrix((t + 1e-9, (iy, to)), shape=(n, n)), indices=KS * MN + MN // 2, return_predecessors=True)
V = A.V.reshape(-1); P = A.P.reshape(-1, 4); F = np.flatnonzero(np.isfinite(g)); B = np.flatnonzero(V < BIG / 2)
print(json.dumps(dict(nodes_fwd=len(F), nodes_bwd=len(B), V_start=round(float(V[KS * MN + MN // 2]), 3))), flush=True)
XB = emb(P[B]); tr = cKDTree(XB); d, jj = tr.query(emb(P[F]), k=8); cost = g[F][:, None] + V[B][jj] + LAM * d
order = np.argsort(cost.ravel()); used = set(); picks = []
for o in order:
    fi, kk = divmod(int(o), 8); jn = int(B[jj[fi, kk]])
    if jn // MN in used: continue
    used.add(jn // MN); picks.append((int(F[fi]), jn, float(cost.ravel()[o]), float(d[fi, kk])))
    if len(picks) >= NB: break
TE = Q.node_edges(A)
for c, (i, j, cst, dist) in enumerate(picks):
    fw = [i]
    while fw[-1] != KS * MN + MN // 2 and pred[fw[-1]] >= 0: fw.append(int(pred[fw[-1]]))
    fw = fw[::-1]; bw = [j]
    for _ in range(200):                                                                      # по рёбрам: лучшее ребро узла, приход → ближайший узел
        if ingoal(P[bw[-1]][:, None])[0] or bw[-1] not in TE: break
        k2, t2, s2 = TE[bw[-1]][0]; nxt = k2 * MN + int(np.clip(np.round((s2 + R) / (2 * R) * (MN - 1)), 0, MN - 1))
        if nxt in bw: break
        bw.append(nxt)
    nodes = fw + bw; S = P[nodes]; S[0] = A.C[KS]; Y, U, H = [], [], []
    for p0, nd in zip(S[:-1], nodes[1:]):
        k1 = nd // MN; p1 = P[nd]
        tt, u1, u2, s2, ok = solve_arcs(p0[:, None], A.C[k1][:, None], A.n[k1][:, None])          # в отрезок споры узла (как ребро атласа)
        if ok[0] and 0 < tt[0] < 3 and abs(u1[0]) <= 1.05 and abs(u2[0]) <= 1.05: Y.append(p0); U.append((u1[0], u2[0])); H.append(tt[0]); continue
        dq = wrap(p1[:2] - p0[:2]); wb = (p0[2:] + p1[2:]) / 2; T0 = float(np.clip(np.linalg.norm(dq) / max(np.linalg.norm(wb), .5), .05, 1.5))
        Y.append(p0); U.append((0., 0.)); H.append(T0)
    Y, U, H = np.array(Y), np.clip(np.array(U), -1, 1), np.array(H)
    np.savez('arcs_%s_b%d.npz' % (sys.argv[2], c), Y=Y, U=U, H=H, T=np.inf)
    print(json.dumps(dict(cand=c, fwd_nodes=len(fw), bwd_nodes=len(bw), g=round(float(g[i]), 3), V=round(float(V[j]), 3), gap=round(dist, 3), arcs=len(H), sumH=round(float(H.sum()), 3))), flush=True)
