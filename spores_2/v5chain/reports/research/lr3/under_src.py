"""lr3: откуда берётся занижение V (V < T_ref). Для каждого узла: значение перехода E3 (по итоговой V), доноры-вершины и их ошибки.
Источник занижения = узел, у которого E3 < T_ref − tol, хотя все вершины ячейки-донора сами не занижены (|V−T_ref| вершин мал)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S
from src.algo import ref_di_disk as R
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1.], 0, par=False); G = S.make_graph(g, atlas, 'N'); G.solve(verbose=True); BIG = S.BIG; rho = G.rho
V = np.where(G.V < BIG / 2, G.V, np.nan); T = R.T_ref(G.P, rho); err = V - T; out = np.linalg.norm(G.P, axis=1) > rho
print('узлов', G.n, 'конечных', np.isfinite(V).sum()); q = lambda x: np.round(np.nanpercentile(x, [1, 10, 25, 50, 75, 90, 99]), 3); print('V−T_ref по узлам', q(err[out]))
for li, u in enumerate(G.us): m = out & (G.layer == li); print(' слой', u, q(err[m]), 'доля < −.05:', round(float((err[m] < -.05).mean()), 3))
# цепочка без переходов: только E1+E2 (точно вдоль своей траектории до круга)
Vc = G.V0.copy(); e1 = G.e1; lev = G.rowid[e1]; o = np.argsort(-lev, kind='stable')
for i in e1[o]: Vc[i] = min(Vc[i], G.cost[i] + Vc[G.nxt[i]])
Vc = np.where(Vc < BIG / 2, Vc, np.nan); ec = Vc - T; print('только своя траектория до круга (без переходов): узлов', np.isfinite(Vc).sum(), 'Vc−T_ref', q(ec[out]), 'min', np.nanmin(ec[out]))
# вклад каждого перехода
rep = []
for li2, u2 in enumerate(G.us):
    pi, verts, w, rank = G.e3[li2]; m = rank == 0; pi, verts, w = pi[m], verts[m], w[m]; Vv = V[verts]; val = (w * Vv).sum(1); d = val - T[pi]; ve = (Vv - T[verts]); cross = G.layer[pi] != li2
    spreadT = T[verts].max(1) - T[verts].min(1); lin_true = (w * T[verts]).sum(1) - T[pi]          # ошибка самой билинейной интерполяции ТОЧНОЙ функции в этой ячейке
    ok = np.isfinite(d) & out[pi]
    print('переход в слой', u2, ': пар', ok.sum(), ' E3−T_ref', q(d[ok]), '\n    интерполяция точной T_ref в тех же ячейках − T_ref', q(lin_true[ok]), '\n    из них через слой (переключение):', q(lin_true[ok & cross]), ' внутри слоя (стык клеток):', q(lin_true[ok & ~cross]))
    src = ok & (lin_true < -.05); print('    пар с занижением интерполяции точной функции > .05:', int(src.sum()), 'из', int(ok.sum()), '; разброс T_ref по вершинам у них (мед):', round(float(np.median(spreadT[src])), 3) if src.any() else None, ' у остальных:', round(float(np.median(spreadT[ok & ~src])), 3))
    for i in np.argsort(np.where(ok, lin_true, 0))[:4]: rep.append((float(lin_true[i]), G.P[pi[i]].round(3).tolist(), float(T[pi[i]].round(3)), G.P[verts[i]].round(3).tolist(), T[verts[i]].round(3).tolist(), w[i].round(2).tolist(), float(G.us[G.layer[pi[i]]]), float(u2)))
for r in sorted(rep)[:6]: print(' худшая: ошибка %.3f узел %s T_ref %.3f слой %s → слой %s\n    вершины %s\n    T_ref вершин %s веса %s' % (r[0], r[1], r[2], r[6], r[7], r[3], r[4], r[5]))
