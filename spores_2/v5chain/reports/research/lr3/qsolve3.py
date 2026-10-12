"""lr3: переключение ВНУТРИ шага строки. Гипотеза: остаточная ошибка Q-решения (+.06 мед.) — оттого, что переключаться можно только в узлах,
а строки стоят редко (шаг по времени до HMAX): узел перед кривой переключения платит за перелёт до следующего узла.
Правка: на каждом ребре вдоль потока K промежуточных точек (точный поток ДИ); Q(узел) = min(…, t_j + интерполяция Q чужого слоя в точке j).
env: K (список), QUAD=1 — квадратичная поперёк."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S, qsolve as QS, qsolve2 as Q2
from src.algo import ref_di_disk as R
BIG = S.BIG

def prep_sub(G, K):
    n = G.e1; c = G.cost[n]; u = np.array(G.us)[G.layer[n]]; own, tt, PP = [], [], []
    for j in range(1, K + 1):
        t = c * j / (K + 1); x, v = G.P[n, 0], G.P[n, 1]; PP.append(np.stack([x + v * t + u * t * t / 2, v + u * t], 1)); own.append(n); tt.append(t)
    if not K: return None
    P = np.concatenate(PP); own = np.concatenate(own); tt = np.concatenate(tt); return own, tt, [G.locate(u2, P) for u2 in G.us], len(P)

def solve_q3(G, sub, pen=0., maxit=3000, tol=1e-10):
    t0 = time.perf_counter(); V = G.V0.copy(); Q = G.V0.copy(); e1 = G.e1; lev = G.rowid[e1]; o = np.argsort(-lev, kind='stable'); e1 = e1[o]; e1n = G.nxt[e1]; e1c = G.cost[e1]; lev = lev[o]
    cuts = np.flatnonzero(np.r_[True, lev[1:] != lev[:-1]]); cuts = np.r_[cuts, len(e1)]
    for it in range(1, maxit + 1):
        Vo = V.copy(); Qo = Q.copy()
        for li2, pairs in enumerate(G.e3):
            Ei = G.evalpairs(Qo, G.n, pairs); Ei = np.where(np.isfinite(Ei), Ei, BIG); same = G.layer == li2; Q[same] = np.minimum(Q[same], Ei[same]); V[~same] = np.minimum(V[~same], Ei[~same] + pen)
        if sub is not None:
            own, tt, prs, ns = sub
            for li2, pairs in enumerate(prs):
                Ei = G.evalpairs(Qo, ns, pairs); m = np.isfinite(Ei) & (G.layer[own] != li2); np.minimum.at(Q, own[m], tt[m] + Ei[m] + pen)
        V = np.minimum(V, Q)
        for a, b in zip(cuts[:-1], cuts[1:]):
            nd = e1[a:b]; Q[nd] = np.minimum(Q[nd], e1c[a:b] + V[e1n[a:b]]); V[nd] = np.minimum(V[nd], Q[nd])
        if max(float(np.abs(np.minimum(V, BIG) - np.minimum(Vo, BIG)).max()), float(np.abs(np.minimum(Q, BIG) - np.minimum(Qo, BIG)).max())) < tol: break
    return V, Q, it, time.perf_counter() - t0

if __name__ == '__main__':
    g = S.load_growN(); layers = [float(x) for x in os.environ.get('LAYERS', '1,-1').split(',')]; atlas, _ = S.build_atlas(layers, 0, par=False); G = S.make_graph(g, atlas, 'N'); rho = G.rho
    T = R.T_ref(G.P, rho); out = np.linalg.norm(G.P, axis=1) > rho; X0 = S.starts(g, 200, 1); Tr = R.T_ref(X0, rho); mx = S.path_extent(X0, rho); ins = mx <= float(g.XLV[0]) + g.GM
    c = G.cost[G.e1]; print('RES шаг строки по времени: мед %.3f p90 %.3f max %.3f; рёбер %d' % (np.median(c), np.percentile(c, 90), c.max(), len(c)), flush=True)
    pols = {'base': dict(), 'base e.02': dict(eps=.02), 'look': dict(look=True)}
    for quad in ([0, 1] if G.M == 3 else [0]):
        if quad: Q2.patch_quad(G)
        for K in [int(x) for x in os.environ.get('K', '0,1,3,7').split(',')]:
            t1 = time.perf_counter(); sub = prep_sub(G, K); tp = time.perf_counter() - t1; V, Q, it, ts = solve_q3(G, sub); G.V = Q; fin = V < BIG / 2; e = (V - T)[out & fin]; Vs, _ = G.Vstar(X0); vm = np.isfinite(Vs); ev = (Vs - Tr)[vm]; name = 'Q %s K=%d' % ('quad' if quad else 'lin', K)
            print('RES %-12s проходов %4d prep %.2fс solve %.2fс | узлы V−T_ref p1/p10/p50/p90/p99 %s <−.05: %.3f >+.05: %.3f |abs| мед %.3f p90 %.3f | V* ош p1/p10/p50/p90/p99 %s' % (name, it, tp, ts, np.round(np.percentile(e, [1, 10, 50, 90, 99]), 3).tolist(), (e < -.05).mean(), (e > .05).mean(), np.median(np.abs(e)), np.percentile(np.abs(e), 90), np.round(np.percentile(ev, [1, 10, 50, 90, 99]), 3).tolist()), flush=True)
            for pn, pk in pols.items():
                Tt, sw = G.run_agent(X0, 3 * Tr + 2, **pk); ok = np.isfinite(Tt); r = Tt[ok & ins] / Tr[ok & ins]; print('RES    %-12s %-9s дошли %.3f (в поле %.3f) T/T_ref мед %.3f p90 %.3f max %.2f перекл %.0f' % (name, pn, ok.mean(), ok[ins].mean(), np.median(r), np.percentile(r, 90), r.max(), sw[ins].mean()), flush=True)
