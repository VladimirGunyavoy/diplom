"""lr3: прототип по схеме пользователя (2026-10-12): V берётся из уже посчитанной области ТОЧНО ВДОЛЬ ПОТОКА донора.
Отличие от qsolve3: значение слоя u2 в точке y ячейки-донора = (линейно поперёк, по НИЖНЕЙ по потоку строке k+1) Q(k+1) + точное время дойти до неё своим потоком:
   Q_u2(y) = Σ_c w_c · [ Q(k+1, c) + (1 − s_t)·(tau(k+1, c) − tau(k, c)) ]      (верхние по потоку вершины в интерполяцию не входят)
— это допустимая стратегия «иду потоком u2 до строки, дальше как там», без интерполяции через крутую сторону излома вдоль потока.
Плюс переключение где угодно: K подточек на каждом шаге (как qsolve3). env: K, M, LAYERS."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S, qsolve3 as Q3
from src.algo import ref_di_disk as R
BIG = S.BIG

def patch_flow(G):
    def evalpairs(V, n, pairs):
        out = np.full(n, np.inf); done = np.zeros(n, bool); pi, verts, w, rank = pairs
        for r in range(G.K):
            m = rank == r
            if not m.any(): continue
            p_, v_, w_ = pi[m], verts[m], w[m]; st = w_[:, 2] + w_[:, 3]; c1 = w_[:, 1] + w_[:, 3]; q0, q1 = V[v_[:, 2]], V[v_[:, 3]]
            d0 = G.tau[v_[:, 2]] - G.tau[v_[:, 0]]; d1 = G.tau[v_[:, 3]] - G.tau[v_[:, 1]]
            ok = (q0 < BIG / 2) & (q1 < BIG / 2) & ~done[p_]; val = (1 - c1) * (q0 + (1 - st) * d0) + c1 * (q1 + (1 - st) * d1)
            out[p_[ok]] = val[ok]; done[p_[ok]] = True
        return out
    G.evalpairs = evalpairs

if __name__ == '__main__':
    g = S.load_growN(); layers = [float(x) for x in os.environ.get('LAYERS', '1,-1').split(',')]; atlas, _ = S.build_atlas(layers, 0, par=False); G = S.make_graph(g, atlas, 'N'); rho = G.rho
    T = R.T_ref(G.P, rho); out = np.linalg.norm(G.P, axis=1) > rho; X0 = S.starts(g, 200, 1); Tr = R.T_ref(X0, rho); mx = S.path_extent(X0, rho); ins = mx <= float(g.XLV[0]) + g.GM
    patch_flow(G); pols = {'base': dict(), 'look': dict(look=True)}
    for K in [int(x) for x in os.environ.get('K', '0,3,7,15').split(',')]:
        t1 = time.perf_counter(); sub = Q3.prep_sub(G, K); tp = time.perf_counter() - t1; V, Q, it, ts = Q3.solve_q3(G, sub); G.V = Q; fin = V < BIG / 2; e = (V - T)[out & fin]; Vs, _ = G.Vstar(X0); vm = np.isfinite(Vs); ev = (Vs - Tr)[vm]; name = 'flow K=%d' % K
        print('RES %-10s M=%d проходов %4d prep %.2fс solve %.2fс конечных %.3f | узлы V−T_ref p1/p10/p50/p90/p99 %s <−.05: %.3f >+.05: %.3f |abs| мед %.3f p90 %.3f max %.3f | V* покр %.3f ош p1/p10/p50/p90/p99 %s' % (name, G.M, it, tp, ts, fin.mean(), np.round(np.percentile(e, [1, 10, 50, 90, 99]), 3).tolist(), (e < -.05).mean(), (e > .05).mean(), np.median(np.abs(e)), np.percentile(np.abs(e), 90), np.abs(e).max(), vm.mean(), np.round(np.percentile(ev, [1, 10, 50, 90, 99]), 3).tolist()), flush=True)
        for pn, pk in pols.items():
            Tt, sw = G.run_agent(X0, 3 * Tr + 2, **pk); ok = np.isfinite(Tt); r = Tt[ok & ins] / Tr[ok & ins]; print('RES    %-10s %-5s дошли %.3f (в поле %.3f) T/T_ref мед %.3f p90 %.3f max %.2f перекл %.0f' % (name, pn, ok.mean(), ok[ins].mean(), np.median(r), np.percentile(r, 90), r.max(), sw[ins].mean()), flush=True)
