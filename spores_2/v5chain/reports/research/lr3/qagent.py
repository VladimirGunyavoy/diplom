"""lr3: агент с ТОЧНЫМ переключением («commit»): идёт своим управлением и переключается только тогда, когда лучший момент переключения — сейчас.
На шаге: вдоль своего потока точки t_j ∈ {0, 1, 2, 5, 10, 20, 40}·dt; g_j = t_j + min по ДРУГИМ слоям Q_u2(x(t_j)).
Переключение в слой argmin, если минимум g достигается при j=0 и g_0 < Q_своего(x) − eps (или Q своего не определено). Иначе — держать своё u.
Сравнение с base/look на двух расчётах: Q lin (qsolve) и «точно вдоль потока» K=3 (qsolve4). env M, LAYERS."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S, qsolve as QS, qsolve3 as Q3, qsolve4 as Q4
from src.algo import ref_di_disk as R
JS = np.array([0, 1, 2, 5, 10, 20, 40])

def run_commit(G, X0, tmax, dt=.01, eps=1e-3):
    n = len(X0); X = X0.copy(); t = np.zeros(n); T = np.full(n, np.nan); sw = np.zeros(n, int); uvec = np.array(G.us); Vl = G.Vlayers(X); cur = Vl.argmin(1); alive = np.isfinite(Vl.min(1)); inn = np.linalg.norm(X, axis=1) <= G.rho; T[inn] = 0; alive &= ~inn
    while alive.any():
        a = np.flatnonzero(alive); u = uvec[cur[a]]; x, v = X[a, 0], X[a, 1]
        best = np.full(len(a), np.inf); bj = np.zeros(len(a), int); bl = cur[a].copy(); qcur = None
        for j in JS:
            tj = j * dt; Pj = np.stack([x + v * tj + u * tj * tj / 2, v + u * tj], 1); Q = G.Vlayers(Pj)
            if j == 0: qcur = Q[np.arange(len(a)), cur[a]].copy()
            Q[np.arange(len(a)), cur[a]] = np.inf; g = tj + Q.min(1); better = g < best - 1e-12; best[better] = g[better]; bj[better] = j; bl[better] = Q.argmin(1)[better]
        chg = (bj == 0) & np.isfinite(best) & ((best < qcur - eps) | ~np.isfinite(qcur)); lost = ~np.isfinite(best) & ~np.isfinite(qcur)
        cur[a[chg]] = bl[chg]; sw[a[chg]] += 1; alive[a[lost]] = False; a = np.flatnonzero(alive)
        if not len(a): break
        u = uvec[cur[a]]; x, v = X[a, 0], X[a, 1]; X[a, 0] = x + v * dt + u * dt * dt / 2; X[a, 1] = v + u * dt; t[a] += dt
        hit = np.linalg.norm(X[a], axis=1) <= G.rho; T[a[hit]] = t[a[hit]]; alive[a[hit]] = False; alive[a[t[a] > tmax[a]]] = False
    return T, sw

if __name__ == '__main__':
    g = S.load_growN(); layers = [float(x) for x in os.environ.get('LAYERS', '1,-1').split(',')]; atlas, _ = S.build_atlas(layers, 0, par=False); G = S.make_graph(g, atlas, 'N'); rho = G.rho
    X0 = S.starts(g, 200, 1); Tr = R.T_ref(X0, rho); mx = S.path_extent(X0, rho); ins = mx <= float(g.XLV[0]) + g.GM; inw = mx <= float(g.XLV[0])
    def rep(name):
        for pn, fn in (('base', lambda: G.run_agent(X0, 3 * Tr + 2)), ('look', lambda: G.run_agent(X0, 3 * Tr + 2, look=True)), ('commit', lambda: run_commit(G, X0, 3 * Tr + 2)), ('commit e.01', lambda: run_commit(G, X0, 3 * Tr + 2, eps=.01))):
            t1 = time.perf_counter(); Tt, sw = fn(); ok = np.isfinite(Tt); r = Tt[ok & ins] / Tr[ok & ins]; rw = Tt[ok & inw] / Tr[ok & inw]
            print('RES %-12s M=%d %-11s дошли %.3f (в росте %.3f, в поле %.3f) T/T_ref мед %.3f p90 %.3f max %.2f | в поле: мед %.3f p90 %.3f | перекл мед %.0f среднее %.1f | %.0f с' % (name, G.M, pn, ok.mean(), ok[ins].mean(), ok[inw].mean(), np.median(r), np.percentile(r, 90), r.max(), np.median(rw), np.percentile(rw, 90), np.median(sw[ins]), sw[ins].mean(), time.perf_counter() - t1), flush=True)
    V, Q, it, ts = QS.solve_q(G, 0.); G.V = Q; rep('Q lin')
    Q4.patch_flow(G); V, Q, it, ts = Q3.solve_q3(G, Q3.prep_sub(G, 3)); G.V = Q; rep('flow K=3')
