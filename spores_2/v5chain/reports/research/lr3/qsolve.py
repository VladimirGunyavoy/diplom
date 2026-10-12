"""lr3: «минимум ПОСЛЕ интерполяции» — переходы по Q, а не по V (прототип, solve8 не меняется).
Q_u(узел) = цена «продолжать своё управление u»: вход в цель / шаг вдоль потока + V(следующего) / стык с клеткой СВОЕГО слоя (интерполяция Q).
V(узел) = min(Q_u(узел), min по другим слоям u2 интерполяции Q_u2 в точке узла [+ цена]).  Запрос: V*(x) = min_u интерп. Q_u(x).
Идея: V имеет излом на кривой переключения, а Q_u по свою сторону гладкая → интерполяция не занижает; излом получается сам как пересечение Q-поверхностей.
Сравнение с базой (lin, lin+.02, maxv): ошибка узлов против T_ref, агент base/look на 200 стартах.  env M."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S
from src.algo import ref_di_disk as R
BIG = S.BIG

def solve_q(G, pen=0., maxit=3000, tol=1e-10):
    t0 = time.perf_counter(); V = G.V0.copy(); Q = G.V0.copy(); e1 = G.e1; lev = G.rowid[e1]; o = np.argsort(-lev, kind='stable'); e1 = e1[o]; e1n = G.nxt[e1]; e1c = G.cost[e1]; lev = lev[o]
    cuts = np.flatnonzero(np.r_[True, lev[1:] != lev[:-1]]); cuts = np.r_[cuts, len(e1)]
    for it in range(1, maxit + 1):
        Vo = V.copy(); Qo = Q.copy()
        for li2, pairs in enumerate(G.e3):
            Ei = G.evalpairs(Qo, G.n, pairs); Ei = np.where(np.isfinite(Ei), Ei, BIG); same = G.layer == li2
            Q[same] = np.minimum(Q[same], Ei[same]); V[~same] = np.minimum(V[~same], Ei[~same] + pen)
        V = np.minimum(V, Q)
        for a, b in zip(cuts[:-1], cuts[1:]):
            nd = e1[a:b]; Q[nd] = np.minimum(Q[nd], e1c[a:b] + V[e1n[a:b]]); V[nd] = np.minimum(V[nd], Q[nd])
        if max(float(np.abs(np.minimum(V, BIG) - np.minimum(Vo, BIG)).max()), float(np.abs(np.minimum(Q, BIG) - np.minimum(Qo, BIG)).max())) < tol: break
    return V, Q, it, time.perf_counter() - t0

if __name__ == '__main__':
    g = S.load_growN(); layers = [float(x) for x in os.environ.get('LAYERS', '1,-1').split(',')]; atlas, _ = S.build_atlas(layers, 0, par=False); G = S.make_graph(g, atlas, 'N'); rho = G.rho
    T = R.T_ref(G.P, rho); out = np.linalg.norm(G.P, axis=1) > rho; X0 = S.starts(g, 200, 1); Tr = R.T_ref(X0, rho); mx = S.path_extent(X0, rho); ins = mx <= float(g.XLV[0]) + g.GM; res = {}
    def report(name, Vn, passes, ts):
        fin = Vn < BIG / 2; e = (Vn - T)[out & fin]; Vs, _ = G.Vstar(X0); vm = np.isfinite(Vs); ev = (Vs - Tr)[vm]; row = dict(passes=passes, t_solve=ts, finite=float(fin.mean()), node_err=np.round(np.percentile(e, [1, 10, 25, 50, 75, 90, 99]), 3).tolist(), under05=float((e < -.05).mean()), over05=float((e > .05).mean()), node_abs_med=float(np.median(np.abs(e))), node_abs_p90=float(np.percentile(np.abs(e), 90)),
                   vstar_cov=float(vm.mean()), vstar_err=np.round(np.percentile(ev, [1, 10, 50, 90, 99]), 3).tolist())
        for pn in ('base', 'look'):
            t1 = time.perf_counter(); Tt, sw = G.run_agent(X0, 3 * Tr + 2, **S.POLICIES[pn]); ok = np.isfinite(Tt); r = Tt[ok & ins] / Tr[ok & ins]; row[pn] = dict(reached=float(ok.mean()), reached_in=float(ok[ins].mean()), T_med=float(np.median(r)) if len(r) else None, T_p90=float(np.percentile(r, 90)) if len(r) else None, sw=float(sw[ins].mean()), ms=(time.perf_counter() - t1) / len(X0) * 1e3)
        res[name] = row; print('RES %-9s проходов %4d solve %.2fс | узлы V−T_ref p1/p10/p25/p50/p75/p90/p99 %s | <−.05: %.3f >+.05: %.3f |abs| мед %.3f p90 %.3f | V* покр %.3f ош p1/p10/p50/p90/p99 %s | base: дошли %.3f(в поле %.3f) T %.3f/%.3f перекл %.0f | look: %.3f(%.3f) T %.3f/%.3f перекл %.0f' % (
            name, passes, ts, row['node_err'], row['under05'], row['over05'], row['node_abs_med'], row['node_abs_p90'], row['vstar_cov'], row['vstar_err'], row['base']['reached'], row['base']['reached_in'], row['base']['T_med'] or 0, row['base']['T_p90'] or 0, row['base']['sw'], row['look']['reached'], row['look']['reached_in'], row['look']['T_med'] or 0, row['look']['T_p90'] or 0, row['look']['sw']), flush=True)
    print('RES слои %s M=%d узлов %d клеток %s стартов в поле %d/200' % (layers, G.M, G.n, {u: len(atlas[u]) for u in G.us}, ins.sum()), flush=True)
    for mode, pen in (('lin', 0.), ('lin', .02), ('maxv', 0.)):
        G.mode, G.pen = mode, pen; G.solve(verbose=False); report('%s+%g' % (mode, pen), G.V.copy(), G.passes, G.t_solve)
    G.mode = 'lin'
    for pen in (0., .02):
        V, Q, it, ts = solve_q(G, pen); G.V = Q; report('Q+%g' % pen, V, it, ts)
    json.dump(res, open(os.environ.get('OUT', 'qsolve') + '.json', 'w'), ensure_ascii=False, indent=1)
