"""lr4: «вдоль потока» с ПРАВИЛЬНЫМ значением на нижней строке. В lr3/qsolve4 значение слоя u2 в точке y = время до строки k+1 + Q(k+1) («продолжать u2 ещё шаг»);
по смыслу E1 должно быть время до строки + V(k+1) («дальше как там лучше»): иначе в ячейке перед кривой переключения цена «продолжать» завышена (горб).
Варианты: донор Q (как было) / донор V; K подточек 3, 7. Замеры: ошибка узлов, горб вдоль оптимальных трасс (как hump.py), агенты look / commit. env M, LAYERS, SEED, OUT. Запуск из v8."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S, qsolve3 as Q3, qsolve4 as Q4, stuck0 as Z, hump as H
from src.algo import ref_di_disk as R
BIG = S.BIG


def solve_flow(G, sub, useV, maxit=3000, tol=1e-10):
    """копия qsolve3.solve_q3; useV — в интерполяцию донора идут V узлов нижней строки, а не Q"""
    t0 = time.perf_counter(); V = G.V0.copy(); Q = G.V0.copy(); e1 = G.e1; lev = G.rowid[e1]; o = np.argsort(-lev, kind='stable'); e1 = e1[o]; e1n = G.nxt[e1]; e1c = G.cost[e1]; lev = lev[o]
    cuts = np.flatnonzero(np.r_[True, lev[1:] != lev[:-1]]); cuts = np.r_[cuts, len(e1)]
    for it in range(1, maxit + 1):
        Vo = V.copy(); Qo = Q.copy(); D = Vo if useV else Qo
        for li2, pairs in enumerate(G.e3):
            Ei = G.evalpairs(D, G.n, pairs); Ei = np.where(np.isfinite(Ei), Ei, BIG); same = G.layer == li2; Q[same] = np.minimum(Q[same], Ei[same]); V[~same] = np.minimum(V[~same], Ei[~same])
        if sub is not None:
            own, tt, prs, ns = sub
            for li2, pairs in enumerate(prs):
                Ei = G.evalpairs(D, ns, pairs); m = np.isfinite(Ei) & (G.layer[own] != li2); np.minimum.at(Q, own[m], tt[m] + Ei[m])
        V = np.minimum(V, Q)
        for a, b in zip(cuts[:-1], cuts[1:]):
            nd = e1[a:b]; Q[nd] = np.minimum(Q[nd], e1c[a:b] + V[e1n[a:b]]); V[nd] = np.minimum(V[nd], Q[nd])
        if max(float(np.abs(np.minimum(V, BIG) - np.minimum(Vo, BIG)).max()), float(np.abs(np.minimum(Q, BIG) - np.minimum(Qo, BIG)).max())) < tol: break
    return V, Q, it, time.perf_counter() - t0


if __name__ == '__main__':
    g = S.load_growN(); layers = [float(x) for x in os.environ.get('LAYERS', '1,-1,0').split(',')]; atlas, _ = S.build_atlas(layers, int(os.environ.get('SEED', '0')), par=False); G = S.make_graph(g, atlas, 'N'); rho = G.rho
    X0 = S.starts(g, 1000, 1); Tr = R.T_ref(X0, rho); mx = S.path_extent(X0, rho); inw = mx <= float(g.XLV[0]); tmax = 3 * Tr + 2; ban = Z.make_ban('none', None, G.L); T = R.T_ref(G.P, rho); out = np.linalg.norm(G.P, axis=1) > rho
    XH = S.starts(g, 600, 1); XH = XH[S.path_extent(XH, rho) <= float(g.XLV[0])]; U0, T1, TT = H.opt_switch(XH, rho); ids = np.flatnonzero(np.isfinite(T1) & (T1 >= .8) & (TT - T1 >= .3)); TG = H.TG; win = (TG >= -.5) & (TG <= 0); res = {}
    PTS = np.zeros((len(ids), len(TG), 2)); TRF = np.zeros((len(ids), len(TG)))
    for r, i in enumerate(ids):
        t = T1[i] + TG; a = np.minimum(t, T1[i]); b = np.maximum(t - T1[i], 0.); u = U0[i]; x1 = XH[i, 0] + XH[i, 1] * a + u * a * a / 2; v1 = XH[i, 1] + u * a; PTS[r, :, 0] = x1 + v1 * b - u * b * b / 2; PTS[r, :, 1] = v1 - u * b; TRF[r] = TT[i] - t
    print('RES слои %s M=%d узлов %d; стартов в поле %d, трасс для горба %d' % (layers, G.M, G.n, inw.sum(), len(ids)), flush=True); Q4.patch_flow(G)
    for k in (3, 7):
        sub = Q3.prep_sub(G, k)
        for useV in (False, True):
            name = 'flow K=%d донор %s' % (k, 'V' if useV else 'Q'); V, Q, it, ts = solve_flow(G, sub, useV); G.V = V if useV else Q; fin = V < BIG / 2; e = (V - T)[out & fin]
            E = np.stack([G.Vlayers(PTS[r]).min(1) - TRF[r] for r in range(len(ids))]); E[~np.isfinite(E)] = np.nan; full = np.isfinite(E[:, 0]) & np.isfinite(E[:, win]).all(1); Ef = E[full]; hump = Ef[:, win].max(1) - Ef[:, 0]; dV = np.diff(Ef - TG[None, :], axis=1); up = (dV[:, (TG[1:] >= -.5) & (TG[1:] <= 0)] > 0).any(1)
            Vs = G.Vlayers(X0).min(1); vm = np.isfinite(Vs) & inw; ev = (Vs - Tr)[vm]
            print('RES %-18s solve %.2f с, проходов %d | узлы V−T_ref: |ош| мед %.3f p90 %.3f, <−.05: %.3f, >+.05: %.3f, p1 %.3f | запрос V* в 780 стартах: мед %+.3f p10 %+.3f p90 %+.3f, <−.05: %.3f | горб мед %.3f p90 %.3f, V* растёт вдоль трассы: %.2f, e до/в t_sw/после: %.3f/%.3f/%.3f' % (
                name, ts, it, np.median(np.abs(e)), np.percentile(np.abs(e), 90), (e < -.05).mean(), (e > .05).mean(), np.percentile(e, 1), np.median(ev), np.percentile(ev, 10), np.percentile(ev, 90), (ev < -.05).mean(), np.median(hump), np.percentile(hump, 90), up.mean(), np.median(Ef[:, 0]), np.median(Ef[:, TG == 0]), np.nanmedian(Ef[:, TG == .2])), flush=True)
            for pn, fn in (('look', Z.run_look), ('commit', Z.run_commit)):
                t1 = time.perf_counter(); Tt, sw, Xe, ce = fn(G, X0, tmax, ban); dtm = time.perf_counter() - t1; ok = np.isfinite(Tt); rw = Tt[ok & inw] / Tr[ok & inw]; stuck = ~ok & (np.abs(Xe[:, 1]) < .05) & (np.linalg.norm(Xe, axis=1) > rho) & inw
                res['%s|%s' % (name, pn)] = dict(reached_field=float(ok[inw].mean()), T_med=float(np.median(rw)), T_p90=float(np.percentile(rw, 90)), T_p99=float(np.percentile(rw, 99)), T_max=float(rw.max()), sw_med=float(np.median(sw[inw])), sw_mean=float(sw[inw].mean()), stuck=int(stuck.sum()), sec=dtm)
                print('RES %-18s %-6s в поле дошли %.3f (застряло на v≈0: %d) | T/T_ref мед %.3f p90 %.3f p99 %.3f max %.2f | перекл мед %.0f ср %.1f | %.0f с' % (name, pn, ok[inw].mean(), stuck.sum(), np.median(rw), np.percentile(rw, 90), np.percentile(rw, 99), rw.max(), np.median(sw[inw]), sw[inw].mean(), dtm), flush=True)
    json.dump(res, open(os.environ.get('OUT', 'flowv') + '.json', 'w'), ensure_ascii=False, indent=1)
