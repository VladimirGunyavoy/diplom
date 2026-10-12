"""lr3: Q-переходы (qsolve.py) + КВАДРАТИЧНАЯ интерполяция поперёк потока (по 3 линиям клетки, M=3; вдоль потока линейно) + гистерезис агента.
Слово пользователя 2026-10-12: «интерполировать можно квадратично, несильно дороже, но даст большую точность»."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S
from src.algo import ref_di_disk as R
import qsolve as QS
BIG = S.BIG

def patch_quad(G):
    """locate → вершины 2 строки × 3 столбца (вся ширина клетки), веса: линейно по времени × Лагранж 2-й степени поперёк"""
    assert G.M == 3; old = G.locate
    def locate(u2, Y, excl=None):
        idx = G.idx[u2]; pi, hid, sc = idx.query(Y); z = (np.zeros(0, int), np.zeros((0, 6), int), np.zeros((0, 6)), np.zeros(0, int))
        if not len(pi): return z
        cid = idx.I[hid, 0]; gcid = G.gcell[u2][cid]
        if excl is not None:
            k = gcid != excl[pi]; pi, hid, sc, cid = pi[k], hid[k], sc[k], cid[k]
            if not len(pi): return z
        cen = np.minimum(sc, 1 - sc).min(1); o = np.lexsort((-cen, pi)); pi, hid, sc, cid = pi[o], hid[o], sc[o], cid[o]; st = np.r_[True, pi[1:] != pi[:-1]]; rank = np.arange(len(pi)) - np.maximum.accumulate(np.where(st, np.arange(len(pi)), 0))
        k = rank < G.K; pi, hid, sc, cid, rank = pi[k], hid[k], sc[k], cid[k], rank[k]; row, col = idx.I[hid, 1], idx.I[hid, 2]; b = G.base[u2][cid]
        xi = col + sc[:, 1]; L = np.stack([(xi - 1) * (xi - 2) / 2, -xi * (xi - 2), xi * (xi - 1) / 2], 1); wt = np.stack([1 - sc[:, 0], sc[:, 0]], 1)
        verts = np.stack([b + (row + r) * 3 + c for r in (0, 1) for c in (0, 1, 2)], 1); w = np.stack([wt[:, r] * L[:, c] for r in (0, 1) for c in (0, 1, 2)], 1); return pi, verts, w, rank
    G.locate = locate; G.mode = 'lin'; G.prepare()

if __name__ == '__main__':
    g = S.load_growN(); layers = [float(x) for x in os.environ.get('LAYERS', '1,-1').split(',')]; atlas, _ = S.build_atlas(layers, 0, par=False); G = S.make_graph(g, atlas, 'N'); rho = G.rho
    T = R.T_ref(G.P, rho); out = np.linalg.norm(G.P, axis=1) > rho; X0 = S.starts(g, 200, 1); Tr = R.T_ref(X0, rho); mx = S.path_extent(X0, rho); ins = mx <= float(g.XLV[0]) + g.GM
    pols = {'base': dict(), 'base e.02': dict(eps=.02), 'base e.05': dict(eps=.05), 'look': dict(look=True), 'look e1': dict(look=True, eps=1.), 'look e3': dict(look=True, eps=3.)}
    def report(name, Vn, passes, ts):
        fin = Vn < BIG / 2; e = (Vn - T)[out & fin]; Vs, _ = G.Vstar(X0); vm = np.isfinite(Vs); ev = (Vs - Tr)[vm]
        print('RES %-10s проходов %4d solve %.2fс | узлы V−T_ref p1/p10/p50/p90/p99 %s <−.05: %.3f >+.05: %.3f |abs| мед %.3f p90 %.3f | V* покр %.3f ош p1/p10/p50/p90/p99 %s' % (name, passes, ts, np.round(np.percentile(e, [1, 10, 50, 90, 99]), 3).tolist(), (e < -.05).mean(), (e > .05).mean(), np.median(np.abs(e)), np.percentile(np.abs(e), 90), vm.mean(), np.round(np.percentile(ev, [1, 10, 50, 90, 99]), 3).tolist()), flush=True)
        for pn, pk in pols.items():
            Tt, sw = G.run_agent(X0, 3 * Tr + 2, **pk); ok = np.isfinite(Tt); r = Tt[ok & ins] / Tr[ok & ins]; print('RES    %-10s %-9s дошли %.3f (в поле %.3f) T/T_ref мед %.3f p90 %.3f max %.2f перекл %.0f' % (name, pn, ok.mean(), ok[ins].mean(), np.median(r), np.percentile(r, 90), r.max(), sw[ins].mean()), flush=True)
    V, Q, it, ts = QS.solve_q(G, 0.); G.V = Q; report('Q lin', V, it, ts)
    patch_quad(G); V, Q, it, ts = QS.solve_q(G, 0.); G.V = Q; report('Q quad', V, it, ts)
    G.solve(verbose=False); report('V quad', G.V.copy(), G.passes, G.t_solve)
