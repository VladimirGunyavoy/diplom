"""Бабочки 4D плоский ДИ, ускоренный поиск пар (PLAN п.3, hub-worker-14). Модель — reports/research/butterfly_4d.py (B4): дуги формулой.
Ускорение: кандидатов даёт cKDTree.sparse_distance_matrix(узлы чанка, ndarray) — вся работа в C++, без списков Python; чанки узлов; в памяти только (qi, k, t, s*) int32/float64;
Беллман — reduceat по отсортированным парам вместо np.minimum.at. Запуск из spores_2/v7: python3 src/cells7/butterfly_4d.py N [m]"""
import sys, os, time, json
sys.path.insert(0, '.'); sys.path.insert(0, '../v5chain/reports/research')
import numpy as np
from scipy.spatial import cKDTree
from butterfly_4d import B4, BIG, RHO
from v7_faces_di import tstar_box


_S = None
def _job(a): return _S._chunk(a)


class B4Fast(B4):
    def _chunk(s, a):
        Yc, own_c, i0, want_a = a; tr = cKDTree(Yc * s.scale)
        M = s.tree.sparse_distance_matrix(tr, np.sqrt(2) + 1e-9, output_type='ndarray'); k = M['i'].astype(np.int64); qi = M['j'].astype(np.int64)
        if own_c is not None: keep = k != own_c[qi]; k, qi = k[keep], qi[keep]
        t, ss, aa = s.arcs(Yc[qi], k); f = np.isfinite(t)
        return (qi[f] + i0).astype(np.int32), k[f].astype(np.int32), t[f], ss[f], (aa[f] if want_a else None)

    def pairs(s, Y, own=None, chunk=None, want_a=True, procs=int(__import__('os').environ.get('PROCS', 1))):
        chunk = chunk or max(100, int(2000 * 8000 / s.K))                      # кандидатов на узел ∝ N ⇒ чанк ∝ 1/N (пик памяти ~ const)
        jobs = [(Y[i:i + chunk], None if own is None else own[i:i + chunk], i, want_a) for i in range(0, len(Y), chunk)]
        if procs > 1 and len(jobs) > 1:
            global _S; _S = s
            with __import__('multiprocessing').get_context('fork').Pool(procs) as P: R = P.map(_job, jobs, chunksize=1)
        else: R = [s._chunk(j) for j in jobs]
        return tuple(np.concatenate([r[i] for r in R]) if (i < 4 or want_a) else None for i in range(5))

    def solve(s, it=200):
        t0 = time.time(); cache = '/tmp/claude-1000/b4pairs_%d_%d.npz' % (s.K, s.m)
        if os.path.exists(cache):
            Z = np.load(cache); starts, nodeid, idx, b, t = Z['starts'], Z['nodeid'], Z['idx'], Z['b'], Z['t']; del Z
        else:
            own = np.repeat(np.arange(s.K), s.m); qi, k, t, ss, _ = s.pairs(s.nodes, own, want_a=False); o = np.argsort(qi, kind='stable')
            qi = qi[o]; k = k[o]; t = t[o]; ss = ss[o]; del o
            starts = np.r_[0, np.nonzero(np.diff(qi))[0] + 1]; nodeid = qi[starts].copy(); del qi
            fpos = (ss + s.r) / (2 * s.r) * (s.m - 1); del ss; j = np.clip(np.floor(fpos).astype(np.int32), 0, s.m - 2); b = fpos - j; del fpos
            idx = k * np.int32(s.m) + j; del k, j
            np.savez(cache, starts=starts, nodeid=nodeid, idx=idx, b=b, t=t)
        s.npairs = len(t); print('pairs', s.npairs, 'на узел', round(s.npairs / len(s.nodes), 1), 'сек', round(time.time() - t0), flush=True)
        V = s.V.copy()
        for n in range(it):
            J = (1 - b) * V[idx]; J += b * V[idx + 1]; J += t; mn = np.minimum.reduceat(J, starts); del J
            Vn = V.copy(); Vn[nodeid] = np.minimum(np.minimum(V[nodeid], BIG), mn); Vn = np.where(s.ingoal, 0., np.minimum(Vn, BIG))
            d = np.max(np.abs(Vn - V)); V = Vn
            if d < 1e-7: break
        s.V = V; s.it = n; return s


if __name__ == '__main__':
    N = int(sys.argv[1]); m = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    t0 = time.time(); S = B4Fast(N=N, m=m).solve(); print('спор', S.K, 'узлов', len(S.nodes), 'итер', S.it, 'BIG', round(float((S.V >= BIG / 2).mean()), 3), 'сек', round(time.time() - t0), flush=True)
    rng = np.random.default_rng(1); Q = np.c_[rng.uniform(-1, 1, (300, 2)), rng.uniform(-.7, .7, (300, 2))]
    Ts = np.maximum(tstar_box(Q[:, 0], Q[:, 2], RHO), tstar_box(Q[:, 1], Q[:, 3], RHO)); ok = Ts > .05; Q, Ts = Q[ok], Ts[ok]
    V, _, _ = S.best(Q); mm = np.isfinite(V) & (V < BIG / 2); r = V[mm] / Ts[mm]
    print(json.dumps(dict(N=N, V_cover=round(float(mm.mean()), 3), V_over_Tstar=dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), max=round(float(r.max()), 3))), ensure_ascii=False), flush=True)
    T, sw = S.rollout(Q[:150]); f = np.isfinite(T); rr = T[f] / Ts[:150][f]
    print(json.dumps(dict(N=N, reach=round(float(f.mean()), 3), T_over_Tstar=dict(mean=round(float(rr.mean()), 4), med=round(float(np.median(rr)), 4), max=round(float(rr.max()), 3)) if f.any() else None, sw_med=float(np.median(sw)), sec=round(time.time() - t0)), ensure_ascii=False), flush=True)
