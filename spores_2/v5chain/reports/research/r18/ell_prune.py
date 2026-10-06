# research-18: прямой solve T̂_s (прототип PLAN 26а) + обрезка атласа до эллипса T̂_s + V ≤ (1+ε)C — сколько клеток остаётся и сохраняется ли T агента.
# Запуск: SYS=pend UM=.3 GS=60 GLIM=0 FRAC=1 python3 ell_prune.py   (атлас — рост только от цели, FIFO + SIDE, как bfs_gen.py)
import os, sys, numpy as np, json, time
import bfs_gen as B
G = B.G
def solve_fwd(A, s, it=20000, r0=None):
    """T̂_s(узел) — время из s: Якоби по стенсилам step(P, u, −1); источник — узлы в коробке r0 вокруг s (T̂ = 0)."""
    I_, IDX_, W_ = [], [], []
    for u in G.US:
        a, b, c_ = A.stencils(G.step(A.P, u, -1.)); I_.append(a); IDX_.append(b); W_.append(c_)
    I = np.concatenate(I_); IDX = np.concatenate(IDX_); W = np.concatenate(W_); o = np.argsort(I, kind='stable'); I, IDX, W = I[o], IDX[o], W[o]
    st = np.flatnonzero(np.r_[True, I[1:] != I[:-1]]); nd = I[st]
    r0 = G.RHOV * 1.5 if r0 is None else r0; src = (np.abs(G.wrapy(A.P - s)) <= r0).all(1); T = np.full(A.N, G.BIG); T[src] = 0.
    for n in range(it):
        val = G.DTN + A.interp(W, T[IDX]); new = T.copy(); new[nd] = np.minimum(T[nd], np.minimum.reduceat(val, st)); new[src] = 0.
        d = np.max(np.abs(new - T)); T = new
        if d < 1e-9: break
    return T, int(src.sum())
def sub_atlas(A, keep):
    A2 = G.Atlas.__new__(G.Atlas); k = 0; A2.layers = []
    for l in A.layers: A2.layers.append([c for c in l if keep[A.cells.index(c)]])
    A2.finish(); return A2
if __name__ == '__main__':
    from multiprocessing import Pool
    A = G.Atlas.__new__(G.Atlas)
    with Pool(len(G.US)) as pool: A.layers = pool.map(B._l, [(u, k) for k, u in enumerate(G.US)])
    A.finish(); A.solve(); Q, ref = G.starts_ref(); EPS = float(os.environ.get('ELLEPS', .2)); NQ = int(os.environ.get('NQ', 5))
    print('полный атлас: клеток', len(A.cells), 'узлов', A.N, flush=True)
    for i in range(NQ):
        s = G.wrapy(Q[i]); T0 = A.rollout(s[None])[0][0]; C = float(A.vstar(s[None])[0]); Ts, nsrc = solve_fwd(A, s)
        cm = np.array([(Ts[c.o:c.o + c.G.shape[0] * G.M ** G.m_] + A.V[c.o:c.o + c.G.shape[0] * G.M ** G.m_]).min() for c in A.cells])
        res = dict(q=i, ref=round(float(ref[i]), 3), C=round(C, 3), T_full=round(float(T0), 3), src=nsrc, ts_c_min=round(float(cm.min()), 3))
        for eps in (EPS, .5):
            keep = cm <= (1 + eps) * C; A2 = sub_atlas(A, keep); A2.solve(); T2 = A2.rollout(s[None])[0][0]
            res['eps%.1f' % eps] = dict(cells=int(keep.sum()), frac=round(float(keep.mean()), 3), T=round(float(T2), 3))
        print(json.dumps(res), flush=True)
