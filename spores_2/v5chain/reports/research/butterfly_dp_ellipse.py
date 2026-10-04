"""research hub-v5chain-research-10: двойной маятник — ЭЛЛИПТИЧЕСКОЕ доращивание атласа бабочек (концепция пользователя: «старт → эллипс до пути →
следующий старт в тот же атлас», concepts.md). База — butterfly_dp_grow.py с FWD (прямое + обратное дерево, перенос нормали).
Раунд: g — время от споры-старта (Дейкстра по рёбрам), V — до цели; эллипс = споры с g + V ≤ (1 + MARG)·T_best; из них растут дети в обе стороны
(дуга ±t, нормаль переносится), новая — только в непокрытую точку (dmin ×DSH за раунд). Пары считаются только у новых спор и у старых рядом с ними.
Без подсказок точного решения: эллипс строится по собственному лучшему пути атласа."""
import numpy as np, sys, os, time, json
sys.path.insert(0, '.')
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from butterfly_dp import wrap, flow, ingoal, BIG, TL, WIN, WMAX
from butterfly_dp_atlas import R, MN
import butterfly_dp_query as Q
from butterfly_dp_grow import emb, normals
MARG, NEW, ROUNDS, DSH = float(os.environ.get('MARG', .1)), int(os.environ.get('NEW', 1000)), int(os.environ.get('ROUNDS', 3)), float(os.environ.get('DSH', .85))
UMAX, KS = .9, 81
GEO, RING = int(os.environ.get('GEO', 0)), float(os.environ.get('RING', .25)); LEVEL = int(os.environ.get('LEVEL', 1))
def edges_add(A, Y_iy, out):
    iy, k, sv, t = out
    f_ = (sv + R) / (2 * R) * (MN - 1); j0 = np.clip(np.floor(f_).astype(int), 0, MN - 2); a = f_ - j0
    return [Y_iy[iy.astype(int)], k.astype(int), j0, a, t]
def g_from_start(A):
    iy, k, j0, a, t = A.e; to = k * MN + np.clip(np.round(j0 + a).astype(int), 0, MN - 1); n = A.K * MN
    G = csr_matrix((t + 1e-9, (iy, to)), shape=(n, n)); return dijkstra(G, indices=KS * MN + MN // 2).reshape(A.K, MN)
def refresh(A):
    A.P = A.C[:, None, :] + A.sn[None, :, None] * A.n[:, None, :]; A.ingoal = ingoal(np.moveaxis(A.P, 2, 0))
    A.X = emb(A.C); A.tree = cKDTree(A.X)
def grow_round(A, rng, dmin, Tb, r=1):
    if GEO:                                                                                  # GEO (концепция пользователя): геометрический эллипс a + b ≤ c_r в вложении, кольца растут
        a = np.linalg.norm(A.X - emb(A.C[KS][None]), axis=1); b = np.linalg.norm(A.X - emb(np.array([[np.pi / 2, 0, 0, 0]])), axis=1)
        c0 = np.linalg.norm(emb(A.C[KS][None]) - emb(np.array([[np.pi / 2, 0, 0, 0]]))); ell = np.flatnonzero(a + b <= c0 * (1 + MARG) + RING * r)
    else:
        g = g_from_start(A).min(1); V = A.V.min(1); ell = np.flatnonzero(g + V <= (1 + MARG) * Tb)
        if Tb >= BIG / 2: ell = np.arange(A.K)                                               # пути ещё нет: растут ОБА дерева (иначе g = ∞ выкидывает обратное)
        if Tb >= BIG / 2 and LEVEL:                                                          # LEVEL: родитель равномерно по уровню (g — прямое, V — обратное) ⇒ фронты глубже
            ell = None; lev = (g, V); sets = (np.flatnonzero(np.isfinite(g)), np.flatnonzero(V < BIG / 2))
    K0 = A.K; C, n = A.C, A.n; added = 0; tries = 0
    while added < NEW and tries < 400:
        tries += 1; b = 400
        if ell is None:
            B = []
            for lv, st in zip(lev, sets):
                o = st[np.argsort(lv[st])]; L = rng.uniform(0, lv[o].max() + .3, b // 2); B.append(o[np.clip(np.searchsorted(lv[o], L) - rng.integers(0, 8, b // 2), 0, len(o) - 1)])
            B = np.concatenate(B)
        else: B = ell[rng.integers(len(ell), size=b)]
        sg = np.where(rng.random(b) < .5, 1., -1.)
        if ell is None: sg = np.r_[np.ones(b // 2), -np.ones(b - b // 2)]                       # прямое растёт вперёд, обратное — назад
        u = rng.uniform(-1, 1, (2, b)); c = rng.random(b) < .5; u[:, c] = np.sign(u[:, c]); u *= UMAX; t = rng.uniform(.2, TL, b)
        Aa = flow(C[B].T, u[0], u[1], sg * t).T; d = (flow((C[B] + 1e-5 * n[B]).T, u[0], u[1], sg * t).T - Aa) / 1e-5
        nA = d / np.linalg.norm(d, axis=1, keepdims=True); Aa[:, :2] = wrap(Aa[:, :2])
        ok = np.isfinite(Aa).all(1) & (np.abs(Aa[:, 2:]) <= WMAX).all(1) & ~ingoal(Aa.T)
        ok &= (np.abs(np.c_[wrap(C[B, :2] - Aa[:, :2]), C[B, 2:] - Aa[:, 2:]]) <= WIN).all(1)
        ok &= cKDTree(emb(C)).query(emb(np.where(ok[:, None], Aa, 0.)))[0] >= dmin
        acc = []
        for i in np.flatnonzero(ok):
            if not acc or np.min(np.linalg.norm(emb(Aa[acc]) - emb(Aa[i:i + 1]), axis=1)) >= dmin: acc.append(i)
        acc = acc[:NEW - added]; C = np.r_[C, Aa[acc]]; n = np.r_[n, nA[acc]]; added += len(acc)
        if len(acc) / b < .02: dmin *= .9
    A.C, A.n = C, n; A.K = len(C); A.V = np.r_[A.V, np.full((A.K - K0, MN), BIG)]; refresh(A); A.V[A.ingoal] = 0.
    # пары: новые узлы → все; старые узлы рядом с новыми → только в новые споры
    Yn = A.P[K0:].reshape(-1, 4); idn = np.arange(K0 * MN, A.K * MN); own = np.repeat(np.arange(K0, A.K), MN)
    e_new = edges_add(A, idn, A.pairs(Yn, own))
    near = np.unique(np.concatenate([np.asarray(b, int) for b in cKDTree(A.X[:K0]).query_ball_point(A.X[K0:], 2 * WIN)] or [np.zeros(0, int)]))
    Yo = A.P[near].reshape(-1, 4); ido = (near[:, None] * MN + np.arange(MN)).reshape(-1); out = A.pairs(Yo)
    keep = out[1].astype(int) >= K0; e_old = edges_add(A, ido, [x[keep] for x in out])
    e = [np.r_[a, b, c] for a, b, c in zip(A.e, e_new, e_old)]; o = np.argsort(e[0], kind='stable'); A.e = [x[o] for x in e]
    return (len(ell) if ell is not None else -1), A.K - K0, dmin
if __name__ == '__main__':
    A = Q.load(sys.argv[1]); rng = np.random.default_rng(1); t0 = time.time(); dmin = float(os.environ.get('DMIN0', .2))
    refresh(A); A.V[A.ingoal] = 0.
    A.solve(); Tb = A.V[KS, MN // 2]; T, arcs, wm = Q.rollout_edges(A, KS, 0.)
    print(json.dumps(dict(round=0, spores=A.K, V=round(float(Tb), 3), T=round(float(T), 3), T_over_OCP=round(float(T / 5.098), 4), arcs=arcs, wmax=round(float(wm), 2))), flush=True)
    for r in range(1, ROUNDS + 1):
        ne, na, dmin = grow_round(A, rng, dmin, Tb, r); dmin *= DSH; A.solve(); Tb = min(Tb, A.V[KS, MN // 2]); A.E = Q.node_edges(A); A.nplan = 0; T, arcs, wm = Q.rollout_edges(A, KS, 0.)
        print(json.dumps(dict(round=r, ellipse=ne, added=na, spores=A.K, pairs_per_node=round(len(A.e[0]) / (A.K * MN), 2), dmin=round(dmin, 3), V=round(float(A.V[KS, MN // 2]), 3),
                              T=round(float(T), 3), T_over_OCP=round(float(T / 5.098), 4), arcs=arcs, wmax=round(float(wm), 2), sec=round(time.time() - t0))), flush=True)
        np.savez(sys.argv[1].replace('.npz', ('_geo%d.npz' if GEO else '_ell%d.npz') % r), C=A.C, n=A.n, V=A.V, e=np.array(A.e, dtype=object), allow_pickle=True)
