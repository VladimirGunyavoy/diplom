"""research-11: дорост в разрыв — раунды Вороного, пока старт не связан. Прямое множество = споры с g < ∞ (достижимы от старта), обратное = с V < BIG.
Раунд: NEW спор, половина вперёд из прямого, половина назад из обратного; родитель — самый дешёвый (g или V) из KN ближайших к случайной точке области,
лучшая дуга веера; |w| ≤ WMAX вдоль дуги; нормаль переносится без нормировки (NT); пары — у новых и у старых рядом. Запуск: G=2 WIN=1.0 WPAIR=1 python3 dp_g2_regrow.py atlas.npz"""
import numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from butterfly_dp import flow, wrap, ingoal, BIG, WIN, WMAX, TL
from butterfly_dp_atlas import R, MN
from scipy.spatial import cKDTree
import butterfly_dp_query as Q, butterfly_dp_ellipse as E
from butterfly_dp_grow import emb
A = Q.load(sys.argv[1]); E.refresh(A); A.V[A.ingoal] = 0.; A.solve(); KS = 81; t0 = time.time(); rng = np.random.default_rng(int(os.environ.get('SEED', 0)))
NEW, ROUNDS, KN, KF, UMAX, OCP = int(os.environ.get('NEW', 1000)), int(os.environ.get('ROUNDS', 8)), int(os.environ.get('KN', 4)), float(os.environ.get('KF', 1.3)), .9, float(os.environ.get('OCP', 7.636))
UG = np.array([(a, c) for a in (-1, 0, 1) for c in (-1, 0, 1)]) * UMAX; TG = np.array([.25, .5, .9, 1.5]); TG = TG[TG <= TL + 1e-9]; UF = np.repeat(UG, len(TG), 0); TF = np.tile(TG, len(UG)); dmin = float(os.environ.get('DMIN0', .13))
for r in range(1, ROUNDS + 1):
    g = E.g_from_start(A).min(1); V = A.V.min(1); K0 = A.K; C, n = A.C, A.n; added = 0; tries = 0
    if V[KS] < BIG / 2 and not int(os.environ.get('KEEP', 0)): break
    while added < NEW and tries < 200:
        tries += 1; b = 300; acc_all = []
        for pool, cost, sg in ((np.flatnonzero(np.isfinite(g)), g, 1.), (np.flatnonzero(V < BIG / 2), V, -1.)):
            X = np.c_[rng.uniform(-np.pi, np.pi, (b, 2)), rng.uniform(-WMAX, WMAX, (b, 2))]; dk, Bk = cKDTree(emb(C[pool])).query(emb(X), k=min(KN, len(pool))); dk, Bk = dk.reshape(b, -1), Bk.reshape(b, -1)
            B = pool[Bk[np.arange(b), np.where(dk <= dk[:, :1] * KF + .05, cost[pool][Bk], np.inf).argmin(1)]]
            Y = np.repeat(C[B], len(TF), 0); Z = flow(Y.T, np.tile(UF[:, 0], b), np.tile(UF[:, 1], b), sg * np.tile(TF, b)).T
            okz = np.isfinite(Z).all(1) & (np.abs(Z[:, 2:]) <= WMAX).all(1) & (np.abs(np.c_[wrap(Y[:, :2] - Z[:, :2]), Y[:, 2:] - Z[:, 2:]]) <= WIN).all(1)
            dz = np.linalg.norm(emb(np.nan_to_num(Z)) - np.repeat(emb(X), len(TF), 0), axis=1); dz[~okz] = np.inf; m = dz.reshape(b, -1).argmin(1); u = UF[m].T; t = TF[m]
            Aa = flow(C[B].T, u[0], u[1], sg * t).T; d = (flow((C[B] + 1e-5 * n[B]).T, u[0], u[1], sg * t).T - Aa) / 1e-5; ld = np.linalg.norm(d, axis=1, keepdims=True); nA = d / ld * np.clip(ld, .3, 3.)
            Aa[:, :2] = wrap(Aa[:, :2]); ok = np.isfinite(Aa).all(1) & (np.abs(Aa[:, 2:]) <= WMAX).all(1) & ~ingoal(Aa.T) & (np.abs(np.c_[wrap(C[B, :2] - Aa[:, :2]), C[B, 2:] - Aa[:, 2:]]) <= WIN).all(1)
            for fr in (.2, .4, .6, .8): Am = flow(C[B].T, u[0], u[1], sg * t * fr).T; ok &= (np.abs(np.nan_to_num(Am[:, 2:], nan=99.)) <= WMAX).all(1)
            ok &= cKDTree(emb(C)).query(emb(np.where(ok[:, None], Aa, 0.)))[0] >= dmin; acc = []
            for i in np.flatnonzero(ok):
                if not acc or np.min(np.linalg.norm(emb(Aa[acc]) - emb(Aa[i:i + 1]), axis=1)) >= dmin: acc.append(i)
            acc = acc[:max(0, NEW - added)]; C = np.r_[C, Aa[acc]]; n = np.r_[n, nA[acc]]; added += len(acc); acc_all += acc
        if len(acc_all) / (2 * b) < .03: dmin *= .9
    A.C, A.n = C, n; A.K = len(C); A.V = np.r_[A.V, np.full((A.K - K0, MN), BIG)]; E.refresh(A); A.V[A.ingoal] = 0.
    Yn = A.P[K0:].reshape(-1, 4); idn = np.arange(K0 * MN, A.K * MN); own = np.repeat(np.arange(K0, A.K), MN); e_new = E.edges_add(A, idn, A.pairs(Yn, own))
    near = np.unique(np.concatenate([np.asarray(b_, int) for b_ in cKDTree(A.X[:K0]).query_ball_point(A.X[K0:], 2 * WIN)] or [np.zeros(0, int)]))
    Yo = A.P[near].reshape(-1, 4); ido = (near[:, None] * MN + np.arange(MN)).reshape(-1); out = A.pairs(Yo); kp = out[1].astype(int) >= K0; e_old = E.edges_add(A, ido, [x[kp] for x in out])
    e = [np.r_[a, b_, c] for a, b_, c in zip(A.e, e_new, e_old)]; o = np.argsort(e[0], kind='stable'); A.e = [x[o] for x in e]; A.solve()
    print(json.dumps(dict(round=r, added=int(A.K - K0), spores=int(A.K), dmin=round(dmin, 3), V=round(float(A.V[KS, MN // 2]), 3), fwd=int(np.isfinite(E.g_from_start(A).min(1)).sum()), bwd=int((A.V.min(1) < BIG / 2).sum()), sec=round(time.time() - t0))), flush=True)
np.savez(sys.argv[1].replace('.npz', '_rg.npz'), C=A.C, n=A.n, V=A.V, e=np.array(A.e, dtype=object), allow_pickle=True)
if A.V[KS, MN // 2] < BIG / 2:
    A.E = Q.node_edges(A); A.nplan = 0; T, arcs, wm = Q.rollout_edges(A, KS, 0.)
    print(json.dumps(dict(V=round(float(A.V[KS, MN // 2]), 3), T=round(float(T), 3), T_over_OCP=round(float(T / OCP), 4), arcs=arcs, wmax=round(float(wm), 2))), flush=True)
