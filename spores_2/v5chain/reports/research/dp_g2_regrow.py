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
from butterfly_dp import S11, S12, S22, MS, G
EGAP, EW, EM = int(os.environ.get('EGAP', 0)), float(os.environ.get('EW', 0.)), float(os.environ.get('EM', .5)); EQ = float(os.environ.get('EQ', 80)); TRUNC = int(os.environ.get('TRUNC', 0))   # research-13: деревья разделены по ЭНЕРГИИ (dp_gap_diag.py) — цели Вороного в полосе разрыва, энергия в метрике
def energy(z): d1, d2 = z[2], z[2] + z[3]; return .5 * (S11 * d1 ** 2 + S22 * d2 ** 2 + 2 * S12 * np.cos(z[1]) * d1 * d2) + G * (MS[0] * np.sin(z[0]) + MS[1] * np.sin(z[0] + z[1]))
def emb2(Y): return np.c_[emb(Y), EW * energy(Y.T)] if EW else emb(Y)
def towardX(b, C, other, sg):                                                              # EGAP=2 (RRT-Connect): цель — спора ДРУГОГО дерева у разрыва (четверть ближайших по энергии) + шум
    e = energy(C[other].T); o = other[e <= np.percentile(e, 25)] if sg > 0 else other[e >= np.percentile(e, 75)]
    X = C[rng.choice(o, b)] + rng.normal(0, .1, (b, 4)); X[:, :2] = wrap(X[:, :2]); return X
def sampleX(b, band):
    X = np.c_[rng.uniform(-np.pi, np.pi, (40 * b, 2)), rng.uniform(-WMAX, WMAX, (40 * b, 2))]
    if band is not None: e = energy(X.T); X = X[(e >= band[0]) & (e <= band[1])]
    return X[:b] if len(X) >= b else np.c_[rng.uniform(-np.pi, np.pi, (b, 2)), rng.uniform(-WMAX, WMAX, (b, 2))]
A = Q.load(sys.argv[1]); E.refresh(A); A.V[A.ingoal] = 0.; A.solve(); KS = 81; t0 = time.time(); rng = np.random.default_rng(int(os.environ.get('SEED', 0)))
if os.environ.get('START'):                                                                 # research-13: новый старт — спорой в ГОТОВЫЙ атлас (общий пул, концепция пользователя), прямое дерево растёт от неё
    from butterfly_dp_grow import normals
    s0 = np.array([float(v) for v in os.environ['START'].split(',')]); K0 = A.K; A.C = np.r_[A.C, s0[None]]; A.n = np.r_[A.n, normals(s0[None], rng)]; A.K += 1; A.V = np.r_[A.V, np.full((1, MN), BIG)]
    E.refresh(A); A.V[A.ingoal] = 0.; KS = E.KS = K0; e_new = E.edges_add(A, np.arange(K0 * MN, A.K * MN), A.pairs(A.P[K0:].reshape(-1, 4), np.repeat([K0], MN)))
    e = [np.r_[a, b_] for a, b_ in zip(A.e, e_new)]; o = np.argsort(e[0], kind='stable'); A.e = [x[o] for x in e]; A.solve()
    print(json.dumps(dict(start=s0.tolist(), KS=int(KS), pairs_from_start=int(len(e_new[0])), V=round(float(A.V[KS, MN // 2]), 3), E_start=round(float(energy(s0)), 2))), flush=True)
NEW, ROUNDS, KN, KF, UMAX, OCP = int(os.environ.get('NEW', 1000)), int(os.environ.get('ROUNDS', 8)), int(os.environ.get('KN', 4)), float(os.environ.get('KF', 1.3)), .9, float(os.environ.get('OCP', 7.636))
UG = np.array([(a, c) for a in (-1, 0, 1) for c in (-1, 0, 1)]) * UMAX; TG = np.array([.25, .5, .9, 1.5]); TG = TG[TG <= TL + 1e-9]; UF = np.repeat(UG, len(TG), 0); TF = np.tile(TG, len(UG)); dmin = float(os.environ.get('DMIN0', .13))
for r in range(1, ROUNDS + 1):
    g = E.g_from_start(A).min(1); V = A.V.min(1); K0 = A.K; C, n = A.C, A.n; added = 0; tries = 0
    if V[KS] < BIG / 2 and not int(os.environ.get('KEEP', 0)): break
    band = None
    if EGAP:
        ec = energy(C.T); lo, hi = np.percentile(ec[np.isfinite(g)], EQ), np.percentile(ec[V < BIG / 2], 100 - EQ); band = (lo - EM, hi + EM) if lo < hi else None; print(json.dumps(dict(round=r, band=None if band is None else [round(float(v), 2) for v in band])), flush=True)
    while added < NEW and tries < 200:
        tries += 1; b = 300; acc_all = []
        for pool, cost, sg in ((np.flatnonzero(np.isfinite(g)), g, 1.), (np.flatnonzero(V < BIG / 2), V, -1.)):
            X = sampleX(b, band) if EGAP < 2 else towardX(b, C, np.flatnonzero(V < BIG / 2) if sg > 0 else np.flatnonzero(np.isfinite(g)), sg); dk, Bk = cKDTree(emb2(C[pool])).query(emb2(X), k=min(KN, len(pool))); dk, Bk = dk.reshape(b, -1), Bk.reshape(b, -1)
            B = pool[Bk[np.arange(b), np.where(dk <= dk[:, :1] * KF + .05, cost[pool][Bk], np.inf).argmin(1)]]
            Y = np.repeat(C[B], len(TF), 0); Z = flow(Y.T, np.tile(UF[:, 0], b), np.tile(UF[:, 1], b), sg * np.tile(TF, b)).T
            if TRUNC:                                                                        # research-13: дугу веера не отбраковывать, а ОБРЕЗАТЬ перед выходом за |w| ≤ WMAX / окно WIN (у фронта по энергии годны были только t = .25)
                tt = np.tile(TF, b); good = np.ones(len(tt), bool); tb = np.zeros(len(tt))
                for fr in np.linspace(.1, 1., 10):
                    z = flow(Y.T, np.tile(UF[:, 0], b), np.tile(UF[:, 1], b), sg * tt * fr).T; z = np.nan_to_num(z, nan=99.)
                    good &= (np.abs(z[:, 2:]) <= WMAX - .02).all(1) & (np.abs(np.c_[wrap(Y[:, :2] - z[:, :2]), Y[:, 2:] - z[:, 2:]]) <= WIN).all(1); tb = np.where(good, tt * fr, tb)
                Z = flow(Y.T, np.tile(UF[:, 0], b), np.tile(UF[:, 1], b), sg * np.maximum(tb, 1e-3)).T; Z[tb < .05] = np.nan
            okz = np.isfinite(Z).all(1) & (np.abs(Z[:, 2:]) <= WMAX).all(1) & (np.abs(np.c_[wrap(Y[:, :2] - Z[:, :2]), Y[:, 2:] - Z[:, 2:]]) <= WIN).all(1)
            dz = np.linalg.norm(emb2(np.nan_to_num(Z)) - np.repeat(emb2(X), len(TF), 0), axis=1); dz[~okz] = np.inf; m = dz.reshape(b, -1).argmin(1); u = UF[m].T; t = tb.reshape(b, -1)[np.arange(b), m] if TRUNC else TF[m]
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
    if int(os.environ.get('CKPT', 0)): np.savez(sys.argv[1].replace('.npz', '_rg.npz'), C=A.C, n=A.n, V=A.V, e=np.array(A.e, dtype=object), allow_pickle=True)   # research-13: снимок каждый раунд (прогон могут убить)
np.savez(sys.argv[1].replace('.npz', '_rg.npz'), C=A.C, n=A.n, V=A.V, e=np.array(A.e, dtype=object), allow_pickle=True)
if A.V[KS, MN // 2] < BIG / 2:
    A.E = Q.node_edges(A); A.nplan = 0; T, arcs, wm = Q.rollout_edges(A, KS, 0.)
    print(json.dumps(dict(V=round(float(A.V[KS, MN // 2]), 3), T=round(float(T), 3), T_over_OCP=round(float(T / OCP), 4), arcs=arcs, wmax=round(float(wm), 2))), flush=True)
