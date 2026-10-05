"""research-13: доля годных дуг веера роста (9 u × 4 t) по полосам энергии — прямое дерево вперёд, обратное назад; причина отказа: |w| > 3 (в конце/вдоль) или окно WIN."""
import numpy as np, sys, json
sys.path.insert(0, '.')
from butterfly_dp import flow, wrap, BIG, WIN, WMAX, TL, G, S11, S12, S22, MS
from butterfly_dp_atlas import MN
import butterfly_dp_query as Q, butterfly_dp_ellipse as E
def energy(z): d1, d2 = z[2], z[2] + z[3]; return .5 * (S11 * d1 ** 2 + S22 * d2 ** 2 + 2 * S12 * np.cos(z[1]) * d1 * d2) + G * (MS[0] * np.sin(z[0]) + MS[1] * np.sin(z[0] + z[1]))
A = Q.load(sys.argv[1]); E.refresh(A); A.V[A.ingoal] = 0.; A.solve(); g = E.g_from_start(A).min(1); V = A.V.min(1)
UG = np.array([(a, c) for a in (-1, 0, 1) for c in (-1, 0, 1)]) * .9; TG = np.array([.25, .5, .9, 1.5]); UF = np.repeat(UG, 4, 0); TF = np.tile(TG, 9); n = len(TF)
for name, pool, sg in (('fwd', np.flatnonzero(np.isfinite(g)), 1.), ('bwd', np.flatnonzero(V < BIG / 2), -1.)):
    C = A.C[pool]; e = energy(C.T); Y = np.repeat(C, n, 0); u1, u2, t = np.tile(UF[:, 0], len(C)), np.tile(UF[:, 1], len(C)), sg * np.tile(TF, len(C))
    wbad = np.zeros(len(Y), bool)
    for fr in (.2, .4, .6, .8, 1.): z = flow(Y.T, u1, u2, t * fr); wbad |= ~(np.abs(np.nan_to_num(z[2:], nan=99.)) <= WMAX).all(0)
    Z = flow(Y.T, u1, u2, t).T; winbad = ~wbad & ~(np.abs(np.c_[wrap(Y[:, :2] - Z[:, :2]), Y[:, 2:] - Z[:, 2:]]) <= WIN).all(1); ok = ~wbad & ~winbad
    dE = np.where(ok, energy(Z.T) - np.repeat(e, n), np.nan) * sg
    ok, wbad, winbad, dE = [x.reshape(len(C), n) for x in (ok, wbad, winbad, dE)]; ok_t = ok.reshape(len(C), 9, 4)
    qs = np.percentile(e, [0, 25, 50, 75, 90, 97, 100])
    for a, b in zip(qs[:-1], qs[1:]):
        m = (e >= a) & (e <= b); best = np.nanmax(np.where(ok[m], dE[m], -np.inf), 1)
        print(json.dumps(dict(tree=name, E=[round(float(a), 2), round(float(b), 2)], n=int(m.sum()), ok=round(float(ok[m].mean()), 3), w_bad=round(float(wbad[m].mean()), 3), win_bad=round(float(winbad[m].mean()), 3),
                              ok_by_t=[round(float(v), 2) for v in ok_t[m].mean((0, 1))], dead=round(float((~ok[m].any(1)).mean()), 3), best_dE_toward_gap_med=round(float(np.median(best[np.isfinite(best)])), 2) if np.isfinite(best).any() else None)), flush=True)
