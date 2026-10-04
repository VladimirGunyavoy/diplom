"""research hub-v5chain-research-9: атлас бабочек двойного маятника (модель и обратная задача — butterfly_dp.py). Окно поиска соседей — по расстоянию
(|Δq|, |Δw| ≤ WIN покомпонентно), время дуги ≤ TL (урок маятника: ограничивать расстоянием, не временем). Пары считаются кусками (память ≤ 1.5 ГБ).
Запрос — эталон research-7: g = 1, старт (−π/2, 0, 0, 0), OCP 5.098, v6 5.28. Плюс 20 случайных стартов (|w| ≤ 1) — самосогласованность V ≈ T."""
import numpy as np, sys, os, time, json
from scipy.spatial import cKDTree
sys.path.insert(0, '.')
from butterfly_dp import wrap, flow, f, solve_arcs, ingoal, C3, RQ, RW_, WMAX, BIG, TL, WIN, G
import butterfly_dp as D
R, MN = float(os.environ.get('R', .15)), int(os.environ.get('MN', 5))
SNAPA = float(os.environ.get('SNAPA', 1e-4))                                                # research-10: приход в узел с ошибкой Ньютона ~1e-8 — считать узлом
def snap(a): return np.where(a < SNAPA, 0., np.where(a > 1 - SNAPA, 1., a))   # (иначе вес 1e-7 у соседа с BIG убивал ребро)
class Atlas:
    def __init__(s, N, seed=0):
        rng = np.random.default_rng(seed); g = np.linspace(-1, 1, 3)
        GC = np.array([[C3[0] + a * RQ * .8, C3[1] + b * RQ * .8, c * RW_ * .8, d * RW_ * .8] for a in g for b in g for c in g for d in g])
        s.C = np.r_[GC, np.c_[rng.uniform(-np.pi, np.pi, (N, 2)), rng.uniform(-WMAX, WMAX, (N, 2))]]; s.K = len(s.C)
        w = s.C[:, 2:]; nn = np.hypot(w[:, 0], w[:, 1]); rnd = rng.normal(size=(s.K, 2)); rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
        s.n = np.zeros((s.K, 4)); s.n[:, 0] = np.where(nn > .05, -w[:, 1] / np.maximum(nn, 1e-9), rnd[:, 0]); s.n[:, 1] = np.where(nn > .05, w[:, 0] / np.maximum(nn, 1e-9), rnd[:, 1])
        s.sn = np.linspace(-R, R, MN); s.P = s.C[:, None, :] + s.sn[None, :, None] * s.n[:, None, :]
        s.ingoal = ingoal(np.moveaxis(s.P, 2, 0)); s.V = np.full((s.K, MN), BIG); s.V[s.ingoal] = 0.
        X = np.c_[np.cos(s.C[:, 0]), np.sin(s.C[:, 0]), np.cos(s.C[:, 1]), np.sin(s.C[:, 1]), s.C[:, 2:] / 2]; s.tree = cKDTree(X); s.X = X
    def emb(s, Y): return np.c_[np.cos(Y[:, 0]), np.sin(Y[:, 0]), np.cos(Y[:, 1]), np.sin(Y[:, 1]), Y[:, 2:] / 2]
    def pairs(s, Y, own=None, chunk=3000):
        out = [[], [], [], []]
        for a0 in range(0, len(Y), chunk):
            y = Y[a0:a0 + chunk]; nb = s.tree.query_ball_point(s.emb(y), 2 * WIN)                     # грубый шар, дальше точная коробка
            iy = np.repeat(np.arange(len(y)), [len(b) for b in nb]); k = np.concatenate([np.asarray(b, int) for b in nb]) if len(iy) else np.zeros(0, int)
            if own is not None: g = k != own[a0 + iy]; iy, k = iy[g], k[g]
            d = np.c_[wrap(s.C[k, :2] - y[iy, :2]), s.C[k, 2:] - y[iy, 2:]]; g = (np.abs(d) <= WIN).all(1); iy, k = iy[g], k[g]
            if not len(iy): continue
            t, u1, u2, sv, ok = solve_arcs(y[iy].T, s.C[k].T, s.n[k].T); ok &= np.abs(sv) <= R
            for o, v in zip(out, (iy[ok] + a0, k[ok], sv[ok], t[ok])): o.append(v)
        return [np.concatenate(o) for o in out] if out[0] else [np.zeros(0)] * 4
    def build(s):
        Y = s.P.reshape(-1, 4); own = np.repeat(np.arange(s.K), MN); iy, k, sv, t = s.pairs(Y, own)
        f_ = (sv + R) / (2 * R) * (MN - 1); j0 = np.clip(np.floor(f_).astype(int), 0, MN - 2); a = f_ - j0
        o = np.argsort(iy, kind='stable'); s.e = [v[o] for v in (iy.astype(int), k.astype(int), j0, a, t)]; return s
    def solve(s, it=3000, tol=1e-7):
        iy, k, j0, a, t = s.e; V = s.V.reshape(-1).copy(); fixed = s.ingoal.reshape(-1); st = np.flatnonzero(np.r_[True, iy[1:] != iy[:-1]]); nodes = iy[st]
        for n in range(it):
            V0, V1 = V[k * MN + j0], V[k * MN + j0 + 1]; a = snap(a); val = t + (1 - a) * V0 + a * V1; val[((V0 >= BIG / 2) & (a < 1 - SNAPA)) | ((V1 >= BIG / 2) & (a > SNAPA))] = BIG
            new = V.copy(); new[nodes] = np.minimum(V[nodes], np.minimum.reduceat(val, st)); new[fixed] = 0.; d = np.max(np.abs(new - V)); V = new
            if d < tol: break
        s.V = V.reshape(s.K, MN); s.n_it = n; return s
    def best(s, y):
        if ingoal(y[:, None])[0]: return 0., 0., 0., 0.
        out = s.pairs(y[None]); iy, k, sv, t = out
        if not len(t): return BIG, 0, 0, 0
        f_ = (sv + R) / (2 * R) * (MN - 1); j0 = np.clip(np.floor(f_).astype(int), 0, MN - 2); a = f_ - j0; k = k.astype(int)
        V0, V1 = s.V[k, j0], s.V[k, j0 + 1]; a = snap(a); val = t + (1 - a) * V0 + a * V1; val[((V0 >= BIG / 2) & (a < 1 - SNAPA)) | ((V1 >= BIG / 2) & (a > SNAPA))] = BIG; i = int(np.argmin(val))
        _, u1, u2, _, ok = solve_arcs(y[:, None], s.C[k[i]][:, None], s.n[k[i]][:, None]); return val[i], u1[0], u2[0], t[i]
    def rollout(s, y0, dt=.05, h=.01, tmax=30.):
        y = np.array(y0, float); t = 0.; wmax = 0.; sw = 0; pu = None
        while t < tmax:
            if ingoal(y[:, None])[0]: return t, wmax, sw
            J, u1, u2, ta = s.best(y)
            if J >= BIG / 2: return np.inf, wmax, sw
            if pu is not None and (abs(u1 - pu[0]) > 1 or abs(u2 - pu[1]) > 1): sw += 1
            pu = (u1, u2); hold = min(dt, ta)
            while hold > 1e-12:
                hh = min(h, hold); y = flow(y[:, None], np.array([u1]), np.array([u2]), hh, n=1)[:, 0]; t += hh; hold -= hh; wmax = max(wmax, np.abs(y[2:]).max())
                if ingoal(y[:, None])[0]: return t, wmax, sw
        return np.inf, wmax, sw
if __name__ == '__main__':
    N = int(sys.argv[1]); t0 = time.time(); A = Atlas(N).build(); tp = time.time() - t0; A.solve()
    print(json.dumps(dict(N=N, G=G, WIN=WIN, TL=TL, R=R, nodes=A.K * MN, pairs=len(A.e[0]), pairs_per_node=round(len(A.e[0]) / (A.K * MN), 1), sec_pairs=round(tp), iters=A.n_it,
                          BIG=round(float((A.V >= BIG / 2).mean()), 3), sec=round(time.time() - t0))), flush=True)
    q = np.array([-np.pi / 2, 0, 0, 0]); V0 = A.best(q)[0]; T, wm, sw = A.rollout(q)
    print(json.dumps(dict(query='висит→вверх', V=round(float(V0), 3), T=round(float(T), 3), T_over_OCP=round(float(T / 5.098), 4), v6=5.28, wmax=round(float(wm), 2), sw=sw), ensure_ascii=False), flush=True)
    rng = np.random.default_rng(3); Q = np.c_[rng.uniform(-np.pi, np.pi, (20, 2)), rng.uniform(-1, 1, (20, 2))]; res = []
    for qq in Q: v = A.best(qq)[0]; T, wm, sw = A.rollout(qq); res.append((v, T, wm, sw))
    R_ = np.array(res); f_ = np.isfinite(R_[:, 1]) & (R_[:, 0] < BIG / 2)
    print(json.dumps(dict(random20_reach=round(float(f_.mean()), 2), T_over_V=dict(mean=round(float((R_[f_, 1] / R_[f_, 0]).mean()), 3), max=round(float((R_[f_, 1] / R_[f_, 0]).max()), 3)) if f_.any() else None,
                          T_med=round(float(np.median(R_[f_, 1])), 2) if f_.any() else None, wmax_over3=int((R_[:, 2] > 3).sum()), sec=round(time.time() - t0)), ensure_ascii=False), flush=True)
    np.save('butterfly_dp_N%d_win%g.npy' % (N, WIN), R_)
