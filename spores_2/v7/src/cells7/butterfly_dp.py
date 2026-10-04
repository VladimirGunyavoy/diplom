"""Бабочки ДВОЙНОГО МАЯТНИКА в v7 (PLAN п.5, перенос прототипов research-9/10: butterfly_dp, _atlas, _grow, _query, _ellipse из v5chain/reports/research).
Модель 2 звена (g=1, моменты |τ|≤1), спора = точка + отрезок вдоль нормали, обратная задача дуги — Ньютон 4×4, Беллман по парам; рост спор обратным/прямым деревом
с переносом нормали (TR=1), агент по рёбрам с запасными дугами, эллиптическое доращивание. Параметры — переменные окружения, как в прототипах.
Запуск из spores_2/v7: `FWD=1500 TR=1 FR=1 python3 -m src.cells7.butterfly_dp grow 1500`, `... ellipse <npz>`. Тяжёлое — systemd-run MemoryMax=1500M."""
import numpy as np, sys, os, time, json
import multiprocessing as mp
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
G = float(os.environ.get('G', 1.)); M1, M2 = 1.5, 1.0; S11, S12, S22 = M1 + M2, M2, M2; MS = np.array([M1 + M2, M2])
C3 = np.array([np.pi / 2, 0.]); RQ, RW_, WMAX = .3, .5, 3.; BIG = 1e3
RRT = int(os.environ.get('RRT', 0))   # RRT=1: родитель со смещением Вороного (research-11, g=2)
PRUNE = int(os.environ.get('PRUNE', 1)); PRUNE_AT = int(os.environ.get('PRUNE_AT', 2)); PRUNE_TH = float(os.environ.get('PRUNE_TH', .1))
TL = float(os.environ.get('TL', 1.5)); WIN = float(os.environ.get('WIN', .5))
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def acc(q1, q2, w1, w2, u1, u2):
    th1, th2 = q1, q1 + q2; d1, d2 = w1, w1 + w2; c = np.cos(th1 - th2); s = np.sin(th1 - th2)
    Q1 = u1 - u2 - G * MS[0] * np.cos(th1) - S12 * s * d2 ** 2; Q2 = u2 - G * MS[1] * np.cos(th2) + S12 * s * d1 ** 2
    det = S11 * S22 - (S12 * c) ** 2; a1 = (S22 * Q1 - S12 * c * Q2) / det; a2 = (S11 * Q2 - S12 * c * Q1) / det
    return a1, a2 - a1
def invdyn(q1, q2, w1, w2, a1, a2):
    """τ, дающие q̈ = (a1, a2) в состоянии."""
    th1, th2 = q1, q1 + q2; d1, d2 = w1, w1 + w2; c = np.cos(th1 - th2); s = np.sin(th1 - th2); t1, t2 = a1, a1 + a2
    Q1 = S11 * t1 + S12 * c * t2; Q2 = S12 * c * t1 + S22 * t2
    u2 = Q2 + G * MS[1] * np.cos(th2) - S12 * s * d1 ** 2; u1 = Q1 + u2 + G * MS[0] * np.cos(th1) + S12 * s * d2 ** 2; return u1, u2
def f(z, u1, u2): a1, a2 = acc(z[0], z[1], z[2], z[3], u1, u2); return np.stack([z[2], z[3], a1, a2])
def flow(z, u1, u2, t, n=6):
    h = t / n
    for _ in range(n):
        k1 = f(z, u1, u2); k2 = f(z + h / 2 * k1, u1, u2); k3 = f(z + h / 2 * k2, u1, u2); k4 = f(z + h * k3, u1, u2); z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z
def ingoal(z): return (np.abs(wrap(z[0] - C3[0])) <= RQ) & (np.abs(wrap(z[1] - C3[1])) <= RQ) & (np.abs(z[2]) <= RW_) & (np.abs(z[3]) <= RW_)

def solve_arcs(Y, C, Nn, newton=8):
    """Y (4,n) старты, C (4,n) центры спор, Nn (4,n) нормали — попарно. Возвращает t, u1, u2, s, ok."""
    dq = wrap(C[:2] - Y[:2]); wb = Y[2:] + C[2:]; t = 2 * (dq * wb).sum(0) / np.maximum((wb ** 2).sum(0), 1e-9)
    r = dq - wb * t / 2; s = (r * Nn[:2]).sum(0)                                                          # остаток вдоль нормали ⇒ s
    t = np.clip(t, 1e-3, TL); a1, a2 = (C[2] - Y[2]) / t, (C[3] - Y[3]) / t; zm = (Y + np.r_[dq + Y[:2], C[2:]][[0, 1, 2, 3]]) / 2
    u1, u2 = invdyn(zm[0], zm[1], zm[2], zm[3], a1, a2); Ct = C.copy(); Ct[:2] = Y[:2] + dq                    # цель без скачка 2π
    def _newton(Y, Nn, Ct, t, u1, u2, s, k):
        for _ in range(k):
            Z = flow(Y, u1, u2, t); R = Z - (Ct + s * Nn)
            e = 1e-5; Z1 = flow(Y, u1 + e, u2, t); Z2 = flow(Y, u1, u2 + e, t); J = np.stack([f(Z, u1, u2), (Z1 - Z) / e, (Z2 - Z) / e, -Nn], -1)   # (4, n, 4)
            J = np.moveaxis(J, 1, 0); rhs = np.moveaxis(R, 0, 1)[..., None]
            bad = ~np.isfinite(J).all((1, 2)) | (np.abs(np.linalg.det(np.nan_to_num(J))) < 1e-12)   # research-10: вырожденные пары (нормаль в касательной трубки) — шаг 0
            J[bad] = np.eye(4); rhs[bad] = 0.; d = np.linalg.solve(J, rhs)[..., 0]
            t = np.clip(t - d[:, 0], 1e-4, 2 * TL); u1 = np.clip(u1 - d[:, 1], -3, 3); u2 = np.clip(u2 - d[:, 2], -3, 3); s = s - d[:, 3]
        return t, u1, u2, s
    if PRUNE and newton > PRUNE_AT:                                    # hub-v5chain-worker-16: после PRUNE_AT шагов отсечь пары с невязкой > PRUNE_TH (86% пар, 0.8% сошедшихся)
        t, u1, u2, s = _newton(Y, Nn, Ct, t, u1, u2, s, PRUNE_AT); R0 = np.abs(flow(Y, u1, u2, t) - (Ct + s * Nn)).max(0); a = np.nonzero(R0 <= PRUNE_TH)[0]
        t2, v1, v2, s2 = _newton(Y[:, a], Nn[:, a], Ct[:, a], t[a], u1[a], u2[a], s[a], newton - PRUNE_AT); t[a], u1[a], u2[a], s[a] = t2, v1, v2, s2
    else: t, u1, u2, s = _newton(Y, Nn, Ct, t, u1, u2, s, newton)
    Z = flow(Y, u1, u2, t); res = np.abs(Z - (Ct + s * Nn)).max(0)
    ok = (res < 1e-7) & (t > 1e-3) & (t <= TL) & (np.abs(u1) <= 1 + 1e-9) & (np.abs(u2) <= 1 + 1e-9)
    return t, u1, u2, s, ok

R, MN = float(os.environ.get('R', .15)), int(os.environ.get('MN', 5))
SNAPA = float(os.environ.get('SNAPA', 1e-4))   # research-10: приход в узел с ошибкой Ньютона — привязать (вес 1e-7 у соседа с BIG убивал ребро); SNAPA=0 — как раньше
def snap(a): return np.where(a < SNAPA, 0., np.where(a > 1 - SNAPA, 1., a))
def _pairs_job(a0):
    s, Y, own, chunk = _PJ; return s._pairs_chunk(a0, Y, own, chunk)
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
    def _pairs_chunk(s, a0, Y, own, chunk):
        y = Y[a0:a0 + chunk]; nb = s.tree.query_ball_point(s.emb(y), 2 * WIN)                     # грубый шар, дальше точная коробка
        iy = np.repeat(np.arange(len(y)), [len(b) for b in nb]); k = np.concatenate([np.asarray(b, int) for b in nb]) if len(iy) else np.zeros(0, int)
        if own is not None: g = k != own[a0 + iy]; iy, k = iy[g], k[g]
        d = np.c_[wrap(s.C[k, :2] - y[iy, :2]), s.C[k, 2:] - y[iy, 2:]]; g = (np.abs(d) <= WIN).all(1); iy, k = iy[g], k[g]
        if not len(iy): return None
        t, u1, u2, sv, ok = solve_arcs(y[iy].T, s.C[k].T, s.n[k].T); ok &= np.abs(sv) <= R
        return [v[ok] for v in (iy + a0, k, sv, t)]
    def pairs(s, Y, own=None, chunk=3000):
        starts = list(range(0, len(Y), chunk)); nproc = int(os.environ.get('NPROC', 1))
        if nproc > 1 and len(starts) > 1:                                                          # NPROC: куски пар по процессам (fork, атлас наследуется)
            global _PJ; _PJ = (s, Y, own, chunk)
            with mp.get_context('fork').Pool(nproc) as pool: res = pool.map(_pairs_job, starts)
        else: res = [s._pairs_chunk(a0, Y, own, chunk) for a0 in starts]
        out = [[], [], [], []]
        for r in res:
            if r is None: continue
            for o, v in zip(out, r): o.append(v)
        return [np.concatenate(o) for o in out] if out[0] else [np.zeros(0)] * 4
    def build(s):
        Y = s.P.reshape(-1, 4); own = np.repeat(np.arange(s.K), MN); iy, k, sv, t = s.pairs(Y, own)
        f_ = (sv + R) / (2 * R) * (MN - 1); j0 = np.clip(np.floor(f_).astype(int), 0, MN - 2); a = f_ - j0
        o = np.argsort(iy, kind='stable'); s.e = [v[o] for v in (iy.astype(int), k.astype(int), j0, a, t)]; return s
    def solve(s, it=3000, tol=1e-7):
        iy, k, j0, a, t = s.e; V = s.V.reshape(-1).copy(); fixed = s.ingoal.reshape(-1); st = np.flatnonzero(np.r_[True, iy[1:] != iy[:-1]]); nodes = iy[st]
        for n in range(it):
            V0, V1 = V[k * MN + j0], V[k * MN + j0 + 1]; a = snap(a); val = t + (1 - a) * V0 + a * V1; val[((V0 >= BIG / 2) & (a < 1 - max(SNAPA, 1e-9))) | ((V1 >= BIG / 2) & (a > max(SNAPA, 1e-9)))] = BIG
            new = V.copy(); new[nodes] = np.minimum(V[nodes], np.minimum.reduceat(val, st)); new[fixed] = 0.; d = np.max(np.abs(new - V)); V = new
            if d < tol: break
        s.V = V.reshape(s.K, MN); s.n_it = n; return s
    def best(s, y):
        if ingoal(y[:, None])[0]: return 0., 0., 0., 0.
        out = s.pairs(y[None]); iy, k, sv, t = out
        if not len(t): return BIG, 0, 0, 0
        f_ = (sv + R) / (2 * R) * (MN - 1); j0 = np.clip(np.floor(f_).astype(int), 0, MN - 2); a = f_ - j0; k = k.astype(int)
        V0, V1 = s.V[k, j0], s.V[k, j0 + 1]; val = t + (1 - a) * V0 + a * V1; val[((V0 >= BIG / 2) & (a < 1 - 1e-9)) | ((V1 >= BIG / 2) & (a > 1e-9))] = BIG; i = int(np.argmin(val))
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
Atlas.pairs.__defaults__ = (None, int(os.environ.get('CHUNK', 800)))                       # куски пар меньше: 4000+4000 при 3000 падали по OOM (1.5 ГБ)
DMIN0 = float(os.environ.get('DMIN', .35)); TR = int(os.environ.get('TR', 0)); UMAX = float(os.environ.get('UMAX', .9)); FR = int(os.environ.get('FR', 0)); FWD = int(os.environ.get('FWD', 0)); START = np.array([-np.pi / 2, 0, 0, 0])
def normals(C, rng):
    w = C[:, 2:]; nn = np.hypot(w[:, 0], w[:, 1]); rnd = rng.normal(size=(len(C), 2)); rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    n = np.zeros((len(C), 4)); n[:, 0] = np.where(nn > .05, -w[:, 1] / np.maximum(nn, 1e-9), rnd[:, 0]); n[:, 1] = np.where(nn > .05, w[:, 0] / np.maximum(nn, 1e-9), rnd[:, 1]); return n
def emb(Y): return np.c_[np.cos(Y[:, 0]), np.sin(Y[:, 0]), np.cos(Y[:, 1]), np.sin(Y[:, 1]), Y[:, 2:] / 2]
class GrowAtlas(Atlas):
    def __init__(s, N, seed=0, batch=400):
        rng = np.random.default_rng(seed); g = np.linspace(-1, 1, 3)
        C = np.array([[C3[0] + a * RQ * .8, C3[1] + b * RQ * .8, c * RW_ * .8, d * RW_ * .8] for a in g for b in g for c in g for d in g])
        n = normals(C, rng); tv = np.zeros(len(C)); dmin = DMIN0; s.grow_log = []; s.lam = []; fw = np.zeros(len(C), bool); it = 0
        if FWD: C = np.r_[C, START[None]]; n = np.r_[n, normals(START[None], rng)]; tv = np.r_[tv, 0.]; fw = np.r_[fw, True]   # FWD: корень прямого дерева — старт запроса
        while len(C) < N + 81 + FWD:
            it += 1; sg = 1. if (FWD and it % 2 == 0 and fw.sum() < FWD + 1) else -1.; pool = np.flatnonzero(fw if sg > 0 else ~fw)
            if sg < 0 and (~fw).sum() >= N + 81: continue
            if sg > 0: B = pool[rng.integers(len(pool), size=batch)]
            elif FR:                                                                           # FR: родитель равномерно по уровню «время до цели по построению» ⇒ фронт уходит дальше
                o = pool[np.argsort(tv[pool])]; L = rng.uniform(0, tv[pool].max() + .3, batch); B = o[np.clip(np.searchsorted(tv[o], L) - rng.integers(0, 8, batch), 0, len(o) - 1)]
            else: B = pool[rng.integers(len(pool), size=batch)]
            u = rng.uniform(-1, 1, (2, batch)); corner = rng.random(batch) < .5
            u[:, corner] = np.sign(u[:, corner]); t = rng.uniform(.2, TL, batch); sv = np.linspace(-R, R, MN)[rng.integers(MN, size=batch)]   # точно в узел споры B: интерполяция V по отрезку не нужна
            if TR: u *= UMAX; sv[:] = 0.                                                     # TR: из центра B, запас по τ для соседних узлов
            if RRT:                                                                          # research-11: смещение Вороного — случайная точка области → ближайшая спора дерева → лучшая дуга веера к ней (dp_g2_voronoi.py)
                UG = np.array([(a, c) for a in (-1, 0, 1) for c in (-1, 0, 1)]) * UMAX; TG = np.array([.25, .5, .9, 1.5]); TG = TG[TG <= TL + 1e-9]; UF = np.repeat(UG, len(TG), 0); TF = np.tile(TG, len(UG))
                X = np.c_[rng.uniform(-np.pi, np.pi, (batch, 2)), rng.uniform(-WMAX, WMAX, (batch, 2))]; B = pool[cKDTree(emb(C[pool])).query(emb(X))[1]]
                Y = np.repeat(C[B], len(TF), 0); Z = flow(Y.T, np.tile(UF[:, 0], batch), np.tile(UF[:, 1], batch), sg * np.tile(TF, batch)).T
                okz = np.isfinite(Z).all(1) & (np.abs(Z[:, 2:]) <= WMAX).all(1) & (np.abs(np.c_[wrap(Y[:, :2] - Z[:, :2]), Y[:, 2:] - Z[:, 2:]]) <= WIN).all(1)
                dz = np.linalg.norm(emb(np.nan_to_num(Z)) - np.repeat(emb(X), len(TF), 0), axis=1); dz[~okz] = np.inf; m = dz.reshape(batch, -1).argmin(1); u = UF[m].T.copy(); t = TF[m].copy(); sv[:] = 0.
            A = flow((C[B] + sv[:, None] * n[B]).T, u[0], u[1], sg * t).T
            if TR:                                                                           # нормаль A = перенос нормали B назад по дуге ⇒ узлы A ложатся на отрезок B
                d = (flow((C[B] + 1e-5 * n[B]).T, u[0], u[1], sg * t).T - A) / 1e-5; nA = d / np.linalg.norm(d, axis=1, keepdims=True); s.lam.append(np.linalg.norm(d, axis=1))
            A[:, :2] = wrap(A[:, :2])
            ok = np.isfinite(A).all(1) & (np.abs(A[:, 2:]) <= WMAX).all(1) & ~ingoal(A.T)
            ok &= (np.abs(np.c_[wrap(C[B, :2] - A[:, :2]), C[B, 2:] - A[:, 2:]]) <= WIN).all(1)      # дуга должна попасть в окно пар
            ok &= cKDTree(emb(C)).query(emb(np.where(ok[:, None], A, 0.)))[0] >= dmin
            acc = []
            for i in np.flatnonzero(ok):                                                     # внутри пачки — тоже не ближе dmin
                if not acc or np.min(np.linalg.norm(emb(A[acc]) - emb(A[i:i + 1]), axis=1)) >= dmin: acc.append(i)
            acc = acc[:(FWD + 1 - fw.sum()) if sg > 0 else (N + 81 - (~fw).sum())]; fw = np.r_[fw, np.full(len(acc), sg > 0)]; C = np.r_[C, A[acc]]; tv = np.r_[tv, tv[B[acc]] + t[acc]]; n = np.r_[n, nA[acc] if TR else normals(A[acc], rng)]
            rate = len(acc) / batch; s.grow_log.append((len(C), round(dmin, 3), round(rate, 3)))
            if rate < .05: dmin *= .9
        s.C, s.n, s.K = C, n, len(C); s.dmin = dmin; s.tv = tv; s.fw = fw
        s.sn = np.linspace(-R, R, MN); s.P = s.C[:, None, :] + s.sn[None, :, None] * s.n[:, None, :]
        s.ingoal = ingoal(np.moveaxis(s.P, 2, 0)); s.V = np.full((s.K, MN), BIG); s.V[s.ingoal] = 0.
        s.X = emb(s.C); s.tree = cKDTree(s.X)
TS = np.array([float(x) for x in os.environ.get('TS', '.25,.5,.9').split(',')]); DEP = int(os.environ.get('DEP', 2)); TOPE = int(os.environ.get('TOPE', 5)); ALLN = int(os.environ.get('ALLN', 1)); PLANFB = int(os.environ.get('PLANFB', 1)); FORCEPLAN = int(os.environ.get('FORCEPLAN', 0))
UU = np.array([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)], float)
def load(path):
    d = np.load(path, allow_pickle=True); A = Atlas.__new__(Atlas); A.C, A.n, A.V = d['C'], d['n'], d['V']; A.K = len(A.C)
    A.sn = np.linspace(-R, R, MN); A.X = emb(A.C); A.tree = cKDTree(A.X)
    A.e = [np.asarray(x, dt) for x, dt in zip(d['e'], (int, int, int, float, float))]; return A
def direct(A, Y):
    """Лучшее значение одной дугой в атлас для каждой строки Y (n, 4); и (t, u1, u2) этой дуги."""
    out = np.full(len(Y), BIG); arc = np.zeros((len(Y), 3))
    if not len(Y): return out, arc
    gi = ingoal(Y.T); out[gi] = 0.
    iy, k, sv, t = A.pairs(Y); iy, k = iy.astype(int), k.astype(int)
    if not len(iy): return out, arc
    f_ = (sv + R) / (2 * R) * (MN - 1); j0 = np.clip(np.floor(f_).astype(int), 0, MN - 2); a = f_ - j0
    V0, V1 = A.V[k, j0], A.V[k, j0 + 1]; val = t + (1 - a) * V0 + a * V1; val[((V0 >= BIG / 2) & (a < 1 - 1e-9)) | ((V1 >= BIG / 2) & (a > 1e-9))] = BIG
    o = np.lexsort((val, iy)); iy, k, val, t = iy[o], k[o], val[o], t[o]; first = np.r_[True, iy[1:] != iy[:-1]]
    for i, kk, v, tt in zip(iy[first], k[first], val[first], t[first]):
        if v < out[i]:
            _, u1, u2, _, ok = solve_arcs(Y[i][:, None], A.C[kk][:, None], A.n[kk][:, None]); out[i] = v; arc[i] = (tt, u1[0], u2[0])
    return out, arc
def children(Y):
    n = len(Y); U = np.repeat(UU, len(TS), 0); T = np.tile(TS, len(UU)); m = len(T)
    Yr = np.repeat(Y, m, 0); Z = flow(Yr.T, np.tile(U[:, 0], n), np.tile(U[:, 1], n), np.tile(T, n)).T; Z[:, :2] = wrap(Z[:, :2])
    return Z, np.tile(T, n), np.tile(U, (n, 1))
def plan(A, y):
    """V(y) и первая дуга (t, u1, u2): прямо в атлас или через дерево глубины DEP."""
    v, arc = direct(A, y[None]); best, barc = v[0], arc[0]
    lv = [(y[None], None, None, None)]; Y = y[None]; par = None
    for d in range(DEP):
        Z, T, U = children(Y); ok = np.isfinite(Z).all(1) & (np.abs(Z[:, 2:]) <= 3).all(1)
        root = np.repeat(np.arange(len(Y)), len(TS) * len(UU)) if par is None else np.repeat(par, len(TS) * len(UU))
        cost = np.repeat(np.zeros(len(Y)) if d == 0 else Ccost, len(TS) * len(UU)) + T
        first = np.c_[T, U] if d == 0 else np.repeat(Farc, len(TS) * len(UU), 0)
        Z, cost, first = Z[ok], cost[ok], first[ok]; vz, _ = direct(A, Z); tot = cost + vz; i = int(np.argmin(tot))
        if tot[i] < best: best, barc = tot[i], first[i]
        Y, Ccost, Farc, par = Z, cost, first, None
    return best, barc
def rollout(A, y0, tmax=25.):
    y = np.array(y0, float); t = 0.; wmax = 0.; nplan = 0; V0 = None
    while t < tmax:
        if ingoal(y[:, None])[0]: return t, wmax, nplan, V0
        J, (ta, u1, u2) = plan(A, y); nplan += 1; V0 = J if V0 is None else V0
        if J >= BIG / 2: return np.inf, wmax, nplan, V0
        for _ in range(max(1, int(round(ta / .01)))):
            y = flow(y[:, None], np.array([u1]), np.array([u2]), ta / max(1, int(round(ta / .01))), n=1)[:, 0]; t += ta / max(1, int(round(ta / .01))); wmax = max(wmax, np.abs(y[2:]).max())
            if ingoal(y[:, None])[0]: return t, wmax, nplan, V0
    return np.inf, wmax, nplan, V0
# --- research-10: агент «по рёбрам» — тёплый Ньютон из точки на отрезке к цели ребра ближайшего узла ---
fdyn = f
def newton_warm(y, c, nn, t, u1, u2, s, it=12):
    for _ in range(it):
        Z = flow(y[:, None], np.array([u1]), np.array([u2]), t)[:, 0]; Rz = Z - (c + s * nn); Rz[:2] = wrap(Rz[:2])
        e = 1e-6; Z1 = flow(y[:, None], np.array([u1 + e]), np.array([u2]), t)[:, 0]; Z2 = flow(y[:, None], np.array([u1]), np.array([u2 + e]), t)[:, 0]
        J = np.c_[fdyn(Z[:, None], np.array([u1]), np.array([u2]))[:, 0], (Z1 - Z) / e, (Z2 - Z) / e, -nn]
        try: d = np.linalg.solve(J, Rz)
        except np.linalg.LinAlgError: return None
        t, u1, u2, s = t - d[0], u1 - d[1], u2 - d[2], s - d[3]
        if not np.isfinite([t, u1, u2, s]).all() or t <= 1e-3 or t > 3: return None
    Z = flow(y[:, None], np.array([u1]), np.array([u2]), t)[:, 0]; Rz = Z - (c + s * nn); Rz[:2] = wrap(Rz[:2])
    if np.abs(Rz).max() > 1e-7 or abs(u1) > 1 + 1e-9 or abs(u2) > 1 + 1e-9 or abs(s) > R: return None
    return t, u1, u2, s
def node_edges(A):
    """Для каждого узла (k, j): лучшее ребро (k2, t, u1, u2, s2) по V атласа."""
    iy, k, j0, a, t = A.e; V = A.V.reshape(-1); val = t + (1 - a) * V[k * MN + j0] + a * V[k * MN + j0 + 1]
    o = np.lexsort((val, iy)); E = {}
    for i in o:                                                                          # research-10: до TOPE рёбер на узел — запасные, если тёплый Ньютон не сошёлся
        if val[i] < BIG / 2 and len(E.setdefault(int(iy[i]), [])) < TOPE: E[int(iy[i])].append((int(k[i]), float(t[i]), float(((j0[i] + a[i]) / (MN - 1)) * 2 * R - R)))
    return E
def edge_value(A, k2, s2):
    f_ = (s2 + R) / (2 * R) * (MN - 1); j0 = min(max(int(np.floor(f_)), 0), MN - 2); a = f_ - j0; return (1 - a) * A.V[k2, j0] + a * A.V[k2, j0 + 1]
def rollout_edges(A, k, s, tmax=40.):
    """Старт на отрезке споры k в точке s. Возвращает T, число дуг, wmax."""
    E = node_edges(A) if not hasattr(A, 'E') else A.E; A.E = E; y = A.C[k] + s * A.n[k]; T = 0.; arcs = 0; wmax = 0.
    while T < tmax:
        if ingoal(y[:, None])[0]: return T, arcs, wmax
        if FORCEPLAN: k = None                                                               # FORCEPLAN: мини-дерево из точки на КАЖДОМ шаге (research-10: р.10 T 5.376 = ×1.054 OCP)
        if k is None:                                                                        # вне отрезка: прямое мини-дерево из точки (plan), первая дуга целиком
            J, (tt, u1, u2) = plan(A, y)
            if J >= BIG / 2 or tt <= 0: return np.inf, arcs, wmax
            nst = max(6, int(tt / .01))
            for _ in range(nst):
                y = flow(y[:, None], np.array([u1]), np.array([u2]), tt / nst, n=1)[:, 0]; wmax = max(wmax, np.abs(y[2:]).max())
                if ingoal(y[:, None])[0]: return T + tt * (_ + 1) / nst, arcs + 1, wmax
            T += tt; arcs += 1; y[:2] = wrap(y[:2]); A.nplan = getattr(A, 'nplan', 0) + 1; continue
        f_ = (s + R) / (2 * R) * (MN - 1); js = sorted({min(max(int(np.floor(f_)), 0), MN - 1), min(max(int(np.ceil(f_)), 0), MN - 1)}); best = None
        if ALLN: js = list(np.argsort(A.V[k]))                                               # ALLN: рёбра всех узлов споры по возрастанию V узла
        for j in js:
            for k2, t0, s20 in E.get(k * MN + j, []):
                P = A.C[k] + A.sn[j] * A.n[k]
                tt, u1, u2, s2, ok = solve_arcs(P[:, None], A.C[k2][:, None], A.n[k2][:, None])
                if not ok[0]: continue
                r = newton_warm(y, A.C[k2], A.n[k2], tt[0], u1[0], u2[0], s2[0])
                if r is None: continue
                v = r[0] + edge_value(A, k2, r[3])
                if v < BIG / 2 and (best is None or v < best[0]): best = (v, k2, r)
                break                                                                    # первое сошедшееся ребро узла (они по возрастанию цены)
        if best is None and ALLN:                                                            # запас: прямые пары из точки (Ньютон из формулы)
            iy, kk, sv, t_ = A.pairs(y[None])
            for kk_, sv_ in zip(kk.astype(int), sv):
                tt, u1, u2, s2, ok = solve_arcs(y[:, None], A.C[kk_][:, None], A.n[kk_][:, None])
                if ok[0] and abs(s2[0]) <= R:
                    v = tt[0] + edge_value(A, kk_, s2[0])
                    if v < BIG / 2 and (best is None or v < best[0]): best = (v, kk_, (tt[0], u1[0], u2[0], s2[0]))
        if best is None:
            if not PLANFB: return np.inf, arcs, wmax
            k = None; continue
        _, k2, (tt, u1, u2, s2) = best; nst = max(6, int(tt / .01)); yy = y.copy()
        for _ in range(nst):
            yy = flow(yy[:, None], np.array([u1]), np.array([u2]), tt / nst, n=1)[:, 0]; wmax = max(wmax, np.abs(yy[2:]).max())
            if ingoal(yy[:, None])[0]: return T + tt * (_ + 1) / nst, arcs + 1, wmax
        T += tt; arcs += 1; k, s = k2, s2; y = A.C[k] + s * A.n[k]
    return np.inf, arcs, wmax

MARG, NEW, ROUNDS, DSH = float(os.environ.get('MARG', .1)), int(os.environ.get('NEW', 1000)), int(os.environ.get('ROUNDS', 3)), float(os.environ.get('DSH', .85))
UMAX, KS = .9, 81
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
def grow_round(A, rng, dmin, Tb):
    g = g_from_start(A).min(1); V = A.V.min(1); ell = np.flatnonzero(g + V <= (1 + MARG) * Tb); K0 = A.K; C, n = A.C, A.n; added = 0; tries = 0
    while added < NEW and tries < 400:
        tries += 1; b = 400; B = ell[rng.integers(len(ell), size=b)]; sg = np.where(rng.random(b) < .5, 1., -1.)
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
    return len(ell), A.K - K0, dmin
def _grow_main(N):
    t0 = time.time(); A = GrowAtlas(N); A.build(); tp = time.time() - t0; A.solve()
    print(json.dumps(dict(N=N, K=A.K, pairs=len(A.e[0]), pairs_per_node=round(len(A.e[0]) / (A.K * MN), 1), sec_pairs=round(tp), iters=A.n_it, BIG=round(float((A.V >= BIG / 2).mean()), 3), sec=round(time.time() - t0))), flush=True)
    out = 'butterfly_dp_grow_N%d_tr%d_fr%d_fwd%d_R%g%s%s.npz' % (N, TR, FR, FWD, R, '' if G == 1 else '_g%g' % G, '_rrt' if RRT else ''); np.savez(out, C=A.C, n=A.n, V=A.V, e=np.array(A.e, dtype=object), allow_pickle=True)
    T, arcs, wm = rollout_edges(A, 81, 0.)
    print(json.dumps(dict(query='висит→вверх (рёбра)', V=round(float(A.V[81, MN // 2]), 3), T=round(float(T), 3), T_over_OCP=round(float(T / 5.098), 4), arcs=arcs, wmax=round(float(wm), 2), sec=round(time.time() - t0)), ensure_ascii=False), flush=True)
def _ellipse_main(path):
    A = load(path); rng = np.random.default_rng(1); t0 = time.time(); dmin = float(os.environ.get('DMIN0', .2)); refresh(A); A.V[A.ingoal] = 0.
    A.solve(); Tb = A.V[KS, MN // 2]; T, arcs, wm = rollout_edges(A, KS, 0.)
    print(json.dumps(dict(round=0, spores=A.K, V=round(float(Tb), 3), T=round(float(T), 3), T_over_OCP=round(float(T / 5.098), 4), arcs=arcs)), flush=True)
    for r in range(1, ROUNDS + 1):
        ne, na, dmin = grow_round(A, rng, dmin, Tb); dmin *= DSH; A.solve(); Tb = min(Tb, A.V[KS, MN // 2]); A.E = node_edges(A); T, arcs, wm = rollout_edges(A, KS, 0.)
        print(json.dumps(dict(round=r, ellipse=ne, added=na, spores=A.K, pairs_per_node=round(len(A.e[0]) / (A.K * MN), 2), dmin=round(dmin, 3), V=round(float(A.V[KS, MN // 2]), 3), T=round(float(T), 3), T_over_OCP=round(float(T / 5.098), 4), arcs=arcs, sec=round(time.time() - t0))), flush=True)
        np.savez(path.replace('.npz', '_ell%d.npz' % r), C=A.C, n=A.n, V=A.V, e=np.array(A.e, dtype=object), allow_pickle=True)
def lazy_nodes(A, rounds=4, cap=4000, ocp=7.636, path=None):
    """research-11 (PLAN п.7б): g=2 — «ленивые узлы». Приход ребра между узлами, один из которых BIG, становится новой спорой (центр в точке прихода);
    ребро «источник → центр новой споры» точное (a=0). Раунды, пока старт не связан; дальше только раздувают атлас."""
    refresh(A); A.V[A.ingoal] = 0.; A.solve(); t0 = time.time()
    print(json.dumps(dict(round=0, spores=int(A.K), V=round(float(A.V[KS, MN // 2]), 3), centers_fin=int((A.V[:, MN // 2] < BIG / 2).sum()))), flush=True)
    for r in range(1, rounds + 1):
        iy, k, j0, a, t = A.e; V = A.V.reshape(-1); V0, V1 = V[k * MN + j0], V[k * MN + j0 + 1]; a_ = snap(a)
        bad = ((V0 >= BIG / 2) & (a_ < 1 - SNAPA)) | ((V1 >= BIG / 2) & (a_ > SNAPA)); fin = (A.V < BIG / 2).any(1)
        cand = np.flatnonzero((V[iy] >= BIG / 2) & bad & fin[k]); sv = A.sn[j0[cand]] + a[cand] * (2 * R / (MN - 1))
        _, u = np.unique(np.c_[k[cand], np.round(sv / .01)], axis=0, return_index=True)         # одна новая спора на (спора, s с шагом .01)
        if len(u) > cap: u = np.random.default_rng(r).choice(u, cap, replace=False)
        if not len(u): print(json.dumps(dict(round=r, new=0))); break
        kk, ss = k[cand][u], sv[u]; K0 = A.K; newC = A.C[kk] + ss[:, None] * A.n[kk]; lab = {(int(x), int(round(y / .01))): K0 + i for i, (x, y) in enumerate(zip(kk, ss))}
        tgt = np.array([lab.get((int(x), int(round(y / .01))), -1) for x, y in zip(k[cand], sv)]); m = tgt >= 0
        A.C = np.r_[A.C, newC]; A.n = np.r_[A.n, A.n[kk]]; A.K = len(A.C); A.V = np.r_[A.V, np.full((A.K - K0, MN), BIG)]; refresh(A); A.V[A.ingoal] = 0.
        out = A.pairs(A.C[K0:], np.arange(K0, A.K)); e_new = edges_add(A, K0 * MN + MN // 2 + MN * np.arange(A.K - K0), out)
        e_src = [iy[cand][m], tgt[m], np.full(m.sum(), MN // 2), np.zeros(m.sum()), t[cand][m]]
        e = [np.r_[x, y, z] for x, y, z in zip(A.e, e_new, e_src)]; o = np.argsort(e[0], kind='stable'); A.e = [x[o] for x in e]; A.solve()
        print(json.dumps(dict(round=r, cand=int(len(cand)), new=int(A.K - K0), new_fin=int((A.V[K0:, MN // 2] < BIG / 2).sum()), spores=int(A.K), V=round(float(A.V[KS, MN // 2]), 3),
                              centers_fin=int((A.V[:, MN // 2] < BIG / 2).sum()), sec=round(time.time() - t0))), flush=True)
        if path: np.savez(path.replace('.npz', '_lazy%d.npz' % r), C=A.C, n=A.n, V=A.V, e=np.array(A.e, dtype=object), allow_pickle=True)
        if A.V[KS, MN // 2] < BIG / 2 and r >= int(os.environ.get('LAZYMIN', 1)): break
    if A.V[KS, MN // 2] < BIG / 2:
        A.E = node_edges(A); T, arcs, wm = rollout_edges(A, KS, 0.)
        print(json.dumps(dict(V=round(float(A.V[KS, MN // 2]), 3), T=round(float(T), 3), T_over_OCP=round(float(T / ocp), 4), arcs=arcs, wmax=round(float(wm), 2))), flush=True)
    return A
def _lazy_main(path):
    A = load(path); lazy_nodes(A, int(os.environ.get('ROUNDS', 8)), int(os.environ.get('CAP', 4000)), float(os.environ.get('OCP', 7.636)), path)
if __name__ == '__main__':
    if sys.argv[1] == 'grow': _grow_main(int(sys.argv[2]))
    elif sys.argv[1] == 'lazy': _lazy_main(sys.argv[2])
    else: _ellipse_main(sys.argv[2])
