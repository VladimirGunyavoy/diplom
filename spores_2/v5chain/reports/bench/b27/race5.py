"""b27: ГОНКА «Картленд» граф-схемой (race_track.md). 5D x,y,θ,v,κ; 9 слоёв = руль по скорости (s_κ∈{-1,0,1}) × педаль-доля сцепления (s_a∈{-1,0,1}).
Схема как у car_val6 (без подбора): обратное дерево v6 (SysN, θ периодический) от финишного отрезка; V=время до финиша итерацией ценностей по графу узлов
(Ŵ(q)=min_K(W_b+KV·|q-b|_scaled)); запрещённые рёбра: sdf<MARG, срыв v²|κ|>SLIPM·μg, v∉[0,vmax]; агент жадный по τ+Ŵ + фильтр безопасности (зазор по пути шага ≥ FM, опц. торможение HB).
Единицы СИ: 1 пкс = .2 м, x = столбец·.2, y = строка·.2 (изображение, y вниз; ход по часовой, финиш — отрезок строка 240, столбцы 410..438, ход вниз = +y).
Запуск: runpc.sh race5.py NB K mode(sel|test) TREESEED [T1]; ENV: KV, RB, TAU, FM, MARG, HB, SLIPM, SC, NST (число стартов), SDF=путь к kartland_sdf_px.npz.
Эталон T_ref: OCP (CasADi) на хабе/aida не установлен ⇒ T_ref = минимум по всем нашим прогонам на старт (агрегатор agg_race.py) — ПОМЕТКА в таблице."""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'v6'))
import numpy as np, numba as nb
from src.atlas6.adaptive_nd import SysN, build_back
NB, KN, MODE, TS = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
KV = float(os.environ.get('KV', .3)); RB = float(os.environ.get('RB', .3)); TAU = float(os.environ.get('TAU', .15)); FM = float(os.environ.get('FM', .1)); MARG = float(os.environ.get('MARG', .3))
HB = float(os.environ.get('HB', 0)); SLIPM = float(os.environ.get('SLIPM', 1.0)); NST = int(os.environ.get('NST', 100)); NSUB = 10
SC = tuple(float(v) for v in os.environ.get('SC', '2.,2.,.4,2.,.08').split(','))
PX = .2; MUG, VMAX, AMAX, BMAX, KMAX = 12., 15., 4., 8., 1. / 3.; KD = 2 * KMAX / .5
_d = np.load(os.environ.get('SDF', os.path.join(HERE, 'kartland_sdf_px.npz'))); SDF = (_d['sdf'] * PX).astype(np.float64); NR, NC = SDF.shape
YF = 240 * PX; X0, X1 = 410 * PX, 438 * PX
ACT = np.array([[sk, sa] for sk in (-1., 0., 1.) for sa in (-1., 0., 1.)])

@nb.njit(inline='always')
def sdfp(x, y):
    c = x / PX; r = y / PX
    if c < 0 or r < 0 or c > NC - 1.001 or r > NR - 1.001: return -10.
    c0 = int(c); r0 = int(r); fc = c - c0; fr = r - r0
    return (SDF[r0, c0] * (1 - fc) + SDF[r0, c0 + 1] * fc) * (1 - fr) + (SDF[r0 + 1, c0] * (1 - fc) + SDF[r0 + 1, c0 + 1] * fc) * fr
@nb.njit(inline='always')
def _f(x, y, th, v, k, sk, sa):
    ad = np.sqrt(max(0., MUG * MUG - (v * v * k) ** 2))
    if sa > 0: a = min(ad, AMAX * (1. - v / VMAX))
    elif sa < 0: a = -min(ad, BMAX)
    else: a = 0.
    if v <= 0 and a < 0: a = 0.
    dk = KD * sk
    if (k >= KMAX and sk > 0) or (k <= -KMAX and sk < 0): dk = 0.
    return v * np.cos(th), v * np.sin(th), v * k, a, dk
@nb.njit
def rk(x, y, th, v, k, sk, sa, h, n):
    for _ in range(n):
        a1 = _f(x, y, th, v, k, sk, sa); a2 = _f(x + h / 2 * a1[0], y + h / 2 * a1[1], th + h / 2 * a1[2], v + h / 2 * a1[3], k + h / 2 * a1[4], sk, sa)
        a3 = _f(x + h / 2 * a2[0], y + h / 2 * a2[1], th + h / 2 * a2[2], v + h / 2 * a2[3], k + h / 2 * a2[4], sk, sa); a4 = _f(x + h * a3[0], y + h * a3[1], th + h * a3[2], v + h * a3[3], k + h * a3[4], sk, sa)
        x += h / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0]); y += h / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1]); th += h / 6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])
        v += h / 6 * (a1[3] + 2 * a2[3] + 2 * a3[3] + a4[3]); k += h / 6 * (a1[4] + 2 * a2[4] + 2 * a3[4] + a4[4])
        if k > KMAX: k = KMAX
        if k < -KMAX: k = -KMAX
    return x, y, th, v, k
@nb.njit(parallel=True)
def flowb(P, sk, sa, t):
    out = np.empty_like(P); n = max(1, int(np.ceil(abs(t) / .02)))
    for i in nb.prange(P.shape[0]):
        r = rk(P[i, 0], P[i, 1], P[i, 2], P[i, 3], P[i, 4], sk, sa, t / n, n)
        for j in range(5): out[i, j] = r[j]
    return out
def flow(P, s, t, dt_max=None):
    P = np.asarray(P, float); u = ACT[s]; return flowb(P.reshape(-1, 5).copy(), u[0], u[1], float(t)).reshape(P.shape)
@nb.njit(parallel=True)
def sdfv(X, Y):
    o = np.empty(X.shape[0])
    for i in nb.prange(X.shape[0]): o[i] = sdfp(X[i], Y[i])
    return o
def blocked(P):
    P = np.asarray(P); sh = P.shape[:-1]; Q = P.reshape(-1, 5); return (sdfv(np.ascontiguousarray(Q[:, 0]), np.ascontiguousarray(Q[:, 1])) < MARG).reshape(sh)
def okv(P):
    P = np.asarray(P); return (P[..., 3] >= 0) & (P[..., 3] <= VMAX) & (np.abs(P[..., 4]) <= KMAX) & (P[..., 3] ** 2 * np.abs(P[..., 4]) <= SLIPM * MUG)
def in_goal(P):
    P = np.asarray(P); return (P[..., 1] >= YF - .6) & (P[..., 1] < YF) & (P[..., 0] >= X0) & (P[..., 0] <= X1) & (np.sin(P[..., 2]) > .3)
def mkS(seed):
    rng = np.random.default_rng(seed); n = 600
    seeds = np.c_[rng.uniform(X0, X1, n), rng.uniform(YF - .6, YF, n), np.pi / 2 + rng.uniform(-.6, .6, n), rng.uniform(0, VMAX, n), rng.uniform(-KMAX, KMAX, n)]
    seeds = seeds[okv(seeds) & ~blocked(seeds) & in_goal(seeds)]
    return SysN(flow, 9, SC, in_goal, seeds, per=(0, 0, 2 * np.pi, 0, 0), ok=okv, blocked=blocked)

@nb.njit
def step_chk(x, y, th, v, k, s, tau, n):
    """шаг слоя s за tau (n подшагов): (состояние x5, мин. sdf, макс. v²|κ|/μg, время пересечения финиша или -1)"""
    h = tau / n; mn = 9.; sl = 0.; hit = -1.
    for j in range(n):
        yo = y; xo = x; x, y, th, v, k = rk(x, y, th, v, k, ACT[s, 0], ACT[s, 1], h, 1)
        mn = min(mn, sdfp(x, y)); sl = max(sl, v * v * abs(k) / MUG)
        if hit < 0 and yo < YF <= y:
            f = (YF - yo) / (y - yo); xc = xo + f * (x - xo)
            if X0 <= xc <= X1: hit = (j + f) * h
    return x, y, th, v, k, mn, sl, hit
@nb.njit(parallel=True)
def edges(P, tau):
    n = P.shape[0]; E = np.zeros((n, 9, 5)); OK = np.zeros((n, 9), np.bool_); GL = np.full((n, 9), -1.)
    for i in nb.prange(n):
        for s in range(9):
            r = step_chk(P[i, 0], P[i, 1], P[i, 2], P[i, 3], P[i, 4], s, tau, NSUB)
            for j in range(5): E[i, s, j] = r[j]
            OK[i, s] = (r[5] >= MARG) and (r[6] <= SLIPM) and r[3] >= 0 and r[3] <= VMAX
            GL[i, s] = r[7]
    return E, OK, GL
@nb.njit(parallel=True)
def vit(J, D, OK, GL, ING, nP, tau, kv, nit, big):
    W = np.full(nP, big); K_ = J.shape[1]
    for i in range(nP):
        if ING[i]: W[i] = 0.
    for it in range(nit):
        Wn = np.empty(nP)
        for i in nb.prange(nP):
            if ING[i]: Wn[i] = 0.; continue
            best = big
            for a in range(9):
                if not OK[i, a]: continue
                if GL[i, a] >= 0: c = GL[i, a]
                else:
                    m = 1e30
                    for k in range(K_):
                        v = W[J[i * 9 + a, k]] + kv * D[i * 9 + a, k]
                        if v < m: m = v
                    c = tau + m
                if c < best: best = c
            Wn[i] = best
        d = np.max(np.abs(Wn - W)); W = Wn
        if d < 1e-4: break
    return W, it, d

class Agent:
    def __init__(self, S, back):
        self.S, self.tree, self.P = S, back[5], np.asarray(back[0]); self.nP = len(self.P)
        t0 = time.perf_counter(); E, OK, GL = edges(self.P, TAU); t1 = time.perf_counter()
        Ew = S.wrap(E.reshape(-1, 5)); d, j = self.tree.query(Ew / S.scale, k=min(KN * 3, len(self.tree.data)), workers=-1)
        J = (j % self.nP).astype(np.int32); D = d.astype(np.float64); t2 = time.perf_counter()
        self.W, it, dres = vit(J, D, OK, GL, in_goal(self.P), self.nP, TAU, KV, int(os.environ.get('NIT', 400)), 1e3); t3 = time.perf_counter(); self.tvi = t3 - t0
        fin = self.W < 1e2
        print('  узлов %d: edges %.1f, knn %.1f, vi %.1f с (it %d res %.1e); W конечных %.1f%%, мед %.2f макс %.2f; безопасных рёбер/узел %.2f' % (self.nP, t1 - t0, t2 - t1, t3 - t2, it, dres, 100 * fin.mean(), np.median(self.W[fin]), self.W[fin].max(), OK.sum(1).mean()), flush=True)
    def act(self, x0):
        x0 = np.asarray(x0, float); Q = np.zeros((9, 5)); cl = np.zeros(9); sc = np.full(9, 1e9); okm = np.zeros(9, bool)
        for s in range(9):
            r = step_chk(x0[0], x0[1], x0[2], x0[3], x0[4], s, TAU, NSUB); Q[s] = r[:5]; mn = r[5]; sl = r[6]
            if HB > 0 and r[7] < 0:   # восстановимость: после шага тормоз + руль в 0 на HB с
                bx = np.array(r[:5])
                for _ in range(int(HB / .05)):
                    sk = -1. if bx[4] > 0 else 1.; res = rk(bx[0], bx[1], bx[2], bx[3], bx[4], sk, -1., .05, 1); bx = np.array(res); mn = min(mn, sdfp(bx[0], bx[1])); sl = max(sl, bx[3] ** 2 * abs(bx[4]) / MUG)
            cl[s] = mn - 2 * max(0., sl - SLIPM); okm[s] = (mn >= FM) and (sl <= SLIPM) and Q[s, 3] >= 0 and Q[s, 3] <= VMAX
            if r[7] >= 0 and okm[s]: sc[s] = r[7]
        if (sc < 1e9).any(): return int(np.argmin(sc))
        if not okm.any(): return int(np.argmax(cl))
        d, j = self.tree.query(self.S.wrap(Q) / self.S.scale, k=min(KN * 3, len(self.tree.data))); j = j % self.nP
        Wq = np.min(self.W[j] + KV * d, axis=1); return int(np.argmin(np.where(okm, Wq, 1e9)))

@nb.njit
def execute(x, y, th, v, k, s, dt, n):
    """шаг агента TAU, подшаг dt: (состояние x5, мин. sdf, макс. v²|κ|/μg, время входа в финиш или -1)"""
    mn = 9.; sl = 0.; hit = -1.
    for j in range(n):
        yo = y; xo = x; x, y, th, v, k = rk(x, y, th, v, k, ACT[s, 0], ACT[s, 1], dt, 1); mn = min(mn, sdfp(x, y)); sl = max(sl, v * v * abs(k) / MUG)
        if hit < 0 and yo < YF <= y:
            f = (YF - yo) / (y - yo); xc = xo + f * (x - xo)
            if X0 <= xc <= X1: hit = (j + f) * dt; break
    return x, y, th, v, k, mn, sl, hit
def episode(ag, x0, tmax=60.):
    x = np.array(x0, float); t = 0.; mn = 9.; sl = 0.; tms = []
    while t < tmax:
        t1 = time.perf_counter(); s = ag.act(x); tms.append(time.perf_counter() - t1)
        r = execute(x[0], x[1], x[2], x[3], x[4], s, .005, int(round(TAU / .005))); x = np.array(r[:5]); mn = min(mn, r[5]); sl = max(sl, r[6])
        if r[7] >= 0: return t + r[7], mn, sl, float(np.mean(tms))
        if mn < -.5: break
        t += TAU
    return None, mn, sl, float(np.mean(tms))

def get_starts():
    """старты по длине оси: гео-расстояние до финиша (Дейкстра по сетке, финиш — стена + источники строка 239), курс = −∇D ± .35 рад, v∈[0,vmax), κ=0; последний — «круг» с ходу"""
    f = os.path.join(HERE, 'starts_race.npy')
    if os.path.exists(f): return np.load(f)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    import scipy.ndimage as ndi
    free = SDF > 0; c0, c1 = 410, 438; free[240:243, c0:c1 + 1] = False
    idx = -np.ones(free.shape, int); idx[free] = np.arange(free.sum()); rr, cc = np.nonzero(free); I = []; J = []; Wt = []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r2, c2 = rr + dr, cc + dc; m = (r2 < NR) & (c2 >= 0) & (c2 < NC); m[m] = free[r2[m], c2[m]]
        I += list(idx[rr[m], cc[m]]); J += list(idx[r2[m], c2[m]]); Wt += [np.hypot(dr, dc)] * int(m.sum())
    n = int(free.sum()); G = coo_matrix((Wt + Wt, (I + J, J + I)), shape=(n, n)).tocsr()
    src = [idx[239, c] for c in range(c0, c1 + 1) if free[239, c]]; Dg = dijkstra(G, indices=src, min_only=True)
    D = np.full(free.shape, np.nan); D[free] = Dg; ii = ndi.distance_transform_edt(~free, return_distances=False, return_indices=True); Df = np.where(free, D, D[ii[0], ii[1]]); Ds = ndi.gaussian_filter(Df, 2)
    gy, gx = np.gradient(Ds); rng = np.random.default_rng(5000); out = []
    ok = np.argwhere(free & (SDF > 1.5) & np.isfinite(D)); q = np.sort(D[ok[:, 0], ok[:, 1]]); tg = np.linspace(q[20], q[-20], NST - 1)
    for t in tg:
        cand = ok[np.abs(D[ok[:, 0], ok[:, 1]] - t) < 3]; r, c = cand[rng.integers(len(cand))]
        th = (np.pi / 2 if (233 <= r <= 247 and c0 - 3 <= c <= c1 + 3) else np.arctan2(-gy[r, c], -gx[r, c])) + rng.uniform(-.35, .35); out.append([c * PX, r * PX, th, rng.uniform(0, float(os.environ.get('VST', 8.))), 0., D[r, c] * PX])
    out.append([(c0 + c1) / 2 * PX, 243 * PX, np.pi / 2, 5., 0., float(np.nanmax(D)) * PX])   # старт «круга»: сразу за финишем
    out = np.array(out); np.save(f, out); return out

if __name__ == '__main__':
    st = get_starts(); out = {}
    for ts in range(TS, int(sys.argv[5]) if len(sys.argv) > 5 else TS + 1):
        S = mkS(ts); t0 = time.perf_counter(); back = build_back(S, TAU, NB, RB); tb = time.perf_counter() - t0; print('ts %d: build %.1f с, узлов %d' % (ts, tb, len(back[0])), flush=True)
        ag = Agent(S, back); res = []
        for i, s0 in enumerate(st):
            te = time.perf_counter(); T, mn, sl, ms = episode(ag, s0[:5]); res.append((i, T, mn, sl, ms, time.perf_counter() - te))
        ok = [r for r in res if r[1] is not None]; viol = sum((r[2] < 0 or r[3] > 1.0 + 1e-9) for r in res)
        print('NB=%d K=%d KV=%g RB=%g TAU=%g MARG=%g HB=%g ts=%d: дошли %d/%d, НАРУШЕНИЙ %d (выезд %d, срыв %d), мин. запас до границы %.3f м, T мед %.2f (круг %s), мс/шаг %.1f, build %.1f с, vi %.1f с' % (
            NB, KN, KV, RB, TAU, MARG, HB, ts, len(ok), len(res), viol, sum(r[2] < 0 for r in res), sum(r[3] > 1 + 1e-9 for r in res), min(r[2] for r in res), np.median([r[1] for r in ok]) if ok else -1, ('%.2f' % res[-1][1]) if res[-1][1] else 'нет',
            1e3 * np.mean([r[4] for r in res]), tb, ag.tvi), flush=True)
        out[ts] = dict(res=res, build=tb, nodes=len(back[0]))
    json.dump(out, open('race_%s_%d_%g_%g_%d.json' % (MODE, NB, KV, RB, TS), 'w'))
