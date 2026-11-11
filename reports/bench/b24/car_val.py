"""b24 СТАТИК-КАР, рецепт круга: обратное дерево v6 (SysN, θ периодический) на 4D (x,y,θ,v); V=время-до-цели итерацией ценностей по графу узлов
(W_i=min_a[τ+Ŵ(flow(i,a,τ))], Ŵ(q)=min_{K ближайших b}(W_b+KV·|q-b|_scaled), диски/стены/|v| — запрещённые рёбра); агент жадный по τ+Ŵ + фильтр безопасности
(зазор до дисков по пути шага ≥ FM). Модель — ref_car100 (K .625, a±1.5, ω±1.8, v∈[-1,2], 3 диска r.7, цель (4,3.6) r.45, стены ±5).
Запуск (aida, рядом с b22/): runpc.sh car_val.py NB K mode(sel|test) TREESEED [T1]; ENV: KV (1.0), RB (.3), TAU (.2), FM (.1), MARG (.12), SC (.25,.25,.4,.3)
sel: старты seed 2000 (первые NSEL=32), эталон b24/ref*/car_q<i>.npy; test: starts100.npy (seed 1000), T_ref из r20 json (TREF=путь)."""
import os, sys, time, json, glob
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'v6'))
import numpy as np, numba as nb
from src.atlas6.adaptive_nd import SysN, build_back
NB, KN, MODE, TS = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
KV = float(os.environ.get('KV', 1.0)); RB = float(os.environ.get('RB', .3)); TAU = float(os.environ.get('TAU', .2)); FM = float(os.environ.get('FM', .1)); MARG = float(os.environ.get('MARG', .12)); HB = float(os.environ.get('HB', 1.0))
SC = tuple(float(v) for v in os.environ.get('SC', '.25,.25,.4,.3').split(','))
K, AM, WM, VL, VH, XL = .625, 1.5, 1.8, -1., 2., 5.
HZ = np.array([[0., .2], [1.8, 1.5], [-1.6, 2.3]]); HR = .70; GOAL = np.array([4., 3.6]); GR = .45
ACT = np.array([[a, w] for a in (-AM, 0., AM) for w in (-WM, 0., WM)])

@nb.njit(inline='always')
def _f(x, y, th, v, a, w): return v * np.cos(th), v * np.sin(th), w, a - K * v
@nb.njit
def rk(x, y, th, v, a, w, h, n):
    for _ in range(n):
        k1 = _f(x, y, th, v, a, w); k2 = _f(x + h / 2 * k1[0], y + h / 2 * k1[1], th + h / 2 * k1[2], v + h / 2 * k1[3], a, w)
        k3 = _f(x + h / 2 * k2[0], y + h / 2 * k2[1], th + h / 2 * k2[2], v + h / 2 * k2[3], a, w); k4 = _f(x + h * k3[0], y + h * k3[1], th + h * k3[2], v + h * k3[3], a, w)
        x += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); y += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]); th += h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]); v += h / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
    return x, y, th, v
@nb.njit
def clear(x, y):
    m = XL - max(abs(x), abs(y))
    for i in range(3): m = min(m, np.sqrt((x - HZ[i, 0]) ** 2 + (y - HZ[i, 1]) ** 2) - HR)
    return m
@nb.njit(parallel=True)
def flowb(P, a, w, t):
    out = np.empty_like(P); n = max(1, int(np.ceil(abs(t) / .02)))
    for i in nb.prange(P.shape[0]):
        x, y, th, v = rk(P[i, 0], P[i, 1], P[i, 2], P[i, 3], a, w, t / n, n); out[i, 0] = x; out[i, 1] = y; out[i, 2] = th; out[i, 3] = v
    return out
def flow(P, s, t, dt_max=None):
    P = np.asarray(P, float); u = ACT[s]
    return flowb(P.reshape(-1, 4).copy(), u[0], u[1], float(t)).reshape(P.shape)
def in_goal(P): P = np.asarray(P); return np.hypot(P[..., 0] - GOAL[0], P[..., 1] - GOAL[1]) <= GR - .05
def blocked(P):
    P = np.asarray(P); m = XL - np.maximum(np.abs(P[..., 0]), np.abs(P[..., 1]))
    for c in HZ: m = np.minimum(m, np.hypot(P[..., 0] - c[0], P[..., 1] - c[1]) - HR)
    return m < MARG
def okv(P): P = np.asarray(P); return (P[..., 3] >= VL) & (P[..., 3] <= VH)
def mkS(seed):
    rng = np.random.default_rng(seed); n = 300; r = .35 * np.sqrt(rng.uniform(size=n)); a = rng.uniform(0, 2 * np.pi, n)
    seeds = np.c_[GOAL[0] + r * np.cos(a), GOAL[1] + r * np.sin(a), rng.uniform(-np.pi, np.pi, n), rng.uniform(VL, VH, n)]
    return SysN(flow, 9, SC, in_goal, seeds, per=(0, 0, 2 * np.pi, 0), ok=okv, blocked=blocked)

@nb.njit(parallel=True)
def edges(P, tau):
    n = P.shape[0]; E = np.zeros((n, 9, 4)); OK = np.zeros((n, 9), np.bool_); GL = np.zeros((n, 9), np.bool_)
    for i in nb.prange(n):
        for s in range(9):
            x, y, th, v = P[i, 0], P[i, 1], P[i, 2], P[i, 3]; ok = True; h = tau / 10
            for _ in range(10):
                x, y, th, v = rk(x, y, th, v, ACT[s, 0], ACT[s, 1], h, 1)
                if clear(x, y) < MARG or v < VL or v > VH: ok = False
                if (x - GOAL[0]) ** 2 + (y - GOAL[1]) ** 2 <= (GR - .05) ** 2: GL[i, s] = True; break
            E[i, s, 0] = x; E[i, s, 1] = y; E[i, s, 2] = th; E[i, s, 3] = v; OK[i, s] = ok
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
                if GL[i, a]: c = tau
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
        from scipy.spatial import cKDTree
        self.S, self.tree, self.P = S, back[5], np.asarray(back[0]); self.nP = len(self.P)
        t0 = time.perf_counter(); E, OK, GL = edges(self.P, TAU); t1 = time.perf_counter()
        Ew = S.wrap(E.reshape(-1, 4)); d, j = self.tree.query(Ew / S.scale, k=min(KN * 3, len(self.tree.data)), workers=-1)
        J = (j % self.nP).astype(np.int32); D = d.astype(np.float64); t2 = time.perf_counter()
        self.W, it, dres = vit(J, D, OK, GL, in_goal(self.P), self.nP, TAU, KV, int(os.environ.get('NIT', 300)), 1e3); t3 = time.perf_counter(); self.tvi = t3 - t2 + (t2 - t1) + (t1 - t0)
        fin = self.W < 1e2
        print('  узлов %d: edges %.1f, knn %.1f, vi %.1f с (it %d res %.1e); W конечных %.1f%%, мед %.2f макс %.2f; G дерева мед %.2f' % (self.nP, t1 - t0, t2 - t1, t3 - t2, it, dres, 100 * fin.mean(), np.median(self.W[fin]), self.W[fin].max(), np.median(back[1])), flush=True)
    def act(self, x0):
        best, bs, mxc, ms = 1e9, -1, -1e9, 0
        Q = np.zeros((9, 4)); sc = np.full(9, 1e9); cl = np.zeros(9)
        for s in range(9):
            x, y, th, v = x0; mn = 9.; hit = None; h = TAU / 10
            for k in range(10):
                x, y, th, v = rk(x, y, th, v, ACT[s, 0], ACT[s, 1], h, 1); mn = min(mn, clear(x, y))
                if hit is None and np.hypot(x - GOAL[0], y - GOAL[1]) <= GR - .05: hit = (k + 1) * h
            if HB > 0:   # торможение после шага (a=-1.5 при v>0, w=0): зазор по пути остановки
                bx, by, bth, bv = x, y, th, v
                for _ in range(int(HB / .05)):
                    bx, by, bth, bv = rk(bx, by, bth, bv, -AM if bv > 0 else AM, 0., .05, 1); mn = min(mn, clear(bx, by) + (FM - FB0 if False else 0))
            cl[s] = mn; Q[s] = (x, y, th, v)
            if hit is not None and mn >= FM: sc[s] = hit
        ok = (cl >= FM) & (Q[:, 3] >= VL) & (Q[:, 3] <= VH)
        if (sc < 1e9).any(): return int(np.argmin(sc))
        if not ok.any(): return int(np.argmax(cl))
        d, j = self.tree.query(self.S.wrap(Q) / self.S.scale, k=min(KN * 3, len(self.tree.data))); j = j % self.nP
        Wq = np.min(self.W[j] + KV * d, axis=1)
        return int(np.argmin(np.where(ok, Wq, 1e9)))

@nb.njit
def execute(x, y, th, v, a, w, dt, n):
    """шаг агента .2 с, подшаг dt: (состояние, мин. зазор, время входа в цель или -1)"""
    mn = 9.; hit = -1.
    for k in range(n):
        x, y, th, v = rk(x, y, th, v, a, w, dt, 1); mn = min(mn, clear(x, y))
        if hit < 0 and (x - GOAL[0]) ** 2 + (y - GOAL[1]) ** 2 <= GR ** 2: hit = (k + 1) * dt; break
    return x, y, th, v, mn, hit
def episode(ag, x0, tmax=60.):
    x = np.array(x0, float); t = 0.; mn = 9.; tms = []
    while t < tmax:
        t1 = time.perf_counter(); s = ag.act(x); tms.append(time.perf_counter() - t1)
        xn, yn, thn, vn, m, hit = execute(x[0], x[1], x[2], x[3], ACT[s, 0], ACT[s, 1], .002, int(round(TAU / .002))); mn = min(mn, m); x = np.array([xn, yn, thn, vn])
        if hit >= 0: return t + hit, mn, float(np.mean(tms))
        t += TAU
    return None, mn, float(np.mean(tms))

def get_starts():
    sys.path.insert(0, HERE)
    import importlib.util
    spec = importlib.util.spec_from_file_location('rc', os.path.join(HERE, 'ref_car100.py')); rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
    if MODE == 'sel':
        S = rc.starts(100, 2000); tr = {}
        for f in glob.glob(os.path.join(HERE, 'b24', 'ref*_*', 'car_q*.npy')): i = int(os.path.basename(f)[5:-4]); tr[i] = min(tr.get(i, 1e9), float(np.load(f)[0]))
        ids = sorted(tr)[:int(os.environ.get('NSEL', 32))]; return [(i, S[i], tr[i]) for i in ids]
    A = np.load(os.path.join(HERE, 'starts100.npy')); S = A[:, :4]; tr = A[:, 4]   # колонка 5 = T_ref (r20)
    return [(i, S[i], tr[i]) for i in range(len(S)) if tr[i]]

if __name__ == '__main__':
    st = get_starts(); out = {}
    for ts in range(TS, int(sys.argv[5]) if len(sys.argv) > 5 else TS + 1):
        S = mkS(ts); t0 = time.perf_counter(); back = build_back(S, TAU, NB, RB); tb = time.perf_counter() - t0; print('ts %d: build %.1f с, узлов %d' % (ts, tb, len(back[0])), flush=True)
        ag = Agent(S, back); res = []
        for i, x0, tref in st:
            te = time.perf_counter(); T, mn, ms = episode(ag, x0); res.append((i, T, tref, mn, ms, time.perf_counter() - te))
        ok = [r for r in res if r[1] is not None]; ratio = np.array([r[1] / r[2] for r in ok]); touch = sum(r[3] < 0 for r in res)
        print('NB=%d K=%d KV=%g RB=%g [%s] ts=%d: дошли %d/%d, касаний %d, T/T_ref мед %.3f p90 %.3f max %.3f среднее %.3f, мс/шаг %.1f, зазор мин %.3f, мс/эпизод %.0f, build %.1f с, vi %.1f с (CPU numba)' % (
            NB, KN, KV, RB, MODE, ts, len(ok), len(res), touch, np.median(ratio), np.percentile(ratio, 90), ratio.max(), ratio.mean(), 1e3 * np.mean([r[4] for r in res]), min(r[3] for r in res), 1e3 * np.mean([r[5] for r in res]), tb, ag.tvi), flush=True)
        out[ts] = dict(res=res, build=tb, nodes=len(back[0]))
    json.dump(out, open('car_val_%s_%d_%g_%g_%d.json' % (MODE, NB, KV, RB, TS), 'w'))
