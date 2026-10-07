"""nD-часть проверки квазинормальных фронтов клетки (laptop-v5chain-worker-1, ТЗ research-1 qn_task.md).
Варианты: V0 плоский (как growN), V2 подгонка времени (МНК), V3(а) одноканальные нормальные кривые, V3(б) + все комбинации (±1,0)^m.
Метрики 1–6 из ТЗ. Код v8 не трогаем: `import growN` из spores_2/v8/src/algo, f патчится счётчиком вызовов.
Запуск (Windows-venv: нужен numpy, tqdm):  python nd_qn.py <SYS> <u1,u2,..> <seed через запятую> [r1,r2]  → дописывает nd_results.json
  python nd_qn.py dd 1,0 0,0,0        python nd_qn.py dd .5,.5 0.5,-0.3,0.7        python nd_qn.py manip 1,1 0,0,.3,-.2
"""
import os, sys, json, time, itertools, math
SYS = sys.argv[1]; U = tuple(float(x) for x in sys.argv[2].split(',')); P0 = [float(x) for x in sys.argv[3].split(',')]
RS = [float(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else [.2, .5]
os.environ['SYS'] = SYS
HERE = os.path.dirname(os.path.abspath(__file__)); V8 = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'v8'))
sys.path.insert(0, V8)
import numpy as np
from tqdm import tqdm
from src.algo import growN as g

CALLS = [0]; _f = g.f
def f_cnt(y, u):
    CALLS[0] += 1 if np.ndim(y) == 1 else int(np.prod(np.shape(y)[:-1])); return _f(y, u)
g.f = f_cnt                                          # rk4/jac/basis внутри growN берут глобальный f → счётчик
N = g.N; m = N - 1; u = U; p = np.array(P0, float)
H, K, STEPS, HS = .1, 11, 15, .05                    # шаг фронта, узлов на ось, шагов, подшаг rk4 (как growN.step: 2 подшага по .05)


def flow(x, T):
    """поток за время T (скаляр или массив по узлам (...,1)) rk4 подшагами ≤ HS"""
    T = np.asarray(T, float); n = max(1, int(math.ceil(float(np.abs(T).max()) / HS - 1e-12))); return g.rk4(x, u, T / n, n)


def proj_f(v, fx):
    fh = fx / np.linalg.norm(fx, axis=-1, keepdims=True); return v - (v * fh).sum(-1, keepdims=True) * fh


def transport(B, fx):
    """перенос базиса (m×n): проекция на f⊥ + Грам–Шмидт"""
    out = []
    for v in proj_f(B, fx):
        for e in out: v = v - (v @ e) * e
        out.append(v / np.linalg.norm(v))
    return np.array(out)


def cosf(t, fx):
    return np.abs((t * fx).sum(-1)) / (np.linalg.norm(t, axis=-1) * np.linalg.norm(fx, axis=-1))


# ---------- V0 / V2: сетка K^m ----------
def grid0(B, r):
    s = np.linspace(-r, r, K); mesh = np.stack(np.meshgrid(*([s] * m), indexing='ij'), -1); return p + mesh @ B


def grid_metrics(X, r):
    fx = g.f(X, u); cs = []
    for a in range(m):
        t = np.gradient(X, axis=a); cs.append(cosf(t, fx).ravel())
    cs = np.concatenate(cs); c = K // 2; w = []
    for a in range(m):
        idx = [c] * m; lo = list(idx); hi = list(idx); lo[a] = 0; hi[a] = K - 1; w.append(np.linalg.norm(X[tuple(hi)] - X[tuple(lo)]))
    return dict(cos_mean=float(cs.mean()), cos_max=float(cs.max()), width=float(np.mean(w) / (2 * r)))


def run_v0(B, r):
    X = grid0(B, r); CALLS[0] = 0; t0 = time.perf_counter()
    for k in range(STEPS): X = flow(X, H)
    ms = (time.perf_counter() - t0) / STEPS * 1e3; calls = CALLS[0] / STEPS
    d = grid_metrics(X, r); d.update(time_spread=0., f_per_front=calls, ms_per_front=ms, nodes=int(X.size // N)); return d


def run_v2(B, r):
    X = grid0(B, r); nn = X.size // N; shp = X.shape[:-1]; c = tuple([K // 2] * m); ci = int(np.ravel_multi_index(c, shp)); Tcum = np.zeros(nn)
    pairs = []; ids = np.arange(nn).reshape(shp)
    for a in range(m):
        sl0 = [slice(None)] * m; sl1 = [slice(None)] * m; sl0[a] = slice(0, K - 1); sl1[a] = slice(1, K)
        pairs.append(np.stack([ids[tuple(sl0)].ravel(), ids[tuple(sl1)].ravel()], 1))
    pairs = np.concatenate(pairs); res_hist = []; CALLS[0] = 0; t0 = time.perf_counter(); dl = np.zeros(nn)
    for k in range(STEPS):
        X = flow(X, H); xf = X.reshape(nn, N); fx = g.f(xf, u); i, j = pairs[:, 0], pairs[:, 1]
        nh = fx[i] + fx[j]; nh /= np.linalg.norm(nh, axis=1, keepdims=True)
        A = np.zeros((len(pairs), nn)); rr = np.arange(len(pairs)); A[rr, j] = (fx[j] * nh).sum(1); A[rr, i] = -(fx[i] * nh).sum(1)
        rhs = -((xf[j] - xf[i]) * nh).sum(1); keep = np.arange(nn) != ci
        sol = np.linalg.lstsq(A[:, keep], rhs, rcond=None)[0]; dl = np.zeros(nn); dl[keep] = sol
        res = rhs - A @ dl; res_hist.append(float(np.sqrt((res ** 2).mean())) / r)         # невязка МНК (в долях r) — неинтегрируемость в nD
        X = flow(xf, dl[:, None]).reshape(X.shape); Tcum += dl
    ms = (time.perf_counter() - t0) / STEPS * 1e3; calls = CALLS[0] / STEPS
    d = grid_metrics(X, r); d.update(time_spread=float(dl.max() - dl.min()), time_spread_cum=float(Tcum.max() - Tcum.min()), lsq_resid_last=res_hist[-1], lsq_resid_mean=float(np.mean(res_hist)),
                                     f_per_front=calls, ms_per_front=ms, nodes=int(nn)); return d


# ---------- V3: нормальные кривые ----------
def dirs_all():
    W = [w for w in itertools.product((-1, 0, 1), repeat=m) if any(w)]; return np.array(W, float)


def build_front(c, B, r, W, order=1):
    """узлы по кривым (nc, 51, n): d ← P_f(x) d, x ← x + ds·d (order=2 — средняя точка)"""
    ds = r / 50; W = W / np.linalg.norm(W, axis=1, keepdims=True); x = np.tile(c, (len(W), 1)); d = W @ B; d /= np.linalg.norm(d, axis=1, keepdims=True); pts = [x.copy()]
    for _ in range(50):
        d = proj_f(d, g.f(x, u)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        if order == 2:
            xm = x + ds / 2 * d; dm = proj_f(d, g.f(xm, u)); dm /= np.linalg.norm(dm, axis=1, keepdims=True); x = x + ds * dm; d = dm
        else: x = x + ds * d
        pts.append(x.copy())
    return np.stack(pts, 1)


def consistency(Fold, Fnew, r, ns=81):
    """расстояние (в долях r) от узла нового фронта до ближайшей траектории из узлов старого (t ∈ [0, 2h]) и время прихода"""
    dt = 2 * H / (ns - 1); traj = [Fold]; x = Fold
    for _ in range(ns - 1): x = g.rk4(x, u, dt, 1); traj.append(x)
    Pt = np.stack(traj, 1).reshape(-1, N); Tt = np.tile(np.arange(ns) * dt, len(Fold)); dist = np.empty(len(Fnew)); tau = np.empty(len(Fnew))
    for a in range(0, len(Fnew), 64):
        D = ((Fnew[a:a + 64, None, :] - Pt[None]) ** 2).sum(-1); jj = D.argmin(1)
        for q, j0 in enumerate(jj):                                            # проекция на ближайший из двух соседних отрезков траектории: время с долей шага
            x = Fnew[a + q]; best = (np.inf, Tt[j0])
            for j1 in (j0 - 1, j0 + 1):
                if 0 <= j1 < len(Pt) and Tt[j1] != 0 or (0 <= j1 < len(Pt) and abs(j1 - j0) == 1 and j0 // ns == j1 // ns):
                    if j0 // ns != j1 // ns: continue
                    A_, B_ = Pt[j0], Pt[j1]; ab = B_ - A_; fr = float(np.clip((x - A_) @ ab / (ab @ ab), 0, 1)); dd_ = np.linalg.norm(x - (A_ + fr * ab))
                    if dd_ < best[0]: best = (dd_, Tt[j0] + fr * (Tt[j1] - Tt[j0]))
            dist[a + q] = min(np.sqrt(D[q, j0]), best[0]) / r; tau[a + q] = best[1] if np.isfinite(best[0]) else Tt[j0]
    return dist, tau


def single_idx(W):
    return np.flatnonzero((W != 0).sum(1) == 1)


def run_v3(B0, r, W, label):
    c = p.copy(); B = B0.copy(); front_prev = None; CALLS[0] = 0; ms_tot = 0.; devs = []; last = None
    for k in range(1, STEPS + 1):
        t0 = time.perf_counter(); c = flow(c, H); B = transport(B, g.f(c, u)); cur = build_front(c, B, r, W); ms_tot += time.perf_counter() - t0
        nodes = np.concatenate([[c], cur[:, 5::5].reshape(-1, N)]);
        if k >= 2:
            c_ = CALLS[0]; t1 = time.perf_counter(); dist, tau = consistency(front_prev, nodes, r); last_cons = (dist, tau); CALLS[0] = c_     # проверка — не часть стоимости фронта
        front_prev = nodes
        # метрика 5: отклонение комбинированных узлов от аддитивной суперпозиции одноканальных кривых (линейная интерполяция по s)
        if label == 'V3b':
            si = single_idx(W); ds = r / 50; dev = []
            for ci in np.flatnonzero((W != 0).sum(1) > 1):
                w = W[ci] / np.linalg.norm(W[ci])
                for j in range(1, K):
                    s = j * 5 * ds; pred = c.copy()
                    for b in np.flatnonzero(w):
                        sg = np.sign(w[b]); row = [q for q in si if W[q][b] == sg][0]; sb = abs(w[b]) * s; t = sb / ds; i0 = int(min(math.floor(t), 49)); fr = t - i0
                        pred = pred + (cur[row, i0] * (1 - fr) + cur[row, i0 + 1] * fr - c)
                    dev.append(np.linalg.norm(cur[ci, j * 5] - pred) / r)
            devs.append((float(np.mean(dev)), float(np.max(dev))))
        last = (c.copy(), cur, nodes)
    calls = CALLS[0] / STEPS; c, cur, nodes = last
    # ортогональность: касательные по кривым (хорды), в узлах 5,10,..,50
    cs = []
    for q in range(len(W)):
        cv = cur[q]
        for i in range(5, 51, 5):
            t = cv[min(i + 1, 50)] - cv[i - 1]; cs.append(cosf(t, g.f(cv[i], u)))
    cs = np.array(cs); si = single_idx(W); wd = []
    for a in range(m):
        lo = [q for q in si if W[q][a] == -1][0]; hi = [q for q in si if W[q][a] == 1][0]; wd.append(np.linalg.norm(cur[hi, 50] - cur[lo, 50]))
    dist, tau = last_cons
    d = dict(cos_mean=float(cs.mean()), cos_max=float(cs.max()), width=float(np.mean(wd) / (2 * r)), time_spread=float(tau.max() - tau.min()), time_mean=float(tau.mean()),
             cons_dist_mean=float(dist.mean()), cons_dist_max=float(dist.max()), f_per_front=calls, ms_per_front=ms_tot / STEPS * 1e3, nodes=int(len(nodes)), curves=int(len(W)))
    if label == 'V3b': d.update(comb_dev_last_mean=devs[-1][0], comb_dev_last_max=devs[-1][1], comb_dev_all_mean=float(np.mean([a for a, b in devs])), comb_dev_all_max=float(np.max([b for a, b in devs])))
    return d


# ---------- метрика 6: голономия ----------
def walk(c, B, steps, ds, order):
    x = c.copy(); Bt = B.copy()
    for a, sg, L in steps:
        for _ in range(int(round(L / ds))):
            if order == 2:
                xm = x + ds / 2 * sg * Bt[a]; Bm = transport(Bt, g.f(xm, u)); dm = Bm[a]; x = x + ds * sg * dm; Bt = transport(Bm, g.f(x, u))
            else: x = x + ds * sg * Bt[a]; Bt = transport(Bt, g.f(x, u))
    return x


def diag_point(c, B, a, b, r, ds, order):
    d = (B[a] + B[b]); d = d / np.linalg.norm(d); x = c.copy(); L = r * math.sqrt(2)
    for _ in range(int(round(L / ds))):
        d = proj_f(d, g.f(x, u)); d /= np.linalg.norm(d)
        if order == 2:
            xm = x + ds / 2 * d; dm = proj_f(d, g.f(xm, u)); dm /= np.linalg.norm(dm); x = x + ds * dm; d = dm
        else: x = x + ds * d
    return x


def holonomy(B0, r, order):
    out = []
    for a, b in itertools.combinations(range(m), 2):
        ds = r / 50; PA = walk(p, B0, [(a, 1, r), (b, 1, r)], ds, order); PB = walk(p, B0, [(b, 1, r), (a, 1, r)], ds, order); PD = diag_point(p, B0, a, b, r, ds, order)
        fd = g.f(PD, u); nf = float(fd @ fd)
        out.append(dict(pair=[a, b], AB_dist_over_r=float(np.linalg.norm(PA - PB) / r), AB_shift_t=float((PA - PB) @ fd / nf), D_A_dist_over_r=float(np.linalg.norm(PD - PA) / r), D_B_dist_over_r=float(np.linalg.norm(PD - PB) / r),
                        D_A_shift_t=float((PD - PA) @ fd / nf), D_B_shift_t=float((PD - PB) @ fd / nf)))
    return out


if __name__ == '__main__':
    B0 = g.basis(p, u); res = dict(SYS=SYS, u=list(U), seed=list(P0), n=N, m=m, H=H, K=K, steps=STEPS, f_seed=[float(v) for v in g.f(p, u)], runs=[], holonomy=[])
    Wb = dirs_all(); Wa = Wb[(Wb != 0).sum(1) == 1]
    for r in RS:
        for name, fn in (('V0', lambda: run_v0(B0, r)), ('V2', lambda: run_v2(B0, r)), ('V3a', lambda: run_v3(B0, r, Wa, 'V3a')), ('V3b', lambda: run_v3(B0, r, Wb, 'V3b'))):
            t0 = time.perf_counter(); d = fn(); d.update(variant=name, r=r, wall_s=round(time.perf_counter() - t0, 2)); res['runs'].append(d); print(SYS, U, name, r, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}, flush=True)
    for r in sorted(set(RS + [.1, .2, .5])):
        for order in (2, 1): res['holonomy'].append(dict(r=r, order=order, pairs=holonomy(B0, r, order)))
    fn = os.path.join(HERE, 'nd_results.json'); allr = json.load(open(fn)) if os.path.exists(fn) else []; allr = [x for x in allr if not (x['SYS'] == SYS and x['u'] == res['u'] and x['seed'] == res['seed'])] + [res]
    json.dump(allr, open(fn, 'w'), indent=1); print('saved', fn)
