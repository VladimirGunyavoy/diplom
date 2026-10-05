"""research hub-v5chain-research-12: потолок семейства дуг для двойного маятника g = 2 (висит → вверх, OCP 7.636 при |u| ≤ 1, |w| ≤ 3).
Вопрос: сколько из ×1.5–1.9 агента бабочек — от самого семейства (K дуг с постоянным моментом, запас UMAX .9), а сколько — от графа/V.
OCP «K дуг»: K отрезков постоянного u_k ∈ [−UB, UB]², своя длительность h_k ∈ [HMIN, HMAX], M подшагов rk4 на дугу; |w| ≤ WM в КАЖДОМ подшаге
(строже ref_dp_ocp.py: там |w| только в узлах). Цель — как в ref_dp_ocp.py: |q − (π/2, 0) − 2πk| ≤ .3, |w| ≤ .5. Допустимость — rk4 numpy dt .002.
Затравки: OCP N=80 (refdp_G2_W3_N80.npy, усреднение u по K кускам) + случайные bang-bang с раскачкой.
Запуск (aida): PYTHONPATH=~/spore_v5/r5/pylib G=2 UB=1 python3 ocp_arcs.py K [M NRAND]"""
import os, sys, time, json
import numpy as np
import casadi as ca

G = float(os.environ.get('G', 2.)); UB = float(os.environ.get('UB', 1.)); WM = float(os.environ.get('WM', 3.)); HMIN = float(os.environ.get('HMIN', .02))
HMAX = float(os.environ.get('HMAX', 3.)); REF = os.environ.get('REF', 'refdp_G2_W3_N80.npy'); MS = [2.5, 1.0]; S11, S12, S22 = 2.5, 1., 1.
C3 = np.array([np.pi / 2, 0.]); RQ, RW = .3, .5; X0 = np.array([-np.pi / 2, 0, 0, 0])


def acc(q1, q2, w1, w2, u1, u2, cos=np.cos, sin=np.sin):
    th1, th2 = q1, q1 + q2; d1, d2 = w1, w1 + w2; c = cos(th1 - th2); s = sin(th1 - th2)
    Q1 = u1 - u2 - G * MS[0] * cos(th1) - S12 * s * d2 ** 2; Q2 = u2 - G * MS[1] * cos(th2) + S12 * s * d1 ** 2
    det = S11 * S22 - (S12 * c) ** 2; a1 = (S22 * Q1 - S12 * c * Q2) / det; a2 = (S11 * Q2 - S12 * c * Q1) / det
    return a1, a2 - a1


def fnp(z, u): a1, a2 = acc(*z, *u); return np.array([z[2], z[3], a1, a2])


def rk4(fx, z, u, h):
    k1 = fx(z, u); k2 = fx(z + h / 2 * k1, u); k3 = fx(z + h / 2 * k2, u); k4 = fx(z + h * k3, u); return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(U, H, dt=.002):
    z = X0.copy(); wm = 0.
    for u, h in zip(U, H):
        n = max(1, int(np.ceil(h / dt)))
        for _ in range(n): z = rk4(fnp, z, u, h / n); wm = max(wm, np.abs(z[2:]).max())
    return z, wm


def make(K, M):
    x = ca.SX.sym('x', 4); u = ca.SX.sym('u', 2); h = ca.SX.sym('h')
    a1, a2 = acc(x[0], x[1], x[2], x[3], u[0], u[1], ca.cos, ca.sin); fx = lambda z, v: ca.vertcat(z[2], z[3], *acc(z[0], z[1], z[2], z[3], v[0], v[1], ca.cos, ca.sin))
    F = ca.Function('F', [x, u, h], [rk4(fx, x, u, h)])
    X = ca.SX.sym('X', 4, K * M + 1); U = ca.SX.sym('U', 2, K); H = ca.SX.sym('H', K); P = ca.SX.sym('P', 2)
    g = [X[:, 0] - X0]
    for k in range(K):
        for j in range(M): i = k * M + j; g.append(F(X[:, i], U[:, k], H[k] / M) - X[:, i + 1])
    g.append(X[:2, -1] - P)
    w = ca.vertcat(ca.vec(X), ca.vec(U), H); S = ca.nlpsol('S', 'ipopt', {'x': w, 'f': ca.sum1(H), 'g': ca.vertcat(*g), 'p': P},
                                                         {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 3000, 'ipopt.tol': 1e-8})
    nx = 4 * (K * M + 1); lbx = np.full(nx + 3 * K, -np.inf); ubx = -lbx.copy(); Xi = np.arange(nx).reshape(-1, 4)
    lbx[Xi[:, 2:].ravel()] = -(WM - .01); ubx[Xi[:, 2:].ravel()] = WM - .01; lbx[Xi[-1, 2:]] = -(RW - .005); ubx[Xi[-1, 2:]] = RW - .005
    lbx[nx:nx + 2 * K] = -UB; ubx[nx:nx + 2 * K] = UB; lbx[nx + 2 * K:] = HMIN; ubx[nx + 2 * K:] = HMAX
    ng = 4 * (K * M + 1) + 2; lbg = np.zeros(ng); ubg = np.zeros(ng); lbg[-2:] = -(RQ - .005); ubg[-2:] = RQ - .005
    return S, lbx, ubx, lbg, ubg, nx


def guess(U, H, M, Ys=None):
    Xg = [X0.copy()]
    for i, (u, h) in enumerate(zip(U, H)):
        z = Xg[-1] if Ys is None else Ys[i].copy()                                            # WARM: дуга — из записанного старта (скачки привязки агента)
        if Ys is not None: z[:2] = Xg[-1][:2] + ((z[:2] - Xg[-1][:2] + np.pi) % (2 * np.pi) - np.pi)
        for _ in range(M): z = rk4(fnp, z, u, h / M); Xg.append(z)
    return np.array(Xg)


def load_warm(path, hmax):
    """Путь агента (dp_traj_arcs.py) → дуги длиной ≤ hmax (длинные режутся на равные куски того же u)."""
    d = np.load(path); Y, U, H = [], [], []
    for y, u, h in zip(d['Y'], d['U'], d['H']):
        n = int(np.ceil(h / hmax - 1e-9)); z = y.copy()
        for _ in range(n):
            Y.append(z.copy()); U.append(u); H.append(h / n)
            for _ in range(20): z = rk4(fnp, z, u, h / n / 20)
    return np.array(Y), np.clip(np.array(U), -UB, UB), np.maximum(np.array(H), HMIN)


def main():
    a = sys.argv[1:]; K = int(a[0]); M = int(a[1]) if len(a) > 1 else 8; NR = int(a[2]) if len(a) > 2 else 12
    global HMAX
    HMAX = min(HMAX, M * float(os.environ.get('DTS', .04)))                  # подшаг rk4 ≤ DTS, иначе IPOPT «выигрывает» на ошибке интегрирования
    WARM = os.environ.get('WARM')
    if WARM: Yw, Uw, Hw = load_warm(WARM, HMAX); K = len(Hw); NR = 0; print('WARM %s: %d дуг, T %.3f' % (WARM, K, Hw.sum()), flush=True)
    S, lbx, ubx, lbg, ubg, nx = make(K, M); seeds = []
    if WARM: seeds.append(('warm', Uw, Hw, Yw))
    elif os.path.exists(REF):
        r = np.load(REF); T0 = r[0]; U80 = r[1:].reshape(-1, 2)
        for kk in range(3):                                                       # три разбиения 80 кусков на K: равные и сдвинутые
            edges = np.unique(np.clip(np.round(np.linspace(0, 80, K + 1) + kk * (80 / K) / 3 * (0 < np.arange(K + 1)) * (np.arange(K + 1) < K)), 0, 80).astype(int))
            if len(edges) != K + 1: continue
            Uk = np.array([U80[edges[i]:edges[i + 1]].mean(0) for i in range(K)]); Hk = np.minimum(np.diff(edges) * T0 / 80, HMAX)
            seeds.append(('ocp%d' % kk, np.clip(Uk, -UB, UB), Hk, None))
    rng = np.random.default_rng(int(os.environ.get('SEED', 0)))
    for s in range(NR):
        T0 = rng.uniform(6, 14); Hk = np.minimum(rng.dirichlet(np.ones(K) * 3) * T0, HMAX); ph = rng.uniform(0, 2 * np.pi, 2); tc = np.cumsum(Hk) - Hk / 2
        Uk = np.stack([np.sign(np.sin(2 * np.pi * tc / rng.uniform(1.5, 3.5) + ph[i])) for i in range(2)], 1) * UB
        seeds.append(('rnd%d' % s, Uk, Hk, None))
    best = None; out = []
    for name, Uk, Hk, Ys in seeds:
        Xg = guess(Uk, Hk, M, Ys); end = Xg[-1, :2]; tgt = end + ((C3 - end + np.pi) % (2 * np.pi) - np.pi)
        Xg[:, 2:] = np.clip(Xg[:, 2:], -WM + .02, WM - .02)
        t0 = time.time(); r = S(x0=np.concatenate([Xg.ravel(), Uk.ravel(), Hk]), p=tgt, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg); st = S.stats()['return_status']
        w = np.array(r['x']).ravel(); U = w[nx:nx + 2 * K].reshape(K, 2); H = w[nx + 2 * K:]; T = H.sum(); ze, wm = simulate(U, H)
        dq = (ze[:2] - C3 + np.pi) % (2 * np.pi) - np.pi; ok = st in ('Solve_Succeeded', 'Solved_To_Acceptable_Level') and np.all(np.abs(dq) <= RQ + 1e-3) \
            and np.all(np.abs(ze[2:]) <= RW + 1e-3) and wm <= WM + 1e-3
        out.append(dict(seed=name, st=st[:12], T=round(T, 4), ok=bool(ok), wmax=round(wm, 3), end=np.round(np.r_[dq, ze[2:]], 3).tolist(), sec=round(time.time() - t0, 1)))
        print(json.dumps(out[-1]), flush=True)
        if ok and (best is None or T < best[0]): best = (T, U, H)
    res = dict(K=K, M=M, G=G, UB=UB, WM=WM, HMIN=HMIN, T=None if best is None else round(best[0], 4), ratio=None if best is None else round(best[0] / 7.636, 4),
               nok=sum(o['ok'] for o in out), n=len(out))
    if best is not None:
        res['H'] = np.round(best[2], 3).tolist(); res['U'] = np.round(best[1], 3).tolist()
        np.save(os.environ.get('OUT', 'ocparcs_G%g_UB%g_K%d.npy' % (G, UB, K)), np.concatenate([[best[0]], best[1].ravel(), best[2]]))
    print('RESULT ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
