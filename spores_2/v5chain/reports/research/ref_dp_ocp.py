"""[hub-research-7: копия ref6d_ocp.py для ДВОЙНОГО МАЯТНИКА n=2, старт «висит вниз в покое» q=(−π/2,0), цель (π/2,0), Rq .3 Rw .5, мультистарт]
Эталон T для 6D (3 звена, manip3dyn) — прямая оптимизация (CasADi/IPOPT, multiple shooting, свободное T, u ∈ [-1,1]^3 кусочно-пост.).
Верхняя оценка T* (допустимая траектория, проверка мелким rk4 numpy) + мультистарт по ветвям 2π и затравкам.
Динамика аффинна по τ, коробка = выпуклая оболочка 8 слоёв ⇒ inf T по слоям = T* по коробке (релаксация) — эталон годен для слоёв.
Постановка = v6/tests/check_corridor_manip6.py (G, DOWN/UP, Rq .3, Rw .6, WM 3, те же 4 старта rng(0)).
Запуск: PYTHONPATH=<casadi> G=0.3 DOWN=1 python3 ref6d_ocp.py [N M NSTART] [q-индексы через запятую]"""
import os, sys, time, itertools
import numpy as np
import casadi as ca
sys.path.insert(0, os.path.expanduser('~/claude-work/projects/spore/spores_2/v6')); sys.path.insert(0, os.environ.get('V6', ''))
from src.atlas6.manip3dyn import f as f_np, MS, L

GG = float(os.environ.get('G', 0)); NJ = 2; C3 = np.array([np.pi / 2, 0.0])
Rq, Rw = 0.3, 0.5; WM = float(os.environ.get('WMP', 3.0))      # WMP — предел |w| вдоль пути (refine коридора его НЕ держит; WMP=50 ≈ без предела)
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi


def starts(nq=4):
    """Те же запросы, что check_corridor_manip6.py (nq=4) и stats_manip6.py без OBST (nq=NQ; первые 4 совпадают)."""
    rng = np.random.default_rng(0)
    for _ in range(32): rng.uniform(-1, 1, 6)                       # затравки окна в тесте — тот же поток rng
    return [np.array([*rng.uniform(-np.pi, np.pi, 3), *rng.uniform(-1, 1, 3)]) for _ in range(nq)]


def f_ca(x, u, g=GG, m=MS, ln=L):
    n = NJ; q, w = x[:n], x[n:]; m = np.asarray(m[:n]); ln = np.asarray(ln[:n])      # MS/L в manip3dyn теперь длины 4 (n=4, worker-8)
    th = [ca.sum1(q[:i + 1]) for i in range(n)]; thd = [ca.sum1(w[:i + 1]) for i in range(n)]
    S = [[float(m[max(i, j):].sum() * ln[i] * ln[j]) for j in range(n)] for i in range(n)]
    M = ca.SX(n, n); rhs = ca.SX(n, 1)
    for i in range(n):
        for j in range(n): M[i, j] = S[i][j] * ca.cos(th[i] - th[j])
        rhs[i] = u[i] - (u[i + 1] if i + 1 < n else 0) - g * ln[i] * float(m[i:].sum()) * ca.cos(th[i]) \
            - sum(S[i][j] * ca.sin(th[i] - th[j]) * thd[j] ** 2 for j in range(n))
    tdd = ca.solve(M, rhs)
    qdd = ca.vertcat(tdd[0], tdd[1] - tdd[0])
    return ca.vertcat(w, qdd)


def make_solver(N, M):
    x = ca.SX.sym('x', 4); u = ca.SX.sym('u', 2); h = ca.SX.sym('h')
    fx = ca.Function('f', [x, u], [f_ca(x, u)])
    z = x
    for _ in range(M):
        k1 = fx(z, u); k2 = fx(z + h / 2 * k1, u); k3 = fx(z + h / 2 * k2, u); k4 = fx(z + h * k3, u); z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    F = ca.Function('F', [x, u, h], [z])
    X = ca.SX.sym('X', 4, N + 1); U = ca.SX.sym('U', 2, N); T = ca.SX.sym('T'); P = ca.SX.sym('P', 6)   # P: x0 (6), центр цели q (3) — уже с ветвью 2π, пусто 3
    h_ = T / (N * M); g = [X[:, 0] - P[:4]]
    for k in range(N): g.append(F(X[:, k], U[:, k], h_) - X[:, k + 1])
    g.append(X[:2, N] - P[4:6])
    w = ca.vertcat(ca.vec(X), ca.vec(U), T)
    nlp = {'x': w, 'f': T, 'g': ca.vertcat(*g), 'p': P}
    S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 1500, 'ipopt.tol': 1e-8,
                                       'ipopt.constr_viol_tol': 1e-9})
    nx = 4 * (N + 1); nu = 2 * N
    lbx = np.full(nx + nu + 1, -np.inf); ubx = np.full(nx + nu + 1, np.inf)
    Xi = np.arange(nx).reshape(N + 1, 4)                                # vec(X) — по столбцам: узел k = 6k..6k+5
    lbx[Xi[:, 2:].ravel()] = -WM; ubx[Xi[:, 2:].ravel()] = WM
    lbx[Xi[-1, 2:]] = -Rw; ubx[Xi[-1, 2:]] = Rw
    lbx[nx:nx + nu] = -1; ubx[nx:nx + nu] = 1; lbx[-1] = 0.05; ubx[-1] = 25
    ng = 4 * (N + 1) + 2; lbg = np.zeros(ng); ubg = np.zeros(ng); lbg[-2:] = -Rq; ubg[-2:] = Rq
    return S, lbx, ubx, lbg, ubg


def simulate(x0, U, T, dt=0.005):
    """Проверка: rk4 numpy (manip3dyn.f) с мелким шагом, u кусочно-пост. на N отрезках."""
    N = U.shape[0]; x = np.array(x0, float); hseg = T / N; k = max(1, int(np.ceil(hseg / dt))); h = hseg / k; wmax = np.max(np.abs(x[2:]))
    for i in range(N):
        for _ in range(k):
            k1 = f_np(x, U[i], GG); k2 = f_np(x + h / 2 * k1, U[i], GG); k3 = f_np(x + h / 2 * k2, U[i], GG); k4 = f_np(x + h * k3, U[i], GG)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        wmax = max(wmax, np.max(np.abs(x[2:])))
    return x, wmax


def solve_query(x0, N=80, M=4, nstart=12, seed=0, solver=None):
    S, lbx, ubx, lbg, ubg = solver or make_solver(N, M)
    d0 = wr(C3 - x0[:2]); rng = np.random.default_rng(seed); best = None; res = []
    for br in itertools.product(*[(d, d - 2 * np.pi * np.sign(d)) if abs(d) > 1e-9 else (0.0, 2 * np.pi, -2 * np.pi) for d in d0]):
        tgt = x0[:2] + np.array(br); P = np.concatenate([x0, tgt])
        for s in range(nstart):
            T0 = [3., 5., 7., 9., 12., 15.][s % 6] * (1 + 0.1 * rng.standard_normal())
            al = np.linspace(0, 1, N + 1)[:, None]; Xg = (1 - al) * x0 + al * np.concatenate([tgt, np.zeros(2)])
            nsw = 1 + s % 4                                                        # затравка раскачки: колебания по q с nsw полупериодами
            Xg[:, :2] += 0.8 * np.sin(np.pi * nsw * al) * rng.uniform(.3, 1.2, 2)
            Xg[:, 2:] = np.clip(np.gradient(Xg[:, :2], T0 / N, axis=0), -WM, WM)
            Ug = np.clip(np.sign(np.sin(np.pi * nsw * (np.arange(N) + .5) / N))[:, None] * np.array([1., 1.]) + 0.3 * rng.standard_normal((N, 2)), -1, 1)
            w0 = np.concatenate([Xg.ravel(), Ug.ravel(), [T0]])
            t0 = time.time(); r = S(x0=w0, p=P, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg); st = S.stats()['return_status']
            wv = np.array(r['x']).ravel(); T = wv[-1]; U = wv[4 * (N + 1):4 * (N + 1) + 2 * N].reshape(N, 2)
            xe, wmax = simulate(x0, U, T)
            ok = st in ('Solve_Succeeded', 'Solved_To_Acceptable_Level') and np.all(np.abs(wr(xe[:2] - C3)) < Rq + 1e-3) \
                and np.all(np.abs(xe[2:]) < Rw + 1e-3) and wmax <= WM + 1e-2
            res.append((ok, T, br, s, st, time.time() - t0))
            if ok and (best is None or T < best[0]): best = (T, U, br)
    return best, res


def warm_start(x0, seq, dts, N=40, M=4, solver=None):
    """Тёплый старт от пути коридора: слои seq (индекс угла коробки, бит i → τ_i = ±1) и длительности dts → затравка U/X/T.
    Ветвь 2π — по концу пути. → (T, U) допустимое (проверка rk4) или None; T ≤ T_corr, если IPOPT не ушёл в худший минимум."""
    S, lbx, ubx, lbg, ubg = solver or make_solver(N, M)
    T0 = float(np.sum(dts)); tb = np.concatenate([[0], np.cumsum(dts)]); tc = (np.arange(N) + 0.5) * T0 / N
    lay = np.array([[1.0 if (s >> i) & 1 else -1.0 for i in range(3)] for s in seq])
    Ug = lay[np.clip(np.searchsorted(tb, tc, side='right') - 1, 0, len(seq) - 1)]
    Xg = [np.array(x0, float)]; h = T0 / N / M
    for k in range(N):
        x = Xg[-1]
        for _ in range(M):
            k1 = f_np(x, Ug[k], GG); k2 = f_np(x + h / 2 * k1, Ug[k], GG); k3 = f_np(x + h / 2 * k2, Ug[k], GG); k4 = f_np(x + h * k3, Ug[k], GG)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Xg.append(x)
    Xg = np.array(Xg); tgt = Xg[-1, :3] + wr(C3 - Xg[-1, :3])                    # ветвь: центр окна рядом с концом пути
    r = S(x0=np.concatenate([Xg.ravel(), Ug.ravel(), [T0]]), p=np.concatenate([x0, tgt, np.zeros(3)]), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    w = np.array(r['x']).ravel(); T = w[-1]; U = w[6 * (N + 1):6 * (N + 1) + 3 * N].reshape(N, 3); xe, wmax = simulate(x0, U, T)
    ok = S.stats()['return_status'] in ('Solve_Succeeded', 'Solved_To_Acceptable_Level') and np.all(np.abs(wr(xe[:3] - C3)) < Rq + 1e-3) \
        and np.all(np.abs(xe[3:]) < Rw + 1e-3) and wmax <= WM + 1e-2
    if os.environ.get('DBG'): print('warm:', S.stats()['return_status'], 'T %.4f' % T, 'конец', np.round(wr(xe[:3] - C3), 4), np.round(xe[3:], 4), 'wmax %.3f' % wmax)
    return (T, U) if ok else None


if __name__ == '__main__':
    a = sys.argv[1:]; N, M, NS = (int(a[0]), int(a[1]), int(a[2])) if len(a) >= 3 else (80, 4, 12)
    x0 = np.array([-np.pi / 2, 0.0, 0.0, 0.0]); sol = make_solver(N, M); t0 = time.time()
    best, res = solve_query(x0, N, M, NS, seed=int(os.environ.get('SEED', 0)), solver=sol)
    Ts = sorted(r[1] for r in res if r[0]); st = {}
    for r in res: st[r[4]] = st.get(r[4], 0) + 1
    print('ДВОЙНОЙ МАЯТНИК висит→вверх G=%.2f WMP=%g N=%d M=%d: T_ref %s  успешных %d/%d  лучшие %s  статусы %s  (%.0f с)' % (GG, WM, N, M, None if best is None else round(best[0], 3),
          len(Ts), len(res), [round(t, 3) for t in Ts[:6]], st, time.time() - t0), flush=True)
    if best is not None: np.save(os.path.join(os.environ.get('OUT', '.'), 'refdp_G%g_W%g_N%d.npy' % (GG, WM, N)), np.concatenate([[best[0]], best[1].ravel()]))
