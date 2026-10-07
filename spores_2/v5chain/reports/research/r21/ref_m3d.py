"""[research-21: копия для 3D двузвенного манипулятора r21/manip3d.py, цель q*=0, NAME m3d] Эталон T для 6D (3 звена, manip3dyn) — прямая оптимизация (CasADi/IPOPT, multiple shooting, свободное T, u ∈ [-1,1]^3 кусочно-пост.).
Верхняя оценка T* (допустимая траектория, проверка мелким rk4 numpy) + мультистарт по ветвям 2π и затравкам.
Динамика аффинна по τ, коробка = выпуклая оболочка 8 слоёв ⇒ inf T по слоям = T* по коробке (релаксация) — эталон годен для слоёв.
Постановка = v6/tests/check_corridor_manip6.py (G, DOWN/UP, Rq .3, Rw .6, WM 3, те же 4 старта rng(0)).
Запуск: PYTHONPATH=<casadi> G=0.3 DOWN=1 python3 ref6d_ocp.py [N M NSTART] [q-индексы через запятую]"""
import os, sys, time, itertools
import numpy as np
import casadi as ca
sys.path.insert(0, os.path.expanduser('~/claude-work/projects/spore/spores_2/v6')); sys.path.insert(0, os.environ.get('V6', ''))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import manip3d as M3
MS, L = None, None
def f_np(x, u, g=0.): return M3.f(np.asarray(x, float), np.asarray(u, float))

GG = float(os.environ.get('G', 0)); C3 = np.zeros(3)
Rq, Rw = 0.3, 0.6; WM = float(os.environ.get('WMP', 3.0))      # WMP — предел |w| вдоль пути (refine коридора его НЕ держит; WMP=50 ≈ без предела)
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi


def starts(nq=4):
    """Те же запросы, что check_corridor_manip6.py (nq=4) и stats_manip6.py без OBST (nq=NQ; первые 4 совпадают)."""
    rng = np.random.default_rng(0)
    for _ in range(32): rng.uniform(-1, 1, 6)                       # затравки окна в тесте — тот же поток rng
    return [np.array([*rng.uniform(-np.pi, np.pi, 3), *rng.uniform(-1, 1, 3)]) for _ in range(nq)]


def f_ca(x, u, g=0., m=None, ln=None):
    """CasADi-копия manip3d.f (проверка совпадения — DBGF=1)"""
    q1, q2, w0, w1, w2 = x[1], x[2], x[3], x[4], x[5]; I0, L1, L2, M1, M2, A, B, D = M3.I0, M3.L1, M3.L2, M3.M1, M3.M2, M3.A, M3.B, M3.D
    r1 = L1 * ca.cos(q1); r2 = r1 + L2 * ca.cos(q1 + q2); J = I0 + M1 * r1 ** 2 + M2 * r2 ** 2; s1, s12 = ca.sin(q1), ca.sin(q1 + q2)
    d1 = -2 * (M1 * r1 * L1 * s1 + M2 * r2 * (L1 * s1 + L2 * s12)); d2 = -2 * M2 * r2 * L2 * s12
    a0 = (u[0] - (d1 * w1 + d2 * w2) * w0) / J; c, s = ca.cos(q2), ca.sin(q2); h = -B * s
    ra = u[1] - h * (2 * w1 * w2 + w2 ** 2) + .5 * d1 * w0 ** 2; rb = u[2] + h * w1 ** 2 + .5 * d2 * w0 ** 2
    m11, m12 = A + 2 * B * c, D + B * c; det = m11 * D - m12 ** 2
    return ca.vertcat(w0, w1, w2, a0, (D * ra - m12 * rb) / det, (-m12 * ra + m11 * rb) / det)


def make_solver(N, M):
    x = ca.SX.sym('x', 6); u = ca.SX.sym('u', 3); h = ca.SX.sym('h')
    fx = ca.Function('f', [x, u], [f_ca(x, u)])
    z = x
    for _ in range(M):
        k1 = fx(z, u); k2 = fx(z + h / 2 * k1, u); k3 = fx(z + h / 2 * k2, u); k4 = fx(z + h * k3, u); z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    F = ca.Function('F', [x, u, h], [z])
    X = ca.SX.sym('X', 6, N + 1); U = ca.SX.sym('U', 3, N); T = ca.SX.sym('T'); P = ca.SX.sym('P', 12)   # P: x0 (6), центр цели q (3) — уже с ветвью 2π, пусто 3
    h_ = T / (N * M); g = [X[:, 0] - P[:6]]
    for k in range(N): g.append(F(X[:, k], U[:, k], h_) - X[:, k + 1])
    g.append(X[:3, N] - P[6:9])
    w = ca.vertcat(ca.vec(X), ca.vec(U), T)
    nlp = {'x': w, 'f': T, 'g': ca.vertcat(*g), 'p': P}
    S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 1500, 'ipopt.tol': 1e-8,
                                       'ipopt.constr_viol_tol': 1e-9})
    nx = 6 * (N + 1); nu = 3 * N
    lbx = np.full(nx + nu + 1, -np.inf); ubx = np.full(nx + nu + 1, np.inf)
    Xi = np.arange(nx).reshape(N + 1, 6)                                # vec(X) — по столбцам: узел k = 6k..6k+5
    lbx[Xi[:, 3:].ravel()] = -WM; ubx[Xi[:, 3:].ravel()] = WM
    lbx[Xi[-1, 3:]] = -Rw; ubx[Xi[-1, 3:]] = Rw
    lbx[nx:nx + nu] = -1; ubx[nx:nx + nu] = 1; lbx[-1] = 0.05; ubx[-1] = 12
    ng = 6 * (N + 1) + 3; lbg = np.zeros(ng); ubg = np.zeros(ng); lbg[-3:] = -Rq; ubg[-3:] = Rq
    return S, lbx, ubx, lbg, ubg


def simulate(x0, U, T, dt=0.005):
    """Проверка: rk4 numpy (manip3dyn.f) с мелким шагом, u кусочно-пост. на N отрезках."""
    N = U.shape[0]; x = np.array(x0, float); hseg = T / N; k = max(1, int(np.ceil(hseg / dt))); h = hseg / k; wmax = np.max(np.abs(x[3:]))
    for i in range(N):
        for _ in range(k):
            k1 = f_np(x, U[i], GG); k2 = f_np(x + h / 2 * k1, U[i], GG); k3 = f_np(x + h / 2 * k2, U[i], GG); k4 = f_np(x + h * k3, U[i], GG)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        wmax = max(wmax, np.max(np.abs(x[3:])))
    return x, wmax


def solve_query(x0, N=50, M=4, nstart=4, seed=0, solver=None):
    S, lbx, ubx, lbg, ubg = solver or make_solver(N, M)
    d0 = wr(C3 - x0[:3]); rng = np.random.default_rng(seed); best = None; res = []
    for br in itertools.product(*[(d, d - 2 * np.pi * np.sign(d)) for d in d0]):
        tgt = x0[:3] + np.array(br); P = np.concatenate([x0, tgt, np.zeros(3)])
        for s in range(nstart):
            T0 = [1.5, 2.5, 3.5, 5.0][s % 4] * (1 + 0.1 * rng.standard_normal())
            al = np.linspace(0, 1, N + 1)[:, None]; Xg = (1 - al) * x0 + al * np.concatenate([tgt, np.zeros(3)])
            Xg[:, 3:] = np.clip(Xg[:, 3:] + (np.array(br) / T0)[None] * np.sin(np.pi * al) * 1.5, -WM, WM)
            Ug = np.clip(0.5 * rng.standard_normal((N, 3)), -1, 1)
            w0 = np.concatenate([Xg.ravel(), Ug.ravel(), [T0]])
            t0 = time.time(); r = S(x0=w0, p=P, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg); st = S.stats()['return_status']
            wv = np.array(r['x']).ravel(); T = wv[-1]; U = wv[6 * (N + 1):6 * (N + 1) + 3 * N].reshape(N, 3)
            xe, wmax = simulate(x0, U, T)
            ok = st in ('Solve_Succeeded', 'Solved_To_Acceptable_Level') and np.all(np.abs(wr(xe[:3] - C3)) < Rq + 1e-3) \
                and np.all(np.abs(xe[3:]) < Rw + 1e-3) and wmax <= WM + 1e-2
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
    a = sys.argv[1:]; N, M, NS = (int(a[0]), int(a[1]), int(a[2])) if len(a) >= 3 else (50, 4, 4)
    qi = [int(v) for v in a[3].split(',')] if len(a) > 3 else range(4)
    Q = starts(int(os.environ.get('NQ', 4))); sol = make_solver(N, M)
    for i in qi:
        t0 = time.time(); best, res = solve_query(Q[i], N, M, NS, seed=i, solver=sol)
        Ts = sorted(r[1] for r in res if r[0])
        print('q%d G=%.1f WMP=%g N=%d M=%d: T_ref %s  успешных %d/%d  лучшие %s  (%.0f с)' % (i + 1, GG, WM, N, M, None if best is None else round(best[0], 3),
              len(Ts), len(res), [round(t, 3) for t in Ts[:5]], time.time() - t0), flush=True)
        if best is not None: np.save(os.path.join(os.environ.get('OUT', '.'), 'refm3d_%s_q%d_G%g_N%d_W%g.npy' % (os.environ.get('NAME', ''), i + 1, GG, N, WM)), np.concatenate([[best[0]], best[1].ravel()]))
if os.environ.get('DBGF'):
    X = ca.SX.sym('x', 6); U = ca.SX.sym('u', 3); F = ca.Function('F', [X, U], [f_ca(X, U)]); rng = np.random.default_rng(5)
    print('max |f_ca − f_np| =', max(np.abs(np.array(F(x, u)).ravel() - f_np(x, u)).max() for x, u in zip(rng.uniform(-3, 3, (200, 6)), rng.uniform(-1, 1, (200, 3)))))
