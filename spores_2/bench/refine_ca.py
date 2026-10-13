"""линия A, SPD6: refine коридора v6 через CasADi/IPOPT вместо SLSQP с конечными разностями (точные производные автодиффом, RK4 с ms шагами на сегмент, dt — переменные).
install(env_name) подменяет corridor_nd.refine. Решение проверяется точным потоком (flow из коридора): окно g ≥ −1e-4, при wlim — |w| ≤ WM по 16 точкам сегмента."""
import os, sys, time
import numpy as np
import casadi as ca
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
Rq, Rw, WM = .3, .6, 3.
class CA:
    def __init__(s, name, ms=10, maxit=60):
        s.ms = ms; s.maxit = maxit; s.fallback = False; s.orig = None; s.cache = {}; s.stat = dict(n=0, t=0., it=0)
        if name == 'm3d': import ocp_m3d_lib as L; s.fca = lambda x, u: L.f_ca(x, u); s.C3 = np.zeros(3); s.nu = 3; import manip3d_dyn as M; s.US = np.array(M.US)
        else:
            os.environ.setdefault('G', '0.3'); import ocp_m3g_lib as L; s.fca = lambda x, u: L.f_ca(x, u, .3); s.C3 = np.array([-np.pi / 2 if name == 'm3gd' else np.pi / 2, 0, 0]); s.nu = 3; import manip3g_dyn as G3; s.US = np.array(G3.LAYERS6)
        x = ca.SX.sym('x', 6); u = ca.SX.sym('u', 3); s.fx = ca.Function('f', [x, u], [s.fca(x, u)])
    def solver(s, n, wl):
        k_ = (n, wl)
        if k_ in s.cache: return s.cache[k_]
        x0 = ca.SX.sym('x0', 6); tau = ca.SX.sym('tau', 3, n); tgt = ca.SX.sym('tgt', 3); dt = ca.SX.sym('dt', n); z = x0; wc = []
        for k in range(n):
            h = dt[k] / s.ms; u = tau[:, k]
            for _ in range(s.ms):
                k1 = s.fx(z, u); k2 = s.fx(z + h / 2 * k1, u); k3 = s.fx(z + h / 2 * k2, u); k4 = s.fx(z + h * k3, u); z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                if wl: wc.append(z[3:])
        g = [z[:3] - tgt, z[3:]] + ([ca.vertcat(*wc)] if wl else []); G = ca.vertcat(*g)
        S = ca.nlpsol('S', 'ipopt', dict(x=dt, f=ca.sum1(dt), g=G, p=ca.vertcat(x0, ca.vec(tau), tgt)), {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': s.maxit, 'ipopt.tol': 1e-7, 'ipopt.acceptable_tol': 1e-5, 'ipopt.sb': 'yes'})
        ng = 6 + (3 * s.ms * n if wl else 0); lb = np.r_[-Rq * np.ones(3), -Rw * np.ones(3), -WM * np.ones(ng - 6)]; s.cache[k_] = (S, lb, -lb); return s.cache[k_]
    def refine(s, flow, x0, path, g, tries=3, seed=0, clear=None, ns=16, wlim=None):
        from src.atlas6.corridor_nd import merge, endpoint
        seq, d0 = merge(path); n = len(seq); wl = wlim is not None; S, lbg, ubg = s.solver(n, wl); rng = np.random.default_rng(seed); best = (np.inf, None); US = flow.__self__ if False else None
        tau = s.US[list(seq)].T
        xe0 = endpoint(flow, x0, seq, d0); tgt = s.C3 + 2 * np.pi * np.round((xe0[:3] - s.C3) / (2 * np.pi)); P = np.r_[np.asarray(x0, float), tau.ravel(order='F'), tgt]
        for k in range(tries if tries else 1):
            di = d0 if k == 0 else d0 * rng.uniform(.8, 1.2, n); t0 = time.perf_counter()
            r = S(x0=di, p=P, lbx=0., ubx=1.5 * d0.sum() + 1, lbg=lbg, ubg=ubg); s.stat['n'] += 1; s.stat['t'] += time.perf_counter() - t0; s.stat['it'] += int(S.stats()['iter_count'])
            if S.stats()['return_status'] not in ('Solve_Succeeded', 'Solved_To_Acceptable_Level'): continue
            d = np.array(r['x']).ravel(); T = float(d.sum())
            if T >= best[0] or np.any(d < -1e-9): continue
            d = np.maximum(d, 0.); xe = endpoint(flow, x0, seq, d)
            if not np.all(np.atleast_1d(g(xe)) >= -1e-4): continue
            if wl:
                x = np.array(x0, float); okw = True
                for sg, t in zip(seq, d):
                    for _ in range(8):
                        x = flow(x, sg, t / 8)
                        if np.abs(x[3:]).max() > WM + 1e-2: okw = False
                if not okw: continue
            best = (T, d)
        ok = best[1] is not None
        if not ok and s.fallback: s.stat['fb'] = s.stat.get('fb', 0) + 1; return s.orig(flow, x0, path, g, tries=tries, seed=seed, clear=clear, ns=ns, wlim=wlim)
        return (best[0] if ok else float(d0.sum())), seq, (best[1] if ok else d0), ok
def install(name, ms=10, maxit=60, prebuild=(), wl=False):
    from src.atlas6 import corridor_nd; c = CA(name, ms, maxit); c.orig = corridor_nd.refine; [c.solver(n, wl) for n in prebuild]; corridor_nd.refine = c.refine; return c
