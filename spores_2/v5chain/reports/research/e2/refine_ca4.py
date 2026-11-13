"""e2: IPOPT-доводчик коридора для 4D-сред dp1 / manip (по образцу bench/refine_ca.py): CasADi/IPOPT вместо SLSQP, точные производные, RK4 с MS шагами на сегмент, dt — переменные.
Отличия от refine_ca.py: n=2 (2 угла + 2 скорости), fca переписан с common.DP1.f / common.Manip.f (в сдвинутых координатах цель в нуле), окно Rq .3 / Rw .5 с запасом CAMARG (.01 = как окно коридора half−.01),
|w| ≤ WMC (по умолчанию 2.96 = 3.01−.05) в MS (по умолчанию 64) точках на сегмент; ФИКС БАГА d4: допуск отрицательных dt −1e-5 (было −1e-9). install(name) подменяет corridor_nd.refine.
Решение проверяется точным потоком коридора: g ≥ −1e-4, |w| ≤ WCHK (3.0+1e-2) в 8 точках сегмента."""
import os, sys, time
import numpy as np
import casadi as ca
Rq, Rw = .3, .5
class CA4:
    def __init__(s, name, ms=64, maxit=60, wmc=2.96):
        s.ms = ms; s.maxit = maxit; s.wmc = wmc; s.fallback = False; s.orig = None; s.cache = {}; s.st = {}; s.stat = dict(n=0, t=0., it=0); s.n = 2; s.nu = 2; s.C3 = np.zeros(2)
        s.US = np.array([[a, b] for a in (-1., 1.) for b in (-1., 1.)]); s.name = name
        if name == 'dp1':
            G1, MS, S11, S12, S22 = 1., (2.5, 1.), 2.5, 1., 1.
            def fca(x, u):
                q1, q2, w1, w2 = x[0] + np.pi / 2, x[1], x[2], x[3]; th1, th2 = q1, q1 + q2; d1, d2 = w1, w1 + w2; c = ca.cos(th1 - th2); sn = ca.sin(th1 - th2)
                Q1 = u[0] - u[1] - G1 * MS[0] * ca.cos(th1) - S12 * sn * d2 ** 2; Q2 = u[1] - G1 * MS[1] * ca.cos(th2) + S12 * sn * d1 ** 2
                det = S11 * S22 - (S12 * c) ** 2; a1 = (S22 * Q1 - S12 * c * Q2) / det; a2 = (S11 * Q2 - S12 * c * Q1) / det
                return ca.vertcat(w1, w2, a1, a2 - a1)
        elif name == 'manip':
            AA, BB, DD = 2.5, .5, 1.
            def fca(x, u):
                c, sn = ca.cos(x[1]), ca.sin(x[1]); w1, w2 = x[2], x[3]; h = -BB * sn; r1 = u[0] - h * (2 * w1 * w2 + w2 ** 2); r2 = u[1] + h * w1 ** 2
                m11, m12, m22 = AA + 2 * BB * c, DD + BB * c, DD; det = m11 * m22 - m12 ** 2
                return ca.vertcat(w1, w2, (m22 * r1 - m12 * r2) / det, (-m12 * r1 + m11 * r2) / det)
        else: raise ValueError(name)
        x = ca.SX.sym('x', 4); u = ca.SX.sym('u', 2); s.fx = ca.Function('f', [x, u], [fca(x, u)])
    def solver(s, n, wl):
        k_ = (n, wl)
        if k_ in s.cache: return s.cache[k_]
        nq = 2; x0 = ca.SX.sym('x0', 4); tau = ca.SX.sym('tau', 2, n); tgt = ca.SX.sym('tgt', nq); dt = ca.SX.sym('dt', n); z = x0; wc = []
        for k in range(n):
            h = dt[k] / s.ms; u = tau[:, k]
            for _ in range(s.ms):
                k1 = s.fx(z, u); k2 = s.fx(z + h / 2 * k1, u); k3 = s.fx(z + h / 2 * k2, u); k4 = s.fx(z + h * k3, u); z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                if wl: wc.append(z[nq:])
        G = ca.vertcat(*([z[:nq] - tgt, z[nq:]] + ([ca.vertcat(*wc)] if wl else [])))
        S = ca.nlpsol('S', 'ipopt', dict(x=dt, f=ca.sum1(dt), g=G, p=ca.vertcat(x0, ca.vec(tau), tgt)), {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': s.maxit, 'ipopt.tol': 1e-7, 'ipopt.acceptable_tol': 1e-5, 'ipopt.sb': 'yes'})
        mg = float(os.environ.get('CAMARG', .01)); ng = 2 * nq + (nq * s.ms * n if wl else 0)
        lb = np.r_[-(Rq - mg) * np.ones(nq), -(Rw - mg) * np.ones(nq), -s.wmc * np.ones(ng - 2 * nq)]; s.cache[k_] = (S, lb, -lb); return s.cache[k_]
    def refine(s, flow, x0, path, g, tries=3, seed=0, clear=None, ns=16, wlim=None):
        from src.atlas6.corridor_nd import merge, endpoint
        seq, d0 = merge(path); n = len(seq); wl = wlim is not None; S, lbg, ubg = s.solver(n, wl); tries0 = tries; rng = np.random.default_rng(seed); best = (np.inf, None)
        tau = s.US[list(seq)].T; xe0 = endpoint(flow, x0, seq, d0); tgt = s.C3 + 2 * np.pi * np.round((xe0[:2] - s.C3) / (2 * np.pi)); P = np.r_[np.asarray(x0, float), tau.ravel(order='F'), tgt]
        negtol = float(os.environ.get('CADNEG', 1e-5))
        for k in range(tries if tries else 1):
            di = d0 if k == 0 else d0 * rng.uniform(.8, 1.2, n); t0 = time.perf_counter()
            r = S(x0=di, p=P, lbx=0., ubx=1.5 * d0.sum() + 1, lbg=lbg, ubg=ubg); s.stat['n'] += 1; s.stat['t'] += time.perf_counter() - t0; s.stat['it'] += int(S.stats()['iter_count'])
            rs = S.stats()['return_status']; s.st[rs] = s.st.get(rs, 0) + 1
            if rs not in ('Solve_Succeeded', 'Solved_To_Acceptable_Level'): continue
            d = np.array(r['x']).ravel(); T = float(d.sum())
            if T >= best[0]: continue
            if np.any(d < -negtol): s.stat['neg_rej'] = s.stat.get('neg_rej', 0) + 1; continue
            d = np.maximum(d, 0.); xe = endpoint(flow, x0, seq, d)
            if not np.all(np.atleast_1d(g(xe)) >= -1e-4): s.stat['rej_goal'] = s.stat.get('rej_goal', 0) + 1; continue
            if wl:
                x = np.array(x0, float); okw = True
                for sg, t in zip(seq, d):
                    for _ in range(8):
                        x = flow(x, sg, t / 8)
                        if np.abs(x[2:]).max() > 3.0 + 1e-2: okw = False
                if not okw: s.stat['rej_w'] = s.stat.get('rej_w', 0) + 1; continue
            best = (T, d)
        ok = best[1] is not None
        if not ok and (s.fallback or (os.environ.get('CAFB') and tries0 > 1)): s.stat['fb'] = s.stat.get('fb', 0) + 1; return s.orig(flow, x0, path, g, tries=tries, seed=seed, clear=clear, ns=ns, wlim=wlim)
        return (best[0] if ok else float(d0.sum())), seq, (best[1] if ok else d0), ok
def install(name, ms=64, maxit=60, wmc=2.96, prebuild=(), wl=True):
    from src.atlas6 import corridor_nd; c = CA4(name, ms, maxit, wmc); c.orig = corridor_nd.refine; [c.solver(n, wl) for n in prebuild]; corridor_nd.refine = c.refine; return c
