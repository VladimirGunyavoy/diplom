"""ENV5 круг: эталоны OCP (CasADi/IPOPT) на редуцированной модели точки 5D (как у спор): (а) все 8 ворот сразу (многофазная, свободная скорость — нижняя граница), (б) ворота за воротами с теми же условиями на воротах (конус/VN), как наш жадный."""
import os, sys, time, json, numpy as np, casadi as ca
sys.argv = ['x', '10', '10', '3']; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spores_pcircle as P
M_, A_, CD, FM, WALL = P.M_, P.A_, P.CD, P.FM, P.WALL; PADO = float(os.environ.get('PADO', .07)); GR = P.RG - float(os.environ.get('GMARG', .05)); NI = int(os.environ.get('NI', 30))
def f(x, u):
    c, s = ca.cos(x[2]), ca.sin(x[2]); F = FM * u[0]; w = u[1]
    return ca.vertcat(x[3], x[4], w, (F * c - CD * x[3]) / M_ + A_ * w * w * c, (F * s - CD * x[4]) / M_ + A_ * w * w * s)
def rk4(x, u, h):
    k1 = f(x, u); k2 = f(x + h / 2 * k1, u); k3 = f(x + h / 2 * k2, u); k4 = f(x + h * k3, u); return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
def solve(phases, x0, T0s, cone=False, last_next=None, ng_list=None):
    """phases: список индексов ворот; один NLP; cone — условия на скорости на воротах (v·e ≥ CONE|v| и VN)."""
    opti = ca.Opti(); X = []; Us = []; Ts = []; obj = 0; xs = ca.DM(x0)
    prev = opti.parameter(5); opti.set_value(prev, x0)
    for pi, gi in enumerate(phases):
        T = opti.variable(); opti.subject_to(T >= .3); opti.set_initial(T, T0s[pi]); Ts.append(T); obj += T
        Xk = opti.variable(5, NI + 1); Uk = opti.variable(2, NI); X.append(Xk); Us.append(Uk)
        opti.subject_to(Xk[:, 0] == prev); h = T / NI
        for k in range(NI):
            opti.subject_to(Xk[:, k + 1] == rk4(Xk[:, k], Uk[:, k], h))
            opti.subject_to(opti.bounded(-1, Uk[0, k], 1)); opti.subject_to(opti.bounded(-P.WS, Uk[1, k], P.WS))
            opti.subject_to(opti.bounded(-(WALL - PADO), Xk[0, k + 1], WALL - PADO)); opti.subject_to(opti.bounded(-(WALL - PADO), Xk[1, k + 1], WALL - PADO))
        g = P.GATES[gi]; xe = Xk[:, NI]; opti.subject_to((xe[0] - g[0]) ** 2 + (xe[1] - g[1]) ** 2 <= GR ** 2)
        if cone:
            e = P.GATES[(gi + 1) % P.NG] - g; e = e / np.hypot(*e); sp = ca.sqrt(xe[3] ** 2 + xe[4] ** 2 + 1e-9)
            opti.subject_to(xe[3] * e[0] + xe[4] * e[1] >= P.CONE * sp)
            for a in (0, 1):
                if abs(g[a]) > .9: opti.subject_to(np.sign(g[a]) * xe[3 + a] <= P.VN)
        # начальное приближение: прямая к воротам
        pa = np.array(x0[:2]) if pi == 0 else P.GATES[phases[pi - 1]]
        for k in range(NI + 1):
            a = k / NI; opti.set_initial(Xk[:, k], [pa[0] + a * (g[0] - pa[0]), pa[1] + a * (g[1] - pa[1]), 0, 0, 0])
        opti.set_initial(Uk, 0); prev = xe
    opti.minimize(obj); opti.solver('ipopt', dict(print_time=False), dict(print_level=0, max_iter=800, tol=1e-6))
    try: sol = opti.solve(); ok = True
    except RuntimeError: sol = opti.debug; ok = False
    return ok, [float(sol.value(T)) for T in Ts], np.array(sol.value(X[-1][:, NI])), sol, X, Us
if __name__ == '__main__':
    th0 = P.ang[0] + np.pi / 2; x0 = [P.GATES[0, 0], P.GATES[0, 1], th0, 0., 0.]; out = {}
    t0 = time.perf_counter(); ph = [(j + 1) % P.NG for j in range(P.NG)]
    ok, Ts, xe, *_ = solve(ph, x0, [1.0] * P.NG, cone=False); out['all_gates'] = dict(ok=ok, T=float(sum(Ts)), phases=Ts, wall=time.perf_counter() - t0); print('OCP все 8 ворот сразу (скорость свободна):', ok, round(sum(Ts), 3), np.round(Ts, 2))
    t0 = time.perf_counter(); x = np.array(x0); Tg = []; oks = []
    for j in range(P.NG):
        gi = (j + 1) % P.NG; ok, Ts, xe, *_ = solve([gi], x, [1.0], cone=True); oks.append(ok); Tg.append(Ts[0]); x = xe
        print(' ворота', gi, ok, round(Ts[0], 3), np.round(xe, 3))
    out['by_gate'] = dict(ok=all(oks), T=float(sum(Tg)), phases=Tg, wall=time.perf_counter() - t0); print('OCP ворота за воротами (конус/VN):', all(oks), round(sum(Tg), 3))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'pcircle_ocp.json'), 'w'), indent=1)
