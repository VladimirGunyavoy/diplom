"""research hub-research-7: эталон DI для цены J = ∫(1+ρu²)dt, |u|≤1, свободное T, цель — квадрат |x|,|v|≤g.
ПМП: H = 1+ρu² + p1 v + p2 u, p1 = const, p2 линейна по t ⇒ u* = −sat(p2/2ρ) = sat(α+βt). Ищем (α,β,T) мультистартом SLSQP с терминальным условием."""
import numpy as np
from scipy.optimize import minimize

def traj(x0, v0, a, b, T, n=400):
    t = (np.arange(n) + .5) * T / n; u = np.clip(a + b * t, -1, 1); h = T / n
    v = v0 + np.concatenate([[0], np.cumsum(u * h)]); x = x0 + np.sum(v[:-1] * h + u * h * h / 2)
    return x, v[-1], np.sum((1 + 0 * u) * h) + 0, u, h

def J(p, x0, v0, rho):
    a, b, T = p; _, _, _, u, h = traj(x0, v0, a, b, T); return T + rho * np.sum(u * u) * h

def ocp_ref(x0, v0, rho, g=.1, Tmax=20.0):
    best = np.inf
    cons = [{'type': 'ineq', 'fun': (lambda p, i=i, sgn=sgn: g - sgn * traj(x0, v0, *p)[i])} for i in (0, 1) for sgn in (1, -1)]
    for a in np.linspace(-2, 2, 5):
        for b in (-2., -.5, .5, 2.):
            for T in (1., 3., 6.):
                r = minimize(J, [a, b, T], args=(x0, v0, rho), method='SLSQP', constraints=cons, bounds=[(-20, 20), (-20, 20), (.01, Tmax)], options=dict(maxiter=200, ftol=1e-10))
                if r.success or True:
                    x, v, _, _, _ = traj(x0, v0, *r.x)
                    if abs(x) <= g + 1e-6 and abs(v) <= g + 1e-6 and r.fun < best: best = r.fun
    return best

if __name__ == '__main__':
    import sys, json
    sys.path.insert(0, '/home/rl/claude-work/projects/spore/spores_2/v5chain/reports/research')
    from v7_faces_di import tstar_box
    for x0, v0 in ((1., 0.), (-1., .5), (.5, 1.2)):                       # проверка: ρ → 0 даёт ≈ T*
        print(x0, v0, 'rho=1e-3', round(ocp_ref(x0, v0, 1e-3), 4), 'T*', round(float(tstar_box(np.array([x0]), np.array([v0]), .1)[0]), 4), 'rho=.5', round(ocp_ref(x0, v0, .5), 4))
