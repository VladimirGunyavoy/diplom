"""r17: общий финиш стрельбой для любой системы growN-вида: <= NA дуг из вершин U, поток rk4 (шаг <= HS), длительности SLSQP, конец в коробке |wrap(y)| <= .9*RHOV.
shoot(y, f, US, RHOV, wrapy, tmax) -> (T, topology) или (inf, None). Проверка: дд — против замкнутых формул (finish.py)."""
import numpy as np, itertools
from scipy.optimize import minimize
HS = .02
def flow(y, u, t, f):
    n = max(1, int(np.ceil(t / HS))); h = t / n; u = np.asarray(u, float)
    for _ in range(n):
        k1 = f(y, u); k2 = f(y + h / 2 * k1, u); k3 = f(y + h / 2 * k2, u); k4 = f(y + h * k3, u); y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y
def shoot(y, f, US, RHOV, wrapy, tmax, NA=3, inits=2):
    best = (np.inf, None); R = .9 * np.asarray(RHOV)
    for tp in (tp for n in range(1, NA + 1) for tp in itertools.product(range(len(US)), repeat=n) if all(tp[i] != tp[i + 1] for i in range(n - 1))):
        def end(d):
            z = np.array(y, float)
            for k, dt in zip(tp, d): z = flow(z, US[k], max(dt, 0.), f)
            return wrapy(z)
        cons = [{"type": "ineq", "fun": lambda d: R - np.abs(end(d))}]
        for d0 in [np.full(len(tp), tmax / (2 * len(tp))), np.full(len(tp), tmax / len(tp))][:inits]:
            r = minimize(lambda d: d.sum(), d0, method="SLSQP", bounds=[(0, tmax)] * len(tp), constraints=cons, options=dict(maxiter=100, ftol=1e-9))
            if r.success and np.all(np.abs(end(r.x)) <= np.asarray(RHOV)) and r.x.sum() < best[0]: best = (r.x.sum(), tp)
    return best
if __name__ == "__main__":
    import sys, os, time; sys.path.insert(0, os.path.expanduser("~/spore_v5/r17/v7/src/cells7")); import grow3 as G
    sys.path.insert(0, os.path.dirname(__file__)); import finish_cf as FC
    rng = np.random.default_rng(5); Y = np.c_[rng.uniform(-.8, .8, (12, 2)), rng.uniform(-1, 1, 12)]; out = []
    for y in Y:
        t0 = time.time(); a = FC.shoot(y, 2.5); t1 = time.time(); b = shoot(y, G.f, G.US, np.full(3, G.RHO), G.wrapy, 2.5); t2 = time.time()
        out.append((a[0], b[0])); print("closed %.4f %s | rk4 %.4f %s | %.1fs %.1fs" % (a[0], a[1], b[0], b[1], t1 - t0, t2 - t1), flush=True)
    o = np.array(out); m = np.isfinite(o).all(1); print("rk4/closed mean %.5f max %.5f, both found %d/12" % ((o[m, 1] / o[m, 0]).mean(), (o[m, 1] / o[m, 0]).max(), m.sum()))
