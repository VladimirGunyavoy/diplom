"""b3: проверка эталона pend: время быстродействия по многодуговым управлениям (K дуг, u_k=.3*clip(c_k,-1,1), длительности d_k), дифф. эволюция. python pend_arcs.py <старт> [K]. Печатает лучшее T (первое попадание в ящик ±.1,±.1)."""
import sys, os, numpy as np
from scipy.optimize import differential_evolution
PI = np.pi; rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-PI, PI, 100), rq.uniform(-2, 2, 100)], 1); ref = np.load(os.environ.get('PEND_REF', 'pend_ref_best_u0.3.npy'))
i = int(sys.argv[1]); K = int(sys.argv[2]) if len(sys.argv) > 2 else 5; DT = .01; TM = 14.
def run(x):
    c, d = x[:K], x[K:]; th, w = Q[i]; t = 0.; best = 1e3
    for k in range(K):
        u = .3 * np.clip(c[k], -1, 1); n = max(int(d[k] / DT), 1); h = d[k] / n
        for _ in range(n):
            f = lambda a, b: (b, np.sin(a) + u)
            k1 = f(th, w); k2 = f(th + h / 2 * k1[0], w + h / 2 * k1[1]); k3 = f(th + h / 2 * k2[0], w + h / 2 * k2[1]); k4 = f(th + h * k3[0], w + h * k3[1])
            th += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); w += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]); t += h
            e = (th + PI) % (2 * PI) - PI
            if abs(e) <= .1 and abs(w) <= .1: return t
            best = min(best, max(abs(e), abs(w)))
    return 20. + 10 * best + t
r = differential_evolution(run, [(-1.5, 1.5)] * K + [(0., 4.)] * K, maxiter=int(os.environ.get('IT', 150)), popsize=25, seed=int(os.environ.get('SEED', 0)), tol=1e-8, polish=False, workers=1, updating='immediate')
print('RES старт %d K=%d seed %s: T=%.3f (T_ref %.3f)  c=%s d=%s' % (i, K, os.environ.get('SEED', 0), r.fun, ref[i], np.round(r.x[:K], 2), np.round(r.x[K:], 2)), flush=True)
