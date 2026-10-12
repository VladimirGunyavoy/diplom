import os, sys, importlib.util
import numpy as np
sys.path.insert(0, os.path.abspath('.'))
from src.algo import solve8 as S
spec = importlib.util.spec_from_file_location('old8', sys.argv[1]); O = importlib.util.module_from_spec(spec); spec.loader.exec_module(O)
rng = np.random.default_rng(0); X = rng.uniform([-2, -2], [2, 2], (4000, 2)); import time
for u in (1., -1., 0.):
    for dt in (3., .02):
        d = np.full(len(X), dt); t0 = time.perf_counter(); a = O.entry_time(X, u, d, .2); t1 = time.perf_counter(); b = S.entry_time(X, u, d, .2); t2 = time.perf_counter(); fa, fb = np.isfinite(a), np.isfinite(b)
        print('RES u=%+.0f dt=%g: конечность разл. %d из %d, max|Δ| %.2e | %.3f с (ДИ) %.3f с (поток)' % (u, dt, (fa != fb).sum(), fa.sum(), np.abs(a[fa & fb] - b[fa & fb]).max() if (fa & fb).any() else 0, t1 - t0, t2 - t1))
