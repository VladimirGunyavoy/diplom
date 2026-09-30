"""H1 n-звенный манипулятор, кинематика (q̇=u, |u_i|≤1, тор): точный эталон T*=max(0, max_i |Δq_i| − R)/1 до окна |q|∞<R. Слои — все ненулевые {−1,0,1}^n. replay_value. Запуск из v6: python3 tests/check_nd_kin.py n."""
import sys, time, itertools; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import *
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3; tau = 0.5; R = 0.2
LY = [np.array(s, float) for s in itertools.product((-1, 0, 1), repeat=n) if any(s)]
fl = lambda P, s, t: P + LY[s] * t
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
S = SysN(fl, len(LY), (np.pi,) * n, lambda P: np.max(np.abs(wr(P)), -1) < R, [np.zeros(n)], per=(2 * np.pi,) * n)
ref = lambda q: max(0.0, np.max(np.abs(wr(q))) - R)
rng = np.random.default_rng(0); Q = [rng.uniform(-np.pi, np.pi, n) for _ in range(10)]; E = [ref(q) for q in Q]
print('n=%d слоёв %d, эталон' % (n, len(LY)), np.round(E, 2), flush=True)
for rho in (0.15,):
  for NB, NF in ((100, 100), (300, 300), (1000, 1000)):
    t0 = time.time(); back = build_back(S, tau, NB, rho); r = []
    for x, e in zip(Q, E):
        V = replay_value(S, tau, x, NB, NF, rho, rho, back=back)[0]; r.append(V / e)
    print('rho %.2f NB %d NF %d: V/T*' % (rho, NB, NF), np.round(r, 2), '(%.0f с)' % (time.time() - t0), flush=True)
