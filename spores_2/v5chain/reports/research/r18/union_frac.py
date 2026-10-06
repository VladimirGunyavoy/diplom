# Объединение эллипсов {T_s + V <= (1+eps) V(s)} по Q стартам: доля поля (DI, k осей). Гипотеза: нижняя оценка T_s = max по осям.
import numpy as np
from region_frac_lib import t2
rng = np.random.default_rng(2); M = 300000
print("k n eps | Q=1  Q=10  Q=60  Q=300 (доля поля в объединении)")
for k in (2, 3, 4):
    P = rng.uniform(-2.5, 2.5, (M, 2*k)); Vp = np.max([t2(P[:, 2*i], P[:, 2*i+1], 0, 0) for i in range(k)], 0)
    for eps in (.05, .2):
        U = np.zeros(M, bool); out = {}
        for q in range(1, 301):
            s = rng.uniform(-2, 2, 2*k); Vs = max(t2(s[2*i], s[2*i+1], 0., 0.) for i in range(k))
            Ts = np.max([t2(s[2*i], s[2*i+1], P[:, 2*i], P[:, 2*i+1]) for i in range(k)], 0)
            U |= Ts + Vp <= (1+eps)*Vs
            if q in (1, 10, 60, 300): out[q] = U.mean()
        print(k, 2*k, eps, "|", "  ".join(f"{out[q]:.2e}" for q in (1, 10, 60, 300)), flush=True)
