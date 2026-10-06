# Двунаправленный рост: шар цели {V <= R_g} общий, шары стартов {T_s <= (1+m) C_s/2}; R_g = max_s (1+m) C_s/2. Доля поля по Q стартам. DI k осей.
import numpy as np
from region_frac_lib import t2
rng = np.random.default_rng(4); M = 300000; m = .1
print("k n | Q=1  10  60 (доля поля: шары цели ∪ стартов, запас 10%) | только шары стартов при Q=60")
for k in (1, 2, 3, 4):
    P = rng.uniform(-2.5, 2.5, (M, 2*k)); Vp = np.max([t2(P[:, 2*i], P[:, 2*i+1], 0, 0) for i in range(k)], 0)
    F = np.zeros(M, bool); Rg = 0; out = {}
    for q in range(1, 61):
        s = rng.uniform(-2, 2, 2*k); Vs = max(t2(s[2*i], s[2*i+1], 0., 0.) for i in range(k))
        Ts = np.max([t2(s[2*i], s[2*i+1], P[:, 2*i], P[:, 2*i+1]) for i in range(k)], 0)
        F |= Ts <= (1+m)*Vs/2; Rg = max(Rg, (1+m)*Vs/2)
        if q in (1, 10, 60): out[q] = np.mean(F | (Vp <= Rg))
    print(k, 2*k, "|", "  ".join(f"{out[q]:.2e}" for q in (1, 10, 60)), "|", f"{F.mean():.2e}", flush=True)
