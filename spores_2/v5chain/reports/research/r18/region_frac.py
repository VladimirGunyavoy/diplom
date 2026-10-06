# Доля объёма поля, которую должен покрыть атлас для ОДНОГО запроса s → 0 (DI, k осей, n = 2k).
# Точные формулы DI по осям: V(p) = max_i T_i(p→0); прямое время s→p — нижняя оценка max_i T_i(s→p) (гипотеза: щель
# времён при v_конца ≠ 0 игнорируется, область — надмножество). Поле [-2.5, 2.5]^n, старты [-2, 2]^n.
import numpy as np, sys
def t2(x0, v0, x1, v1):
    """мин. время DI (|u|≤1) из (x0,v0) в (x1,v1), векторно"""
    d = x1 - x0; s = (v0**2 + v1**2) / 2
    a = d + s; vm = np.sqrt(np.maximum(a, 0)); ok = (a >= 0) & (vm >= np.maximum(v0, v1) - 1e-12)
    tA = np.where(ok, 2*vm - v0 - v1, np.inf)
    b = s - d; wm = -np.sqrt(np.maximum(b, 0)); ok = (b >= 0) & (wm <= np.minimum(v0, v1) + 1e-12)
    tB = np.where(ok, v0 + v1 - 2*wm, np.inf)
    return np.minimum(tA, tB)
rng = np.random.default_rng(1); M = int(sys.argv[1]) if len(sys.argv) > 1 else 400000; NS = 20
print("k n | V<=Vs (обратный Дейкстра) | двунапр. V<=Vs/2 или Ts<=Vs/2 | эллипс eps 0 / .05 / .2 / .5   (доли поля, медиана по стартам)")
for k in (1, 2, 3, 4):
    P = rng.uniform(-2.5, 2.5, (M, 2*k)); Vp = np.max([t2(P[:, 2*i], P[:, 2*i+1], 0, 0) for i in range(k)], 0)
    R = []
    for j in range(NS):
        s = rng.uniform(-2, 2, 2*k); Vs = max(t2(s[2*i], s[2*i+1], 0., 0.) for i in range(k))
        Ts = np.max([t2(s[2*i], s[2*i+1], P[:, 2*i], P[:, 2*i+1]) for i in range(k)], 0)
        r = [np.mean(Vp <= Vs), np.mean((Vp <= Vs/2) | (Ts <= Vs/2))] + [np.mean(Ts + Vp <= (1+e)*Vs) for e in (1e-3, .05, .2, .5)]
        R.append(r)
    m = np.median(R, 0)
    print(f"{k} {2*k} | {m[0]:.2e} | {m[1]:.2e} | " + " / ".join(f"{x:.2e}" for x in m[2:]), flush=True)
