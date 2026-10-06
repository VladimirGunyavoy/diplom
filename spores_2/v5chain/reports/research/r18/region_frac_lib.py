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
