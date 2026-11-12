"""research-37: точный эталон ДИ 4D в коробку |x|,|v| <= RHO (замена SLSQP из r17/di4_ref.py, который на ~10% стартов не сходился
и возвращал время в точку). Ось: T_box(x0,v0) = min по точкам ГРАНИЦЫ коробки (вход в коробку — через границу) замкнутой формулы
DI «точка → точка» (одно переключение, |u| <= 1); граница — 4·NB точек + локальное уточнение. T* = max(T1, T2) (r17: множество времён оси — [Ti*, ∞)).
Проверка: на стенде r22/di4_ref_60_rho35.npy. Запуск: python3 di4_ref_exact.py [check|NS SEED]"""
import sys, numpy as np
from scipy.optimize import minimize_scalar
RHO = .35; NB = 2000

def tpp(x0, v0, xf, vf):
    """min время DI из (x0,v0) в (xf,vf), |u|<=1; векторно по xf,vf"""
    best = np.full(np.broadcast(xf, vf).shape, np.inf); dx = xf - x0; s = (v0 * v0 + vf * vf) / 2
    a = s + dx; vs = np.sqrt(np.maximum(a, 0)); ok = (a >= 0) & (vs >= np.maximum(v0, vf) - 1e-12); best = np.where(ok, np.minimum(best, 2 * vs - v0 - vf), best)   # +1 затем −1
    b = s - dx; vs2 = -np.sqrt(np.maximum(b, 0)); ok = (b >= 0) & (vs2 <= np.minimum(v0, vf) + 1e-12); best = np.where(ok, np.minimum(best, v0 + vf - 2 * vs2), best)   # −1 затем +1
    return best

def bnd(t):
    """t ∈ [0,4): точка на границе квадрата [-R,R]^2"""
    k = np.floor(t).astype(int) % 4; f = t - np.floor(t); R = RHO; c = -R + 2 * R * f
    x = np.select([k == 0, k == 1, k == 2, k == 3], [c, np.full_like(c, R), -c, np.full_like(c, -R)])
    v = np.select([k == 0, k == 1, k == 2, k == 3], [np.full_like(c, -R), c, np.full_like(c, R), -c]); return x, v

def tbox(x0, v0):
    if abs(x0) <= RHO and abs(v0) <= RHO: return 0.
    t = np.arange(4 * NB) / NB; xf, vf = bnd(t); T = tpp(x0, v0, xf, vf); i = int(np.argmin(T)); best = T[i]
    r = minimize_scalar(lambda s: float(tpp(x0, v0, *bnd(np.array([s % 4])))[0]), bounds=(t[i] - 1. / NB, t[i] + 1. / NB), method='bounded', options=dict(xatol=1e-10))
    return min(best, r.fun)

def tstar(q): return tbox(q[0], q[2]), tbox(q[1], q[3])

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        a = np.load(sys.argv[2]); T = np.array([max(tstar(q)) for q in a[:, :4]]); d = T - a[:, 6]
        print('стенд: |ΔT*| max %.2e, новое ≤ старого везде: %s, max(старое−новое) %.3e' % (np.abs(d).max(), bool((d <= 1e-6).all()), (-d).max()))
    else:
        NS, SEED = int(sys.argv[1]), int(sys.argv[2]); Q = np.random.default_rng(SEED).uniform(-2, 2, (NS, 4))
        TT = np.array([tstar(q) for q in Q]); Ts = TT.max(1)
        np.save('di4_ref_%d_rho35_s%d.npy' % (NS, SEED), np.c_[Q, TT, Ts]); print('T*: mean %.3f min %.3f max %.3f' % (Ts.mean(), Ts.min(), Ts.max()))
