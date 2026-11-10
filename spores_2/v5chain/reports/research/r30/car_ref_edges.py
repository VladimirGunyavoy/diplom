"""research-30 (задача senior-1): доля времени эталона статик-кара (r21 OCP) на краю дисков (зазор < 1 см) и на v = v_max; плюс насыщение |a|, |ω|.
Вход: папка ref_car_N* (car_q<i>.npy = [T, U(N×2)]); прокатка rk4 dt .002, u кусочно-постоянно на N отрезках (как в ref_car.py)."""
import sys, os, numpy as np
import types; sys.modules.setdefault("casadi", types.ModuleType("casadi"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'r21'))
from ref_car import starts, f_np, HZ, HR, GOAL, GR, VH, AM, WM
D = sys.argv[1]; ref = np.load(os.path.join(os.path.dirname(__file__), '..', 'r21', 'car_ref20.npy')); S = starts(20)
rows = []
for i in range(20):
    fn = os.path.join(D, 'car_q%d.npy' % i)
    if not os.path.exists(fn): continue
    a = np.load(fn); T = a[0]; U = a[1:].reshape(-1, 2); N = len(U); x = S[i].copy(); h = .002; n = int(np.ceil(T / h)); h = T / n
    clr = []; vs = []; ua = []
    for k in range(n):
        u = U[min(int(k * h / (T / N)), N - 1)]
        k1 = f_np(x, u); k2 = f_np(x + h/2*k1, u); k3 = f_np(x + h/2*k2, u); k4 = f_np(x + h*k3, u); x = x + h/6*(k1+2*k2+2*k3+k4)
        clr.append((np.linalg.norm(HZ - x[:2], axis=1) - HR).min()); vs.append(x[3]); ua.append(u)
    clr = np.array(clr); vs = np.array(vs); ua = np.array(ua)
    rows.append((i, T, ref[i], np.mean(clr < .01), clr.min(), np.mean(vs > VH - .01), np.mean(np.abs(ua[:, 0]) > AM - 1e-3), np.mean(np.abs(ua[:, 1]) > WM - 1e-3), round(float(np.linalg.norm(x[:2] - GOAL) - GR), 3)))
print('старт | T | T_ref(car_ref20) | доля t: зазор<1см | мин. зазор | доля t: v>vmax-.01 | |a|=AM | |ω|=WM | dist-GR в конце')
for r in rows: print(' | '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in r))
R = np.array([r[3:8] for r in rows], float); print('медиана:', ' '.join(f'{v:.3f}' for v in np.median(R, 0)), '| среднее:', ' '.join(f'{v:.3f}' for v in R.mean(0)))
