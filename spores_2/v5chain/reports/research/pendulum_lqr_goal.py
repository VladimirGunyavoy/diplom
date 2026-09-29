"""research: клетка цели наверху маятника (θ̈ = −sin θ + u, |u| ≤ u_max) — LQR и её радиус.
φ = θ − π: φ̈ = sin φ + u. Линеаризация A = [[0,1],[1,0]], B = [0,1]ᵀ. Эллипс xᵀPx ≤ c без насыщения: c = u_max²/(K P⁻¹ Kᵀ).
Проверка: 400 точек на границе эллипса (и ×1.5, ×2), нелинейная замкнутая система с насыщением, RK4 dt=0.01, 30 с → в |x| < 1e-3?"""
import numpy as np, json
from scipy.linalg import solve_continuous_are
A = np.array([[0., 1.], [1., 0.]]); B = np.array([[0.], [1.]])
def sim(x, K, um, T=30.0, dt=0.01):
    f = lambda x: np.array([x[1], np.sin(x[0]) + np.clip(-(K @ x).item(), -um, um)])
    for _ in range(int(T/dt)):
        k1 = f(x); k2 = f(x+dt/2*k1); k3 = f(x+dt/2*k2); k4 = f(x+dt*k3); x = x + dt/6*(k1+2*k2+2*k3+k4)
        if np.abs(x).max() > 10: return False
    return np.hypot(*x) < 1e-3
out = {}
for um in (0.3, 0.5, 0.8):
    for R in (0.1, 1.0, 10.0):
        P = solve_continuous_are(A, B, np.eye(2), R*np.eye(1)); K = np.linalg.solve(R*np.eye(1), B.T @ P)
        c = um**2 / (K @ np.linalg.solve(P, K.T)).item()
        L = np.linalg.cholesky(np.linalg.inv(P/c))           # эллипс = L·единичный круг
        ang = np.linspace(0, 2*np.pi, 60, endpoint=False); ring = (L @ np.stack([np.cos(ang), np.sin(ang)])).T
        ok = {s: float(np.mean([sim(s*p, K, um) for p in ring])) for s in (1.0, 1.5, 2.0)}
        r_phi = float(np.sqrt(c / P[0, 0] * 1.0)) if False else float(np.sqrt(c*np.linalg.inv(P)[0, 0]))   # макс |φ| на эллипсе
        out[f"u={um} R={R}"] = dict(phi_max=r_phi, dphi_max=float(np.sqrt(c*np.linalg.inv(P)[1, 1])), ok_x1=ok[1.0], ok_x15=ok[1.5], ok_x2=ok[2.0])
        print(f"u={um} R={R}", {k: round(v, 3) for k, v in out[f'u={um} R={R}'].items()}, flush=True)
json.dump(out, open(__file__.replace('.py', '.json'), 'w'), indent=1)
