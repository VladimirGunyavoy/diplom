"""research: лечит ли клетка цели зависание агента на крупных клетках? (продолжение di_gradV_switch.py)
Внутри круга радиуса R_goal вокруг цели — точное правило (кривая переключения x = -v|v|/2), вне — ∇V из 5 точек клетки."""
import numpy as np, json
from di_gradV_switch import Ts, flow, build, starts

def run(x0, v0, C, G, R_goal, dt=0.01, tol=0.05):
    x, v, t, T0 = x0, v0, 0.0, Ts(x0, v0)
    while t < 3*T0 + 1:
        if np.hypot(x, v) < tol: return t/T0, True
        if np.hypot(x, v) < R_goal:
            s = x + v*abs(v)/2; u = -np.sign(s) if abs(s) > 1e-9 else -np.sign(v)
        else:
            i = np.argmin((C[:, 0]-x)**2 + (C[:, 1]-v)**2); u = -np.sign(G[i, 1]) or 1.0
        x, v = flow(x, v, u, dt); t += dt
    return t/T0, False

out = {}
for h in (0.4, 0.2):
    C, G = build(h, r=h/2, tau=h/2)
    for k in (1, 2, 3):
        res = [run(x, v, C, G, R_goal=k*h) for x, v in starts]
        ok = [r for r, a in res if a]
        out[f"h={h} R={k}h"] = dict(arrived=len(ok), ratio_mean=float(np.mean(ok)) if ok else None, ratio_max=float(np.max(ok)) if ok else None)
        print(f"h={h} R={k}h", out[f"h={h} R={k}h"], flush=True)
json.dump(out, open(__file__.replace('.py', '.json'), 'w'), indent=1)
