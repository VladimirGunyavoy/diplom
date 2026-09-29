"""research: эталон для дифдрайва с ромбовым U (|v|/vmax + |ω|/ωmax ≤ 1): по Balkcom & Mason (IJRR 2002) оптимум — повороты на месте (T)
и прямые (G), до 5 отрезков. Здесь: (а) TGT — поворот-прямая-поворот (замкнутая формула, вперёд/назад), (б) TGTGT — численно (SLSQP,
многостарт). Если (б) заметно лучше (а) — эталон должен быть 5-отрезочным. vmax = ωmax = 1. Цель — (0,0,0)."""
import numpy as np, json
from scipy.optimize import minimize
wrap = lambda a: (a + np.pi) % (2*np.pi) - np.pi
def tgt(x, y, th):
    d = np.hypot(x, y); best = np.inf
    if d < 1e-12: return abs(wrap(th))
    phi = np.arctan2(-y, -x)                       # направление из старта в цель
    for back in (0, np.pi):                        # ехать передом или задом
        h = wrap(phi + back); best = min(best, abs(wrap(h - th)) + d + abs(wrap(0 - h)))
    return best
def pose(p, a):                                    # применить T(a) или G(s) к позе
    x, y, th = p
    return (x, y, th + a[1]) if a[0] == 'T' else (x + a[1]*np.cos(th), y + a[1]*np.sin(th), th)
def tgtgt(x, y, th, starts=40, rng=np.random.default_rng(0)):
    def end(z):
        p = (x, y, th)
        for k, val in zip('TGTGT', z): p = pose(p, (k, val))
        return p
    cost = lambda z: np.sum(np.sqrt(z**2 + 1e-9))
    cons = [{'type': 'eq', 'fun': lambda z: [end(z)[0], end(z)[1], np.sin(end(z)[2])]}]
    best = np.inf
    for _ in range(starts):
        r = minimize(cost, rng.uniform(-3, 3, 5), constraints=cons, method='SLSQP', options=dict(maxiter=300))
        e = end(r.x)
        if np.hypot(e[0], e[1]) < 1e-5 and abs(wrap(e[2])) < 1e-5: best = min(best, cost(r.x))
    return best
cases = [(1, 0, 0), (-1, 0, 0), (0, 0.3, 0), (0, 1, 0), (0, 0.3, np.pi/2), (1, 1, np.pi), (2, 0.5, 0.3), (0.2, 0.1, np.pi)]
out = []
for c in cases:
    a, b = tgt(*c), tgtgt(*c); out.append(dict(start=c, TGT=a, TGTGT=b, gain=a - b))
    print(c, f"TGT {a:.3f}  TGTGT {b:.3f}  выигрыш {a-b:+.3f}", flush=True)
json.dump(out, open(__file__.replace('.py', '.json'), 'w'), indent=1, default=float)
