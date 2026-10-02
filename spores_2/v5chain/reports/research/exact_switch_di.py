"""research hub-research-7: точный момент переключения внутри шага (слово пользователя: «не размазывание переключений»), DI.
Агент в слое k; D(s) = [h + V(φ_k(p_s,h))] − [h + V(φ_k'(p_s,h))], p_s = φ_k(q,s). Если D(0) ≤ 0 < D(dt) — бисекция s* (12 итер.), до s* слой k, потом k'.
Управление чистое ±1. V — V* клеток (v7_spore_di, τ .4, r .1) или точное T* (проверка принципа)."""
import numpy as np, json, sys, time
sys.path.insert(0, '.')
from v7_spore_di import Cover
from v7_faces_di import tstar_box
def phi(Q, u, h): return np.stack([Q[:, 0] + Q[:, 1] * h + u * h * h / 2, Q[:, 1] + u * h], 1)
def rollout(Vf, Q0, dt=.06, h=.06, rho=.1, tmax=30, exact=True):
    q = np.array(Q0, float); n = len(q); T = np.zeros(n); k = np.zeros(n); done = (np.abs(q) <= rho).all(1); sw = np.zeros(n); dead = np.zeros(n, bool)
    c0 = np.stack([Vf(phi(q, u, h)) for u in (1., -1.)], 1); k = np.where(c0[:, 0] <= c0[:, 1], 1., -1.)
    def D(p, kk): return Vf(phi(p, kk, h)) - Vf(phi(p, -kk, h))
    for _ in range(int(tmax / dt)):
        a = np.nonzero(~done & ~dead)[0]
        if not len(a): break
        qa, ka = q[a], k[a]; d0 = D(qa, ka)
        if exact:
            d1 = D(phi(qa, ka, dt), ka); s = np.full(len(a), dt)
            now = d0 > 0; ka = np.where(now, -ka, ka); sw[a[now]] += 1                 # уже хуже — переключиться сразу
            cross = ~now & (d1 > 0)                                                    # станет хуже внутри шага — найти момент
            lo, hi = np.zeros(len(a)), np.full(len(a), dt)
            for _b in range(12):
                mid = (lo + hi) / 2; dm = D(phi(qa, ka, mid), ka); hi = np.where(cross & (dm > 0), mid, hi); lo = np.where(cross & (dm <= 0), mid, lo)
            s = np.where(cross, hi, dt)
        else:
            now = d0 > 0; ka = np.where(now, -ka, ka); sw[a[now]] += 1; cross = np.zeros(len(a), bool); s = np.full(len(a), dt)
        fin = ~np.isfinite(d0); dead[a[fin]] = True
        for part in (0, 1):                                                            # [0,s] слоем ka, затем [s,dt] слоем −ka (если cross)
            hh = s if part == 0 else np.where(cross, dt - s, 0.); uu = ka if part == 0 else -ka
            for _j in range(4):
                g = (np.abs(qa) <= rho).all(1); step = np.where(g, 0., hh / 4); qa = phi(qa, uu, step) if False else np.stack([qa[:, 0] + qa[:, 1] * step + uu * step * step / 2, qa[:, 1] + uu * step], 1); T[a] += step
        sw[a[cross]] += 1; ka = np.where(cross, -ka, ka); q[a], k[a] = qa, ka; done[a] |= (np.abs(qa) <= rho).all(1)
    T[~done] = np.inf; return T, sw
if __name__ == '__main__':
    rng = np.random.default_rng(1); Q = rng.uniform(-1.5, 1.5, (300, 2)); Ts = tstar_box(Q[:, 0], Q[:, 1], .1); ok0 = Ts > .05
    Texact = lambda P: tstar_box(P[:, 0], P[:, 1], .1, n=801)
    Cv = Cover(tau=.4, r=.1, seeds=6000); Cv.solve()
    for nm, Vf in (('T* точное', Texact), ('V* клеток', Cv.Vstar)):
        for ex in (False, True):
            t0 = time.time(); T, sw = rollout(Vf, Q, exact=ex); ok = ok0 & np.isfinite(T); r = T[ok] / Ts[ok]
            print(json.dumps(dict(V=nm, switch='точный момент' if ex else 'по шагам dt', reach=round(float(np.isfinite(T).mean()), 3), T_Tstar=dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), max=round(float(r.max()), 3)),
                  sw_med=float(np.median(sw[ok])), sec=round(time.time() - t0)), ensure_ascii=False), flush=True)
