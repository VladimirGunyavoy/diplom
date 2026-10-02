"""research hub-research-7: спектр управлений и обобщённая цена на клетках из точных траекторий (DI). Слово пользователя 2026-10-02:
спектр нужен у финиша и при выходе на кривую переключения; при штрафе за управление bang-bang не оптимален — проверить.
Клетки — по bang-слоям ±1 (сетка для V*), Беллман и агент — по набору U: V*(узел) = min_{u∈U} [Δt(1+ρ|u|) [L1] + V*(φ_u(узел, Δt))].
Эталон для ρ>0 — мелкая полулагранжева сетка HJB (тот же оператор, сетка 361², 41 значение u)."""
import numpy as np, sys, json, time
sys.path.insert(0, ''+__import__("os").path.dirname(__import__("os").path.abspath(__file__))+'')
from v7_spore_di import Cover, BIG
from v7_faces_di import tstar_box

def solve_U(Cv, U, rho, tol=1e-9, it=20000):
    P = np.stack([Cv.X.ravel(), Cv.Vv.ravel()], 1); dt = np.repeat(Cv.C[:, 4], Cv.m * Cv.nt) * (Cv.tgn[1] - Cv.tgn[0])
    lay = np.repeat(Cv.C[:, 2], Cv.m * Cv.nt); last = (np.arange(Cv.N) % Cv.nt) == Cv.nt - 1; nxt = np.minimum(np.arange(Cv.N) + 1, Cv.N - 1); F = []
    for u in U:
        qi, k, idx, w = Cv.pairs(Cv.phi(P, u, dt)); o = np.argsort(qi, kind='stable'); qi, idx, w = qi[o], idx[o], w[o]; uq, st = np.unique(qi, return_index=True)
        F.append((u, uq, st, idx, w, (lay == u) & ~last))
    G = Cv.goal(P[:, 0], P[:, 1]); V = np.full(Cv.N, BIG); V[G] = 0
    for n in range(it):
        Vn = np.full(Cv.N, BIG)
        for u, uq, st, idx, w, same in F:
            c = np.full(Cv.N, BIG); c[uq] = np.minimum.reduceat(np.sum(w * V[idx], -1), st); c = np.where(same, V[nxt], c); Vn = np.minimum(Vn, dt * (1 + rho * abs(u)) + c)
        Vn[G] = 0; Vn = np.minimum(Vn, BIG); d = np.max(np.abs(Vn - V)); V = Vn
        if d < tol: break
    Cv.V = V; Cv.iters = n; return V

def rollout_U(Cv, Q0, U, rho, dt, tmax=30.0):
    """J = ∫(1+ρ|u|) [L1]dt до цели; число переключений (|Δu|>.5), полная вариация u."""
    q = np.array(Q0, float); n = len(q); J = np.zeros(n); T = np.zeros(n); done = Cv.goal(q[:, 0], q[:, 1]); dead = np.zeros(n, bool); sw = np.zeros(n); tv = np.zeros(n); pu = np.full(n, np.nan)
    for _ in range(int(tmax / dt)):
        act = np.nonzero(~done & ~dead)[0]
        if not len(act): break
        c = np.stack([dt * (1 + rho * abs(u)) + Cv.Vstar(Cv.phi(q[act], u, dt)) for u in U], 1); b = np.argmin(c, 1); u = np.asarray(U)[b]
        nod = ~np.isfinite(c[np.arange(len(act)), b]); dead[act[nod]] = True; act, u = act[~nod], u[~nod]
        p = pu[act]; has = np.isfinite(p); sw[act] += has & (np.abs(u - np.where(has, p, u)) > .5); tv[act] += np.where(has, np.abs(u - np.where(has, p, u)), 0); pu[act] = u
        for _s in range(4):
            h = dt / 4; x, v = q[act, 0], q[act, 1]; q[act, 0] = x + v * h + u * h * h / 2; q[act, 1] = v + u * h; T[act] += h; J[act] += h * (1 + rho * abs(u))
            g = Cv.goal(q[act, 0], q[act, 1]); done[act[g]] = True; act, u = act[~g], u[~g]
    J[~done] = np.inf; T[~done] = np.inf; return J, T, sw, tv

def hjb_grid(rho, L=2.5, n=361, U=np.linspace(-1, 1, 41), goal=.1, dt=.02, tol=1e-8, it=20000):
    """Эталон: полулагранжева сетка, V(x) = min_u [dt(1+ρ|u|) [L1] + V(φ_u(x,dt))], билинейно; BIG-конечное."""
    g = np.linspace(-L, L, n); hg = g[1] - g[0]; X, Vv = np.meshgrid(g, g, indexing='ij'); X, Vv = X.ravel(), Vv.ravel(); N = len(X)
    G = (np.abs(X) <= goal) & (np.abs(Vv) <= goal); V = np.full(N, BIG); V[G] = 0; F = []
    for u in U:
        x1 = X + Vv * dt + u * dt * dt / 2; v1 = Vv + u * dt; out = (np.abs(x1) > L) | (np.abs(v1) > L)
        fx = np.clip((x1 + L) / hg, 0, n - 1 - 1e-9); fv = np.clip((v1 + L) / hg, 0, n - 1 - 1e-9); i = fx.astype(int); j = fv.astype(int); a = fx - i; b = fv - j
        F.append((dt * (1 + rho * abs(u)), i * n + j, a, b, out))
    for k in range(it):
        Vn = np.full(N, BIG)
        for c0, base, a, b, out in F:
            val = (1 - a) * (1 - b) * V[base] + (1 - a) * b * V[base + 1] + a * (1 - b) * V[base + n] + a * b * V[base + n + 1]
            Vn = np.minimum(Vn, np.where(out, BIG, c0 + val))
        Vn[G] = 0; d = np.max(np.abs(Vn - V)); V = Vn
        if d < tol: break
    def at(Q):
        fx = np.clip((Q[:, 0] + L) / hg, 0, n - 1 - 1e-9); fv = np.clip((Q[:, 1] + L) / hg, 0, n - 1 - 1e-9); i = fx.astype(int); j = fv.astype(int); a = fx - i; b = fv - j; base = i * n + j
        return (1 - a) * (1 - b) * V[base] + (1 - a) * b * V[base + 1] + a * (1 - b) * V[base + n] + a * b * V[base + n + 1]
    return at, k

if __name__ == '__main__':
    rng = np.random.default_rng(1); Q = rng.uniform(-1.5, 1.5, (300, 2)); res = []
    Cv = Cover(tau=.4, r=.1, seeds=6000); dt = Cv.dt0
    sets = {'bang': (-1.0, 1.0), 'tri': (-1.0, 0.0, 1.0), 'spec11': tuple(np.linspace(-1, 1, 11))}
    for rho in [float(a) for a in sys.argv[1:]] or (0.0, 0.5, 2.0):
        ref, kref = hjb_grid(rho) if rho > 0 else (None, 0)
        Jref = ref(Q) if rho > 0 else tstar_box(Q[:, 0], Q[:, 1], .1)
        for nm, U in sets.items():
            t0 = time.time(); solve_U(Cv, U, rho); J, T, sw, tv = rollout_U(Cv, Q, U, rho, dt); ok = np.isfinite(J) & (Jref > .05) & (Jref < BIG / 2); r = J[ok] / Jref[ok]
            out = dict(rho=rho, U=nm, reach=round(float(np.isfinite(J).mean()), 3), J_ref=dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), min=round(float(r.min()), 4), max=round(float(r.max()), 3)),
                       sw_med=float(np.median(sw[ok])), tv_med=round(float(np.median(tv[ok])), 2), iters=Cv.iters, ref_iters=kref, sec=round(time.time() - t0))
            print(json.dumps(out), flush=True); res.append(out)
    json.dump(res, open(''+__import__("os").path.dirname(__import__("os").path.abspath(__file__))+'/v7_spore_spec.json', 'w'), indent=1)
