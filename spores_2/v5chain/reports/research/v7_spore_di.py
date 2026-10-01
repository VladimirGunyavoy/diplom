"""research hub-research-7: v7 клетки из точных траекторий споры + гало 10% + свободное переключение (идея пользователя 2026-10-02), DI |u|<=1.
Клетка слоя u: спора c, нормальный отрезок s ∈ [-r_h, r_h] (r_h = 1.1 r) ⟂ полю в c, клоны y(s) = c + s·n интегрируются точно по t ∈ [-.1τ, 1.1τ]
(ядро |s|<=r, 0<=t<=τ; гало 10% по обеим осям). Узлы (s_j, t_i) — опорные точки; внутри — интерполяция по (s,t) (для DI поток точный).
V в узле = min_k [Δt + V*(φ_k(узел, Δt))] — монотонная полулагранжева схема (первая версия с переходом «в ту же точку другой клетки» за 0 времени давала диффузию поперёк траекторий).
Неизвестное = BIG конечное (урок маятника). Агент: точное движение, каждые dt — слой argmin [dt + V*(φ_k(q,dt))] (свободное переключение).
Покрытие: новая клетка — только если спора вне ядер клеток своего слоя."""
import numpy as np, sys, json, time
sys.path.insert(0, '/home/rl/claude-work/projects/spore/spores_2/v5chain/reports/research')
from v7_faces_di import tstar_box

BIG = 1e3

class Cover:
    def __init__(s, L=2.5, tau=.4, r=.08, m=7, nt=9, halo=.1, seeds=6000, rho=.1, seed=0, ovl=None, fill=True):
        s.L, s.tau, s.r, s.rh, s.m, s.nt, s.rho = L, tau, r, r * (1 + halo), m, nt, rho
        s.h = halo; s.sgn = np.linspace(-1, 1, m); s.tgn = np.linspace(-halo, 1 + halo, nt)        # сетки в долях r_h и τ (у каждой клетки свои r, τ)
        s.sg = s.sgn * s.rh; s.tg = s.tgn * tau; s.dt0 = s.tg[1] - s.tg[0]
        rng = np.random.default_rng(seed); C = []                                   # (x0, v0, u, r, τ)
        sc, tc = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(0, 1, 5)); sc, tc = sc.ravel(), tc.ravel()
        def corepts(c, u, rr, tt):
            fn = np.sqrt(c[1]**2 + 1); S_ = sc * rr; T_ = tc * tt; xs = c[0] - S_ * u / fn; vs = c[1] + S_ * c[1] / fn; return np.stack([xs + vs * T_ + u * T_ * T_ / 2, vs + u * T_], 1)
        for u in (1.0, -1.0):
            mine = []
            for c in rng.uniform(-L, L, (seeds, 2)):
                if mine:
                    A = np.array(mine)
                    if ovl is None:
                        _, _, ok = s._locate(A, c[None, :], core=True)
                        if ok.any(): continue
                    else:                                                            # доля ядра кандидата в чужих ядрах своего слоя ≤ ovl
                        _, _, ok = s._locate(A, corepts(c, u, r, tau), core=True)
                        if ok.any(1).mean() > ovl: continue
                mine.append((c[0], c[1], u, r, tau))
            s.nfill = 0
            if fill and ovl is not None:                                             # дыра (вне ядра+гало) — клетка в эту точку, ПОДРЕЗАННАЯ до перекрытия ≤ ovl
                for _ in range(400):
                    P = rng.uniform(-L, L, (4000, 2)); _, _, okf = s._locate(np.array(mine), P); hole = ~okf.any(1)
                    if hole.mean() < 2e-3: break
                    for c in P[hole][:50]:
                        A = np.array(mine); _, _, ok = s._locate(A, c[None, :])
                        if ok.any(): continue
                        for f in (1.0, .7, .5, .35, .25):
                            _, _, ok = s._locate(A, corepts(c, u, r * f, tau * f), core=True)
                            if ok.any(1).mean() <= ovl: break
                        mine.append((c[0], c[1], u, r * f, tau * f)); s.nfill += 1
            C += mine
        s.C = np.array(C); s.K = len(s.C)
        # узлы
        x0, v0, u = s.C[:, 0, None, None], s.C[:, 1, None, None], s.C[:, 2, None, None]; rk, tk = s.C[:, 3, None, None], s.C[:, 4, None, None]
        fn = np.sqrt(v0**2 + 1); nx, nv = -u / fn, v0 / fn
        sj = s.sgn[None, :, None] * rk * (1 + s.h); xs = x0 + sj * nx; vs = v0 + sj * nv; t = s.tgn[None, None, :] * tk
        s.X = xs + vs * t + u * t * t / 2; s.Vv = vs + u * t                           # (K, m, nt)
        s.N = s.K * m * nt

    def _locate(s, A, Q, core=False):
        """(s,t) точки Q (n,2) в клетках A (k,3) — через инвариант DI x − v²/(2u). Возвращает S,T (n,k), ok (n,k): в ядре (core) или ядре+гало."""
        x0, v0, u, rr, tt = A[None, :, 0], A[None, :, 1], A[None, :, 2], A[None, :, 3], A[None, :, 4]; xq, vq = Q[:, 0, None], Q[:, 1, None]
        fn = np.sqrt(v0**2 + 1); nx, nv = -u / fn, v0 / fn
        a = -nv**2 / (2 * u); b = nx - v0 * nv / u; cc = (x0 - v0**2 / (2 * u)) - (xq - vq**2 / (2 * u))
        with np.errstate(invalid='ignore', divide='ignore'):
            disc = b * b - 4 * a * cc; sq = np.sqrt(np.maximum(disc, 0))
            lin = np.abs(a) < 1e-12
            r1 = np.where(lin, -cc / b, (-b + sq) / (2 * a)); r2 = np.where(lin, -cc / b, (-b - sq) / (2 * a))
        S = np.where(np.abs(r1) <= np.abs(r2), r1, r2); S = np.where(disc < 0, np.inf, S)
        T = (vq - (v0 + S * nv)) / u
        if core: ok = (np.abs(S) <= rr) & (T >= 0) & (T <= tt)
        else: ok = (np.abs(S) <= rr * (1 + s.h) + 1e-12) & (T >= -s.h * tt - 1e-12) & (T <= (1 + s.h) * tt + 1e-12)
        return S, T, ok

    def _bil(s, k, S, T):
        """Индексы 4 узлов и веса билинейной интерполяции в клетке k по (S,T)."""
        rr, tt = s.C[k, 3] * (1 + s.h), s.C[k, 4]; fs = (S / rr + 1) / (s.sgn[1] - s.sgn[0]); ft = (T / tt - s.tgn[0]) / (s.tgn[1] - s.tgn[0])
        j = np.clip(np.floor(fs).astype(int), 0, s.m - 2); i = np.clip(np.floor(ft).astype(int), 0, s.nt - 2)
        a = np.clip(fs - j, 0, 1); b = np.clip(ft - i, 0, 1); base = (k * s.m + j) * s.nt + i
        idx = np.stack([base, base + 1, base + s.nt, base + s.nt + 1], -1); w = np.stack([(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b], -1)
        return idx, w

    def pairs(s, Q, exclude=None, chunk=1500):
        """Для точек Q: все (точка, клетка) с точкой в ядре+гало клетки → (qi, k, idx4, w4)."""
        QI, KK, ID, W = [], [], [], []
        for a0 in range(0, len(Q), chunk):
            S, T, ok = s._locate(s.C, Q[a0:a0 + chunk])
            if exclude is not None: ok &= (np.arange(s.K)[None, :] != exclude[a0:a0 + chunk, None])
            qi, k = np.nonzero(ok); idx, w = s._bil(k, S[qi, k], T[qi, k])
            QI.append(qi + a0); KK.append(k); ID.append(idx); W.append(w)
        return np.concatenate(QI), np.concatenate(KK), np.concatenate(ID), np.concatenate(W)

    def goal(s, x, v): return (np.abs(x) <= s.rho) & (np.abs(v) <= s.rho)

    @staticmethod
    def phi(Q, u, h): return np.stack([Q[:, 0] + Q[:, 1] * h + u * h * h / 2, Q[:, 1] + u * h], 1)

    def _red(s, qi, val, n):
        """min по парам для каждой точки (inf, если пар нет)."""
        out = np.full(n, BIG)
        if len(qi): o = np.argsort(qi, kind='stable'); a, v = qi[o], val[o]; uq, st = np.unique(a, return_index=True); out[uq] = np.minimum.reduceat(v, st)
        return out

    def solve(s, tol=1e-9, it=20000):
        """Монотонная полулагранжева схема: V*(узел) = min_k [dt + V*(φ_k(узел, dt))]; V* в точке — min по клеткам (любого слоя), содержащим её
        (билинейно по узлам клетки). Свой слой — следующий узел той же траектории (точно, без интерполяции). Переходов с нулевым временем нет."""
        P = np.stack([s.X.ravel(), s.Vv.ravel()], 1); dt = np.repeat(s.C[:, 4], s.m * s.nt) * (s.tgn[1] - s.tgn[0]); s.dt = s.dt0
        lay = np.repeat(s.C[:, 2], s.m * s.nt); last = (np.arange(s.N) % s.nt) == s.nt - 1
        F = []
        for u in (1.0, -1.0):
            Fp = s.phi(P, u, dt); qi, k, idx, w = s.pairs(Fp); o = np.argsort(qi, kind='stable'); qi, idx, w = qi[o], idx[o], w[o]
            uq, st = np.unique(qi, return_index=True); same = (lay == u) & ~last                  # свой слой не последний узел — берём следующий узел
            F.append((u, uq, st, idx, w, same)); s.npairs = getattr(s, 'npairs', 0) + len(qi)
        G = s.goal(P[:, 0], P[:, 1]); V = np.full(s.N, BIG); V[G] = 0; nxt = np.minimum(np.arange(s.N) + 1, s.N - 1)
        for n in range(it):
            Vn = np.full(s.N, BIG)
            for u, uq, st, idx, w, same in F:
                c = np.full(s.N, BIG); c[uq] = np.minimum.reduceat(np.sum(w * V[idx], -1), st); c = np.where(same, V[nxt], c)
                Vn = np.minimum(Vn, dt + c)
            Vn[G] = 0; Vn = np.minimum(Vn, BIG); d = np.max(np.abs(Vn - V)); V = Vn
            if d < tol: break
        s.V = V; s.iters = n; return V

    def Vstar(s, Q):
        qi, k, idx, w = s.pairs(Q); v = s._red(qi, np.sum(w * s.V[idx], -1), len(Q)); return np.where(v > BIG / 2, np.inf, v)

    def rollout(s, Q0, dt=None, tmax=30.0):
        """Агент: точное движение; каждые dt — слой k = argmin [dt + V*(φ_k(q, dt))] (свободное переключение с разрешением dt)."""
        dt = dt or s.dt; q = np.array(Q0, float); T = np.zeros(len(q)); done = s.goal(q[:, 0], q[:, 1]); dead = np.zeros(len(q), bool); sw = np.zeros(len(q)); pu = np.zeros(len(q))
        for _ in range(int(tmax / dt)):
            act = np.nonzero(~done & ~dead)[0]
            if not len(act): break
            c = np.stack([s.Vstar(s.phi(q[act], u, dt)) for u in (1.0, -1.0)], 1); best = np.min(c, 1); u = np.where(c[:, 0] <= c[:, 1], 1.0, -1.0)
            nod = ~np.isfinite(best); dead[act[nod]] = True; act, u = act[~nod], u[~nod]
            sw[act] += (pu[act] != 0) & (pu[act] != u); pu[act] = u
            for _s in range(4):
                h = dt / 4; x, v = q[act, 0], q[act, 1]; q[act, 0] = x + v * h + u * h * h / 2; q[act, 1] = v + u * h; T[act] += h
                g = s.goal(q[act, 0], q[act, 1]); done[act[g]] = True; act, u = act[~g], u[~g]
        T[~done] = np.inf; return T, sw

if __name__ == '__main__':
    rng = np.random.default_rng(1); Q = rng.uniform(-1.5, 1.5, (400, 2)); Tq = tstar_box(Q[:, 0], Q[:, 1], .1); okq = Tq > .05; res = []
    for a in sys.argv[2:] or ('.4,.1', '.4,.05', '.2,.05'):
        a = list(map(float, a.split(','))); tau, r = a[0], a[1]; ovl = a[2] if len(a) > 2 else None
        t0 = time.time(); Cv = Cover(tau=tau, r=r, ovl=ovl, seeds=6000 if ovl is None else 20000); tc = time.time() - t0
        t0 = time.time(); V = Cv.solve(); ts = time.time() - t0
        P = rng.uniform(-2, 2, (3000, 2)); core = np.zeros(len(P)); full = np.zeros(len(P)); ov, inc = [], []
        for l, uu in ((0, 1.0), (1, -1.0)):
            A = Cv.C[Cv.C[:, 2] == uu]; _, _, okc = Cv._locate(A, P, core=True); _, _, okf = Cv._locate(A, P); core += okc.sum(1); full += (okf.sum(1) > 0)
            nc = okc.sum(1); ov.append(float((nc[nc > 0] > 1).mean())); inc.append(float((nc > 0).mean()))
        t0 = time.time(); T, sw = Cv.rollout(Q); tr = time.time() - t0; ok = np.isfinite(T) & okq; rr = T[ok] / Tq[ok]
        out = dict(tau=tau, r=r, ovl=ovl, cells=int(Cv.K), filled=int(getattr(Cv, "nfill", 0)), small_cells=round(float((Cv.C[:, 3] < r * .99).mean()), 3), nodes=int(Cv.N), pairs=int(Cv.npairs), iters=Cv.iters, finV=round(float((V < BIG / 2).mean()), 3),
                   cores_per_pt_per_layer=round(float(core.mean() / 2), 2), core_overlap_share=round(float(np.mean(ov)), 3), core_covered=round(float(np.mean(inc)), 3), covered_both_layers=round(float((full == 2).mean()), 3),
                   reach=round(float(np.isfinite(T).mean()), 3), T_Tstar=dict(mean=round(float(rr.mean()), 4), med=round(float(np.median(rr)), 4), max=round(float(rr.max()), 3), min=round(float(rr.min()), 4)) if ok.any() else None,
                   sw_med=float(np.median(sw[ok])) if ok.any() else None, sec=dict(cover=round(tc), solve=round(ts), roll=round(tr)))
        print(json.dumps(out), flush=True); res.append(out)
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else 'v7_spore_di.json', 'w'), indent=1)
