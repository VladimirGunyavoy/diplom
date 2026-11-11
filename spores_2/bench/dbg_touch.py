"""b15: агент схемы А (knowledge/research/dyn_obstacles.md §Две схемы) — общий для статик/динамик кара (σ=0, диски стоят — частный случай).
Атлас carfree (V₀ без дисков) = дальняя цена; облако дисков — ограничение на ближнем горизонте H. Каждый шаг (.08 с, их среда): кандидаты = (u0 из 6 слоёв, держать c ∈ COMMITS шагов по DTN .1), затем жадно по V₀
среди безопасных u; безопасно = расстояние до центра прогноза m(τ)=p+wτ ≥ R + PAD + κσ(τ), σ²=σ₀²+qτ³/3 (q — по наблюдениям: изотропно, лаг 2 с, окно 4 с; q₀=.01 до 6 с истории), κ=√(−2 ln ε).
Выбор: min [время до конца горизонта + V₀(x(H))] среди кандидатов без нарушений; нет — max запаса (уход). STATIC=1: диски стоят, σ=0.
Запуск (aida 2, из bench/): MODE=static|fixed|random LOAD=<carfree.pkl> [EPS=.03 H=2 PAD=.1] M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 PESS=1 GS=300 SYS=car NODISC=1 NP=8 python envs_agentA.py → data/env_A_<MODE>.json"""
import os, sys, json, time
os.environ.setdefault('SYS', 'car'); os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'v7', 'src', 'cells7'))
import numpy as np
import envs_dyn as D
from envs2026 import their_step
HR, GOAL, GR, DT = D.HR, D.GOAL, D.GR, D.DT; C0 = D.C0
class StaticHaz:
    def pos(s, t): return C0.copy() if np.ndim(t) == 0 else np.broadcast_to(C0, (len(t), 3, 2))
class AgentA:
    def __init__(s, G, A, H=2., eps=.03, pad=.1, static=False, sig0=0., q0=.01, commits=(1, 3, 6), smax=.5, look=0):
        s.G, s.A, s.H, s.pad, s.static, s.sig0, s.q0, s.commits, s.smax = G, A, H, pad, static, sig0, q0, commits, smax; s.kap = np.sqrt(-2 * np.log(eps)); s.look = look; s.side = {}; s.nsw = 0; s.hist = []; s.US = np.array(G.US, float); s.nfb = 0; s.nstep = 0; s.lsinfo = []; s.inst = []
    def observe(s, p): s.hist.append(np.array(p, float))
    def qest(s):
        if s.static: return 0.
        L = int(round(2. / DT)); W = int(round(4. / DT)); h = s.hist
        if len(h) < L + W // 2 + 2: return s.q0
        P = np.array(h); e = []
        for k in range(max(1, len(P) - W - L), len(P) - L):
            w = (P[k] - P[k - 1]) / DT; e.append(P[k + L] - P[k] - w * L * DT)
        e = np.array(e); return float(max((3 * (e ** 2).mean((0, 1)) / (L * DT) ** 3).max(), 1e-4)) if len(e) else s.q0
    def sipp_sides(s, y0, p, w, q, cell=.3, dt=.25, nl=100, vmax=1.2):
        """b16 схема Б (грубый SIPP): достижимость по слоям времени на сетке xy (ожидание разрешено), диск — m(τ)=p+wτ, радиус R+pad+κσ(τ)+cell/2;
        кратчайший по времени путь к цели → сторона каждого диска (знак относительно нормали к линии цели в точке наибольшего сближения). None — пути нет."""
        n = int(10 / cell) + 1; ax = -5. + cell * np.arange(n); XX, YY = np.meshgrid(ax, ax, indexing='ij'); P = np.stack([XX, YY], -1)
        i0 = tuple(np.clip(np.round((y0[:2] + 5.) / cell).astype(int), 0, n - 1)); goal = np.linalg.norm(P - GOAL, axis=-1) <= GR
        R = np.zeros((n, n), bool); R[i0] = True; layers = [R]; jg = None
        for j in range(1, nl + 1):
            tau = j * dt; sg = min(np.sqrt(s.sig0 ** 2 + q * tau ** 3 / 3), s.smax); m = p + w * tau; occ = np.zeros((n, n), bool)
            for k in range(len(m)): occ |= np.linalg.norm(P - m[k], axis=-1) < HR + s.pad + s.kap * sg + cell / 2
            Rn = R.copy(); r = int(vmax * dt / cell + .999) if False else 1
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0: continue
                    sh = np.zeros_like(R); sh[max(di, 0):n + min(di, 0), max(dj, 0):n + min(dj, 0)] = R[max(-di, 0):n + min(-di, 0), max(-dj, 0):n + min(-dj, 0)]; Rn |= sh
            R = Rn & ~occ; layers.append(R)
            if (R & goal).any(): jg = j; break
            if not R.any(): return None
        if jg is None: return None
        c = np.argwhere(layers[jg] & goal)[0]; path = [c]
        for j in range(jg - 1, -1, -1):
            best = None
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    u = (c[0] + di, c[1] + dj)
                    if 0 <= u[0] < n and 0 <= u[1] < n and layers[j][u]:
                        d_ = abs(di) + abs(dj)
                        if best is None or d_ < best[0]: best = (d_, u)
            c = np.array(best[1]); path.append(c)
        path = path[::-1]; pts = np.array([P[tuple(c)] for c in path]); out = {}
        dg = GOAL - y0[:2]; ug = dg / max(np.linalg.norm(dg), 1e-9); nr = np.array([-ug[1], ug[0]])
        for k in range(len(p)):
            d = np.array([np.linalg.norm(pts[j] - (p[k] + w[k] * j * dt)) for j in range(len(pts))]); j = int(d.argmin()); out[k] = 1 if np.dot(pts[j] - (p[k] + w[k] * j * dt), nr) >= 0 else -1
        return out
    def delay_field(s, p, w, q, cell=.3, dt=.25, N=80):
        """b16 схема Б (обратный DP по времени на сетке xy, ожидание разрешено, все диски): Tg[j][ячейка] — мин. остаток времени до цели из (ячейка, слой j);
        возвращает функцию задержки dl(xy, j) = Tg[j] − Tg0 (Tg0 — без дисков). σ ≤ smax (облако κσ(τ))."""
        n = int(10 / cell) + 1
        if not hasattr(s, '_g'):
            ax = -5. + cell * np.arange(n); XX, YY = np.meshgrid(ax, ax, indexing='ij'); P = np.stack([XX, YY], -1); goal = np.linalg.norm(P - GOAL, axis=-1) <= GR
            def relax(T):
                Tp = np.pad(T, 1, constant_values=np.inf); best = T.copy()
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1): best = np.minimum(best, Tp[1 + di:1 + di + n, 1 + dj:1 + dj + n])
                return best
            T0 = np.where(goal, 0., np.inf)
            for _ in range(2 * n):
                Tn = np.where(goal, 0., relax(T0) + dt)
                if np.array_equal(Tn, T0): break
                T0 = Tn
            s._g = (P, goal, relax, T0)
        P, goal, relax, T0 = s._g; Tg = [None] * (N + 1); Tg[N] = T0.copy(); inf = np.inf
        for j in range(N - 1, -1, -1):
            tau = (j + 1) * dt; sg = min(np.sqrt(s.sig0 ** 2 + q * tau ** 3 / 3), s.smax); m = p + w * tau; occ = np.zeros((n, n), bool)
            for k in range(len(m)): occ |= np.linalg.norm(P - m[k], axis=-1) < HR + s.pad + s.kap * sg + cell / 2
            Tg[j] = np.where(goal, 0., np.where(occ, inf, relax(Tg[j + 1]) + dt))
        def dl(xy, j):
            ij = np.clip(np.round((xy + 5.) / cell).astype(int), 0, n - 1); d = Tg[min(j, N)][ij[:, 0], ij[:, 1]] - T0[ij[:, 0], ij[:, 1]]
            return np.where(np.isfinite(d), np.maximum(d, 0.), 20.)
        return dl
    def act(s, y, t_unused=0.):
        G = s.G; s.nstep += 1; p = s.hist[-1]; w = np.zeros_like(p) if (s.static or len(s.hist) < 2) else (s.hist[-1] - s.hist[-2]) / DT; q = s.qest()
        nst = int(round(s.H / G.DTN)); cands = [(i, c) for i in range(len(s.US)) for c in s.commits]; nc = len(cands); i0 = np.array([c[0] for c in cands]); cm = np.array([c[1] for c in cands])
        Y = np.tile(np.asarray(y, float), (nc, 1)); t_el = np.zeros(nc); done = G.ingoal(Y); viol = np.zeros(nc, bool); marg = np.full(nc, np.inf); first = i0.copy()
        blk = []; y0 = np.asarray(y, float)
        if s.look == 5:   # b17: Б-LS — пересчёт V области. LSDELAY=1/LSPIPE=1: моделированная задержка (V состояния шага t доступна с шага t+ceil(t_LS/DT); LSTLS — подмена t_LS, с)
            import envs_ls; kk = int(os.environ.get('LSK', 12)); pipe = int(os.environ.get('LSPIPE', 0)); dly = int(os.environ.get('LSDELAY', 0)) or pipe
            kw = dict(sig0=s.sig0, pad=s.pad, HR=HR, T_PR=float(os.environ.get('TPR', 2.)), smax=min(s.smax, .25), pe=float(os.environ.get('LSPE', .3)), ngs=int(os.environ.get('LSGS', 1)), MARG=(None if os.environ.get('MARG') == 'none' else float(os.environ.get('MARG', .3))))
            if not dly:
                if (s.nstep - 1) % kk == 0: r = envs_ls.refresh(_Z, G, s.A, y0, p, w, q, **kw); r.pop('_V', None); s.lsinfo.append(r)
            else:
                if not hasattr(s, 'pend'): s.pend = None; s.lastst = -10 ** 9
                if int(os.environ.get('LSPRE', 0)) and s.nstep == 1: r0 = envs_ls.refresh(_Z, G, s.A, y0, p, w, q, install=True, **kw); r0.pop('_V', None); r0['pre'] = 1; s.lsinfo.append(r0)   # b17: план до отправления — V_Б по состоянию t=0 до старта (в T не входит)
                if int(os.environ.get('LSCOARSE', 0)) and s.nstep == 1:   # b17: старт по А + грубая первая V_Б (оболочка T_пр LSCTPR, t_LS = узлы × LSNODEUS)
                    kw1 = dict(kw); kw1['T_PR'] = float(os.environ.get('LSCTPR', 1.)); r1 = envs_ls.refresh(_Z, G, s.A, y0, p, w, q, install=False, **kw1); Vn1 = r1.pop('_V', None)
                    if Vn1 is not None:
                        tl1 = r1['n'] * float(os.environ.get('LSNODEUS', 2.3e-6)); d1 = max(1, int(np.ceil(tl1 / DT))); s.pend = (s.nstep + d1, Vn1); s.lastst = s.nstep; s.tl_prev = tl1; r1['delay'] = d1; r1['tl'] = tl1; r1['coarse'] = 1
                    s.lsinfo.append(r1)
                if s.pend is not None and s.nstep >= s.pend[0]:
                    if not int(os.environ.get("LSNOINST", 0)): s.A.V = s.pend[1]   # b18: LSNOINST=1 — пересчёты идут, V_Б не устанавливается (контроль)
                    s.pend = None; s.inst.append(s.nstep)
                if s.pend is None and (pipe or s.nstep - s.lastst >= kk):
                    r = envs_ls.refresh(_Z, G, s.A, y0, p, w, q, install=False, tshift=(getattr(s, 'tl_prev', float(os.environ.get('LSTL0', 5.))) if int(os.environ.get('LSB', 0)) else 0.), **kw); Vn = r.pop('_V', None)
                    if Vn is not None:
                        tl = float(os.environ['LSTLS']) if os.environ.get('LSTLS') else r['tot']; d = int(np.ceil(tl / DT)); s.pend = (s.nstep + d, Vn); s.lastst = s.nstep; s.tl_prev = tl; r['delay'] = d; r['tl'] = tl
                    s.lsinfo.append(r)
        if s.look and s.look != 5:
            dg = GOAL - y0[:2]; Lg = np.linalg.norm(dg); ug = dg / max(Lg, 1e-9); nrm = np.array([-ug[1], ug[0]])
            for k in range(len(p)):
                tc = min(max(np.dot(p[k] - y0[:2], ug), 0.) / 1.5, s.H); sgc = min(np.sqrt(s.sig0 ** 2 + q * tc ** 3 / 3), s.smax); c = p[k] + w[k] * tc; pr = np.dot(c - y0[:2], ug); off = np.dot(c - y0[:2], nrm)
                rr_ = HR + s.pad + s.kap * sgc + .3; inb = 0 < pr < Lg and abs(off) < rr_
                if s.look == 2 and k in s.side:   # b16: фиксация стороны — держится, пока диск не остался позади по проекции на линию к цели
                    inb = np.dot(p[k] - y0[:2], ug) > -(HR + s.pad)
                if inb: blk.append((k, c, nrm, rr_, tc))
            for k in list(s.side):
                if k not in [b[0] for b in blk]: del s.side[k]
            plan = None
            for k, c, nr_, rr, tc in blk:
                cs = []
                for sd in (1, -1):
                    wp = c + sd * nr_ * rr; th = np.arctan2(*(GOAL - wp)[::-1]); vw = s.A.vstar(np.array([[wp[0], wp[1], th, 1.5]]))[0]; cs.append(np.linalg.norm(wp - y0[:2]) / 1.5 + (1e3 if vw >= G.BIG / 2 else vw))
                new = 1 if cs[0] <= cs[1] else -1; old = s.side.get(k)
                if s.look == 3:
                    if plan is None and k == blk[0][0]: plan = s.sipp_sides(y0, p, w, q) or {}
                    if k in plan:
                        if old is not None and old != plan[k]: s.nsw += 1
                        s.side[k] = plan[k]; continue
                if old is None or (s.look != 2 and s.look != 3 and old != new and cs[(0 if old == 1 else 1)] - cs[(0 if new == 1 else 1)] > .5):
                    if old is not None: s.nsw += 1
                    s.side[k] = new
        bmin = [np.full(nc, np.inf) for _ in blk]; bpos = [np.zeros((nc, 2)) for _ in blk]
        def mg(Yn, tau):
            sg = np.sqrt(s.sig0 ** 2 + q * tau ** 3 / 3); sg = min(sg, s.smax); m = p + w * tau
            return (np.linalg.norm(Yn[:, None, :2] - m[None], axis=2) - HR - s.pad - s.kap * sg).min(1)
        for j in range(nst):
            tau = (j + 1) * G.DTN
            hold = j < cm
            # жадный выбор для кандидатов вне held-части
            g = ~hold & ~done
            Ynext = np.empty_like(Y)
            hm = hold & ~done
            if hm.any():
                Yt = Y[hm].copy()
                for u in np.unique(i0[hm]):
                    sel = i0[hm] == u; Yt[sel] = G.step(Yt[sel], s.US[u])
                Ynext[hm] = Yt
            if g.any():
                Yg = Y[g]; ng = len(Yg); Y6 = np.repeat(Yg, len(s.US), 0); U6 = np.tile(np.arange(len(s.US)), ng); Yn6 = np.empty_like(Y6)
                for u in range(len(s.US)): m_ = U6 == u; Yn6[m_] = G.step(Y6[m_], s.US[u])
                Yn6 = G.wrapy(Yn6); Yn6[:, 3] = np.clip(Yn6[:, 3], G.VLO, G.VHI); v6 = s.A.vstar(Yn6); J = np.where(v6 >= G.BIG / 2, np.inf, G.DTN + v6); m6 = mg(Yn6, tau); safe = m6 >= 0
                J = np.where(safe, J, np.inf).reshape(ng, -1); mm = m6.reshape(ng, -1); k = np.where(np.isfinite(J.min(1)), J.argmin(1), mm.argmax(1)); Ynext[g] = Yn6.reshape(ng, len(s.US), 4)[np.arange(ng), k]
            Ynext[done] = Y[done]; Ynext = G.wrapy(Ynext); Ynext[:, 3] = np.clip(Ynext[:, 3], G.VLO, G.VHI)
            act_ = ~done; mgn = np.full(nc, np.inf); mgn[act_] = mg(Ynext[act_], tau); viol |= (mgn < 0); marg = np.minimum(marg, mgn)
            for bi, (k, c, nr_, rr, tc) in enumerate(blk):
                dk = np.linalg.norm(Ynext[:, :2] - (p[k] + w[k] * tau), axis=1); upd = act_ & (dk < bmin[bi]); bmin[bi] = np.where(upd, dk, bmin[bi]); bpos[bi][upd] = Ynext[upd, :2]
            t_el = np.where(act_, t_el + G.DTN, t_el); Y = Ynext; done = done | G.ingoal(Y)
        V = s.A.vstar(Y); V = np.where(done, 0., V); cost = t_el + np.where(V >= G.BIG / 2, 1e3, V)
        if s.look == 4 and not s.static:
            dl = s.delay_field(p, w, q); cost = cost + float(os.environ.get('DW', 1.)) * np.where(done, 0., dl(Y[:, :2], int(round(s.H / .25))))
        for bi, (k, c, nr_, rr, tc) in enumerate(blk):
            sd_ = np.sign(np.einsum('ij,j->i', bpos[bi] - c, nr_)); cost = cost + 3. * (sd_ != s.side[k]) * np.isfinite(bmin[bi])
        ok = ~viol
        if ok.any(): k = int(np.argmin(np.where(ok, cost, np.inf)))
        else: k = int(np.argmax(marg)); s.nfb += 1
        return s.US[first[k]], dict(nsw=s.nsw, feas=bool(ok.any()), cost=float(cost[k]), q=q)

def episode(ag, haz, x0, tmax=20., mode_dyn=True):
    y = np.array(x0, float); t = 0.; Y = [y.copy()]; clr = 1e9; L = 0.; ok = False; tms = []; nofeas = 0; inside = 0
    while t < tmax:
        if np.linalg.norm(y[:2] - GOAL) <= GR: ok = True; break
        ag.observe(haz.pos(t)[:, :]); t0 = time.perf_counter(); u, inf = ag.act(y); tms.append(time.perf_counter() - t0); nofeas += (not inf['feas'])
        yn = their_step(y, u)
        for s_ in np.linspace(0, 1, 21)[1:]:
            q = y[:2] * (1 - s_) + yn[:2] * s_; c = (np.linalg.norm(haz.pos(t + s_ * DT) - q, axis=1) - HR).min(); clr = min(clr, c, 5. - np.abs(q).max())
        L += np.linalg.norm(yn[:2] - y[:2]); y = yn; t += DT; Y.append(y.copy())
    return dict(ok=ok, T=t if ok else None, lsn=ag.lsinfo[:40], nref=len(ag.lsinfo), inst=ag.inst, touch=bool(clr < -1e-3), clr=float(clr), L=float(L), ms=float(np.mean(tms) * 1e3) if tms else 0., nofeas=nofeas, nsw=ag.nsw, steps=len(tms), Y=np.array(Y)[::2].tolist())

_G = _A = _Z = None
def one(a):
    i, mode, seed = a; x = X0[i]; haz = StaticHaz() if mode == 'static' else D.Hazards(mode, seed)
    ag = AgentA(_G, _A, H=float(os.environ.get('H', 2.)), eps=float(os.environ.get('EPS', .03)), pad=float(os.environ.get('PAD', .1)), static=(mode == 'static'), smax=float(os.environ.get('SMAX', .5)), look=int(os.environ.get('LOOK', 0)))
    r = episode(ag, haz, x); r.update(i=i, mode=mode, seed=seed); return r
if __name__ == '__main__':
    import pickle, multiprocessing as mp, growCar as G, common, inspect, __main__
    for nm, o in vars(G).items():
        if inspect.isclass(o): setattr(__main__, nm, o)
    _G = G; d = pickle.load(open(os.path.expanduser(os.environ['LOAD']), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']; _A = A
    if int(os.environ.get('NODISCS', 0)): C0[:] = 50.   # b15: остаток без дисков — диски унесены за арену
    mode = os.environ.get('MODE', 'static')
    if int(os.environ.get('LOOK', 0)) == 5:
        import envs_ls; _Z = envs_ls.prep_zone(G, A, C0, HR, tube=float(os.environ.get('TUBE', .5)), cache=os.environ.get('ZCACHE')); envs_ls.prep_rev(_Z); print('обратный индекс зоны %.0f с;' % _Z['t_rev'], end=' '); print('зона %d узлов, рёбер %d, стенсилы %.0f с' % (_Z['nz'], len(_Z['X']), _Z['t_prep']), flush=True)
    e = common.make('car'); X0 = e.X0 if mode == 'static' else D.starts(20); Tref = e.Tref
    if os.environ.get('NST'): X0 = X0[:int(os.environ['NST'])]; Tref = Tref[:int(os.environ['NST'])]
    A.stencils(np.asarray(X0[0], float)[None])   # прогрев индекса ячеек до fork (иначе каждый воркер строит его сам, ~20 с)
    if int(os.environ.get('LOOK', 0)) == 5: A.V = _Z['V0']; envs_ls.refresh(_Z, G, A, np.asarray(X0[0], float), C0.copy(), np.zeros_like(C0), 0., install=False); A.V = _Z['V0']; print('прогрев numba ok', flush=True)
    tag = os.environ.get("DTAG", "A"); out = {}
    for mode_ in os.environ.get("DMODES", "fixed,random").split(","):
        for i in [int(a) for a in os.environ.get("DST", "18,19").split(",")]:
            haz = StaticHaz() if mode_ == "static" else D.Hazards(mode_, 0); ag = AgentA(_G, _A, H=float(os.environ.get("H", 2.)), eps=float(os.environ.get("EPS", .03)), pad=float(os.environ.get("PAD", .1)), static=(mode_ == "static"), smax=float(os.environ.get("SMAX", .5)), look=int(os.environ.get("LOOK", 0)))
            y = np.array(X0[i], float); t = 0.; rec = []
            while t < 20.:
                if np.linalg.norm(y[:2] - GOAL) <= GR: break
                ag.observe(haz.pos(t)[:, :]); u, inf = ag.act(y); yn = their_step(y, u); cd = 1e9; cw = 1e9; jj = -1
                for s_ in np.linspace(0, 1, 21)[1:]:
                    q = y[:2] * (1 - s_) + yn[:2] * s_; dd = np.linalg.norm(haz.pos(t + s_ * DT) - q, axis=1) - HR
                    if dd.min() < cd: cd = float(dd.min()); jj = int(dd.argmin())
                    cw = min(cw, float(5. - np.abs(q).max()))
                vd = haz.vel(t)[jj] if hasattr(haz, "vel") else np.zeros(2); rec.append(dict(k=len(rec), t=t, cd=cd, cw=cw, j=jj, feas=bool(inf["feas"]), vd=np.linalg.norm(vd), va=float(y[3]), ang=float(np.degrees(np.arccos(np.clip(vd @ (np.array([np.cos(y[2]), np.sin(y[2])])) / max(np.linalg.norm(vd), 1e-9), -1, 1)))), n_inst=len(ag.inst), y=[float(v) for v in y]))
                y = yn; t += DT
            out["%s_%d" % (mode_, i)] = dict(rec=rec, inst=ag.inst)
            r = min(rec, key=lambda z: min(z["cd"], z["cw"])); k = r["k"]
            print(tag, mode_, i, "шагов", len(rec), "мин зазор: диск %.3f стена %.3f на шаге %d (t=%.2f), диск %d, v_диска %.2f v_аг %.2f угол(vд,курс) %.0f°, feas на шаге %s, нет-feas шагов %d (шаги %s), V_Б установлена к шагу: %s, inst %s" % (r["cd"], r["cw"], k, r["t"], r["j"], r["vd"], r["va"], r["ang"], r["feas"], sum(not z["feas"] for z in rec), [z["k"] for z in rec if not z["feas"]][:8], r["n_inst"] > 0, ag.inst[:3]), flush=True)
    json.dump(out, open(os.environ["OUT"], "w"))
