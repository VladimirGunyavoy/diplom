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
    def __init__(s, G, A, H=2., eps=.03, pad=.1, static=False, sig0=0., q0=.01, commits=(1, 3, 6), smax=.5):
        s.G, s.A, s.H, s.pad, s.static, s.sig0, s.q0, s.commits, s.smax = G, A, H, pad, static, sig0, q0, commits, smax; s.kap = np.sqrt(-2 * np.log(eps)); s.hist = []; s.US = np.array(G.US, float); s.nfb = 0; s.nstep = 0
    def observe(s, p): s.hist.append(np.array(p, float))
    def qest(s):
        if s.static: return 0.
        L = int(round(2. / DT)); W = int(round(4. / DT)); h = s.hist
        if len(h) < L + W // 2 + 2: return s.q0
        P = np.array(h); e = []
        for k in range(max(1, len(P) - W - L), len(P) - L):
            w = (P[k] - P[k - 1]) / DT; e.append(P[k + L] - P[k] - w * L * DT)
        e = np.array(e); return float(max((3 * (e ** 2).mean((0, 1)) / (L * DT) ** 3).max(), 1e-4)) if len(e) else s.q0
    def act(s, y, t_unused=0.):
        G = s.G; s.nstep += 1; p = s.hist[-1]; w = np.zeros_like(p) if (s.static or len(s.hist) < 2) else (s.hist[-1] - s.hist[-2]) / DT; q = s.qest()
        nst = int(round(s.H / G.DTN)); cands = [(i, c) for i in range(len(s.US)) for c in s.commits]; nc = len(cands); i0 = np.array([c[0] for c in cands]); cm = np.array([c[1] for c in cands])
        Y = np.tile(np.asarray(y, float), (nc, 1)); t_el = np.zeros(nc); done = G.ingoal(Y); viol = np.zeros(nc, bool); marg = np.full(nc, np.inf); first = i0.copy()
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
            t_el = np.where(act_, t_el + G.DTN, t_el); Y = Ynext; done = done | G.ingoal(Y)
        V = s.A.vstar(Y); V = np.where(done, 0., V); cost = t_el + np.where(V >= G.BIG / 2, 1e3, V)
        ok = ~viol
        if ok.any(): k = int(np.argmin(np.where(ok, cost, np.inf)))
        else: k = int(np.argmax(marg)); s.nfb += 1
        return s.US[first[k]], dict(feas=bool(ok.any()), cost=float(cost[k]), q=q)

def episode(ag, haz, x0, tmax=20., mode_dyn=True):
    y = np.array(x0, float); t = 0.; Y = [y.copy()]; clr = 1e9; L = 0.; ok = False; tms = []; nofeas = 0; inside = 0
    while t < tmax:
        if np.linalg.norm(y[:2] - GOAL) <= GR: ok = True; break
        ag.observe(haz.pos(t)[:, :]); t0 = time.perf_counter(); u, inf = ag.act(y); tms.append(time.perf_counter() - t0); nofeas += (not inf['feas'])
        yn = their_step(y, u)
        for s_ in np.linspace(0, 1, 21)[1:]:
            q = y[:2] * (1 - s_) + yn[:2] * s_; c = (np.linalg.norm(haz.pos(t + s_ * DT) - q, axis=1) - HR).min(); clr = min(clr, c, 5. - np.abs(q).max())
        L += np.linalg.norm(yn[:2] - y[:2]); y = yn; t += DT; Y.append(y.copy())
    return dict(ok=ok, T=t if ok else None, touch=bool(clr < -1e-3), clr=float(clr), L=float(L), ms=float(np.mean(tms) * 1e3) if tms else 0., nofeas=nofeas, steps=len(tms), Y=np.array(Y)[::2].tolist())

_G = _A = None
def one(a):
    i, mode, seed = a; x = X0[i]; haz = StaticHaz() if mode == 'static' else D.Hazards(mode, seed)
    ag = AgentA(_G, _A, H=float(os.environ.get('H', 2.)), eps=float(os.environ.get('EPS', .03)), pad=float(os.environ.get('PAD', .1)), static=(mode == 'static'))
    r = episode(ag, haz, x); r.update(i=i, mode=mode, seed=seed); return r
if __name__ == '__main__':
    import pickle, multiprocessing as mp, growCar as G, common, inspect, __main__
    for nm, o in vars(G).items():
        if inspect.isclass(o): setattr(__main__, nm, o)
    _G = G; d = pickle.load(open(os.path.expanduser(os.environ['LOAD']), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']; _A = A
    if int(os.environ.get('NODISCS', 0)): C0[:] = 50.   # b15: остаток без дисков — диски унесены за арену
    mode = os.environ.get('MODE', 'static'); e = common.make('car'); X0 = e.X0 if mode == 'static' else D.starts(20); Tref = e.Tref
    A.stencils(np.asarray(X0[0], float)[None])   # прогрев индекса ячеек до fork (иначе каждый воркер строит его сам, ~20 с)
    jobs = [(i, mode, int(os.environ.get('SEED', 0))) for i in range(len(X0))]; t0 = time.perf_counter()
    with mp.get_context('fork').Pool(int(os.environ.get('NP', 8))) as p: res = p.map(one, jobs)
    res.sort(key=lambda r: r['i']); ok = [r for r in res if r['ok']]; rt = np.array([r['T'] / Tref[r['i']] for r in ok]) if ok else np.array([np.nan])
    print(mode, 'дошли', len(ok), '/', len(res), 'касаний', sum(r['touch'] for r in res), 'T/Tref мед %.3f p90 %.3f' % (np.median(rt), np.percentile(rt, 90)), 'мин.зазор мед %.3f' % np.median([r['clr'] for r in res]), 'мс/шаг %.0f' % np.mean([r['ms'] for r in res]), 'застряли', len(res) - len(ok), 'wall %.0f' % (time.perf_counter() - t0))
    json.dump(dict(res=res, Tref=Tref.tolist()), open(os.path.join(common.DATA, 'env_A_%s%s%s.json' % (mode, '_nodisc' if int(os.environ.get('NODISCS', 0)) else '', os.environ.get('TAG', ''))), 'w'), default=float)
