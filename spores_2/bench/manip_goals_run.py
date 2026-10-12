"""CA4: споры на манипуляторе 4D с переменной целью, интерфейс b9 (v7/src/cells7/spd.py: GoalSolver.retarget; growN.GC — центр цели).
Слои и рёбра графа (стенсилы) от цели не зависят: заготовка = слои (270 с, CPU) + один полный solve с SAVEE (~300 с, строит кэш рёбер ~4 ГБ) — цифры b9 (spd1.md); рёбра берём готовые b9 (E_g300so_*).
На новую цель: retarget (tgoal у границы + GPU-Якоби tol 1e-4, lcut 4.5, fp32, ~11 с); запрос — агент (rollout_rec) с целью GC. Без досева клеток цели (SPD1b у b9 — потом).
FINBOX=1 (по умолч.): финиш-дуги целиком в коробке |ω| ≤ 3 (b9 FIN-ω); FINBOX=0 — как до FIN-ω (данные mg_*_spores_prefin.json).
Запуск (aida, venv-sb3, под flock GPU): ~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python manip_goals_run.py run A|B [K] → data/mg_{A,B}_spores.json; эталон — data/mg_{A,B}_ref.json (OCP IPOPT, вне таймера)."""
import os, sys, time, json, pickle
ENV = dict(SYS='manip', KF='11', M='3', FRAC='.5', OVH='.5', MINROWS='3', GLIM='0', RMAX='.6', TMAX='3', SBCAUS='1', SBLAY='0', NOLATCH='1', FINGRID='2', VF='2', PESS='1', GPUMAX='1000000', GS='300', DELTA='.3', MAXC='3000', SONE='1', TQDM_MI='1000', OMP_NUM_THREADS='4')
for k, v in ENV.items(): os.environ.setdefault(k, v)
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.expanduser('~/spore_v5/c1/manip_b9'); sys.path.insert(0, HERE); sys.path.insert(0, SRC)
import numpy as np, common
import growN as G, finish_gen as FG, spd
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
EDG = os.path.expanduser('~/spore_v5/b9/E_g300so'); LCUT = 4.5

def shoot_pol_rec(y, tmax, NA, inits=2, H=.1):                      # как в spores_manip, но остаток до центра цели GC
    from scipy.optimize import minimize
    T0, tp = FG.shoot_grid(y, G.f, G.US, G.RHOV, G.wrapg, tmax, NA, H=H)
    if tp is None: return T0, tp, None
    R = .9 * np.asarray(G.RHOV); best = (T0, tp, None); n = len(tp)
    def end(d):
        z = np.array(y, float)
        for k, dt in zip(tp, d): z = FG.flow(z, G.US[k], max(dt, 0.), G.f)
        return G.wrapg(z)
    def path_free(d):                                                  # FIN-ω (b9): вся дуга внутри коробки состояний (|ω| ≤ WM)
        if FG.FREE is None: return True
        z = np.array(y, float)
        for k, dt in zip(tp, d):
            ns = max(1, int(np.ceil(max(dt, 0.) / FG.HS)))
            for _ in range(ns):
                z = FG.flow(z, G.US[k], max(dt, 0.) / ns, G.f)
                if not FG.FREE(z): return False
        return True
    cons = [{"type": "ineq", "fun": lambda d: R - np.abs(end(d))}]
    for d0 in (np.full(n, T0 / n), np.r_[T0 * .6, np.full(n - 1, T0 * .4 / max(n - 1, 1))][:n]):
        r = minimize(lambda d: d.sum(), d0, method="SLSQP", bounds=[(0, tmax)] * n, constraints=cons, options=dict(maxiter=60, ftol=1e-7))
        if r.success and np.all(np.abs(end(r.x)) <= np.asarray(G.RHOV)) and r.x.sum() < best[0] and path_free(r.x): best = (float(r.x.sum()), tp, r.x.copy())
    return best

def rollout_rec(A, Q, tmax=40.):
    FG.FREE = (lambda y: G.inbox(y)) if int(os.environ.get('FINBOX', 1)) else None
    E = os.environ.get; DTN = G.DTN; US = G.US; VF = G.VF
    Y = G.wrapy(Q); n = len(Y); T = np.zeros(n); done = G.ingoal(Y); tried = np.zeros(n, bool); fin = np.zeros(n, bool); seg = [[] for _ in range(n)]
    for _ in range(int(tmax / DTN)):
        if done.all(): break
        tgs = np.stack([A.tgoal(Y, u) for u in US], 1); J = np.minimum(tgs, DTN + np.stack([A.vstar(G.step(Y, u)) for u in US], 1)); k = J.argmin(1); tg = tgs[np.arange(n), k]
        for i in np.flatnonzero(~done & ~tried & (J.min(1) <= VF)):
            tried[i] = True; Ts, tp, dd = shoot_pol_rec(Y[i], G.FTMAX, int(E('NA', 3)))
            if np.isfinite(Ts) and dd is not None: T[i] += Ts; done[i] = True; fin[i] = True; seg[i] += [(US[kk], float(d)) for kk, d in zip(tp, dd)]
            elif np.isfinite(Ts): T[i] += Ts; done[i] = True; fin[i] = True; seg[i] += [(US[kk], float(H_)) for kk, H_ in zip(tp, [.1] * len(tp))]
        stuck = J.min(1) >= G.BIG / 2; act = ~done & ~stuck; Yn = Y.copy()
        for ki, ui in enumerate(US):
            m = act & (k == ki)
            if not m.any(): continue
            hh = np.where(np.isfinite(tg[m]), tg[m], DTN); y8 = Y[m].copy()
            for i in range(1, 9): y_i = G.rk4(y8, ui, DTN / 8, 1); y8 = np.where((hh >= DTN * i / 8 - 1e-12)[:, None], y_i, y8)
            Yn[m] = y8
        for i in np.flatnonzero(act): seg[i].append((US[k[i]], float(tg[i]) if np.isfinite(tg[i]) else DTN))
        Y = G.wrapy(np.where(act[:, None], Yn, Y)); T += np.where(act, np.where(np.isfinite(tg), tg, DTN), 0.); T[~done & stuck] = np.inf; done |= G.ingoal(Y) | stuck
    T[~G.ingoal(Y) & ~fin] = np.inf; return T, seg

def load_atlas():
    A = G.Atlas.__new__(G.Atlas); A.layers = pickle.load(open(os.path.join(SRC, 'L_g300so.pkl'), 'rb'))['layers']; return A

def solve_goal(A, S, q):
    import io, contextlib; t = time.perf_counter(); S.retarget(np.array([q[0], q[1], 0., 0.]), tol=1e-4, lcut=LCUT, fp32=True, log=lambda *a: None); return time.perf_counter() - t

def stage_run(kind, K):
    import manip_goals as MGm
    tr, _ = MGm.transitions(kind); tr = tr[:K]; ref = {r['k']: r['Tref'] for r in json.load(open(os.path.join(common.DATA, 'mg_%s_ref.json' % kind)))}
    A = load_atlas(); A.finish(); S = spd.GoalSolver(A, EDG); res = []; seen = {}
    for k, (st, g) in enumerate(tr):
        key = tuple(np.round(g, 9)); tg = None
        if key not in seen:
            tg = solve_goal(A, S, g); seen[key] = A.V.copy()
            print('цель', k, 'retarget %.1f с' % tg, 'конечных %.3f' % float((A.V < G.BIG / 2).mean()), 'узлов цели', int(A.goal.sum()), flush=True)
        else: G.GC = np.array([g[0], g[1], 0., 0.]); A.goal = G.ingoal(A.P); A.V = seen[key]    # V этой цели уже решена (A: M решений, не K)
        e = MGm.MG(g, st); t1 = time.perf_counter(); T, seg = rollout_rec(A, e.X0); dtq = time.perf_counter() - t1; sg = seg[0]
        Tv, why = common.verify(e, e.X0[0], [list(u) for u, d in sg], [d for u, d in sg])
        res.append(dict(k=k, start=list(map(float, st)), goal=list(map(float, g)), Tref=ref.get(k), T_agent=float(T[0]) if np.isfinite(T[0]) else None, T_verify=Tv, why=why, t_goal=tg, t_query=dtq))
        print(k, res[-1], flush=True)
    json.dump(res, open(os.path.join(common.DATA, 'mg_%s_spores.json' % kind), 'w'), indent=1, default=float)

if __name__ == '__main__':
    a = sys.argv[1:]
    stage_run(a[1], int(a[2]) if len(a) > 2 else 99)
