"""CA4: споры на манипуляторе 4D с переменной целью. Слои и рёбра графа (стенсилы) от цели не зависят — строятся один раз (слои L_g300so.pkl, рёбра edges stage);
на новую цель: s.goal = узлы в ящике вокруг q_g, Vg = tgoal (CPU), решение V на готовых рёбрах (GPU, LOADE), запрос агента с целью-центром.
Этапы (aida, ~/spore_v5/c1/manip_run, cwd там же; growN/finish_gen/L_g300so.pkl копии wb5):
  edges: python3 manip_goals_run.py edges        (CPU, один раз; SAVEE → edges_*.npy)
  run:   venv-sb3 python manip_goals_run.py run A|B [K]   (GPU; на каждую цель решить V, запросы переходов; → data/mg_{A,B}_spores.json)
Эталон — data/mg_{A,B}_ref.json (ocp_casadi, вне таймера). Времена: t_edges (раз), t_goal (Vg+solve на новую цель), t_query на переход."""
import os, sys, time, json, pickle
ENV = dict(SYS='manip', KF='11', M='3', FRAC='.5', OVH='.5', MINROWS='3', GLIM='0', RMAX='.6', TMAX='3', SBCAUS='1', SBLAY='0', NOLATCH='1', FINGRID='2', VF='2', PESS='1', GPUMAX='1000000', GS='300', DELTA='.3', MAXC='3000', SONE='1', TQDM_MI='1000', OMP_NUM_THREADS='4')
for k, v in ENV.items(): os.environ.setdefault(k, v)
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.expanduser('~/spore_v5/c1/manip_run'); sys.path.insert(0, HERE); sys.path.insert(0, SRC)
import numpy as np, common
import growN as G, finish_gen as FG
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
EDG = os.path.join(SRC, 'edges'); GC = np.zeros(4)

def set_goal(q):
    """центр цели (q1, q2, 0, 0): подмена проверок цели в модуле growN (динамика f — в абсолютных q, не меняется)"""
    global GC; GC = np.array([q[0], q[1], 0., 0.])
    G.ingoal = lambda y: (np.abs(G.wrapy(np.asarray(y, float) - GC)) <= G.RHOV + 1e-9).all(-1)
def wgoal(y): return G.wrapy(np.asarray(y, float) - GC)

def shoot_pol_rec(y, tmax, NA, inits=2, H=.1):                      # как в spores_manip, но остаток до центра цели GC
    from scipy.optimize import minimize
    T0, tp = FG.shoot_grid(y, G.f, G.US, G.RHOV, wgoal, tmax, NA, H=H)
    if tp is None: return T0, tp, None
    R = .9 * np.asarray(G.RHOV); best = (T0, tp, None); n = len(tp)
    def end(d):
        z = np.array(y, float)
        for k, dt in zip(tp, d): z = FG.flow(z, G.US[k], max(dt, 0.), G.f)
        return wgoal(z)
    cons = [{"type": "ineq", "fun": lambda d: R - np.abs(end(d))}]
    for d0 in (np.full(n, T0 / n), np.r_[T0 * .6, np.full(n - 1, T0 * .4 / max(n - 1, 1))][:n]):
        r = minimize(lambda d: d.sum(), d0, method="SLSQP", bounds=[(0, tmax)] * n, constraints=cons, options=dict(maxiter=60, ftol=1e-7))
        if r.success and np.all(np.abs(end(r.x)) <= np.asarray(G.RHOV)) and r.x.sum() < best[0]: best = (float(r.x.sum()), tp, r.x.copy())
    return best

def rollout_rec(A, Q, tmax=40.):
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

def stage_edges():
    set_goal([0., 0.]); A = load_atlas(); A.finish(); os.environ['SAVEE'] = EDG; t = time.perf_counter()
    try: A.solve(it=1)
    except Exception as ex: print('solve it=1:', repr(ex)[:200])
    print('EDGESDONE t_edges %.0f с' % (time.perf_counter() - t), 'узлов', A.N, flush=True)

def solve_goal(A, q):
    """Vg + V на готовых рёбрах для цели q → (t_vg, t_solve)"""
    t0 = time.perf_counter(); set_goal(q); A.finish(); Vg = np.full(A.N, np.inf)
    for u in G.US: Vg = np.minimum(Vg, A.tgoal(A.P, u))
    np.save(EDG + 'g_Vg.npy', Vg)
    for k in ('I', 'IDX', 'W'):
        if not os.path.exists(EDG + 'g_%s.npy' % k): os.symlink(EDG + '_%s.npy' % k, EDG + 'g_%s.npy' % k)
    t1 = time.perf_counter(); os.environ['LOADE'] = EDG + 'g'; os.environ['SOLVEGPU'] = '1'; A.solve(); return t1 - t0, time.perf_counter() - t1

def stage_run(kind, K):
    import manip_goals as MGm
    tr, _ = MGm.transitions(kind); tr = tr[:K]; ref = {r['k']: r['Tref'] for r in json.load(open(os.path.join(common.DATA, 'mg_%s_ref.json' % kind)))}
    A = load_atlas(); res = []; seen = {}
    for k, (st, g) in enumerate(tr):
        key = tuple(np.round(g, 9)); tg = None
        if key not in seen:
            tv, ts = solve_goal(A, g); seen[key] = A.V.copy(); tg = tv + ts
            print('цель', k, 'Vg %.1f с solve %.1f с' % (tv, ts), 'конечных %.3f' % float((A.V < G.BIG / 2).mean()), 'узлов цели', int(A.goal.sum()), flush=True)
        else: set_goal(g); A.finish(); A.V = seen[key]    # V этой цели уже решена (A: M решений, не K)
        e = MGm.MG(g, st); t1 = time.perf_counter(); T, seg = rollout_rec(A, e.X0); dtq = time.perf_counter() - t1; sg = seg[0]
        Tv, why = common.verify(e, e.X0[0], [list(u) for u, d in sg], [d for u, d in sg])
        res.append(dict(k=k, start=list(map(float, st)), goal=list(map(float, g)), Tref=ref.get(k), T_agent=float(T[0]) if np.isfinite(T[0]) else None, T_verify=Tv, why=why, t_goal=tg, t_query=dtq))
        print(k, res[-1], flush=True)
    json.dump(res, open(os.path.join(common.DATA, 'mg_%s_spores.json' % kind), 'w'), indent=1, default=float)

if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'edges': stage_edges()
    else: stage_run(a[1], int(a[2]) if len(a) > 2 else 99)
