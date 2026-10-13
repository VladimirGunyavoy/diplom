"""C3: RRT и SST из OMPL (control::RRT, control::SST) на Env bench. Запуск (aida, ~/venv-ompl): python ompl_run.py ENV ALGO [--budgets .3,1,3,10,60] [--sel .2 --prune .1] [--starts 0,1,2] [--procs 16] [--tag x]
Протокол: goal bias .05, шаг propagation .05, длительность 1..20 шагов, управление — равномерно в U, метрика — взвешенная (вес 1/полуширина коробки; углы SO2), стоимость — время (измерение t с весом 0).
SST — в sst_own.py (в pip-OMPL нет конструктора control::SST). SST: anytime — один планировщик, solve() досчитывается до накопленных бюджетов, после каждого — лучший путь (проверяльщик). RRT: до первого решения (≤ max budget)."""
import os, sys, time, json, argparse, numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
from ompl import base as ob, control as oc, util as ou
import common
PSTEP = .05; ENV = None; ARGS = None
def _set(s, x):
    try: s[0] = x
    except TypeError: s.value = x      # SO2State в этой сборке OMPL не поддерживает s[0] = x
def _get(s):
    try: return s[0]
    except TypeError: return s.value


def plan(e, y0, algo, budgets, sel, prune, seed):
    ou.RNG.setSeed(seed); sc = (e.hi - e.lo) / 2; cs = ob.CompoundStateSpace(); subs = []
    for i in range(e.n):
        if e.period[i] > 0: sp = ob.SO2StateSpace(); cs.addSubspace(sp, float(e.period[i]) / 2 / np.pi * 0 + 1.0 / np.pi)
        else:
            sp = ob.RealVectorStateSpace(1); b = ob.RealVectorBounds(1); b.setLow(float(e.lo[i])); b.setHigh(float(e.hi[i])); sp.setBounds(b); cs.addSubspace(sp, float(1.0 / sc[i]))
        subs.append(sp)
    tsp = ob.RealVectorStateSpace(1); b = ob.RealVectorBounds(1); b.setLow(0); b.setHigh(1e4); tsp.setBounds(b); cs.addSubspace(tsp, 0.0)
    cspace = oc.RealVectorControlSpace(cs, e.m); cb = ob.RealVectorBounds(e.m)
    for k in range(e.m): cb.setLow(k, float(e.umin[k])); cb.setHigh(k, float(e.umax[k]))
    cspace.setBounds(cb); si = oc.SpaceInformation(cs, cspace); n = e.n
    def getY(st): return np.array([_get(st[i]) for i in range(n)])
    def propagate(st, ctrl, dur, res):
        y = getY(st)[None]; u = np.array([[ctrl[k] for k in range(e.m)]]); k = max(1, int(round(dur / PSTEP)))
        y = e.step(y, u, dur / k) if getattr(e, 'exact', False) else _multi(e, y, u, dur, k)
        for i in range(n): _set(res[i], float(y[0, i]))
        res[n][0] = st[n][0] + dur
    def _multi(e, y, u, dur, k):
        h = dur / k
        for _ in range(k): y = e.step(y, u, h)
        return y
    si.setStatePropagator(propagate); si.setPropagationStepSize(PSTEP); si.setMinMaxControlDuration(1, 20)
    si.setup()
    grng = np.random.default_rng(seed)
    class G(ob.GoalSampleableRegion):
        def __init__(s, si): super().__init__(si); s.setThreshold(0.)
        def distanceGoal(s, st): return float(max(0., e.goal_dist(getY(st)[None])[0]))
        def maxSampleCount(s): return 100000
        def sampleGoal(s, st):
            if e.ball is not None:
                v = grng.normal(size=e.n); v *= e.ball * grng.random() ** (1. / e.n) / np.linalg.norm(v)
            else: v = grng.uniform(-e.half, e.half)
            for i in range(n): _set(st[i], float(v[i]))
            st[n][0] = 0.
    class Cost(ob.OptimizationObjective):
        def __init__(s, si): super().__init__(si)
        def motionCost(s, a, b): return ob.Cost(float(b[n][0] - a[n][0]))
    pdef = ob.ProblemDefinition(si); st0 = cs.allocState()
    for i in range(n): _set(st0[i], float(y0[i]))
    st0[n][0] = 0.; pdef.addStartState(st0); pdef.setGoal(G(si)); obj = Cost(si); pdef.setOptimizationObjective(obj)
    pl = oc.RRT(si)
    pl.setGoalBias(.05)
    pl.setProblemDefinition(pdef); pl.setup(); out = []; t_used = 0.; t_cum = 0.; e.nf = 0; t0 = time.perf_counter()
    for B in budgets:
        remaining = B - (time.perf_counter() - t0)
        if remaining > 0 and not (out and out[-1]['has']):
            pl.solve(ob.timedPlannerTerminationCondition(remaining))
        el = time.perf_counter() - t0; has = pdef.hasExactSolution(); rec = dict(budget=B, t=el, has=bool(has), nf=int(e.nf))
        if has:
            p = pdef.getSolutionPath(); us = [[p.getControl(i)[k] for k in range(e.m)] for i in range(p.getControlCount())]; dts = [p.getControlDuration(i) for i in range(p.getControlCount())]
            T, why = common.verify(e, y0, us, dts); rec.update(T=T, why=why, T_plan=float(p.getState(p.getStateCount() - 1)[n][0]))
        out.append(rec)
    return out

def work(a):
    i, algo, budgets, sel, prune = a; e = common.make(ENV); out = plan(e, e.X0[i], algo, budgets, sel, prune, 1000 + i); return dict(i=i, runs=out)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('env'); p.add_argument('algo'); p.add_argument('--budgets', default='.3,1,3,10,60'); p.add_argument('--sel', type=float, default=.2); p.add_argument('--prune', type=float, default=.1)
    p.add_argument('--starts', default=''); p.add_argument('--procs', type=int, default=16); p.add_argument('--tag', default=''); a = p.parse_args(); ENV = a.env; e = common.make(a.env)
    idx = [int(x) for x in a.starts.split(',')] if a.starts else list(range(len(e.X0))); budgets = [float(x) for x in a.budgets.split(',')]
    t0 = time.time()
    with Pool(a.procs) as pool: R = pool.map(work, [(i, a.algo, budgets, a.sel, a.prune) for i in idx])
    per = []
    for bi, B in enumerate(budgets):
        Ts = [None] * len(e.X0)
        for r in R: Ts[r['i']] = r['runs'][bi].get('T') if r['runs'][bi]['has'] else None
        sub = np.zeros(len(e.X0), bool); sub[idx] = True; e2 = e; keep0 = e.keep; e.keep = keep0 & sub; s = common.summarize(e, Ts); e.keep = keep0
        s.update(budget=B, t_wall_sum=float(sum(r['runs'][bi]['t'] for r in R)), nf_med=float(np.median([r['runs'][bi]['nf'] for r in R]))); per.append(s)
        print(a.algo, 'budget', B, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
    os.makedirs(common.DATA, exist_ok=True); json.dump(dict(env=a.env, algo=a.algo, sel=a.sel, prune=a.prune, starts=idx, budgets=per, raw=R, Tref=e.Tref.tolist(), hw='CPU 1 proc/запрос (OMPL, Python-колбэк)'), open(os.path.join(common.DATA, f'{a.env}_{a.algo}{a.tag}.json'), 'w'), indent=1, default=float)
