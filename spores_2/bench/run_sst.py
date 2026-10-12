"""C3: SST (sst_own) — параллельно по стартам (1 процесс на запрос), anytime-бюджеты. python run_sst.py ENV [--budgets .3,1,3,10,60] [--prune .1 (sel=2·prune)] [--starts 0,1] [--procs 16] [--tag _x]"""
import os, sys, json, argparse, time, numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import common, sst_own, rrtstar_di
ENV = None; ALGO = 'sst'
def work(a):
    i, budgets, sel, prune = a; e = common.make(ENV); snaps = sst_own.sst(e, e.X0[i], budgets, sel=sel, prune=prune, seed=1000 + i) if ALGO == 'sst' else sst_own.rrt(e, e.X0[i], budgets, seed=1000 + i) if ALGO == 'rrt' else rrtstar_di.rrtstar(e, e.X0[i], budgets, seed=1000 + i); out = []
    for s in snaps:
        r = dict(budget=s['budget'], t=s['t'], has=s['has'], nodes=s['nodes'], nf=s['nf'])
        if s['has']: T, why = common.verify(e, e.X0[i], s['us'], s['dts']); r.update(T=T, why=why)
        out.append(r)
    return dict(i=i, runs=out)
if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('env'); p.add_argument('--algo', default='sst'); p.add_argument('--budgets', default='.3,1,3,10,60'); p.add_argument('--prune', type=float, default=.1); p.add_argument('--starts', default=''); p.add_argument('--procs', type=int, default=16); p.add_argument('--tag', default=''); a = p.parse_args(); ENV = a.env; ALGO = a.algo
    e = common.make(a.env); idx = [int(x) for x in a.starts.split(',')] if a.starts else list(range(len(e.X0))); budgets = [float(x) for x in a.budgets.split(',')]
    with Pool(a.procs) as pool: R = pool.map(work, [(i, budgets, 2 * a.prune, a.prune) for i in idx])
    per = []
    for bi, B in enumerate(budgets):
        Ts = [None] * len(e.X0)
        for r in R: Ts[r['i']] = r['runs'][bi].get('T') if r['runs'][bi]['has'] else None
        sub = np.zeros(len(e.X0), bool); sub[idx] = True; k0 = e.keep; e.keep = k0 & sub; s = common.summarize(e, Ts); e.keep = k0
        s.update(budget=B, t_wall_sum=float(sum(r['runs'][bi]['t'] for r in R)), nf_med=float(np.median([r['runs'][bi]['nf'] for r in R]))); per.append(s)
        print(a.algo, 'prune', a.prune, 'budget', B, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
    os.makedirs(common.DATA, exist_ok=True); json.dump(dict(env=a.env, algo=a.algo, prune=a.prune, sel=2 * a.prune, starts=idx, budgets=per, raw=R, Tref=e.Tref.tolist(), hw='CPU 1 proc/запрос, numpy'), open(os.path.join(common.DATA, f'{a.env}_{a.algo}{a.tag}.json'), 'w'), indent=1, default=float)
