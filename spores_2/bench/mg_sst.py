"""CA4: SST на каждый переход манипулятора (MG, старт = предыдущая цель, w=0), prune .2 как в manip_sst, бюджеты .3/1/3/10/60 с, 1 процесс на переход (aida, CPU).
python mg_sst.py A|B [--procs 16] → data/mg_{A,B}_sst.json (на переход и бюджет: T_verify, успех, время)."""
import os, sys, json, argparse, numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import common, sst_own, manip_goals as MGm
BUD = [.3, 1., 3., 10., 60.]
def work(a):
    kind, k = a; tr, _ = MGm.transitions(kind); st, g = tr[k]; e = MGm.MG(g, st); snaps = sst_own.sst(e, e.X0[0], BUD, sel=.4, prune=.2, seed=1000 + k); out = []
    for s in snaps:
        r = dict(budget=s['budget'], t=s['t'], has=s['has'], nf=s['nf'])
        if s['has']: T, why = common.verify(e, e.X0[0], s['us'], s['dts']); r.update(T=T, why=why)
        out.append(r)
    return dict(k=k, runs=out)
if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('kind'); p.add_argument('--procs', type=int, default=16); a = p.parse_args(); n = len(MGm.transitions(a.kind)[0])
    with Pool(a.procs) as pool: R = pool.map(work, [(a.kind, k) for k in range(n)])
    json.dump(R, open(os.path.join(common.DATA, 'mg_%s_sst.json' % a.kind), 'w'), indent=1, default=float)
    ref = {r['k']: r['Tref'] for r in json.load(open(os.path.join(common.DATA, 'mg_%s_ref.json' % a.kind)))}
    for bi, B in enumerate(BUD):
        ok = [r['runs'][bi].get('T') for r in R]; rr = [t / ref[r['k']] for t, r in zip(ok, R) if t is not None]
        print('SST', a.kind, 'budget', B, 'дошли %d/%d' % (len(rr), n), 'T/Tref мед %.2f' % (np.median(rr) if rr else np.nan), flush=True)
    print('MGSSTDONE')
