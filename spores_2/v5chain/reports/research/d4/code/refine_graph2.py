"""d4 [4]: путь графа m3d (g6d: us,dts) -> IPOPT-доводка коридора SPD6 (refine_ca.CA, wlim, запас .004). Пороги senior-3: мед<=1.05 и >=19/20."""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.expanduser('~/spore_v5/d4/spores_2/bench'); sys.path.insert(0, HERE)
import numpy as np, common, envs_6d
sys.path.insert(0, os.path.join(HERE, '..', 'v6'))
import manip3d_dyn as M3
import refine_ca_d4 as refine_ca, multiprocessing as mp
JS = sys.argv[1]; OUT = sys.argv[2]
e = common.make('m3d'); US_ = M3.US; WM = 3.
fl = lambda P, s, t: M3.flow(P, s, t, dt_max=.05)
Rq, Rw = .3, .6; wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
g = lambda x: np.array([Rq ** 2 - wr(x[0]) ** 2, Rq ** 2 - wr(x[1]) ** 2, Rq ** 2 - wr(x[2]) ** 2, Rw ** 2 - x[3] ** 2, Rw ** 2 - x[4] ** 2, Rw ** 2 - x[5] ** 2])
D = json.load(open(JS)); CA = refine_ca.CA('m3d', ms=10, maxit=int(os.environ.get('CAIT', 60)))
def idx(u): return int(np.argmin(np.abs(US_ - np.asarray(u)).sum(1)))
def work(r):
    i = r['i']; path = [(idx(u), t) for u, t in zip(r['us'], r['dts'])]; t0 = time.perf_counter()
    Tg, whyg = common.verify(e, e.X0[i], r['us'], r['dts'])
    try: T, seq, d, ok = CA.refine(fl, e.X0[i], path, g, tries=3, wlim=((3, 4, 5), WM))
    except Exception as ex: return dict(i=i, err=str(ex), Tg=Tg)
    Tv, why = common.verify(e, e.X0[i], [US_[s] for s in seq], list(d)) if ok else (None, 'refine_fail')
    return dict(i=i, Tg=Tg, whyg=str(whyg), Tref=float(e.Tref[i]), Tr=Tv, why=str(why), ok=bool(ok), nseg=len(seq), t=time.perf_counter() - t0)
if __name__ == '__main__':
    with mp.get_context('fork').Pool(int(os.environ.get('PROCS', 4))) as p: res = p.map(work, D['res'], chunksize=1)
    for r in res: print(r, flush=True)
    rg = np.array([r['Tg'] / r['Tref'] for r in res if r.get('Tg') is not None and r['Tg'] <= 3 * r['Tref']]); rr = np.array([r['Tr'] / r['Tref'] for r in res if r.get('Tr') is not None and r['Tr'] <= 3 * r['Tref']])
    print('ГРАФ (до): дошли %d/%d мед %.3f p90 %.3f max %.3f' % (len(rg), len(res), np.median(rg), np.percentile(rg, 90), rg.max()))
    print('ГРАФ+IPOPT (после): дошли %d/%d мед %.3f p90 %.3f max %.3f; viol box %d' % (len(rr), len(res), np.median(rr), np.percentile(rr, 90), rr.max(), sum(1 for r in res if r.get('why') == 'box')))
    json.dump(res, open(OUT, 'w'), default=float)
