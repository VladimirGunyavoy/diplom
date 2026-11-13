"""d4: путь графа (g6d: us,dts) -> IPOPT-доводка SPD6 (refine_ca_d4.CA, wlim, запас .004) для m3d/m3gu/m3gd. usage: refine_graph3.py ENV g6d.json out.json [CAIT, CADNEG, PROCS из env]"""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.expanduser('~/spore_v5/d4/spores_2/bench'); sys.path.insert(0, HERE)
import numpy as np, common, envs_6d
sys.path.insert(0, os.path.join(HERE, '..', 'v6'))
import refine_ca_d4 as refine_ca, multiprocessing as mp
ENV = sys.argv[1]; JS = sys.argv[2]; OUT = sys.argv[3]
e = common.make(ENV); WM = 3.; Rq, Rw = .3, .6; wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi; C3 = np.array(getattr(e, 'C3', np.zeros(6)), float)
if ENV == 'm3d':
    import manip3d_dyn as M3; US_ = M3.US; fl = lambda P, s, t: M3.flow(P, s, t, dt_max=.05)
else:
    import manip3g_dyn as G3; US_ = np.array(G3.LAYERS6); fl = lambda P, s, t: G3.flow(P, s, t, dt_max=.05, g=e.G)
g = lambda x: np.array([Rq ** 2 - wr(x[0] - C3[0]) ** 2, Rq ** 2 - wr(x[1]) ** 2, Rq ** 2 - wr(x[2]) ** 2, Rw ** 2 - x[3] ** 2, Rw ** 2 - x[4] ** 2, Rw ** 2 - x[5] ** 2])
D = json.load(open(JS)); CA = refine_ca.CA(ENV, ms=10, maxit=int(os.environ.get('CAIT', 60)))
def idx(u): return int(np.argmin(np.abs(US_ - np.asarray(u)).sum(1)))
def work(r):
    i = r['i']; path = [(idx(u), t) for u, t in zip(r['us'], r['dts'])]; t0 = time.perf_counter()
    Tg, whyg = common.verify(e, e.X0[i], r['us'], r['dts'])
    try: T, seq, d, ok = CA.refine(fl, e.X0[i], path, g, tries=3, wlim=((3, 4, 5), WM))
    except Exception as ex: return dict(i=i, err=str(ex), Tg=Tg, Tref=float(e.Tref[i]), tq=r['t_query'])
    Tv, why = common.verify(e, e.X0[i], [US_[s] for s in seq], list(d)) if ok else (None, 'refine_fail')
    return dict(i=i, Tg=Tg, whyg=str(whyg), Tref=float(e.Tref[i]), Tr=Tv, why=str(why), ok=bool(ok), nseg=len(seq), t_ref=time.perf_counter() - t0, tq=r['t_query'])
def st(x, n):
    x = np.array([v for v in x if v is not None]); x = x[x <= 3]
    return 'дошли %d/%d мед %.3f p90 %.3f max %.3f' % (len(x), n, np.median(x), np.percentile(x, 90), x.max()) if len(x) else 'дошли 0/%d' % n
if __name__ == '__main__':
    print('ПАСПОРТ refine_graph3 env=%s CAIT=%s CADNEG=%s tries=3 wlim запас .004; вход %s; проверяльщик common.verify, допуск |w|<=3+%g' % (ENV, os.environ.get('CAIT', 60), os.environ.get('CADNEG', '1e-9'), JS, e.TOLW), flush=True)
    with mp.get_context('fork').Pool(int(os.environ.get('PROCS', 4))) as p: res = p.map(work, D['res'], chunksize=1)
    n = len(res); Rg = [r['Tg'] / r['Tref'] if r.get('Tg') is not None else None for r in res]; Rr = [r['Tr'] / r['Tref'] if r.get('Tr') is not None else None for r in res]
    com = [i for i in range(n) if Rg[i] is not None and Rg[i] <= 3 and Rr[i] is not None and Rr[i] <= 3]
    print('ГРАФ (до): ' + st(Rg, n)); print('ГРАФ+IPOPT (после): ' + st(Rr, n) + '; viol box %d' % sum(1 for r in res if r.get('why') == 'box'))
    print('на ОБЩИХ дошедших (%d): до мед %.3f p90 %.3f | после мед %.3f p90 %.3f' % (len(com), np.median([Rg[i] for i in com]), np.percentile([Rg[i] for i in com], 90), np.median([Rr[i] for i in com]), np.percentile([Rr[i] for i in com], 90)))
    miss = [i for i in range(n) if Rg[i] is None or Rg[i] > 3]; print('недошедшие графом: %d; из них доводка вытянула %d: %s' % (len(miss), sum(1 for i in miss if Rr[i] is not None and Rr[i] <= 3), [i for i in miss if Rr[i] is not None and Rr[i] <= 3]))
    print('время: запрос графа мед %.2f с, доводка мед %.2f / p90 %.2f с, сумма запрос+доводка мед %.2f с' % (np.median([r['tq'] for r in res]), np.median([r['t_ref'] for r in res if 't_ref' in r]), np.percentile([r['t_ref'] for r in res if 't_ref' in r], 90), np.median([r['tq'] + r.get('t_ref', 0) for r in res])))
    json.dump(res, open(OUT, 'w'), default=float)
