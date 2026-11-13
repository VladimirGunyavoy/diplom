"""d4 [А]: путь графа (tvg_lp, LP-агент, WMARG) -> SLSQP-доводка коридора (corridor_nd.refine, как e2/corridor_env.py) на dp1/manip. usage: refine_sl.py ENV NB 8 SEED  (env WMARG=.05, PROCS)."""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('W43', '~/spore_v5/d4')
ARGS = sys.argv[1:5]; HERE_ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE_ + '/spores_2/w39n'); sys.path.insert(0, os.path.expanduser('~/spore_v5/d4/spores_2/w39n'))
sys.argv = ['x'] + ARGS
import numpy as np
import tvg_lp as L
T = L.T; env = T.env; d = T.d; ts = T.TS
sys.path.insert(0, os.path.expanduser('~/spore_v5/d4/e2')); import corridor_env as CE   # argv[1]=ENV, argv[2] -> tag (безвредно)
E, S2, US, FL, G, MISS, WLIM = CE.setup()
from src.atlas6.corridor_nd import refine
import common
S = T.mkS(ts); rng_ = np.random.default_rng(1000 + ts); lo_ = np.where(env.period > 0, -env.period / 2, T.LO); hi_ = np.where(env.period > 0, env.period / 2, T.HI)
P_ = rng_.uniform(lo_, hi_, (2 * T.NB, d)); P_ = P_[T.okv(P_)][:T.NB]; back = (P_, np.zeros(len(P_)), None, None, None, T.cKDTree(T._scaled(S, T._images(S, P_))))
ag = L.AgentLP(S, back); ag.mode = 'ip'; T._MG[0] = T.WMARG
_rec = {}; _ov = common.verify
def _v(env_, y0, us_, dts_, *a, **k): _rec['us'] = list(us_); _rec['dts'] = list(dts_); return _ov(env_, y0, us_, dts_, *a, **k)
def idx(u): return int(np.argmin(np.abs(US - np.asarray(u)).sum(1)))
def work(i):
    x0 = env.X0[i]; common.verify = _v; t0 = time.perf_counter(); Tv, why, m = T.episode(ag, x0); common.verify = _ov; tq = time.perf_counter() - t0
    us = _rec['us']; dts = _rec['dts']; path = [(idx(u), t) for u, t in zip(us, dts)]; Tg, whyg = common.verify(E, E.X0[i], us, dts); t1 = time.perf_counter()
    try: Tr, seq, dd, ok = refine(FL, E.X0[i], path, G, tries=3, wlim=WLIM)
    except Exception as ex: return dict(i=i, err=str(ex)[:100], Tg=Tg, whyg=str(whyg), Tref=float(E.Tref[i]), tq=tq)
    Tv2, why2 = common.verify(E, E.X0[i], [US[s] for s in seq], list(dd)) if ok else (None, 'refine_fail')
    return dict(i=i, Tg=Tg, whyg=str(whyg), Tref=float(E.Tref[i]), Tr=Tv2, why=str(why2), ok=bool(ok), nseg=len(seq), t_ref=time.perf_counter() - t1, tq=tq)
def st(x, n):
    x = np.array([v for v in x if v is not None]); x = x[x <= 3]
    return 'дошли %d/%d мед %.3f p90 %.3f max %.3f' % (len(x), n, np.median(x), np.percentile(x, 90), x.max()) if len(x) else 'дошли 0/%d' % n
if __name__ == '__main__':
    import multiprocessing as mp
    print('ПАСПОРТ refine_sl env=%s NB=%s seed=%d WMARG=%s; путь графа (tvg_lp LP-агент) -> corridor_nd.refine SLSQP tries 3 wlim=%s, окно half-WIN(%.3f); старты/эталон common.make(env) n=%d md5 Tref %s; проверяльщик common.verify' % (ARGS[0], ARGS[1], ts, T.WMARG, WLIM, CE.WIN, len(E.X0), __import__('hashlib').md5(np.asarray(E.Tref, float).tobytes()).hexdigest()[:8]), flush=True)
    with mp.get_context('fork').Pool(int(os.environ.get('PROCS', 4))) as p: res = p.map(work, range(len(E.X0)), chunksize=1)
    n = len(res); Rg = [r['Tg'] / r['Tref'] if r.get('Tg') is not None else None for r in res]; Rr = [r['Tr'] / r['Tref'] if r.get('Tr') is not None else None for r in res]
    Rf = [Rr[i] if Rr[i] is not None else Rg[i] for i in range(n)]
    for r in res: print(r, flush=True)
    print('ГРАФ (до): ' + st(Rg, n) + '; viol box %d' % sum(1 for r in res if r.get('whyg') == 'box')); print('ГРАФ+SLSQP: ' + st(Rr, n) + '; viol box %d' % sum(1 for r in res if r.get('why') == 'box')); print('ГРАФ+SLSQP с откатом на путь графа: ' + st(Rf, n))
    com = [i for i in range(n) if Rg[i] is not None and Rg[i] <= 3 and Rr[i] is not None and Rr[i] <= 3]
    if com: print('на ОБЩИХ дошедших (%d): до мед %.3f p90 %.3f | после мед %.3f p90 %.3f' % (len(com), np.median([Rg[i] for i in com]), np.percentile([Rg[i] for i in com], 90), np.median([Rr[i] for i in com]), np.percentile([Rr[i] for i in com], 90)))
    print('время: доводка мед %.1f / p90 %.1f с; запрос графа мед %.2f с' % (np.median([r.get('t_ref', 0) for r in res]), np.percentile([r.get('t_ref', 0) for r in res], 90), np.median([r['tq'] for r in res])))
    json.dump(res, open(os.path.expanduser('~/spore_v5/d4/out/rsl_%s_%s_%d.json' % (ARGS[0], ARGS[1], ts)), 'w'), default=float)
