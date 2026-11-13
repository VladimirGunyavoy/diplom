"""d4 [А'] 2x2: путь графа (tvg_lp, WMARG .05) -> SLSQP-доводка (копия corridor_nd.refine, параметры) ; (i) split: дробить пути до >=4 отрезков; (ii) dense: ns 16->64 точек |w| с запасом .05. Варианты A=ни одного(контроль=прежняя refine), B=split, C=dense, D=оба."""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('W43', '~/spore_v5/d4')
ARGS = sys.argv[1:5]; sys.path.insert(0, os.path.expanduser('~/spore_v5/d4/spores_2/w39n')); sys.argv = ['x'] + ARGS
import numpy as np
import tvg_lp as L
T = L.T; env = T.env; d = T.d; ts = T.TS
sys.path.insert(0, os.path.expanduser('~/spore_v5/d4/e2')); import corridor_env as CE
E, S2, US, FL, G, MISS, WLIM = CE.setup()
from src.atlas6.corridor_nd import merge, endpoint
from scipy.optimize import minimize
import common
S = T.mkS(ts); rng_ = np.random.default_rng(1000 + ts); lo_ = np.where(env.period > 0, -env.period / 2, T.LO); hi_ = np.where(env.period > 0, env.period / 2, T.HI)
P_ = rng_.uniform(lo_, hi_, (2 * T.NB, d)); P_ = P_[T.okv(P_)][:T.NB]; back = (P_, np.zeros(len(P_)), None, None, None, T.cKDTree(T._scaled(S, T._images(S, P_))))
ag = L.AgentLP(S, back); ag.mode = 'ip'; T._MG[0] = T.WMARG
_rec = {}; _ov = common.verify
def _v(env_, y0, us_, dts_, *a, **k): _rec['us'] = list(us_); _rec['dts'] = list(dts_); return _ov(env_, y0, us_, dts_, *a, **k)
def idx(u): return int(np.argmin(np.abs(US - np.asarray(u)).sum(1)))
def refine_v(flow, x0, path, g, split, ns, wmarg, wlim, tries=3, seed=0):
    seq, d0 = merge(path); seq = list(seq); d0 = np.array(d0, float)
    if split:
        while len(seq) < 4:
            i = int(np.argmax(d0)); d0[i] /= 2; seq.insert(i + 1, seq[i]); d0 = np.insert(d0, i + 1, d0[i])
    rng = np.random.default_rng(seed); best = (np.inf, None); WMm = wlim[1] - wmarg
    def gg(dv):
        x = np.array(x0, float); pts = []
        for sg, t in zip(seq, dv):
            for _ in range(ns): x = flow(x, sg, t / ns); pts.append(x)
        P = np.array(pts); return np.concatenate([np.atleast_1d(g(x)), np.ravel(WMm - np.abs(P[:, wlim[0]]))])
    con = {'type': 'ineq', 'fun': gg}
    for k in range(tries):
        di = d0 if k == 0 else d0 * rng.uniform(0.8, 1.2, len(d0))
        r = minimize(lambda dv: dv.sum(), di, jac=lambda dv: np.ones_like(dv), bounds=[(0, 1.5 * d0.sum() + 1)] * len(d0), constraints=[con], method='SLSQP', options=dict(maxiter=80, ftol=1e-9))
        if np.all(con['fun'](r.x) >= -1e-6) and r.x.sum() < best[0]: best = (float(r.x.sum()), r.x)
    ok = best[1] is not None
    return (best[0] if ok else float(d0.sum())), seq, (best[1] if ok else d0), ok
VAR = dict(A=(False, 16, 0.), B=(True, 16, 0.), C=(False, 64, .05), D=(True, 64, .05))
def work(i):
    x0 = env.X0[i]; common.verify = _v; t0 = time.perf_counter(); Tv, why, m = T.episode(ag, x0); common.verify = _ov; tq = time.perf_counter() - t0
    us = _rec['us']; dts = _rec['dts']; path = [(idx(u), t) for u, t in zip(us, dts)]; Tg, whyg = common.verify(E, E.X0[i], us, dts)
    out = dict(i=i, Tg=Tg, whyg=str(whyg), Tref=float(E.Tref[i]), tq=tq, nseg0=len(merge(path)[0]), us=[list(map(float, u)) for u in us], dts=[float(t) for t in dts])
    for v, (sp, ns, wm) in VAR.items():
        t1 = time.perf_counter()
        try: Tr, seq, dd, ok = refine_v(FL, E.X0[i], path, G, sp, ns, wm, WLIM)
        except Exception as ex: out[v] = dict(err=str(ex)[:80]); continue
        Tv2, why2 = common.verify(E, E.X0[i], [US[s] for s in seq], list(dd)) if ok else (None, 'refine_fail')
        out[v] = dict(Tr=Tv2, why=str(why2), ok=bool(ok), nseg=len(seq), t=time.perf_counter() - t1)
    return out
def st(x, n):
    x = np.array([v for v in x if v is not None]); x = x[x <= 3]
    return 'дошли %d/%d мед %.3f p90 %.3f max %.3f' % (len(x), n, np.median(x), np.percentile(x, 90), x.max()) if len(x) else 'дошли 0/%d' % n
if __name__ == '__main__':
    import multiprocessing as mp
    print('ПАСПОРТ refine_sl2 env=%s NB=%s seed=%d WMARG=%s; путь графа (tvg_lp) -> копия corridor_nd.refine SLSQP tries 3 maxiter 80; варианты A(split0,ns16,запас0) B(split>=4 отрезков) C(ns64, запас .05) D(B+C); окно half-WIN(%.3f); wlim=%s; старты/эталон common.make(env) n=%d; проверяльщик common.verify' % (ARGS[0], ARGS[1], ts, T.WMARG, CE.WIN, WLIM, len(E.X0)), flush=True)
    with mp.get_context('fork').Pool(int(os.environ.get('PROCS', 4))) as p: res = p.map(work, range(len(E.X0)), chunksize=1)
    n = len(res); Rg = [r['Tg'] / r['Tref'] if r.get('Tg') is not None else None for r in res]; print('ГРАФ (без отката): ' + st(Rg, n) + '; viol box %d; пути с <4 отрезками: %d' % (sum(1 for r in res if r['whyg'] == 'box'), sum(1 for r in res if r['nseg0'] < 4)))
    for v in VAR:
        Rr = [r[v]['Tr'] / r['Tref'] if r[v].get('Tr') is not None else None for r in res]; Rf = [Rr[i] if Rr[i] is not None else Rg[i] for i in range(n)]
        print('%s: %s; viol box %d; refine_fail %d; с откатом: %s; время мед %.1f / p90 %.1f с' % (v, st(Rr, n), sum(1 for r in res if r[v].get('why') == 'box'), sum(1 for r in res if r[v].get('why') == 'refine_fail'), st(Rf, n), np.median([r[v].get('t', 0) for r in res]), np.percentile([r[v].get('t', 0) for r in res], 90)), flush=True)
    json.dump(res, open(os.path.expanduser('~/spore_v5/d4/out/rsl2_%s_%s_%d.json' % (ARGS[0], ARGS[1], ts)), 'w'), default=float)
