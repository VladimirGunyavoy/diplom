"""R20×100 динамик кар: OCP-ясновидец на каждую пару (сид дисков k, старт i). Запуск: STARTS=starts100.npy NP=24 python r20_dyn_ref.py k0 k1 [TIGHT=1 — тесная раскладка].
Результат (по сиду, возобновляемый): data/r20_ref_s{k}{_tight}.json = список {k,i,T,wall,clr0}. T=None — не сошёлся; clr0<0 — старт внутри диска при t=0 (пара выкидывается из итога явно, не молча)."""
import os, sys, json, time, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, envs_dyn as D
X0 = None
def one(a):
    k, i = a; x0 = X0[i, :4]; H = D.Hazards('random', k); clr0 = float((np.linalg.norm(H.pos(0.)[:, :] - x0[:2], axis=1) - D.HR).min())
    if clr0 < 0: return dict(k=k, i=i, T=None, wall=0., clr0=clr0)
    rng = np.random.default_rng(i); t0 = time.perf_counter()
    try: r = D.make_oracle()(x0, H, D.vias(x0, rng))
    except Exception as ex: r = None
    return dict(k=k, i=i, T=None if r is None else float(r[0]), wall=time.perf_counter() - t0, clr0=clr0)
if __name__ == '__main__':
    import common
    X0 = np.load(os.path.expanduser(os.environ['STARTS'])) if os.environ.get('STARTS') else D.starts(20)
    if os.environ.get('NST'): X0 = X0[:int(os.environ['NST'])]
    suf = '_tight' if int(os.environ.get('TIGHT', 0)) else ''
    for k in range(int(sys.argv[1]), int(sys.argv[2])):
        f = os.path.join(common.DATA, 'r20_ref_s%d%s.json' % (k, suf))
        if os.path.exists(f): continue
        t0 = time.perf_counter()
        with mp.get_context('fork').Pool(int(os.environ.get('NP', 24))) as p: res = p.map(one, [(k, i) for i in range(len(X0))], chunksize=1)
        json.dump(res, open(f, 'w')); Ts = [r['T'] for r in res if r['T']]
        print('сид %d: сошлись %d/%d, внутри диска %d, wall %.0f с, медиана T %.2f' % (k, len(Ts), len(res), sum(r['clr0'] < 0 for r in res), time.perf_counter() - t0, np.median(Ts) if Ts else float('nan')), flush=True)
