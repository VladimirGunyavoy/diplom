"""d4 Г3: A* по состояниям с эвристикой V̂ графа (g6d) только чтобы ДОЙТИ; потом IPOPT-доводка (refine_ca_d4, CAIT/CADNEG из env). env: ENVN NB SEED как у g6d; ASB бюджет с (6), ASBATCH (8)."""
import os, sys, time, json, heapq
os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g6d as GG
import numpy as np, multiprocessing as mp
import refine_ca_d4 as refine_ca
env = GG.env; ENVN = GG.ENVN; BUD = float(os.environ.get('ASB', 6.)); B = int(os.environ.get('ASBATCH', 8)); RES = np.array([.25] * 3 + [.4] * 3)
G = GG.Graph(); W, tv, it, res_ = G.vi(); print('VI %.1f с, конечных %.1f%%' % (tv, 100 * (W < GG.VCAP - 1e-9).mean()), flush=True)
WM = 3.; Rq, Rw = .3, .6; wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi; C3 = np.array(getattr(env, 'C3', np.zeros(6)), float)
if ENVN == 'm3d':
    import manip3d_dyn as M3; fl = lambda P, s, t: M3.flow(P, s, t, dt_max=.05)
else:
    import manip3g_dyn as G3; fl = lambda P, s, t: G3.flow(P, s, t, dt_max=.05, g=env.G)
gwin = lambda x: np.array([Rq ** 2 - wr(x[0] - C3[0]) ** 2, Rq ** 2 - wr(x[1]) ** 2, Rq ** 2 - wr(x[2]) ** 2, Rw ** 2 - x[3] ** 2, Rw ** 2 - x[4] ** 2, Rw ** 2 - x[5] ** 2])
CA = refine_ca.CA(ENVN, ms=10, maxit=int(os.environ.get('CAIT', 60)))
key = lambda x: tuple(np.floor(GG.wrap_ang(x[None])[0] / RES).astype(int))
def astar(x0):
    t0 = time.perf_counter(); st = [np.asarray(x0, float)]; gc = [0.]; par = [-1]; act = [-1]; hh = [float(GG.vhat(G, W, st[0][None])[0])]; best = {key(st[0]): 0.}; op = [(hh[0], 0)]; nexp = 0; goal = None
    while op and goal is None and time.perf_counter() - t0 < BUD:
        pop = [heapq.heappop(op)[1] for _ in range(min(B, len(op)))]; X = np.array([st[k] for k in pop]); Zs, OK, GL = GG.edges(X); nexp += len(pop); ch = []
        for r, k in enumerate(pop):
            fin = np.flatnonzero(GL[r] > 0)
            if len(fin): a = int(fin[np.argmin(GL[r][fin])]); goal = (k, a); break
            for a in np.flatnonzero(OK[r]):
                z = Zs[r, a, -1].astype(float); c = gc[k] + GG.TAUV[a]; kk = key(z)
                if kk in best and best[kk] <= c: continue
                best[kk] = c; ch.append((k, int(a), z, c))
        if goal is not None: break
        if ch:
            H = GG.vhat(G, W, np.array([c[2] for c in ch]))
            for (k, a, z, c), h in zip(ch, H): st.append(z); gc.append(c); par.append(k); act.append(a); hh.append(float(h)); heapq.heappush(op, (c + float(h), len(st) - 1))
    reached = goal is not None
    if reached: k, a = goal; nodes = [(k, a)]
    else:
        k = int(np.argmin(hh)); nodes = []; a = None
    path = []
    if reached: path.append((int(GG.SIDX[a]), float(GG.TAUV[a])))
    while par[k] >= 0: path.append((int(GG.SIDX[act[k]]), float(GG.TAUV[act[k]]))); k = par[k]
    path.reverse(); return path, reached, nexp, time.perf_counter() - t0
def work(i):
    x0 = env.X0[i]; path, reached, nexp, ta = astar(x0); us = [GG.ACT[s] for s, _ in path]; dts = [t for _, t in path]
    import common
    Tg, whyg = common.verify(env, x0, us, dts); t1 = time.perf_counter()
    try: T, seq, d, ok = CA.refine(fl, x0, path, gwin, tries=3, wlim=((3, 4, 5), WM))
    except Exception as ex: return dict(i=i, err=str(ex), Tg=Tg, whyg=str(whyg), reached=reached, nexp=nexp, ta=ta, Tref=float(env.Tref[i]))
    Tv, why = common.verify(env, x0, [GG.ACT[s] for s in seq], list(d)) if ok else (None, 'refine_fail')
    return dict(i=i, Tg=Tg, whyg=str(whyg), reached=reached, nexp=nexp, ta=ta, Tref=float(env.Tref[i]), Tr=Tv, why=str(why), ok=bool(ok), nseg=len(seq), t_ref=time.perf_counter() - t1)
def st_(x, n):
    x = np.array([v for v in x if v is not None]); x = x[x <= 3]
    return 'дошли %d/%d мед %.3f p90 %.3f max %.3f' % (len(x), n, np.median(x), np.percentile(x, 90), x.max()) if len(x) else 'дошли 0/%d' % n
if __name__ == '__main__':
    print('ПАСПОРТ astar_g6d env=%s NB=%d seed=%d бюджет A* %.1f с, пачка %d, дедуп-клетка угол .25/скорость .4, эвристика V̂ (LP-интерп.), f=g+h; CAIT=%s CADNEG=%s; проверяльщик common.verify' % (ENVN, GG.NB, GG.SEED, BUD, B, os.environ.get('CAIT', 60), os.environ.get('CADNEG', '1e-9')), flush=True)
    with mp.get_context('fork').Pool(int(os.environ.get('PROCS', 4))) as p: res = p.map(work, range(len(env.X0)), chunksize=1)
    n = len(res); Rg = [r['Tg'] / r['Tref'] if r.get('Tg') is not None else None for r in res]; Rr = [r['Tr'] / r['Tref'] if r.get('Tr') is not None else None for r in res]
    for r in res: print(r, flush=True)
    print('A* (до доводки): ' + st_(Rg, n) + '; viol box %d' % sum(1 for r in res if r['whyg'] == 'box')); print('A*+IPOPT: ' + st_(Rr, n) + '; viol box %d' % sum(1 for r in res if r.get('why') == 'box'))
    print('время: A* мед %.2f / макс %.2f с, доводка мед %.2f с, узлов раскрыто мед %d' % (np.median([r['ta'] for r in res]), max(r['ta'] for r in res), np.median([r.get('t_ref', 0) for r in res]), np.median([r['nexp'] for r in res])))
    json.dump(res, open(os.path.expanduser('~/spore_v5/d4/out/astar_%s_%d_%d%s.json' % (ENVN, GG.NB, GG.SEED, os.environ.get('TAG', ''))), 'w'), default=float)
