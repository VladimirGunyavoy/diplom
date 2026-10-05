"""«Трейн» для 2D систем на общем ядре v5chain/reports/research/grow_cells2d.py: считает и пишет данные в <эксперимент>/data/ (картинку рисует plot.py).
Запуск из этой папки: SYS=di [GROW=2 DELTA=.03 CUT=1 GOALB=1 ...] python3 compute.py NN_имя (копия pend/compute.py — общее ядро grow_cells2d.py)"""
import numpy as np, sys, os, json, time, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '../../../v5chain/reports/research'))
import grow_cells2d as G
tag = sys.argv[1]; D = os.path.join(HERE, tag, 'data'); os.makedirs(D, exist_ok=True); t0 = time.time()
def dump(name, obj, js=True):
    tmp = os.path.join(D, name + '.tmp'); (json.dump(obj, open(tmp, 'w')) if js else pickle.dump(obj, open(tmp, 'wb'))); os.replace(tmp, os.path.join(D, name))
PAR = dict(SYS=G.SYS, US=list(G.US), TMAX=G.TMAX, RMAX=G.RMAX, OVL=G.OVL, RHO=G.RHO, ADAPT=G.ADAPT, NORM=G.NORM, DELTA=G.DELTA, SEL=G.SEL, CORE=G.CORE, JUMP=G.JUMP, JMODE=G.JMODE, JAG=G.JAG, JDIR=G.JDIR, GROW=G.GROW, OVH=G.OVH, FRAC=G.FRAC, GOALB=int(os.environ.get('GOALB', 0)), M=G.M, DTN=G.DTN, XL=G.XL, WL=G.WL, PER=G.PER)
def status(stage, **kw): dump('status.json', dict(stage=stage, sec=round(time.time() - t0, 1), params=PAR, **kw))
def cells_dump(cells): dump('cells.pkl', [dict(u=float(c.u), c=c.c, r=float(c.r), n=c.n, G=c.G.astype(np.float32)) for c in cells], js=False)
done = []
def log(u, cells):
    if len(cells) % 5 == 0: cells_dump(done + cells); status('строю атлас u = %+g' % u, cells=len(done) + len(cells))
for fn in ('cells.pkl', 'value.npz', 'agent.npz', 'paths.pkl'):
    if os.path.exists(os.path.join(D, fn)): os.remove(os.path.join(D, fn))
CUT = float(os.environ.get('CUT', 0)); REFINE = int(os.environ.get('REFINE', 0)) or (1 if CUT > 0 else 0); RTOL = float(os.environ.get('RTOL', .05)); RFT = int(os.environ.get('RFT', 1)); HB = .1     # REFINE (research-14): проходы измельчения по невязке Беллмана
NX, NW = int(np.ceil(2 * G.XL / HB)), int(np.ceil(2 * G.WL / HB)); LEV = np.zeros((NX, NW), int)
def binof(Y): Y = np.atleast_2d(Y); return np.clip(((G.wrap(Y[:, 0]) + G.XL) / HB).astype(int), 0, NX - 1), np.clip(((Y[:, 1] + G.WL) / HB).astype(int), 0, NW - 1)
def limits(p, u): l = LEV[binof(p)][0]; return max(G.RMAX / 2 ** l, .02), (G.TMAX / 2 ** l if RFT else G.TMAX)
G.LIMITS = limits; hist = []
GOALB = int(os.environ.get('GOALB', 0)); GB = None                                           # GOALB=1 (research-14): край цели ±RHO — стена с самого начала (клетка не лежит поперёк края цели)
if GOALB:
    from scipy.spatial import cKDTree; e_ = np.linspace(-G.RHO, G.RHO, 41); GB = np.r_[np.c_[e_, e_ * 0 - G.RHO], np.c_[e_, e_ * 0 + G.RHO], np.c_[e_ * 0 - G.RHO, e_], np.c_[e_ * 0 + G.RHO, e_]]; G.BARRIER = cKDTree(GB)
for ps in range(REFINE + 1):
    status('старт' if not ps else 'проход %d' % ps, refine=hist); rng = np.random.default_rng(int(os.environ.get('SEED', 0))); A = G.Atlas.__new__(G.Atlas); A.layers = []; A.idx = []; done = []
    for u in G.US: l, ix = G.build_layer(u, rng, log); A.layers.append(l); A.idx.append(ix); done += l; cells_dump(done)
    if ps == REFINE: break
    A.finish(); A.solve(); Y, Vo = [], []
    if CUT > 0:                                                                                # CUT: середины пар соседних узлов (поперёк и вдоль) с перепадом V > CUT — барьер для прохода 1
        from scipy.spatial import cKDTree; DB = []
        for c in A.cells:
            g = c.G; v = A.V[c.o:c.o + len(g) * c.m].reshape(len(g), c.m)
            for a_, b_, va, vb in ((g[:, :-1], g[:, 1:], v[:, :-1], v[:, 1:]), (g[:-1], g[1:], v[:-1], v[1:])):
                k = (np.abs(va - vb) > CUT) & (va < G.BIG / 2) & (vb < G.BIG / 2); DB.append(((a_ + b_) / 2)[k])
        DB = np.concatenate(DB + ([GB] if GB is not None else [])); DB[:, 0] = G.wrap(DB[:, 0]); G.BARRIER = cKDTree(DB); hist.append(dict(cells=len(done), nodes=int(A.N), barrier_pts=len(DB))); print('проход', ps, hist[-1], flush=True); continue                                                       # невязка: V в центре четырёхугольника (среднее 4 узлов) против шага Беллмана из этой точки
    for c in A.cells:
        g = c.G; nt = len(g); v = A.V[c.o:c.o + nt * c.m].reshape(nt, c.m); Y.append(((g[:-1, :-1] + g[1:, :-1] + g[:-1, 1:] + g[1:, 1:]) / 4).reshape(-1, 2)); Vo.append(((v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:]) / 4).ravel())
    Y, Vo = np.concatenate(Y), np.concatenate(Vo); rhs = np.full(len(Y), np.inf)
    for u in G.US: tg = A.tgoal(Y, u); rhs = np.minimum(rhs, np.where(np.isfinite(tg), tg, G.DTN + A.vstar(G.step(Y, u))))
    ok = (Vo < G.BIG / 2) & (rhs < G.BIG / 2); res = np.abs(Vo - rhs); bad = ok & (res > RTOL); M_ = np.zeros_like(LEV, bool); M_[binof(Y[bad])] = True
    M_ = M_ | np.roll(M_, 1, 0) | np.roll(M_, -1, 0); M_ = M_ | np.roll(M_, 1, 1) | np.roll(M_, -1, 1); LEV[M_] += 1
    hist.append(dict(cells=len(done), nodes=int(A.N), res_med=round(float(np.median(res[ok])), 4), res_p95=round(float(np.quantile(res[ok], .95)), 4), bad=round(float(bad.mean()), 3), lev_max=int(LEV.max()), lev_area=round(float((LEV > 0).mean()), 3)))
    print('проход', ps, hist[-1], flush=True); np.save(os.path.join(D, 'lev.npy'), LEV)
A.finish(); con = [G.contact_stats(l, ix) for l, ix in zip(A.layers, A.idx)]; X = np.c_[rng.uniform(-G.XL, G.XL, 20000), rng.uniform(-G.WL, G.WL, 20000)]; cov = [round(float(ix.covered(X).mean()), 3) for ix in A.idx]
status('считаю цену V', cells=len(done), cells_by_layer=[len(l) for l in A.layers], nodes=int(A.N), contact=con, cover=cov); A.solve()
gx, gw = np.linspace(-G.XL, G.XL, 161), np.linspace(-G.WL, G.WL, 141); GX, GW = np.meshgrid(gx, gw); VV = A.vstar(np.c_[GX.ravel(), GW.ravel()]).reshape(GX.shape)
np.savez(os.path.join(D, 'value.tmp.npz'), gx=gx, gw=gw, V=VV); os.replace(os.path.join(D, 'value.tmp.npz'), os.path.join(D, 'value.npz'))
base = dict(refine=hist, cells=len(done), cells_by_layer=[len(l) for l in A.layers], nodes=int(A.N), iters=int(A.n_it), contact=con, cover=cov, big_nodes=round(float((A.V >= G.BIG / 2).mean()), 3)); status('агент', **base)
Q, ref = G.starts_ref(); ok = ref > .05; Q, ref = Q[ok], ref[ok]; T, sw, path = A.rollout(Q)
np.savez(os.path.join(D, 'agent.tmp.npz'), Q=Q, T=T, ref=ref, sw=sw); os.replace(os.path.join(D, 'agent.tmp.npz'), os.path.join(D, 'agent.npz'))
fz = np.isfinite(T); r = T[fz] / ref[fz]
if G.SYS == 'pend':                                                                           # пути для картинки: 7 стартов + их зеркала (−φ, −ω) — видно закрутку в обе стороны
    S0 = np.array([(2.2, 0.), (2.8, .8), (1.5, 1.2), (.8, -1.5), (2.5, -1.), (3.0, .3), (1.9, -.6)]); S = np.r_[S0, -S0]
else: S = np.array([[-1.3, -.6], [1.4, 1.0], [-.4, 1.3], [.9, -1.2], [-1.4, .9], [1.3, .6], [-1.4, -1.0], [.4, -1.3]])
T2, _, p2 = A.rollout(S); paths = []
for i in range(len(S)):
    pp = p2[:, i]; st_ = np.flatnonzero(np.all(np.abs(pp[1:] - pp[:-1]) < 1e-12, axis=1)); paths.append(dict(p=pp[:(st_[0] + 1 if len(st_) else len(pp))].astype(np.float32), T=float(T2[i]), ref=None))
dump('paths.pkl', paths, js=False)
status('готово', reach=round(float(fz.mean()), 3), T_mean=round(float(r.mean()), 4), T_med=round(float(np.median(r)), 4), T_max=round(float(r.max()), 3), T_min=round(float(r.min()), 3), sw_med=float(np.median(sw[fz])), **base)
print(open(os.path.join(D, 'status.json')).read())
