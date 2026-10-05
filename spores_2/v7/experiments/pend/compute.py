"""«Трейн» для 2D систем на общем ядре v5chain/reports/research/grow_cells2d.py: считает и пишет данные в <эксперимент>/data/ (картинку рисует plot.py).
Запуск из этой папки: [SYS=pend] [TMAX=1.5] [OVL=.05] ... python3 compute.py 01_имя-эксперимента"""
import numpy as np, sys, os, json, time, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '../../../v5chain/reports/research'))
import grow_cells2d as G
tag = sys.argv[1]; D = os.path.join(HERE, tag, 'data'); os.makedirs(D, exist_ok=True); t0 = time.time()
def dump(name, obj, js=True):
    tmp = os.path.join(D, name + '.tmp'); (json.dump(obj, open(tmp, 'w')) if js else pickle.dump(obj, open(tmp, 'wb'))); os.replace(tmp, os.path.join(D, name))
PAR = dict(SYS=G.SYS, US=list(G.US), TMAX=G.TMAX, RMAX=G.RMAX, OVL=G.OVL, RHO=G.RHO, ADAPT=G.ADAPT, M=G.M, DTN=G.DTN, XL=G.XL, WL=G.WL, PER=G.PER)
def status(stage, **kw): dump('status.json', dict(stage=stage, sec=round(time.time() - t0, 1), params=PAR, **kw))
def cells_dump(cells): dump('cells.pkl', [dict(u=float(c.u), c=c.c, r=float(c.r), n=c.n, G=c.G.astype(np.float32)) for c in cells], js=False)
done = []
def log(u, cells):
    if len(cells) % 5 == 0: cells_dump(done + cells); status('строю атлас u = %+g' % u, cells=len(done) + len(cells))
for fn in ('cells.pkl', 'value.npz', 'agent.npz', 'paths.pkl'):
    if os.path.exists(os.path.join(D, fn)): os.remove(os.path.join(D, fn))
status('старт'); rng = np.random.default_rng(int(os.environ.get('SEED', 0))); A = G.Atlas.__new__(G.Atlas); A.layers = []; A.idx = []
for u in G.US: l, ix = G.build_layer(u, rng, log); A.layers.append(l); A.idx.append(ix); done += l; cells_dump(done)
A.finish(); con = [G.contact_stats(l, ix) for l, ix in zip(A.layers, A.idx)]; X = np.c_[rng.uniform(-G.XL, G.XL, 20000), rng.uniform(-G.WL, G.WL, 20000)]; cov = [round(float(ix.covered(X).mean()), 3) for ix in A.idx]
status('считаю цену V', cells=len(done), cells_by_layer=[len(l) for l in A.layers], nodes=int(A.N), contact=con, cover=cov); A.solve()
gx, gw = np.linspace(-G.XL, G.XL, 161), np.linspace(-G.WL, G.WL, 141); GX, GW = np.meshgrid(gx, gw); VV = A.vstar(np.c_[GX.ravel(), GW.ravel()]).reshape(GX.shape)
np.savez(os.path.join(D, 'value.tmp.npz'), gx=gx, gw=gw, V=VV); os.replace(os.path.join(D, 'value.tmp.npz'), os.path.join(D, 'value.npz'))
base = dict(cells=len(done), cells_by_layer=[len(l) for l in A.layers], nodes=int(A.N), iters=int(A.n_it), contact=con, cover=cov, big_nodes=round(float((A.V >= G.BIG / 2).mean()), 3)); status('агент', **base)
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
