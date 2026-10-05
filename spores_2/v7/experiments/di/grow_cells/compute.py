"""«Трейн»: считает и пишет данные в data/ по мере счёта (картинку рисует plot.py). ДИ, три отдельных атласа u ∈ {−1, 0, +1}, споры растут до
допустимого наложения (ядро алгоритма — v5chain/reports/research/di_grow_cells.py). Этапы: cells.json (растёт по ходу) → value.npz → agent.npz, paths.json.
Запуск из этой папки: [TMAX=.5] [RMAX=.1] [OV=.2] python3 compute.py [метка]   → data/<метка>/"""
import numpy as np, sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, '../../../../v5chain/reports/research'); sys.path.insert(0, R); os.chdir(R)
import di_grow_cells as G
from v7_faces_di import tstar_box
tag = sys.argv[1] if len(sys.argv) > 1 else 'run'; D = os.path.join(HERE, 'data', tag); os.makedirs(D, exist_ok=True); t0 = time.time()
def dump(name, obj):                                                                          # атомарно: plot.py читает параллельно
    tmp = os.path.join(D, name + '.tmp'); json.dump(obj, open(tmp, 'w')); os.replace(tmp, os.path.join(D, name))
def cells_json(cells): return [dict(c=[round(float(v), 4) for v in c.c], n=[round(float(v), 4) for v in c.n], r=round(float(c.r), 4), u=int(c.u), nb=int(c.nb), nf=int(c.nf)) for c in cells]
def status(stage, **kw): dump('status.json', dict(stage=stage, sec=round(time.time() - t0, 1), params=dict(TMAX=None if G.TMAX > 1e8 else G.TMAX, RMAX=G.RMAX, RHO=G.RHO, OVL=G.OVL, OV=G.OV, M=G.M, DTN=G.DTN), **kw))
done = []
def log(u, cells):
    if len(cells) % 3 == 0: dump('cells.json', cells_json(done + cells)); status('строю атлас u = %+d' % u, cells=len(done) + len(cells))
for f in ('cells.json', 'value.npz', 'agent.npz', 'paths.json'):
    if os.path.exists(os.path.join(D, f)): os.remove(os.path.join(D, f))
status('старт'); rng = np.random.default_rng(int(os.environ.get('SEED', 0))); layers = []
for u in G.US: l = G.build_layer(u, rng, log); layers.append(l); done += l; dump('cells.json', cells_json(done))
A = G.Atlas.__new__(G.Atlas); A.layers = layers; A.cells = done; A.off = []; N = 0; P = []
for c in A.cells: nd, sn, tt = c.nodes(); c.sn, c.tt, c.o = sn, tt, N; N += nd.shape[0] * G.M; P.append(nd.reshape(-1, 2))
A.P = np.concatenate(P); A.N = N; A.goal = G.ingoal(A.P); A.V = np.full(N, G.BIG); A.V[A.goal] = 0.
status('считаю цену V', cells=len(done), nodes=int(N)); A.solve()
g = np.linspace(-G.L, G.L, 121); GX, GV = np.meshgrid(g, g); VV = A.vstar(np.c_[GX.ravel(), GV.ravel()]).reshape(GX.shape); X = rng.uniform(-G.L, G.L, (20000, 2)); cov = [float(G.covered(l, X).mean()) for l in layers]
np.savez(os.path.join(D, 'value.tmp.npz'), g=g, V=VV, P=A.P, Vn=A.V); os.replace(os.path.join(D, 'value.tmp.npz'), os.path.join(D, 'value.npz'))
con = [G.contact_stats(l) for l in layers]
status('агент: 200 стартов', cells=len(done), nodes=int(N), contact=con, cover=[round(v, 3) for v in cov], big_nodes=round(float((A.V >= G.BIG / 2).mean()), 3))
Q = np.random.default_rng(1).uniform(-1.5, 1.5, (200, 2)); Ts = tstar_box(Q[:, 0], Q[:, 1], G.RHO); ok = Ts > .05; Q, Ts = Q[ok], Ts[ok]; T, sw, path = A.rollout(Q); Vs = A.vstar(Q)
np.savez(os.path.join(D, 'agent.tmp.npz'), Q=Q, T=T, Ts=Ts, sw=sw, Vs=Vs); os.replace(os.path.join(D, 'agent.tmp.npz'), os.path.join(D, 'agent.npz'))
S = np.array([[-1.3, -.6], [1.4, 1.0], [-.4, 1.3], [.9, -1.2], [-1.4, .9]]); T2, _, p2 = A.rollout(S); Ts2 = tstar_box(S[:, 0], S[:, 1], G.RHO)
dump('paths.json', [dict(p=p2[:, i].round(4).tolist(), T=None if not np.isfinite(T2[i]) else round(float(T2[i]), 3), Ts=round(float(Ts2[i]), 3)) for i in range(len(S))])
f = np.isfinite(T); r = T[f] / Ts[f]; dur = [(c.nb + c.nf) * G.DTN for c in done]
status('готово', cells=len(done), contact=con, cells_by_layer=[len(l) for l in layers], nodes=int(N), cover=[round(v, 3) for v in cov], big_nodes=round(float((A.V >= G.BIG / 2).mean()), 3), reach=round(float(f.mean()), 3),
       T_mean=round(float(r.mean()), 4) if f.any() else None, T_med=round(float(np.median(r)), 4) if f.any() else None, T_max=round(float(r.max()), 3) if f.any() else None, sw_med=float(np.median(sw[f])) if f.any() else None,
       dur_med=round(float(np.median(dur)), 2), dur_max=round(float(np.max(dur)), 2))
print(open(os.path.join(D, 'status.json')).read())
