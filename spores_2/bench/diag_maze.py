"""CA-MAZE диагностика провалов спор (шаблон r26 diag_stuck/hole_width): V*(старт) BIG? покрыт? где застрял (все u в стену / нет V у соседей); доля конечных V узлов по корзинам gfree и геодезического расстояния до цели.
aida: FORB2B=1 SRC_DIR=~/spore_v5/c1/maze_src SOLVEGPU=1 SONE=1 STGPU=1 LAYERS=~/spore_v5/c1/maze_layers_b.pkl python3 diag_maze.py  (solve под замком, V → maze_V_b.pkl)"""
import os, sys, json, pickle, fcntl, numpy as np
sys.argv = ['x']; os.environ.setdefault('XX', '1')
import spores_maze as SM          # окружение growN + подмена GF
import growN as G, maze, common
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
e = common.make('maze'); M = SM.M; A = G.Atlas.__new__(G.Atlas); LP = os.path.expanduser(os.environ['LAYERS']); VP = LP.replace('layers', 'V')
A.layers = pickle.load(open(LP, 'rb'))['layers']; A.finish()
if os.path.exists(VP): A.V = pickle.load(open(VP, 'rb')); A.n_it = 0; A.edges = 0
else:
    lk = open('/tmp/gpu.lock', 'w'); fcntl.flock(lk, fcntl.LOCK_EX); A.solve(); fcntl.flock(lk, fcntl.LOCK_UN); pickle.dump(A.V, open(VP, 'wb'))
fin = A.V < G.BIG / 2; print('узлов', A.N, 'конечных V', fin.mean().round(3), flush=True)
# геодезическое расстояние до цели по пикселям (8-связность, свободные gfree>0)
S = M.sdf.shape[0]; free = M.sdf > maze.RROB; idx = -np.ones((S, S), int); idx[free] = np.arange(free.sum()); r, c = np.nonzero(free); rows = []; cols = []; w = []
for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
    r2, c2 = r + di, c + dj; ok = (r2 < S) & (c2 >= 0) & (c2 < S); ok[ok] = free[r2[ok], c2[ok]]; rows += list(idx[r[ok], c[ok]]); cols += list(idx[r2[ok], c2[ok]]); w += [maze.PXM * np.hypot(di, dj)] * int(ok.sum())
Gm = csr_matrix((w, (rows, cols)), shape=(free.sum(), free.sum())); g0 = idx[int(round(M.G0[1] / maze.PXM - .5)), int(round(M.G0[0] / maze.PXM - .5))]; geo = np.full((S, S), np.inf); geo[free] = dijkstra(Gm, directed=False, indices=g0)
def geod(y): px = np.clip(np.round((y[:, 0] + M.G0[0]) / maze.PXM - .5).astype(int), 0, S - 1); py = np.clip(np.round((y[:, 1] + M.G0[1]) / maze.PXM - .5).astype(int), 0, S - 1); return geo[py, px]
P = G.wrapy(A.P); gf = M.gfree(P[:, :2]); gd = geod(P)
print('доля конечных V по gfree узла:', {'%.2f-%.2f' % (a, b): round(float(fin[(gf >= a) & (gf < b)].mean()), 3) for a, b in ((0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, 1))}, flush=True)
print('доля конечных V по геод. расстоянию до цели:', {'%d-%d' % (a, b): round(float(fin[(gd >= a) & (gd < b)].mean()), 3) for a, b in ((0, 2), (2, 4), (4, 6), (6, 8), (8, 12), (12, 30)) if ((gd >= a) & (gd < b)).any()}, flush=True)
Q = e.X0; vs = A.vstar(Q); res = json.load(open(os.path.join(common.DATA, 'maze_spores_b.json')))['per_start']; bad = [r['i'] for r in res if not r['T_verify']]
print('провалов', len(bad), 'из них V*(старт)=BIG:', int((vs[bad] >= G.BIG / 2).sum()), flush=True)
sg = geod(Q); print('геод. расстояние до цели: ok-старты', np.round(np.median(sg[[r['i'] for r in res if r['T_verify']]]), 2), 'провалы', np.round(np.median(sg[bad]), 2), flush=True)
T, sw, path = A.rollout(Q[bad][:12])
for j, i in enumerate(bad[:12]):
    Pp = path[:, j]; mv = np.flatnonzero(np.abs(np.diff(Pp, axis=0)).sum(1) > 0); last = (mv[-1] + 1) if len(mv) else 0; Y = Pp[last][None]; st = []
    for u in G.US:
        ok = bool(G.edge_ok(Y, u)[0]); yn = G.step(Y, u); v1 = A.vstar(yn)[0]; st.append('u=%s edge_ok=%s V*(y+)=%s' % (tuple(np.round(u, 1)), ok, 'BIG' if v1 >= G.BIG / 2 else round(float(v1), 2)))
    print('старт %d: V*(старт)=%s gfree=%.3f geod=%.2f; шагов %d, y_конец=%s gfree=%.3f V*(y)=%s | %s' % (i, 'BIG' if vs[i] >= G.BIG / 2 else round(float(vs[i]), 2), M.gfree(Q[i:i + 1, :2])[0], sg[i], last, np.round(Y[0], 2).tolist(), M.gfree(Y[:, :2])[0], 'BIG' if A.vstar(Y)[0] >= G.BIG / 2 else round(float(A.vstar(Y)[0]), 2), ' ; '.join(st)), flush=True)
