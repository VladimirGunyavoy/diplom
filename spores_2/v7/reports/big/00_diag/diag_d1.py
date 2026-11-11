"""Д1 (multi_control_cell.md): доля узлов клетки, где собственный u даёт min в уравнении V (допуск 1e-6); «работает» >10%; попадание на пути 100 стартов; выброс неработающих."""
import sys, os, json; sys.path.insert(0, os.path.join(os.getcwd(), '..', 'v7', 'reports', 'big'))
import big_lib as B, numpy as np
G = B.G; A = B.make_atlas(); ts = B.solve(A); s0, path = B.eval100(A); print('база', s0, flush=True)
US = list(G.US); P = A.P; N = A.N
val = np.full((len(US), N), np.inf)
for k, u in enumerate(US):
    I, IDX, W = A.stencils(G.step(P, u)); v = G.DTN + A.interp(W, A.V[IDX]) if len(I) else np.zeros(0); np.minimum.at(val[k], I, v)
    val[k] = np.minimum(val[k], A.tgoal(P, u))
own = np.zeros(N, int); off = np.array([c.o for c in A.cells] + [N]); cu = np.array([US.index(c.u) for c in A.cells])
for ci, c in enumerate(A.cells):                                                  # точное ребро вдоль столбца собственной траектории (NFEDGE): учтено через V[eb] + dt — берём как доп. кандидат для своего u
    pass
nodeown = np.repeat(cu, np.diff(off)); mn = val.min(0); vown = val[nodeown, np.arange(N)]
work_node = (vown <= mn + 1e-6) & (A.V < G.BIG / 2) & ~A.goal
frac = np.array([work_node[off[i]:off[i + 1]].mean() for i in range(len(A.cells))]); 
P_ = path.reshape(-1, 2); I, IDX, W = A.stencils(P_); onpath = np.zeros(len(A.cells), bool); onpath[np.unique(np.searchsorted(off, IDX[:, 0], 'right') - 1)] = True
rows = []; tot = dict(cells=len(A.cells), nodes=N)
for k, u in enumerate(US):
    ci = np.flatnonzero(cu == k); w = ci[frac[ci] > .1]; rows.append(dict(u=u, cells=len(ci), working=len(w), onpath=int(onpath[ci].sum()), working_onpath=int(onpath[w].sum()), nodes_all=int(sum(off[i + 1] - off[i] for i in ci)), nodes_working=int(sum(off[i + 1] - off[i] for i in w))))
print(rows, flush=True); res = dict(base=s0, rows=rows, frac_hist=np.histogram(frac, bins=10, range=(0, 1))[0].tolist())
for name, keep in (('drop_nonworking', np.flatnonzero(frac > .1)), ('keep_working_or_onpath', np.flatnonzero((frac > .1) | onpath)), ('keep_onpath_only', np.flatnonzero(onpath))):
    A2 = B.sub(A, keep); t2 = B.solve(A2); s, _ = B.eval100(A2); res[name] = dict(cells=len(A2.cells), nodes=int(A2.N), t_solve=t2, **s); print(name, res[name], flush=True)
json.dump(res, open(os.environ['BIGOUT'] + '/pend_d1.json', 'w'), indent=1)
np.savez(os.environ['BIGOUT'] + '/pend_d1_cells.npz', frac=frac, onpath=onpath, cu=cu, vol=np.array([B.cell_vol(c) for c in A.cells]))
