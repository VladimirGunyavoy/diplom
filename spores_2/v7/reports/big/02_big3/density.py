"""Плотность узлов и верхняя оценка выигрыша BIG3 на готовом атласе (pkl слоёв). Запуск из bench/: PKL=~/spore_v5/a31/big1p_vmin0.8_s0.pkl python3 ../v7/reports/big/02_big3/density.py
Выводит: узлов на единицу объёма клетки (медиана/по слоям), долю узлов в зонах двойного покрытия (узел лежит в ядре >1 клетки своего слоя), долю строк с |V_k − (V_{k−1}+V_{k+1})/2| < tol, столбцов — то же."""
import sys, os, json, pickle; sys.path.insert(0, os.path.join(os.getcwd(), '..', 'v7', 'reports', 'big'))
import big_lib as B, numpy as np
G = B.G; d = pickle.load(open(os.path.expanduser(os.environ['PKL']), 'rb')); A = B.make_atlas.__globals__['G'].Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); B.solve(A)
s100, _ = B.eval100(A); out = dict(cells=len(A.cells), nodes=int(A.N), s100=s100)
vol = np.array([B.cell_vol(c) for c in A.cells]); nodes = np.array([c.G.shape[0] * c.G.shape[1] for c in A.cells])
out['nodes_per_vol_med'] = float(np.median(nodes / vol)); out['nodes_per_vol_total'] = float(nodes.sum() / vol.sum()); out['m'] = sorted(set(int(c.m) for c in A.cells)); out['nrows_med'] = float(np.median([c.G.shape[0] for c in A.cells]))
dbl = 0
for l in A.layers:                                   # узел двойного покрытия: лежит в области (|боковая| ≤ r) ещё одной клетки того же слоя
    for ci, c in enumerate(l):
        X = c.G.reshape(-1, 2); n = np.zeros(len(X), int)
        for cj, c2 in enumerate(l):
            if cj == ci: continue
            for sh in (-2 * np.pi, 0., 2 * np.pi):
                y = X - np.array([sh, 0.]); q = np.flatnonzero((y[:, 0] >= c2.bb[0]) & (y[:, 0] <= c2.bb[1]) & (y[:, 1] >= c2.bb[2]) & (y[:, 1] <= c2.bb[3]))
                if len(q): n[q] += c2.incore(y[q])
        dbl += int((n > 0).sum())
out['nodes_double_cover_frac'] = dbl / A.N
for tol in (1e-3, 3e-3, 1e-2):
    fr = []; fc = []; nr_aff = 0; nrt = 0
    for c in A.cells:
        nt, m = c.G.shape[:2]; V = A.V[c.o:c.o + nt * m].reshape(nt, m); ok = (V < G.BIG / 2)
        if nt > 2: e = np.abs(V[1:-1] - (V[:-2] + V[2:]) / 2); good = ok[1:-1].all(1) & ok[:-2].all(1) & ok[2:].all(1) & (e.max(1) < tol); nr_aff += int(good.sum()); nrt += nt - 2
        if m > 2: e = np.abs(V[:, 1:-1] - (V[:, :-2] + V[:, 2:]) / 2); good = ok[:, 1:-1] & ok[:, :-2] & ok[:, 2:] & (e < tol); fc.append((int(good.sum()), V.shape[0] * (m - 2)))
    out['rows_affine_%g' % tol] = nr_aff / max(nrt, 1); out['cols_affine_%g' % tol] = sum(a for a, b in fc) / max(sum(b for a, b in fc), 1)
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(os.environ['BIGOUT'], 'density_%s.json' % os.path.basename(os.environ['PKL'])), 'w'), indent=1)
