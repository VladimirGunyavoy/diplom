"""BIG п.2/(в): в работающих (Д1) и неработающих клетках базы 249 кл.: доля строк с |ΔV|<1e-3 (центральная вторая разность) и подгонка V(s), Φ(c+s·n,t) по s полиномом степени 1/2/3 (по строке, s — номер клона): доля строк с остатком ≤ tol (V — tol секунд, положение — tol)."""
import os, sys, json; sys.path.insert(0, os.path.join(os.getcwd(), '..', 'v7', 'reports', 'big'))
import big_lib as B, numpy as np
G = B.G; A = B.make_atlas(); B.solve(A); d = np.load(os.environ['BIGOUT'].replace('03_dens', '00_diag') + '/pend_d1_cells.npz'); frac = d['frac']; out = {}
for grp, sel in (('работающие', frac > .1), ('неработающие', frac <= .1)):
    for tol in (1e-3, 3e-3):
        row = dict(cells=0, nodes=0, rows=0, aff=0); poly = {k: [0, 0] for k in (1, 2, 3)}; cells_ok = {k: 0 for k in (1, 2, 3)}; big = 0
        for ci, c in enumerate(A.cells):
            if not sel[ci]: continue
            nt, m = c.G.shape[:2]; V = A.V[c.o:c.o + nt * m].reshape(nt, m); row['cells'] += 1; row['nodes'] += nt * m
            if nt > 2: ok = (V < G.BIG / 2); e = np.abs(V[1:-1] - (V[:-2] + V[2:]) / 2).max(1); good = ok[1:-1].all(1) & ok[:-2].all(1) & ok[2:].all(1) & (e < tol); row['rows'] += nt - 2; row['aff'] += int(good.sum())
            if m >= 9:
                s_ = np.linspace(-1, 1, m); big += nt * m
                for k in (1, 2, 3):
                    Vm = np.where(V < G.BIG / 2, V, 0.); Va = np.vander(s_, k + 1)
                    def res(Z): return np.abs(Va @ np.linalg.lstsq(Va, Z.T, rcond=None)[0] - Z.T).max(0)       # Z (nt, m) → остаток по каждой строке
                    rv = res(Vm); rg = np.maximum(res(c.G[..., 0]), res(c.G[..., 1]))
                    okr = ((rv <= tol) | (V.max(1) >= G.BIG / 2)) & (rg <= tol); poly[k][0] += int(okr.sum()); poly[k][1] += nt; cells_ok[k] += int(okr.mean() > .9)
        out['%s_tol%g' % (grp, tol)] = dict(cells=row['cells'], nodes=row['nodes'], rows_affine_V=row['aff'] / max(row['rows'], 1), cells_m_ge9_rowpoly_deg={k: dict(rows_ok=poly[k][0] / max(poly[k][1], 1), cells_90pct_ok=cells_ok[k]) for k in poly}, nodes_in_m_ge9=big); print(grp, tol, out['%s_tol%g' % (grp, tol)], flush=True)
json.dump(out, open(os.environ['BIGOUT'] + '/affine_work.json', 'w'), indent=1, default=float)
