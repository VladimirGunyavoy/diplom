"""BIG3 «грубо→мельче»: строки по времени. Старт — каждая S-я строка клетки (+ концы + строки с узлами цели); solve; невязка Беллмана в середине каждого интервала строк:
|V_интерп(x_mid) − min_u(DTN + V_интерп(step_u x_mid))| (по всем столбцам, max) > tol → строка добавляется (исходная строка клетки); P проходов (solve между ними). Без эталона полного атласа.
Запуск из bench/: BASE=pkl:<pkl>|all S=8 TOL=3e-3 PASSES=3 python3 ../v7/reports/big/02_big3/refine_big3.py"""
import sys, os, json, copy, pickle, time; sys.path.insert(0, os.path.join(os.getcwd(), '..', 'v7', 'reports', 'big'))
import big_lib as B, numpy as np
G = B.G
def make(A0, keeps):
    A2 = copy.copy(A0); A2.layers = []; ci = 0
    for l in A0.layers:
        nl = []
        for c in l:
            c2 = copy.copy(c); X = np.ascontiguousarray(c.G[sorted(keeps[ci])]); ci += 1
            c2.G = X; c2.bb = (X[..., 0].min() - 1e-6, X[..., 0].max() + 1e-6, X[..., 1].min() - 1e-6, X[..., 1].max() + 1e-6)
            c2.Q = [X[:-1, :-1].reshape(-1, 2), X[:-1, 1:].reshape(-1, 2), X[1:, 1:].reshape(-1, 2), X[1:, :-1].reshape(-1, 2)]; nl.append(c2)
        A2.layers.append(nl)
    if hasattr(A2, 'qx'): delattr(A2, 'qx')
    A2.finish(); return A2
def backup(A, Y):
    out = np.full(len(Y), np.inf)
    for u in G.US: out = np.minimum(out, np.minimum(A.tgoal(Y, u), G.DTN + A.vstar(G.step(Y, u))))
    return out
if __name__ == '__main__':
    base = os.environ.get('BASE', 'all'); S = int(os.environ.get('S', 8)); TOL = float(os.environ.get('TOL', 3e-3)); PASSES = int(os.environ.get('PASSES', 3)); tag = os.environ.get('TAG', 'x')
    if base.startswith('pkl:'): A0 = G.Atlas.__new__(G.Atlas); A0.layers = pickle.load(open(os.path.expanduser(base[4:]), 'rb'))['layers']; A0.finish()
    else: A0 = B.make_atlas()
    cells = A0.cells; keeps = []
    for c in cells:
        nt = c.G.shape[0]; k = set(range(0, nt, S)) | {nt - 1} | set(np.flatnonzero(G.ingoal(c.G).any(1)).tolist()); k |= {max(i - 1, 0) for i in k} | {min(i + 1, nt - 1) for i in k if i in set(np.flatnonzero(G.ingoal(c.G).any(1)))}; keeps.append(k)
    hist = []
    for p in range(PASSES + 1):
        A = make(A0, keeps); t0 = time.perf_counter(); ts = B.solve(A); s, _ = B.eval100(A)
        h = dict(pass_=p, nodes=int(A.N), t_solve=ts, **s); hist.append(h); print(h, flush=True)
        if p == PASSES: break
        add = 0
        for ci, c in enumerate(cells):
            kk = sorted(keeps[ci]); X = c.G; nt = X.shape[0]; mids = []
            for a, b in zip(kk[:-1], kk[1:]):
                if b - a >= 2: mids.append((a + b) // 2)
            if not mids: continue
            Y = X[mids].reshape(-1, 2); vi = A.vstar(Y); vb = backup(A, Y)
            res = np.where((vi >= G.BIG / 2) | (vb >= G.BIG / 2), np.inf, np.abs(vi - vb)).reshape(len(mids), -1).max(1)
            for m_, r_ in zip(mids, res):
                if r_ > TOL: keeps[ci].add(m_); add += 1
        print('pass', p, 'added rows', add, flush=True)
        if add == 0: break
    json.dump(dict(base=tag, S=S, tol=TOL, hist=hist), open(os.environ['BIGOUT'] + '/refine_%s_S%d_tol%g.json' % (tag, S, TOL), 'w'), indent=1, default=float)
