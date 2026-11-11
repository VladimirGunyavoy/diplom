"""BIG3: прореживание узлов в готовых клетках маятника по ошибке линейной интерполяции V и положения узла.
Клетка: G (nt, m, 2) — строки по времени × m столбцов поперёк. 1) столбцы: наибольший шаг k | (m−1), при котором удалённые столбцы восстанавливаются линейной интерполяцией
между оставленными (|ΔV| ≤ tol и |ΔG| ≤ tolg во всех строках); 2) строки: Дуглас–Пекер по индексу строки (концы остаются; строки с целью/BIG не удаляются). V — из решения полного атласа.
Затем solve заново на прореженном атласе, запросы. Запуск из bench/: BASE=all|K25 TOLS=1e-3,3e-3,1e-2 python3 ../v7/reports/big/02_big3/big3.py"""
import sys, os, json, copy; sys.path.insert(0, os.path.join(os.getcwd(), '..', 'v7', 'reports', 'big'))
import big_lib as B, numpy as np
G = B.G
def err_cols(V, X, k):
    """(nt, m), (nt, m, 2): ошибка восстановления удалённых столбцов при шаге k"""
    m = V.shape[1]; ev = 0.; eg = 0.; idx = np.arange(0, m, k)
    for a, b in zip(idx[:-1], idx[1:]):
        for j in range(a + 1, b):
            w = (j - a) / (b - a); Vi = (1 - w) * V[:, a] + w * V[:, b]; Xi = (1 - w) * X[:, a] + w * X[:, b]
            bad = (V[:, [a, j, b]] >= G.BIG / 2).any(1)
            if (bad & ~(V[:, [a, j, b]] >= G.BIG / 2).all(1)).any(): return np.inf, np.inf
            ev = max(ev, float(np.abs(V[~bad, j] - Vi[~bad]).max()) if (~bad).any() else 0.); eg = max(eg, float(np.abs(X[:, j] - Xi).max()))
    return ev, eg
def rows_dp(V, X, protect, tol, tolg):
    nt = V.shape[0]; keep = {0, nt - 1}; st = [(0, nt - 1)]
    while st:
        a, b = st.pop()
        if b - a < 2: continue
        w = ((np.arange(a + 1, b) - a) / (b - a))[:, None]; Vi = (1 - w) * V[a] + w * V[b]; Xi = (1 - w[..., None]) * X[a] + w[..., None] * X[b]
        bigs = (V[a:b + 1] >= G.BIG / 2).any(1); e = np.abs(V[a + 1:b] - Vi).max(1); e = np.where(bigs[1:-1] & ~(bigs[1:-1] & bigs[0] & bigs[-1]), np.inf, e); g = np.abs(X[a + 1:b] - Xi).max((1, 2))
        score = np.maximum(e / tol, g / tolg); score = np.where(protect[a + 1:b], np.inf, score); k = int(np.argmax(score))
        if score[k] > 1: keep.add(a + 1 + k); st += [(a, a + 1 + k), (a + 1 + k, b)]
    return sorted(keep)
def thin(A, tol, tolg=None, cols=True, rows=True):
    tolg = tol if tolg is None else tolg; A2 = copy.copy(A); A2.layers = []; stat = dict(nodes_before=int(A.N))
    for l in A.layers:
        nl = []
        for c in l:
            c2 = copy.copy(c); nt, m = c.G.shape[:2]; V = A.V[c.o:c.o + nt * m].reshape(nt, m); X = c.G; goal = G.ingoal(X)
            if cols and m > 3:
                best = 1
                for k in sorted([d for d in range(1, m) if (m - 1) % d == 0 and (m - 1) // d >= 2], reverse=True):
                    ev, eg = err_cols(V, X, k)
                    if ev <= tol and eg <= tolg: best = k; break
                if best > 1: V, X, goal = V[:, ::best], X[:, ::best], goal[:, ::best]; c2.sn = c.sn[::best]; c2.m = len(c2.sn)
            if rows and nt > 2:
                kr = rows_dp(V, X, goal.any(1), tol, tolg); V, X = V[kr], X[kr]
            c2.G = np.ascontiguousarray(X); c2.bb = (X[..., 0].min() - 1e-6, X[..., 0].max() + 1e-6, X[..., 1].min() - 1e-6, X[..., 1].max() + 1e-6)
            c2.Q = [X[:-1, :-1].reshape(-1, 2), X[:-1, 1:].reshape(-1, 2), X[1:, 1:].reshape(-1, 2), X[1:, :-1].reshape(-1, 2)]; nl.append(c2)
        A2.layers.append(nl)
    if hasattr(A2, 'qx'): delattr(A2, 'qx')
    A2.finish(); stat['nodes_after'] = int(A2.N); return A2, stat
if __name__ == '__main__':
    base = os.environ.get('BASE', 'all'); TOLG = float(os.environ.get('TOLG', 0)) or None
    if base.startswith('pkl:'):
        import pickle; A = G.Atlas.__new__(G.Atlas); A.layers = pickle.load(open(os.path.expanduser(base[4:]), 'rb'))['layers']; A.finish(); base = os.path.basename(base[4:]).replace('.pkl', '')
    else:
        A = B.make_atlas(); vol = np.array([B.cell_vol(c) for c in A.cells]); order = np.argsort(-vol)
        if base.startswith('K'): A = B.sub(A, order[:int(base[1:])])
    ts = B.solve(A); s0, _ = B.eval100(A); s20 = B.evalA(A); res = {'base': dict(cells=len(A.cells), nodes=int(A.N), t_solve=ts, **s0, ratio20_med=s20['ratio_med'])}; print(res, flush=True)
    for t in [float(x) for x in os.environ.get('TOLS', '1e-3,3e-3,1e-2').split(',')]:
        for mode, kw in (('cols+rows', {}), ('cols', dict(rows=False)), ('rows', dict(cols=False))):
            A2, st = thin(A, t, tolg=TOLG, **kw); ts2 = B.solve(A2); s, _ = B.eval100(A2); s2 = B.evalA(A2)
            res['%g_%s' % (t, mode)] = dict(tol=t, mode=mode, cells=len(A2.cells), nodes=st['nodes_after'], t_solve=ts2, **s, reached20=s2['reached'], ratio20_med=s2['ratio_med']); print(t, mode, res['%g_%s' % (t, mode)], flush=True)
            json.dump(res, open(os.environ['BIGOUT'] + '/big3_%s%s.json' % (base, os.environ.get('SFX', '')), 'w'), indent=1, default=float)
