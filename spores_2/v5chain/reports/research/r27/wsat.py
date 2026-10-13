"""research-27: гипотеза о провале спор 8D на стартах 13/16 — оптимум идёт вдоль предела |w|=WM (дуга на ограничении),
а дерево ±1 с шагом tau=.4 такую дугу не представляет. По траектории OCP: доля времени у предела, доля внутренних u.
python wsat.py (aida, venv-ompl; читает bench из a27)"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); B = os.path.expanduser('~/spore_v5/a27/spores_2/bench'); sys.path.insert(0, B)
import numpy as np, common, envs_6d, ocp_m8_lib as L
from multiprocessing import get_context
try: from tqdm import tqdm
except ImportError: tqdm = lambda it, **kw: it          # в venv-ompl tqdm нет — прогресс строками print
e = common.make('m8'); n = 4; SOL = None
def stats(x0, U, T):
    N = len(U); hseg = T / N; k = 8; h = hseg / k; x = np.array(x0, float); W = []; f = lambda x, u: L.f_np(x, u, L.GG)
    for i in range(N):
        for _ in range(k):
            k1 = f(x, U[i]); k2 = f(x + h / 2 * k1, U[i]); k3 = f(x + h / 2 * k2, U[i]); k4 = f(x + h * k3, U[i]); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); W.append(np.abs(x[n:]).copy())
    W = np.array(W); sat = (W > 2.9).any(1).mean(); inner = (np.abs(U) < .95).mean(); sw = int((np.diff(np.sign(U), axis=0) != 0).sum())
    return dict(sat=float(sat), satj=(W > 2.9).mean(0).round(2).tolist(), wmax=W.max(0).round(2).tolist(), inner=float(inner), innerj=(np.abs(U) < .95).mean(0).round(2).tolist(), sw=sw)
def work(i):
    t0 = time.perf_counter(); best, res = L.solve_query(e.X0[i], 50, 4, 2, seed=i, solver=SOL, nearest=True)
    if best is None and i in (13,): best, res = L.solve_query(e.X0[i], 50, 4, 1, seed=i, solver=SOL, nearest=False)
    if best is None: return dict(i=i, T=None, t=time.perf_counter() - t0)
    T, U, br = best; d = stats(e.X0[i], U, T); d.update(i=i, T=float(T), t=time.perf_counter() - t0, w0=np.abs(e.X0[i][n:]).max()); return d
if __name__ == '__main__':
    SOL = L.make_solver(50, 4); R = []
    with get_context('fork').Pool(6) as p:
        for r in tqdm(p.imap_unordered(work, range(20), chunksize=1), total=20, desc='wsat', mininterval=10): R.append(r); print(r, flush=True)
    json.dump(sorted(R, key=lambda r: r['i']), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wsat.json'), 'w'), indent=1, default=float)
