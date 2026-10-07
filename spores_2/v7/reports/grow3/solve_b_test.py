"""w24 п.34: вёдра vs Якоби/ГЗ на готовом атласе grow3 (env как у прогона атласа: те же RS/GM/OBST...). Запуск: python3 solve_b_test.py atlas.pkl"""
import sys, os, time, pickle, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src/cells7')); import grow3 as G
import __main__
for n_ in dir(G):
    if n_[0].isupper() and isinstance(getattr(G, n_), type): setattr(__main__, n_, getattr(G, n_))
d = pickle.load(open(sys.argv[1], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = np.full(A.N, G.BIG); A.V[A.goal] = 0.; V0 = A.V.copy()
out = dict(cells=len(A.cells), nodes=int(A.N))
os.environ['GSN'] = os.environ.get('GSN', '64'); t = time.time(); A.solve_j(); out['gs'] = dict(t=round(time.time() - t, 1), it=int(A.n_it)); Vj = A.V.copy()
for tol in os.environ.get('SBTOLS', '1e-4,1e-6').split(','):
    for D in os.environ.get('SBDS', '%g' % (2 * G.DTN)).split(','):
        A.V = V0.copy(); t = time.time(); A.solve_bucket(float(D), float(tol)); fin = (Vj < G.BIG / 2) | (A.V < G.BIG / 2); dv = np.abs(A.V - Vj)[fin]
        out['b_tol%s_D%s' % (tol, D)] = dict(t=round(time.time() - t, 1), dV_max=float(dv.max()), dV_mean=float(dv.mean()), big_j=int((Vj >= G.BIG / 2).sum()), big_b=int((A.V >= G.BIG / 2).sum()), **A.solve_info)
print(json.dumps(out), flush=True)
