"""п.23 (b4): Гаусс–Зейдель в Atlas.solve (env GS=k) против Якоби — на готовом DUMP (layers + V от Якоби). python3 solve_gs_test.py DUMP.pkl; env как у прогона (SYS=manip ... GS=20)."""
import sys, os, time, pickle, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src/cells7'))
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]   # DUMP писался из __main__
d_ = pickle.load(open(sys.argv[1], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); Vref = d_['V']; A.V = np.full(A.N, G.BIG); A.V[A.goal] = 0.
t0 = time.time(); A.solve(); dt = time.time() - t0; V = A.V; f = (Vref < G.BIG / 2) & (V < G.BIG / 2)
print('nodes', A.N, 'edges', A.edges, 'sweeps', A.n_it, 'solve %.0f s' % dt, 'max|V-Vref| %.3g' % np.abs(V[f] - Vref[f]).max(), 'finite ref/new', int((Vref < G.BIG / 2).sum()), int((V < G.BIG / 2).sum()), 'mismatch finite', int(((Vref < G.BIG / 2) != (V < G.BIG / 2)).sum()), flush=True)
