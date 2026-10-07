"""п.37 (b5): причинный граф solve (SBCAUS/SBLAY) на готовых слоях. LAYERS=L.pkl VREF=V.pkl env как у прогона слоёв; сравнение V с VREF (Якоби, все стенсилы)."""
import sys, os, time, pickle, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src/cells7'))
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
d_ = pickle.load(open(os.environ['LAYERS'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish()
vr = pickle.load(open(os.environ['VREF'], 'rb')) if os.environ.get('VREF') else None; Vref = (vr['V'] if isinstance(vr, dict) else vr) if vr is not None else d_.get('V')
t0 = time.time(); A.solve(); dt = time.time() - t0; V = A.V
if Vref is None: Vref = np.where(V < G.BIG / 2, V, G.BIG)   # нет эталона V (новые слои) — сравнение с собой
f = (Vref < G.BIG / 2) & (V < G.BIG / 2)
print('RESULT nodes', A.N, 'edges', A.edges, 'sweeps', A.n_it, 'solve %.0f s' % dt, 'max|V-Vref| %.3g' % np.abs(V[f] - Vref[f]).max(), 'mean|dV| %.3g' % np.abs(V[f] - Vref[f]).mean(), 'finite ref/new', int((Vref < G.BIG / 2).sum()), int((V < G.BIG / 2).sum()), flush=True)
if os.environ.get('SAVEV'): pickle.dump(dict(layers=A.layers, V=V), open(os.environ['SAVEV'], 'wb'))   # формат LOAD growN (rollout: LOAD=… DI4REF=…)
