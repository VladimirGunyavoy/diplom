"""research-23: V*(старт)/эталон против T агента/эталон на 20 стартах manip (A_g50d1p1.pkl) — где теряется хвост: граф (V*) или агент."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
d_ = pickle.load(open('A_g50d1p1.pkl', 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); A.V = d_['V']
R = np.load(os.path.expanduser('~/spore_v5/r23/manip_ref_ocp_20.npy')) if os.path.exists(os.path.expanduser('~/spore_v5/r23/manip_ref_ocp_20.npy')) else np.load(os.path.expanduser('~/spore_v5/wb4/spores_2/v5chain/reports/research/r23/manip_ref_ocp_20.npy'))
Q, ref = R[:, :4], R[:, 4]; agent = np.array([1.02, 1.12, 1.15, 1.05, 1.01, 1.18, 1.03, 1.18, 1.11, 1.07, 1.05, 1.17, 1.06, 1.08, 1.04, 1.22, 1.01, 1.08, 1.03, 1.51])
vs = A.vstar(Q); I, IDX, W = A.stencils(Q); npair = np.bincount(I, minlength=len(Q))
for q in range(len(Q)): print('старт %2d | эталон %.3f | V*/эт %.3f | агент T/эт %.2f | стенсилов %d' % (q, ref[q], vs[q] / ref[q], agent[q], npair[q]), flush=True)
