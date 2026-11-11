import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'cells7')); G = __import__('growCar')
import __main__; [setattr(__main__, k, v) for k, v in vars(G).items() if isinstance(v, type)]
O = os.path.expanduser('~/spore_v5/a36/car/'); a = pickle.load(open(O + 'Vc_a.pkl', 'rb'))['V']; b = pickle.load(open(O + 'Vc_b.pkl', 'rb'))['V']; f = (a < 5e5) & (b < 5e5)
print('maxdV', float(np.abs(a - b)[f].max()), 'mean', float(np.abs(a - b)[f].mean()), 'BIG-расхождений', int(((a < 5e5) != (b < 5e5)).sum()), 'N', len(a))
