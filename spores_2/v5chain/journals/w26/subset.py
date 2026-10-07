import os, sys, pickle, __main__
sys.path.insert(0, os.path.expanduser('~/spore_v5/w25/v7/src/cells7')); import growN as G
for n in dir(G):
    if n[0].isupper() and isinstance(getattr(G, n), type): setattr(__main__, n, getattr(G, n))
K = int(sys.argv[1]); d = pickle.load(open(os.path.expanduser('~/spore_v5/w25/dp1_L2000g300.pkl'), 'rb'))
d['layers'] = [l[:K] for l in d['layers']]; pickle.dump(d, open(os.path.expanduser('~/spore_v5/w25/dp1_L%dg300.pkl' % K), 'wb')); print('ok', [len(l) for l in d['layers']])
