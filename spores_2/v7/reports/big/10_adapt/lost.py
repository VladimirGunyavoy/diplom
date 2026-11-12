"""Разбор потерянных стартов ДИ4D по pkl (layers+V) из DUMP: per-start T, покрыт ли старт клетками атласа, V* в старте. Запуск: cd v7; TAGS=baseVF_s0,f1_s0 OUTD=… env как di4.sh; python3 reports/big/10_adapt/lost.py"""
import os, sys, json, pickle, numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'cells7')); G = __import__(os.environ.get('MOD', 'growNa'))
import __main__; [setattr(__main__, k, v) for k, v in vars(G).items() if isinstance(v, type)]
Q, ref = G.starts_ref(); res = {}
for tag in os.environ['TAGS'].split(','):
    d = pickle.load(open(os.path.join(os.environ['OUTD'], tag + '.pkl'), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']; A.n_it = 0; A.edges = 0
    A._init_qx(); cov = A.qx.covered(G.wrapy(Q)); vs = A.vstar(Q); T, sw, path = A.rollout(Q); last = np.array(path[-1]); vl = A.vstar(last)
    res[tag] = dict(T=[None if not np.isfinite(x) else float(x) for x in T], cov=[bool(c) for c in cov], vstar=[float(x) if x < G.BIG / 2 else None for x in vs], nodes=int(A.N), last=last.tolist(), vlast=[float(x) if x < G.BIG / 2 else None for x in vl]); print(tag, 'reach', int(np.isfinite(T).sum()), 'cov', int(cov.sum()), flush=True); del A, d
res['ref'] = [float(x) for x in ref]; res['Q'] = Q.tolist(); json.dump(res, open(os.path.join(os.environ['OUTD'], 'lost_s0.json'), 'w'))
