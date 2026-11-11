"""Двухпроходная ДИ 4D, шаг 0: карта argmin БАЗЫ (DTN .1 M 3, дамп V ~/spore_v5/r30/di4d1/base.pkl) на NPROBE случайных узлах (V<BIG, не цель) — argmin как в r30/di4/d1_di4.py.
Запуск из ~/spore_v5/b12/spores_2/v7 (env DTN=.1 M=3 + общий env di4): LOAD=… OUT=… python3 probes_base.py"""
import os, sys, pickle, time, numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'cells7')); import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
d = pickle.load(open(os.environ['LOAD'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']; A._init_qx(); US = list(G.US)
ok = np.flatnonzero((A.V < G.BIG / 2) & ~A.goal); rng = np.random.default_rng(1); ix = np.sort(rng.choice(ok, min(int(os.environ.get('NPROBE', 20000)), len(ok)), replace=False)); P = A.P[ix]; t0 = time.time()
J = np.stack([np.minimum(A.tgoal(P, u), G.DTN + A.vstar(G.step(P, u))) for u in US], 1); arg = J.argmin(1).astype(np.int8); print('probes', len(P), 'DTN', G.DTN, 't %.0f' % (time.time() - t0), 'доли argmin', np.bincount(arg) / len(arg), flush=True)
np.savez(os.path.join(os.environ['OUT'], 'probes_base.npz'), P=P, arg=arg, Jmin=J.min(1))
