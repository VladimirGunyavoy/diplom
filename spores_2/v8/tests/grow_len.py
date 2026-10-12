"""№10: длина клеток pend (1 спора на слой) при разных DELTA/RMAX/TMAX: python tests/grow_len.py  (env переопределяют). Печатает строки и длину центральной линии; причины остановок — grow.log-строки [stop]."""
import os, sys, time
V8 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, V8)
for k, v in dict(SYS='pend', M='5', KF='11', NFAIL='400', DELTA='.2', RMAX='.3', TMAX='8', TQDM_MI='1000', GS='0', GLIM='0', GOALB='0', GOALSHAPE='ball', GSEED='1', RHO='.1', UM='.3', COVTOL='.003', OVL='.05', GROW='2', OVH='.5', FRAC='.5', DTN='.06', NORM='gram', SEL='stop', STOPS='0', SOLVE8_PLUS='1', NORMFRONT='2', MAXC='1', V8_MAXCELLS='1').items(): os.environ.setdefault(k, v)
import numpy as np
from src.algo import growN as g
for k, u in enumerate([u for u in g.US if u != 0]):
    t0 = time.perf_counter(); res = g.build_layer(u, np.random.default_rng(k), seeds=None, only_queue=False)[0]; cl = [g.cell_dict(c) for c in res]
    for c in cl:
        G = np.asarray(c['G'], float); mid = G.shape[1] // 2; L = float(np.linalg.norm(np.diff(G[:, mid], axis=0), axis=1).sum()); W = float(np.linalg.norm(np.diff(G[G.shape[0] // 2], axis=0), axis=1).sum())
        print('u=%+g cell rows %d cols %d  centre-line length %.2f  width %.2f  (%.1f s) DELTA=%s RMAX=%s TMAX=%s' % (u, G.shape[0], G.shape[1], L, W, time.perf_counter() - t0, os.environ['DELTA'], os.environ['RMAX'], os.environ['TMAX']), flush=True)
