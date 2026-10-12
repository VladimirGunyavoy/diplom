"""затравки на линии ω=0: рост одного слоя без окна, печать центров клеток по порядку: python tests/grow_line.py (env: SEEDLINE, MAXC)"""
import os, sys
V8 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, V8)
for k, v in dict(SYS='pend', M='5', KF='11', NFAIL='400', DELTA='.2', RMAX='.3', TMAX='8', TQDM_MI='1000', GS='0', GLIM='0', GOALB='0', GOALSHAPE='ball', GSEED='1', RHO='.1', UM='.3', COVTOL='.003', OVL='.05', GROW='2', OVH='.5', FRAC='.5', DTN='.06', NORM='gram', SEL='stop', STOPS='0', SOLVE8_PLUS='1', NORMFRONT='2', SEEDLINE='.2', MAXC='3').items(): os.environ.setdefault(k, v)
import numpy as np
from src.algo import growN as g
for k, u in enumerate([u for u in g.US if u != 0]):
    cl = [g.cell_dict(c) for c in g.build_layer(u, np.random.default_rng(k), seeds=None, only_queue=False)[0]]
    print('u=%+g cells %d:' % (u, len(cl)), [tuple(round(float(x), 2) for x in np.asarray(c['G'], float)[len(c['G']) // 2, np.asarray(c['G']).shape[1] // 2]) for c in cl], 'rows', [len(c['G']) for c in cl], flush=True)
