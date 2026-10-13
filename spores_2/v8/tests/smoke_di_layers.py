"""ДИ офскрин: рост слоёв +1/−1/0 с env окна run.py, время и число клеток: python tests/smoke_di_layers.py [MAXC]"""
import os, sys, time
V8 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, V8)
os.environ.setdefault('V8_WINDOW', 'offscreen')
import re
src = open(os.path.join(V8, 'src/stepper/live.py'), encoding='utf-8').read(); m = re.search(r"for k, v in dict\((.*?)\)\.items\(\)", src, re.S).group(1)
for k, v in re.findall(r"(\w+)='([^']*)'", m): os.environ.setdefault(k, v)
if len(sys.argv) > 1: os.environ['MAXC'] = sys.argv[1]
import numpy as np
from src.algo import growN as g
print('M', g.M, 'MAXC', g.MAXC, 'US', g.ULAYERS, 'SEEDNEAR', g.SEEDNEAR, 'ZACEP', g.ZACEP, 'FMIN', g.FMIN, flush=True)
for k, u in enumerate(g.ULAYERS):
    t0 = time.time(); cl = g.build_layer(u, np.random.default_rng(k))[0]; print('u=%+g cells %d in %.1f s' % (u, len(cl), time.time() - t0), flush=True)
