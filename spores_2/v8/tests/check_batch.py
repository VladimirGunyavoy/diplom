"""пачками = одним пакетом (побитно). python3 tests/check_batch.py [batch]"""
import os, sys, time
os.environ.setdefault('V8_MAXCELLS', '0'); V8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, V8)
from src.stepper import live
import numpy as np
from src.algo import growN as g
from src.stepper import vsurf as VS
cl = {u: [g.cell_dict(c) for c in g.build_layer(u, np.random.default_rng(k))[0]] for k, u in enumerate(live.LAYERS)}
Vs, info = VS.solve_layers(cl, calc='q'); G = info.pop('G'); B = int(sys.argv[1]) if len(sys.argv) > 1 else 25
t = time.perf_counter(); r1 = VS.agent_paths(G, cl=cl); t1 = time.perf_counter() - t; got = []
t = time.perf_counter(); r2 = VS.agent_paths(G, cl=cl, batch=B, on_batch=lambda p, d, n: got.append(d)); t2 = time.perf_counter() - t
def eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and (np.array_equal(a, b) if a.dtype == object or a.dtype.kind in 'US' else np.array_equal(a, b, equal_nan=True))
bad = [k for k in r1 if not (all(eq(a, b) for a, b in zip(r1[k], r2[k])) if isinstance(r1[k], list) else eq(r1[k], r2[k]))]
print('batch', B, 'batches', got[:3], '...', got[-1], 'one-shot %.1fs batched %.1fs' % (t1, t2), 'DIFF keys:', bad or 'none (bit-identical)')
