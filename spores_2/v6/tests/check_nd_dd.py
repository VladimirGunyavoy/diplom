"""H1 дифдрайв 3D (ромб): replay_value против эталона min(TGT,TGTGT) (до точки цели). Окно R=.25, Rθ=.26. Запуск из v6."""
import sys, time, importlib.util; sys.path.insert(0, '.')
import numpy as np
from src.atlas6.adaptive_nd import *
from src.atlas6.dd3 import flow
from src.atlas6.dd_atlas import LAYERS
sp = importlib.util.spec_from_file_location('ref', '../v5chain/reports/research/dd_rhombus_ref.py'); ref = importlib.util.module_from_spec(sp); sp.loader.exec_module(ref)
tau = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25; R = .25; Rth = .26
fl = lambda P, s, t: flow(P, LAYERS[s], t)
dth = lambda th: (th + np.pi) % (2 * np.pi) - np.pi
ing = lambda P: (np.hypot(P[..., 0], P[..., 1]) < R) & (np.abs(dth(P[..., 2])) < Rth)
S = SysN(fl, 4, (1.0, 1.0, np.pi), ing, [(0.0, 0.0, 0.0)], per=(0, 0, 2 * np.pi), ok=lambda p: abs(p[0]) <= 3 and abs(p[1]) <= 3)
def eref(p):
    a = ref.tgt(*p); b = ref.tgtgt(*p); b = b[0] if isinstance(b, tuple) else b; return min(a, b)
rng = np.random.default_rng(3); Q = [np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)]) for _ in range(10)]
E = [eref(p) for p in Q]; print('эталон', np.round(E, 2), flush=True)
for rho in (0.1, 0.06):
  for NB, NF in ((300, 300), (1000, 1000)):
    t0 = time.time(); back = build_back(S, tau, NB, rho); r = []
    for x, e in zip(Q, E):
        V, nb, nf, path = replay_value(S, tau, x, NB, NF, rho, rho, back=back); r.append(V / e)
    print('rho %.2f NB %d NF %d: V/эталон' % (rho, NB, NF), np.round(r, 2), '(%.0f с)' % (time.time() - t0), flush=True)
