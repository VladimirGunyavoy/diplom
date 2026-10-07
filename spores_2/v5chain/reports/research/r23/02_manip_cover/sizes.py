"""research-23: размеры клеток manip c3000 — полуширины r (3 оси сечения), длина (nb+nf)·DTN, средняя |f| ⇒ объём ядра ≈ 2³Πr·L·|f|."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.expanduser('~/spore_v5/wb4/spores_2/v7/src/cells7'))
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
L = pickle.load(open(os.path.expanduser('~/spore_v5/wb4/manip_c3000_cor.pkl'), 'rb'))['layers']
for j, l in enumerate(L):
    R = np.array([np.sort(c.r)[::-1] for c in l]); T = np.array([(c.nb + c.nf) * G.DTN for c in l]); F = np.array([np.linalg.norm(G.f(c.c, c.u)) for c in l])
    vol = 8 * R.prod(1) * T * F; q = lambda a: np.round(np.percentile(a, [10, 50, 90]), 3).tolist()
    print('слой', j, 'r1', q(R[:, 0]), 'r2', q(R[:, 1]), 'r3', q(R[:, 2]), '| длина, с', q(T), '| |f|', q(F), '| объём', q(vol), 'сумма %.1f' % vol.sum(), '| доля r3 = RMIN: %.2f' % (R[:, 2] <= G.RMIN * 1.01).mean(), flush=True)
