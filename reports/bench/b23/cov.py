"""b23: покрытие дерева у покоя: dNN и V̂ состояний покоя (vr=vt=0) на сетке (ρ,α); деревья 12 против 11,13. runpc.sh cov.py NB trees..."""
import os, sys
os.environ.setdefault('FM','.1'); os.environ.setdefault('HS','.5')
nb_ = sys.argv[1]; trees = [int(x) for x in sys.argv[2:]]; sys.argv = ['x', nb_, '0.3', '8', 'sel', '0']
import numpy as np, canal_val as V
from canal_val import C, CT
rr, aa = np.meshgrid(np.linspace(.1, 1.05, 12), np.linspace(0, 2*np.pi, 24, endpoint=False), indexing='ij')
Q = np.column_stack([rr.ravel(), aa.ravel(), np.zeros(rr.size), np.zeros(rr.size)])
for ts in trees:
    S = CT.mkS(ts); back = V.build_back(S, C.TAU, int(nb_), .3); ag = V.AgentV(S, back, 8)
    d, j = ag.tree.query(S.wrap(Q) / S.scale, k=1); Vq = ag.V[j % ag.nP]
    print('TREE %d: покой dNN мед %.2f p90 %.2f max %.2f; V̂ покоя мед %.2f мин %.2f; узлов с v<.3: %d; узлов V<=-29 (тупик): %d' % (ts, np.median(d), np.percentile(d, 90), d.max(), np.median(Vq), Vq.min(), int((np.abs(np.asarray(back[0])[:, 2:]).max(1) < .3).sum()), int((ag.V <= -29).sum())), flush=True)
